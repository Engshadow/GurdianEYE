from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass


# =====================================================================
# 1) GEOMETRY HELPERS
# =====================================================================
# Return area of a box (x1, y1, x2, y2)
def box_area(box): 
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0

#I depend in containment_ratio cuz it is more accurate in this case cuz the helmet area is smaller than the person area, so the intersection area divided by the helmet area is more accurate than the intersection area divided by the person area.
def containment_ratio(inner, outer):
    a = box_area(inner)
    return intersection_area(inner, outer) / a if a > 0 else 0.0


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_distance(a, b):
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


# =====================================================================
# 2) SAFETY STATUS
# =====================================================================

SAFE = "SAFE"
NO_HELMET = "NO_HELMET"
NO_VEST = "NO_VEST"
CRITICAL = "CRITICAL_VIOLATION"
UNKNOWN = "UNKNOWN"


def classify_status(has_helmet: bool, has_vest: bool) -> str:
    """4 agreed safety classes."""
    if has_helmet and has_vest:
        return SAFE
    if not has_helmet and not has_vest:
        return CRITICAL
    if not has_helmet:
        return NO_HELMET
    return NO_VEST


# =====================================================================
# 3) CONFIG
# =====================================================================

@dataclass
class MatcherConfig:
    # Verified from the trained model: model.names shows
    # Hardhat, NO-Hardhat, Person, NO-Safety Vest, Safety Vest, etc.
    person_classes: tuple = ("person", "person_id")
    helmet_classes: tuple = ("helmet", "hardhat", "hard_hat")
    vest_classes: tuple = ("vest", "safety_vest", "safety_vest_1")
    no_helmet_classes: tuple = (
        "no_helmet",
        "no_hardhat",
        "no_hardhat_v2",
        "head",
    )
    no_vest_classes: tuple = (
        "no_vest",
        "no_safety_vest",
        "no_safety_vest_v2",
    )

    # Tuned to work with real  data 
    min_person_confidence: float = 0.4
    min_ppe_confidence: float = 0.4
    containment_threshold: float = 0.5
    max_distance_ratio: float = 0.8
    min_match_score: float = 0.35
    head_region_ratio: float = 0.5
    torso_region: tuple = (0.25, 0.9)
    edge_margin_px: int = 5
    stability_frames: int = 5
    temporal_filter_frames: int = 4


# =====================================================================
# 4) MATCHER
# =====================================================================

class PPEMatcher:

    def __init__(self, config: MatcherConfig | None = None):
        self.cfg = config or MatcherConfig()
        self._history = []
        self._violation_history = defaultdict(
            lambda: deque(maxlen=self.cfg.temporal_filter_frames)
        )

    def match(self, detections: list[dict], frame_shape=None) -> list[dict]:
        """Return per-person results in the team-agreed format."""
        persons, helmets, vests, no_helmets, no_vests = self._split(detections)

        helmet_owner = self._assign(persons, helmets, region="head")#we use assign to assign helmets to persons based on the head region and no duplicte assignmets 
        vest_owner = self._assign(persons, vests, region="torso")
        nohelmet_owner = self._assign(persons, no_helmets, region="head")
        novest_owner = self._assign(persons, no_vests, region="torso")

        results = []
        for i, p in enumerate(persons):
            has_helmet, helmet_ev = self._resolve(
                i, helmet_owner, helmets, nohelmet_owner, no_helmets
            )
            has_vest, vest_ev = self._resolve(
                i, vest_owner, vests, novest_owner, no_vests
            )

            truncated = self._is_truncated(p["bbox"], frame_shape) if frame_shape else False
            head_out_of_frame = truncated and p["bbox"][1] <= self.cfg.edge_margin_px

            if head_out_of_frame and not has_helmet:
                status = UNKNOWN
            else:
                status = classify_status(has_helmet, has_vest)

            missing = []
            if not has_helmet:
                missing.append("helmet")
            if not has_vest:
                missing.append("vest")

            results.append(
                {
                    "person_id": p.get("track_id", i),
                    "bbox": p["bbox"],
                    "confidence": p["confidence"],
                    "has_helmet": has_helmet,
                    "has_vest": has_vest,
                    "missing_ppe": missing,
                    "status": status,
                    "helmet": helmet_ev,
                    "vest": vest_ev,
                    "truncated": truncated,
                }
            )
        results = self._stabilize_results(results)
        return self._apply_temporal_filter(results)

    def _apply_temporal_filter(self, results):
        """Expose a violation only after four consecutive violating frames."""
        active_ids = set()
        violation_statuses = {NO_HELMET, NO_VEST, CRITICAL}

        for result in results:
            person_id = result["person_id"]
            active_ids.add(person_id)
            history = self._violation_history[person_id]
            is_violation = result["status"] in violation_statuses
            history.append(is_violation)

            if is_violation and (
                len(history) < self.cfg.temporal_filter_frames
                or not all(history)
            ):
                result["status"] = UNKNOWN

        for person_id in list(self._violation_history):
            if person_id not in active_ids:
                del self._violation_history[person_id]

        return results

    def _split(self, detections):
        persons, helmets, vests, no_helmets, no_vests = [], [], [], [], []

        for d in detections:
            cls = str(d.get("class", "")).lower().strip()
            cls = cls.replace("-", "_").replace(" ", "_")
            conf = float(d.get("confidence", 0.0))
            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            if cls in self.cfg.person_classes:
                if conf >= self.cfg.min_person_confidence:
                    persons.append(d)
            elif cls in self.cfg.helmet_classes:
                if conf >= self.cfg.min_ppe_confidence:
                    helmets.append(d)
            elif cls in self.cfg.vest_classes:
                if conf >= self.cfg.min_ppe_confidence:
                    vests.append(d)
            elif cls in self.cfg.no_helmet_classes:
                if conf >= self.cfg.min_ppe_confidence:
                    no_helmets.append(d)
            elif cls in self.cfg.no_vest_classes:
                if conf >= self.cfg.min_ppe_confidence:
                    no_vests.append(d)

        return persons, helmets, vests, no_helmets, no_vests

    def _score(self, person_box, ppe_box, region: str) -> float:
        if box_area(ppe_box) <= 0:
            return 0.0

        containment = containment_ratio(ppe_box, person_box)

        px1, py1, px2, py2 = person_box
        diag = math.hypot(px2 - px1, py2 - py1) or 1.0
        dist_ratio = center_distance(person_box, ppe_box) / diag

        if containment < self.cfg.containment_threshold and dist_ratio > self.cfg.max_distance_ratio: 
            #return 0 if the containment is less than the threshold and the distance ratio is greater than the max distance ratio

            return 0.0
        #the score is a weighted sum of the containment and the distance ratio, with containment being weighted more heavily cuz it is more important 
        score = 0.7 * containment + 0.3 * max(0.0, 1.0 - dist_ratio) #TUNED VALLUE


        #this part is to give a bonus score if the ppe is in the correct region of the person, for example if the helmet is in the head region or if the vest is in the torso region. This is to help with cases where the containment and distance ratio are not enough to determine if the ppe belongs to the person.
        _, cy = center(ppe_box)
        rel_y = (cy - py1) / max(1.0, (py2 - py1))
        if region == "head" and rel_y <= self.cfg.head_region_ratio:
            score += 0.10
        elif region == "torso" and self.cfg.torso_region[0] <= rel_y <= self.cfg.torso_region[1]:
            score += 0.10

        return min(score, 1.0)

    def _assign(self, persons, ppes, region: str) -> dict:
        candidates = []
        for pi, p in enumerate(persons):
            for di, d in enumerate(ppes):
                s = self._score(p["bbox"], d["bbox"], region)
                if s >= self.cfg.min_match_score:
                    candidates.append((s, pi, di))

        candidates.sort(key=lambda t: t[0], reverse=True)

        owner, used = {}, set()
        for _, pi, di in candidates:
            if pi in owner or di in used:
                continue
            owner[pi] = di
            used.add(di)
        return owner

    def _resolve(self, pi, pos_owner, pos_dets, neg_owner, neg_dets):
        pos = pos_dets[pos_owner[pi]] if pi in pos_owner else None
        neg = neg_dets[neg_owner[pi]] if pi in neg_owner else None

        if pos is not None and neg is not None:
            return True, pos
        if pos is not None:
            return True, pos
        if neg is not None:
            return False, neg
        return False, None

    def _stabilize_results(self, results):
        for result in results:
            previous = self._find_previous(result["bbox"])
            if previous is not None and previous["age"] < self.cfg.stability_frames:
                if previous["has_vest"] and not result["has_vest"]:
                    result["has_vest"] = True
                    result["vest"] = previous["vest"]
                if previous["has_helmet"] and not result["has_helmet"]:
                    result["has_helmet"] = True
                    result["helmet"] = previous["helmet"]

            result["missing_ppe"] = []
            if not result["has_helmet"]:
                result["missing_ppe"].append("helmet")
            if not result["has_vest"]:
                result["missing_ppe"].append("vest")
            result["status"] = classify_status(result["has_helmet"], result["has_vest"])

        updated_history = []
        for result in results:
            previous = self._find_previous(result["bbox"])
            updated_history.append(
                {
                    "bbox": result["bbox"],
                    "has_vest": result["has_vest"],
                    "has_helmet": result["has_helmet"],
                    "vest": result["vest"],
                    "helmet": result["helmet"],
                    "age": 0 if previous is None else previous["age"],
                }
            )

        for previous in self._history:
            if not any(iou(previous["bbox"], result["bbox"]) >= 0.2 for result in results):
                previous["age"] += 1
                if previous["age"] < self.cfg.stability_frames:
                    updated_history.append(previous)

        self._history = updated_history
        return results

    def _find_previous(self, bbox):
        matches = [item for item in self._history if iou(item["bbox"], bbox) >= 0.2]
        return max(matches, key=lambda item: iou(item["bbox"], bbox), default=None)

    def _is_truncated(self, box, frame_shape) -> bool:
        if not frame_shape or len(frame_shape) < 2:
            return False
        h, w = frame_shape[0], frame_shape[1]
        m = self.cfg.edge_margin_px
        x1, y1, x2, y2 = box
        return x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m


# =====================================================================
# 5) SUMMARY HELPER
# =====================================================================

def summarize(results: list[dict]) -> dict:
    return {
        "workers": len(results),
        "safe": sum(1 for r in results if r["status"] == SAFE),
        "violations": sum(1 for r in results if r["status"] in (NO_HELMET, NO_VEST, CRITICAL)),
        "unknown": sum(1 for r in results if r["status"] == UNKNOWN),
    }


# =====================================================================
# 6) INTEGRATION WITH PERSON 2
# =====================================================================

try:
    from detection.detector import Detector

    _DETECTOR = Detector(
        model_path="models/best.pt",
        confidence_threshold=0.5,
        iou_threshold=0.3,
        img_size=640,
    )
except Exception:
    _DETECTOR = None


def get_detections(frame) -> list[dict]:
    """Use the real Person 2 detector output from the agreed format."""
    if _DETECTOR is None:
        return []
    detections, _ = _DETECTOR.detect_frame(frame)
    return detections


def process_frame(frame, matcher: PPEMatcher | None = None) -> list[dict]:
    """One-stop helper for Person 3 to process a full frame."""
    if frame is None:
        return []
    current_matcher = matcher or PPEMatcher()
    detections = get_detections(frame)
    return current_matcher.match(detections, frame_shape=frame.shape)


def yolo_result_to_detections(yolo_result) -> list[dict]:
    """Optional bridge for direct Ultralytics outputs."""
    detections = []
    names = yolo_result.names
    for box in yolo_result.boxes:
        cls_id = int(box.cls[0])
        detections.append(
            {
                "class": str(names[cls_id]).lower(),
                "confidence": float(box.conf[0]),
                "bbox": [float(v) for v in box.xyxy[0].tolist()],
            }
        )
    return detections


__all__ = [
    "MatcherConfig",
    "PPEMatcher",
    "SAFE",
    "NO_HELMET",
    "NO_VEST",
    "CRITICAL",
    "UNKNOWN",
    "classify_status",
    "summarize",
    "get_detections",
    "process_frame",
    "yolo_result_to_detections",
]
