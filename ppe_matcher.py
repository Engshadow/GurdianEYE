from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment


# ============================================================
# STATUS CONSTANTS
# ============================================================

SAFE = "SAFE"
NO_HELMET = "NO_HELMET"
NO_VEST = "NO_VEST"
CRITICAL = "CRITICAL_VIOLATION"
UNKNOWN = "UNKNOWN"


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def area(box):
    """Return the area of a bounding box."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection(box1, box2):
    """Return the intersection area of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box1, box2):
    """Return the Intersection over Union of two boxes."""
    intersection_area = intersection(box1, box2)
    union_area = area(box1) + area(box2) - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def containment(inner_box, outer_box):
    """
    Return the percentage of inner_box that is inside outer_box.

    This is more useful than IoU for PPE because helmets and vests
    are much smaller than the person bounding box.
    """
    inner_area = area(inner_box)

    if inner_area == 0:
        return 0.0

    return intersection(inner_box, outer_box) / inner_area


def get_center(box):
    """Return the center point of a bounding box."""
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def distance_between(box1, box2):
    """Return the Euclidean distance between two box centers."""
    x1, y1 = get_center(box1)
    x2, y2 = get_center(box2)

    return math.hypot(x1 - x2, y1 - y2)


# ============================================================
# STATUS CLASSIFICATION
# ============================================================

def classify_status(has_helmet, has_vest):
    """Determine a worker's PPE safety status."""

    if has_helmet and has_vest:
        return SAFE

    if not has_helmet and not has_vest:
        return CRITICAL

    if not has_helmet:
        return NO_HELMET

    return NO_VEST


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class MatcherConfig:
    """Configuration for PPE matching and temporal filtering."""

    # Detection classes
    person_classes: tuple = (
        "person",
        "person_id",
    )

    helmet_classes: tuple = (
        "helmet",
        "hardhat",
        "hard_hat",
        "hat",
    )

    vest_classes: tuple = (
        "vest",
        "safety_vest",
        "safety_vest_1",
    )

    # Confidence thresholds
    min_person_confidence: float = 0.45
    min_ppe_confidence: float = 0.30

    # PPE matching
    containment_threshold: float = 0.35
    max_distance_ratio: float = 0.8
    min_match_score: float = 0.35

    # Body regions
    head_region_ratio: float = 0.30
    torso_region: tuple = (0.25, 0.90)

    # Frame edges
    edge_margin_px: int = 5

    # Temporal stability
    stability_frames: int = 5
    temporal_filter_frames: int = 5
    max_filter_dropouts: int = 1

    # Tracking
    iou_id_threshold: float = 0.30


# ============================================================
# PPE MATCHER
# ============================================================

class PPEMatcher:

    def __init__(self, config=None):
        self.cfg = config or MatcherConfig()

        # Previous frame history used for PPE stability.
        self.history = []

        # Violation history for each person.
        self.violation_history = defaultdict(
            lambda: deque(
                maxlen=self.cfg.temporal_filter_frames
            )
        )

        # IDs for people without a tracker-provided ID.
        self._pseudo_counter = 0

        # Previous untracked people.
        self._last_untracked = []

    # --------------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------------

    def match(self, detections, frame_shape=None):
        """
        Match detected helmets and vests to people.

        Returns one result dictionary per detected person.
        """

        persons, helmets, vests = self.split_detections(detections)

        self._assign_stable_ids(persons)

        helmet_matches = self.assign_ppe(
            persons,
            helmets,
            region="head",
        )

        vest_matches = self.assign_ppe(
            persons,
            vests,
            region="torso",
        )

        results = []

        for person_index, person in enumerate(persons):
            has_helmet, helmet_data = self.resolve_ppe(
                person_index,
                helmet_matches,
                helmets,
            )

            has_vest, vest_data = self.resolve_ppe(
                person_index,
                vest_matches,
                vests,
            )

            truncated = False

            if frame_shape:
                truncated = self.is_truncated(
                    person["bbox"],
                    frame_shape,
                )

            status = classify_status(
                has_helmet,
                has_vest,
            )

            results.append({
                "person_id": person["uid"],
                "track_id": person.get("track_id"),
                "bbox": person["bbox"],
                "confidence": person["confidence"],
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "missing_ppe": self.get_missing_ppe(
                    has_helmet,
                    has_vest,
                ),
                "status": status,
                "helmet": helmet_data,
                "vest": vest_data,
                "truncated": truncated,
            })

        results = self.stabilize(results)
        results = self.apply_temporal_filter(results)

        return results

    # --------------------------------------------------------
    # PERSON ID HANDLING
    # --------------------------------------------------------

    def _assign_stable_ids(self, persons):
        """
        Assign a stable ID to every person.

        If a tracker already provides track_id, use it.
        Otherwise, match the person to the previous frame using IoU.
        """

        used_ids = set()
        next_untracked = []

        for person in persons:
            track_id = person.get("track_id")

            # Prefer the ID provided by the tracker.
            if track_id is not None:
                person["uid"] = track_id
                used_ids.add(track_id)
                continue

            # Try to match an untracked person to the previous frame.
            best_id = None
            best_overlap = self.cfg.iou_id_threshold

            for pseudo_id, previous_box in self._last_untracked:
                if pseudo_id in used_ids:
                    continue

                overlap = iou(previous_box, person["bbox"])

                if overlap > best_overlap:
                    best_id = pseudo_id
                    best_overlap = overlap

            # Create a new temporary ID if no previous match exists.
            if best_id is None:
                self._pseudo_counter += 1
                best_id = f"u{self._pseudo_counter}"

            person["uid"] = best_id
            used_ids.add(best_id)

            next_untracked.append(
                (best_id, person["bbox"])
            )

        self._last_untracked = next_untracked

    # --------------------------------------------------------
    # DETECTION SPLITTING
    # --------------------------------------------------------
    # Separate detections into people, helmets, and vests

    def split_detections(self, detections):
    
        persons = []
        helmets = []
        vests = []

        for detection in detections:
            class_name = self.normalize_class(
                detection.get("class", "")
            )

            confidence = float(
                detection.get("confidence", 0)
            )

            bbox = detection.get("bbox")

            if not bbox or len(bbox) != 4:
                continue

            if (
                class_name in self.cfg.person_classes
                and confidence >= self.cfg.min_person_confidence
            ):
                persons.append(detection)

            elif (
                class_name in self.cfg.helmet_classes
                and confidence >= self.cfg.min_ppe_confidence
            ):
                helmets.append(detection)

            elif (
                class_name in self.cfg.vest_classes
                and confidence >= self.cfg.min_ppe_confidence
            ):
                vests.append(detection)

        return persons, helmets, vests

    @staticmethod
    def normalize_class(class_name):
        """Normalize a detector class name."""
        return (
            str(class_name)
            .lower()
            .strip()
            .replace("-", "_")
            .replace(" ", "_")
        )

    # --------------------------------------------------------
    # PPE MATCHING
    # --------------------------------------------------------

    def assign_ppe(self, persons, ppe_items, region):
        
        # Match PPE detections to people using the Hungarian algorithm.
        

        if not persons or not ppe_items:
            return {}

        scores = [
            [
                self.calculate_score(
                    person["bbox"],
                    ppe["bbox"],
                    region,
                )
                for ppe in ppe_items
            ]
            for person in persons
        ]

        # Hungarian algorithm minimizes cost, so use negative scores.
        rows, columns = linear_sum_assignment(
            [
                [-score for score in row]
                for row in scores
            ]
        )

        return {
            row: column
            for row, column in zip(rows, columns)
            if scores[row][column] >= self.cfg.min_match_score
        }

    def calculate_score(self, person_box, ppe_box, region):
        """Calculate how likely a PPE box belongs to a person."""

        if area(ppe_box) <= 0:
            return 0.0

        # How much of the PPE box is inside the person box.
        containment_score = containment(
            ppe_box,
            person_box,
        )

        # Check whether the PPE center is inside the person.
        ppe_center = get_center(ppe_box)

        center_inside = (
            person_box[0] <= ppe_center[0] <= person_box[2]
            and person_box[1] <= ppe_center[1] <= person_box[3]
        )

        if (
            containment_score < self.cfg.containment_threshold
            and not center_inside
        ):
            return 0.0

        # Normalize distance based on person size.
        px1, py1, px2, py2 = person_box

        person_diagonal = math.hypot(
            px2 - px1,
            py2 - py1,
        )

        person_diagonal = max(person_diagonal, 1)

        distance_ratio = (
            distance_between(person_box, ppe_box)
            / person_diagonal
        )

        score = (
            0.65 * containment_score
            + 0.35 * max(
                0.0,
                1.0 - distance_ratio,
            )
        )

        # Add body-region and size bonuses.
        score += self.region_bonus(
            person_box,
            ppe_box,
            region,
        )

        score += self.scale_prior(
            person_box,
            ppe_box,
            region,
        )

        return min(score, 1.0)

    def scale_prior(self, person_box, ppe_box, region):
        """Reward PPE boxes with a reasonable size relative to the person."""

        person_width = max(
            1,
            person_box[2] - person_box[0],
        )

        ppe_width = max(
            1,
            ppe_box[2] - ppe_box[0],
        )

        width_ratio = ppe_width / person_width

        if region == "head":
            lower, upper = 0.10, 0.60
        else:
            lower, upper = 0.30, 0.95

        if lower <= width_ratio <= upper:
            return 0.10

        if width_ratio < lower:
            return -0.10 * min(
                1.0,
                (lower - width_ratio) / lower,
            )

        return -0.10 * min(
            1.0,
            (width_ratio - upper) / upper,
        )

    def region_bonus(self, person_box, ppe_box, region):
        """Reward PPE located in the expected body region."""

        _, person_top, _, person_bottom = person_box
        _, ppe_y = get_center(ppe_box)

        person_height = max(
            1,
            person_bottom - person_top,
        )

        relative_y = (
            ppe_y - person_top
        ) / person_height

        # Helmet should be near the top of the person.
        if region == "head":
            if relative_y <= 0.15:
                return 0.15

            if relative_y <= self.cfg.head_region_ratio:
                return 0.08

        # Vest should be around the torso.
        elif region == "torso":
            torso_start, torso_end = self.cfg.torso_region

            if torso_start <= relative_y <= torso_end:
                return 0.10

        return 0.0

    # --------------------------------------------------------
    # PPE RESULT HANDLING
    # --------------------------------------------------------

    @staticmethod
    def resolve_ppe(person_index, matches, ppe_items):
        """Return the PPE detection assigned to a person."""

        if person_index not in matches:
            return False, None

        ppe_index = matches[person_index]

        return True, ppe_items[ppe_index]

    # --------------------------------------------------------
    # TEMPORAL STABILITY
    # --------------------------------------------------------

    def stabilize(self, results):
        """
        Keep previously detected PPE for a few frames.

        This prevents a single missed helmet/vest detection from
        immediately becoming a violation.
        """

        for result in results:
            previous = self.find_previous(result)

            if (
                previous
                and previous["age"] < self.cfg.stability_frames
            ):
                # Keep the previous helmet if the current frame missed it.
                if (
                    previous["has_helmet"]
                    and not result["has_helmet"]
                ):
                    result["has_helmet"] = True
                    result["helmet"] = previous["helmet"]

                # Keep the previous vest if the current frame missed it.
                if (
                    previous["has_vest"]
                    and not result["has_vest"]
                ):
                    result["has_vest"] = True
                    result["vest"] = previous["vest"]

            self._update_status(result)

        self.update_history(results)

        return results

    @staticmethod
    def _update_status(result):
        """Update missing PPE and status after stabilization."""

        result["missing_ppe"] = PPEMatcher.get_missing_ppe(
            result["has_helmet"],
            result["has_vest"],
        )

        result["status"] = classify_status(
            result["has_helmet"],
            result["has_vest"],
        )

    def update_history(self, results):
        """Store current detections for future temporal stabilization."""

        new_history = []

        for result in results:
            previous = self.find_previous(result)

            new_history.append({
                "track_id": result.get("track_id"),
                "bbox": result["bbox"],
                "has_helmet": result["has_helmet"],
                "has_vest": result["has_vest"],
                "helmet": result["helmet"],
                "vest": result["vest"],
                "age": 0 if previous is None else 0,
            })

        # Keep people that disappeared for a few frames.
        for previous in self.history:
            still_visible = any(
                iou(
                    previous["bbox"],
                    result["bbox"],
                ) >= 0.2
                for result in results
            )

            if not still_visible:
                previous["age"] += 1

                if previous["age"] < self.cfg.stability_frames:
                    new_history.append(previous)

        self.history = new_history

    def find_previous(self, result):
        """Find the best matching person from the previous history."""

        track_id = result.get("track_id")

        # Prefer the tracker ID when available.
        if track_id is not None:
            for item in self.history:
                if item.get("track_id") == track_id:
                    return item

        # Otherwise use bounding-box IoU.
        bbox = result["bbox"]

        matches = [
            item
            for item in self.history
            if iou(item["bbox"], bbox) >= 0.2
        ]

        if not matches:
            return None

        return max(
            matches,
            key=lambda item: iou(
                item["bbox"],
                bbox,
            ),
        )

    # --------------------------------------------------------
    # TEMPORAL VIOLATION FILTER
    # --------------------------------------------------------

    def apply_temporal_filter(self, results):
        """
        Require a violation to persist across multiple frames.

        A temporary detection is changed to UNKNOWN instead of
        immediately being reported as a violation.
        """

        violation_states = {
            NO_HELMET,
            NO_VEST,
            CRITICAL,
        }

        active_ids = set()

        for result in results:
            person_id = result["person_id"]
            active_ids.add(person_id)

            history = self.violation_history[person_id]

            # A safe frame resets the violation history.
            if result["status"] == SAFE:
                history.clear()
                continue

            is_violation = result["status"] in violation_states

            history.append(is_violation)

            if not is_violation:
                continue

            enough_frames = (
                len(history)
                >= self.cfg.temporal_filter_frames
            )

            continuous_violation = (
                sum(history)
                >= len(history) - self.cfg.max_filter_dropouts
            )

            if not enough_frames or not continuous_violation:
                result["status"] = UNKNOWN

        # Remove people who are no longer visible.
        for person_id in list(self.violation_history):
            if person_id not in active_ids:
                del self.violation_history[person_id]

        return results

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    @staticmethod
    def get_missing_ppe(has_helmet, has_vest):
        """Return a list of missing PPE items."""

        missing = []

        if not has_helmet:
            missing.append("helmet")

        if not has_vest:
            missing.append("vest")

        return missing

    def is_truncated(self, box, frame_shape):
        """Return True if the person touches the frame boundary."""

        if not frame_shape:
            return False

        frame_height, frame_width = frame_shape[:2]
        margin = self.cfg.edge_margin_px

        x1, y1, x2, y2 = box

        return (
            x1 <= margin
            or y1 <= margin
            or x2 >= frame_width - margin
            or y2 >= frame_height - margin
        )


# ============================================================
# SUMMARY
# ============================================================

def summarize(results):
    """Return a summary of worker safety statuses."""

    return {
        "workers": len(results),

        "safe": sum(
            result["status"] == SAFE
            for result in results
        ),

        "violations": sum(
            result["status"] in {
                NO_HELMET,
                NO_VEST,
                CRITICAL,
            }
            for result in results
        ),

        "unknown": sum(
            result["status"] == UNKNOWN
            for result in results
        ),
    }


# ============================================================
# DETECTOR INTEGRATION
# ============================================================

try:
    from detection.detector import Detector

    detector = Detector(
        model_path="models/best.pt",
        confidence_threshold=0.5,
        iou_threshold=0.3,
        img_size=640,
    )

except Exception:
    detector = None


def get_detections(frame):
    """Run the configured detector on a frame."""

    if detector is None:
        return []

    detections, _ = detector.detect_frame(frame)

    return detections


def process_frame(frame, matcher=None):
    """Detect PPE and match it to workers in a frame."""

    if frame is None:
        return []

    if matcher is None:
        matcher = PPEMatcher()

    detections = get_detections(frame)

    return matcher.match(
        detections,
        frame_shape=frame.shape,
    )


# ============================================================
# ULTRALYTICS YOLO BRIDGE
# ============================================================

def yolo_result_to_detections(yolo_result):
    """Convert an Ultralytics YOLO result into our detection format."""

    detections = []
    class_names = yolo_result.names

    for box in yolo_result.boxes:
        class_id = int(box.cls[0])

        detections.append({
            "class": str(
                class_names[class_id]
            ).lower(),

            "confidence": float(
                box.conf[0]
            ),

            "bbox": [
                float(value)
                for value in box.xyxy[0].tolist()
            ],
        })

    return detections


# ============================================================
# PUBLIC API
# ============================================================

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
