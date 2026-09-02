from ultralytics import YOLO


class Detector:
    """
    Person 2 responsibility: pure detection.
    Loads the YOLO model and runs inference on frames.
    Returns detections in the standard team-agreed format.
    """

    def __init__(
        self,
        model_path="models/best.pt",
        confidence_threshold=0.5,
        iou_threshold=0.5,
        img_size=940,
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Lower than YOLO's default (0.7) to more aggressively merge
        # duplicate/overlapping boxes for the same object (NMS IoU).
        self.iou_threshold = iou_threshold

        # Inference resolution — lower = faster, but can miss small
        # objects (e.g. a small hardhat far from the camera).
        self.img_size = img_size

        # All class names, read dynamically — nothing hard-coded
        self.class_names = self.model.names
        self._recent_vests = []
        self._recent_persons = []

    def detect_frame(self, frame):
        """
        Runs YOLO on a single frame.
        Returns:
            detections: list of dicts (class, confidence, bbox)
            results: raw YOLO results (used for drawing boxes)
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False
        )

        detections = []

        for result in results:
            keep_indices = self._filter_conflicting_ppe_boxes(result.boxes)
            result.boxes = result.boxes[keep_indices]

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.class_names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                })

        raw_detections = detections
        detections = self._stabilize_vest_detections(raw_detections)

        if not any(self._normalise_class_name(d["class"]) in {"person", "person_id"} for d in detections):
            inferred_persons = self._infer_person_detections(raw_detections, frame.shape)
            detections.extend(self._stabilize_person_detections(inferred_persons))

        return detections, results

    @staticmethod
    def _normalise_class_name(class_name):
        return str(class_name).lower().replace("-", "_").replace(" ", "_")

    def _filter_conflicting_ppe_boxes(self, boxes):
        """Keep the stronger prediction when positive and negative PPE boxes overlap."""
        if len(boxes) < 2:
            return list(range(len(boxes)))

        names = [self._normalise_class_name(self.class_names[int(box.cls[0])]) for box in boxes]
        ppe_groups = {
            "helmet": {"hardhat", "no_hardhat", "no_hardhat_v2"},
            "vest": {"safety_vest", "no_safety_vest", "no_safety_vest_v2"},
        }
        boxes_xyxy = [box.xyxy[0].tolist() for box in boxes]
        suppressed = set()

        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                group = next(
                    (values for values in ppe_groups.values()
                     if names[first] in values and names[second] in values),
                    None,
                )
                if group is None:
                    continue

                a = boxes_xyxy[first]
                b = boxes_xyxy[second]
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
                area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                overlap = intersection / min(area_a, area_b) if min(area_a, area_b) else 0

                if overlap >= 0.5:
                    weaker = first if float(boxes[first].conf[0]) < float(boxes[second].conf[0]) else second
                    suppressed.add(weaker)

        return [index for index in range(len(boxes)) if index not in suppressed]

    def _infer_person_detections(self, detections, frame_shape):
        """Build approximate worker regions from best.pt PPE boxes when needed."""
        height, width = frame_shape[:2]
        worker_seeds = []
        vest_classes = {
            "safety_vest",
            "no_safety_vest",
            "no_safety_vest_v2",
        }

        for detection in detections:
            class_name = self._normalise_class_name(detection["class"])
            if class_name in vest_classes:
                x1, y1, x2, y2 = detection["bbox"]
                box_width = max(1, x2 - x1)
                box_height = max(1, y2 - y1)
                person_height = max(box_height * 2.0, height * 0.45)
                person_width = max(box_width * 1.6, person_height * 0.45)
                center_x = (x1 + x2) / 2
                person_box = [
                    max(0, int(center_x - person_width / 2)),
                    max(0, int(y1 - box_height * 0.35)),
                    min(width, int(center_x + person_width / 2)),
                    min(height, int(y1 + person_height)),
                ]
                worker_seeds.append((detection["confidence"], person_box))

        return [
            {
                "class": "Person",
                "confidence": min(0.5, float(confidence)),
                "bbox": person_box,
            }
            for confidence, person_box in worker_seeds
        ]

    def _stabilize_vest_detections(self, detections):
        positive_classes = {"safety_vest"}
        negative_classes = {"no_safety_vest", "no_safety_vest_v2"}
        filtered = []

        for detection in detections:
            class_name = self._normalise_class_name(detection["class"])
            if class_name in negative_classes and any(
                self._box_iou(detection["bbox"], previous["bbox"]) >= 0.2
                for previous in self._recent_vests
            ):
                continue
            filtered.append(detection)

        current_vests = [
            {"bbox": detection["bbox"], "age": 0}
            for detection in filtered
            if self._normalise_class_name(detection["class"]) in positive_classes
        ]
        for previous in self._recent_vests:
            if not any(self._box_iou(previous["bbox"], current["bbox"]) >= 0.2 for current in current_vests):
                previous["age"] += 1
                if previous["age"] < 5:
                    current_vests.append(previous)
        self._recent_vests = current_vests
        return filtered

    @staticmethod
    def _box_iou(first, second):
        intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
            0, min(first[3], second[3]) - max(first[1], second[1])
        )
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0

    def _stabilize_person_detections(self, detections):
        current = [{"bbox": detection["bbox"], "age": 0} for detection in detections]
        for previous in self._recent_persons:
            if not any(self._box_iou(previous["bbox"], item["bbox"]) >= 0.2 for item in current):
                previous["age"] += 1
                if previous["age"] < 5:
                    current.append(previous)
        self._recent_persons = current
        return [
            {
                "class": "Person",
                "confidence": 0.5,
                "bbox": item["bbox"],
            }
            for item in current
        ]