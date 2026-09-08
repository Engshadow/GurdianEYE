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

    def detect_frame(self, frame):
        """
        Runs YOLO on a single frame.
        Returns:
            detections: list of dicts (class, confidence, bbox)
            results: raw YOLO results (used for drawing boxes)
        """
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            persist=True,
            tracker="bytetrack.yaml",
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

                detection = {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
                if box.id is not None:
                    detection["track_id"] = int(box.id[0])
                detections.append(detection)

        return detections, results

    @staticmethod
    def _normalise_class_name(class_name):
        return str(class_name).lower().replace("-", "_").replace(" ", "_")

    def _filter_conflicting_ppe_boxes(self, boxes):
        """Keep the stronger prediction when overlapping PPE boxes conflict."""
        if len(boxes) < 2:
            return list(range(len(boxes)))

        names = [self._normalise_class_name(self.class_names[int(box.cls[0])]) for box in boxes]
        ppe_groups = {
            "helmet": {"helmet", "hardhat", "hard_hat", "hat"},
            "vest": {"vest", "safety_vest", "safety_vest_1"},
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

    @staticmethod
    def _box_iou(first, second):
        intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
            0, min(first[3], second[3]) - max(first[1], second[1])
        )
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0

