from pathlib import Path

import cv2
import pytest

from detection.detector import Detector


@pytest.mark.skipif(
    not Path("videos/test.mp4").exists(),
    reason="videos/test.mp4 is not available",
)
def test_detector_reads_first_video_frame():
    detector = Detector(
        model_path="models/best.pt",
        confidence_threshold=0.5,
    )

    cap = cv2.VideoCapture("videos/test.mp4")
    try:
        assert cap.isOpened(), "Could not open videos/test.mp4"
        ret, frame = cap.read()
        assert ret, "Could not read the first video frame"

        detections, _ = detector.detect_frame(frame)

        assert isinstance(detections, list)
    finally:
        cap.release()