import cv2

from detection.detector import Detector


# Create detector
detector = Detector(
    model_path="models/best.pt",
    confidence_threshold=0.5
)


# Open video
cap = cv2.VideoCapture("videos/test.mp4")


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# Read ONE frame only
ret, frame = cap.read()


if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()


# Run detection
detections, results = detector.detect_frame(frame)


# Print detections
print("\nDetections in first frame:\n")

for detection in detections:
    print(detection)


# Release
cap.release()