import cv2
from ultralytics import YOLO

# Load PPE model
model = YOLO("models/best.pt")

# Open video
video_path = "videos/test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    # Run the PPE model directly
    results = model(
        frame,
        conf=0.1,
        verbose=False
    )

    # Print detections
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            print(
                f"Class: {class_name}, "
                f"Confidence: {confidence:.2f}, "
                f"BBox: [{x1}, {y1}, {x2}, {y2}]"
            )

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow(
        "Direct PPE Model Test",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()