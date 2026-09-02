from ultralytics import YOLO

model = YOLO("models/best.pt")

print("Model classes:")

for class_id, class_name in model.names.items():
    print(class_id, "->", class_name)