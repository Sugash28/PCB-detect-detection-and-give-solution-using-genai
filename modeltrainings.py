from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the Nano model, you can choose 'yolov8s.pt', 'yolov8m.pt', etc.

# Train the model
model.train(data='data.yaml', epochs=10, imgsz=640, batch=16)


