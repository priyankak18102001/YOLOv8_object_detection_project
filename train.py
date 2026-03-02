
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="data.yaml",
    epochs=30,
    imgsz=256,
    batch=8
)
import torch
print(torch.__version__)
