import os
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "results", "models")

os.makedirs(model_path, exist_ok=True)

# Load model 1 lần duy nhất khi import module
_model = YOLO(os.path.join(model_path, "yolov8n.pt"))

def run_object_detection(img_path):
    results = _model(img_path)
    return results[0]