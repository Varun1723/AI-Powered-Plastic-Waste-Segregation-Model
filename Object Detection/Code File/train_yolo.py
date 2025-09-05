# train_yolo.py

from ultralytics import YOLO
import torch
import os

def main():
    # Confirm GPU
    if not torch.cuda.is_available():
        raise SystemError("CUDA GPU not available.")
    print(f"🟢 Using GPU: {torch.cuda.get_device_name(0)}")

    # Resolve the path to your data.yaml
    # Option A: Relative to this script file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "..", "data.yaml")  # goes up one level

    # Option B: Absolute path (hardcode your actual path)
    # yaml_path = r"C:\Varun\Coding\Waste_Detection\YOLO-Waste-Detection-1\data.yaml"

    print(f"Loading dataset config from: {yaml_path}")

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data=yaml_path,
        epochs=15,
        imgsz=640,
        batch=8,
        device=0,
        project="waste_detection",
        name="baseline_train",
        save=True
    )

    # Validate
    results = model.val()
    print("✅ Validation metrics:", results)

if __name__ == "__main__":
    main()
