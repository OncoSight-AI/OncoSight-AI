
from pathlib import Path
from ultralytics import YOLO
import torch

DATA = r"D:\OncoSightData\yolo_seg_50\data.yaml"
PROJECT = r"D:\OncoSightData\yolo_runs"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n-seg.pt")

    model.train(
        data=DATA,
        epochs=50,
        imgsz=512,
        batch=2,
        device=0,
        workers=2,
        project=PROJECT,
        name="lung_nodule_yolov8n_seg",
        exist_ok=False,
        pretrained=True,
        patience=10,
        amp=True,
        cache=False,
        plots=True,
        seed=42,
        deterministic=True,
    )

    print("\nTraining completed.")
    print("Results saved to:", PROJECT)

if __name__ == "__main__":
    main()
