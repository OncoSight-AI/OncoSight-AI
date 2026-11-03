
from ultralytics import YOLO
import multiprocessing

MODEL = r"D:\OncoSightData\yolo_runs\lung_nodule_yolov8n_seg\weights\best.pt"
DATA = r"D:\OncoSightData\yolo_seg_50\data.yaml"
PROJECT = r"D:\OncoSightData\yolo_runs"

def main():
    print("Loading best YOLO model...")

    model = YOLO(MODEL)

    print("Starting evaluation on the TEST split...")

    results = model.val(
        data=DATA,
        split="test",
        imgsz=512,
        batch=2,
        device=0,
        workers=0,
        plots=True,
        project=PROJECT,
        name="lung_nodule_test_evaluation",
        exist_ok=True
    )

    print("\nTEST SET EVALUATION COMPLETED")
    print(results.results_dict)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
