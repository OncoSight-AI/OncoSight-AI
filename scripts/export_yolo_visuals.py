
from pathlib import Path
import shutil
import cv2
from ultralytics import YOLO

# Existing trained model and test dataset
MODEL = Path(r"D:\OncoSightData\yolo_runs\lung_nodule_yolov8n_seg\weights\best.pt")
TEST_IMAGES = Path(r"D:\OncoSightData\yolo_seg_50\test\images")
TEST_LABELS = Path(r"D:\OncoSightData\yolo_seg_50\test\labels")
EVAL_DIR = Path(r"D:\OncoSightData\yolo_runs\lung_nodule_test_evaluation")

# GitHub report folders
REPORT = Path("reports/yolo_test_evaluation")
GRAPHS = REPORT / "graphs"
PREDICTIONS = REPORT / "predictions"

def main():
    GRAPHS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    print("Collecting evaluation graphs...")

    graph_count = 0
    for file in EVAL_DIR.rglob("*"):
        if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            destination = GRAPHS / file.name
            if not destination.exists():
                shutil.copy2(file, destination)
                graph_count += 1

    print(f"Graphs copied: {graph_count}")

    print("Selecting test images...")

    positive = []
    negative = []

    for image in sorted(TEST_IMAGES.glob("*")):
        if image.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        label = TEST_LABELS / (image.stem + ".txt")

        if label.exists() and label.stat().st_size > 0:
            positive.append(image)
        else:
            negative.append(image)

    selected = positive[:8] + negative[:4]

    if not selected:
        print("No test images found. Check the dataset paths.")
        return

    print(f"Selected {len(positive[:8])} positive and {len(negative[:4])} negative images.")

    print("Loading trained YOLO model...")
    model = YOLO(str(MODEL))

    print("Generating YOLO prediction images...")

    results = model.predict(
        source=[str(p) for p in selected],
        imgsz=512,
        conf=0.25,
        device=0,
        verbose=False,
        stream=False
    )

    for index, result in enumerate(results, start=1):
        annotated = result.plot()
        output = PREDICTIONS / f"test_prediction_{index:02d}.jpg"
        cv2.imwrite(str(output), annotated)
        print(f"Saved: {output}")

    print("\nEXPORT COMPLETED")
    print(f"Graphs: {GRAPHS.resolve()}")
    print(f"Predictions: {PREDICTIONS.resolve()}")

if __name__ == "__main__":
    main()
