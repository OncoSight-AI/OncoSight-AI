
from pathlib import Path
import cv2

ROOT = Path(r"D:\OncoSightData\yolo_seg_50")

def validate_split(split):
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    total = 0
    empty = 0
    invalid = 0
    unreadable = 0

    for image_path in image_dir.glob("*.png"):
        total += 1
        label_path = label_dir / (image_path.stem + ".txt")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            unreadable += 1
            continue

        if not label_path.exists():
            invalid += 1
            continue

        content = label_path.read_text(encoding="utf-8").strip()

        if not content:
            empty += 1
            continue

        for line in content.splitlines():
            values = line.split()

            try:
                numbers = [float(x) for x in values]
            except ValueError:
                invalid += 1
                break

            if len(numbers) < 7 or (len(numbers) - 1) % 2 != 0:
                invalid += 1
                break

            if numbers[0] != 0:
                invalid += 1
                break

            coords = numbers[1:]

            if any(x < 0 or x > 1 for x in coords):
                invalid += 1
                break

    print(f"\n{split.upper()}")
    print(f"Total images: {total}")
    print(f"Empty labels (negative slices): {empty}")
    print(f"Invalid labels: {invalid}")
    print(f"Unreadable images: {unreadable}")

for split in ("train", "val", "test"):
    validate_split(split)
