
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"D:\OncoSightData\yolo_seg_50")
OUTPUT = Path(r"D:\OncoSightData\yolo_previews")
OUTPUT.mkdir(parents=True, exist_ok=True)

samples = {
    "train": [
        "LIDC-IDRI-0001_9720603192_slice_0086",
        "LIDC-IDRI-0001_9720603192_slice_0000",
    ],
    "val": [
        "LIDC-IDRI-0013_2640380786_slice_0082",
        "LIDC-IDRI-0013_2640380786_slice_0000",
    ],
    "test": [
        "LIDC-IDRI-0011_6468423138_slice_0027",
        "LIDC-IDRI-0011_6468423138_slice_0000",
    ],
}

for split, filenames in samples.items():
    for name in filenames:
        image_path = ROOT / split / "images" / f"{name}.png"
        label_path = ROOT / split / "labels" / f"{name}.txt"

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None or not label_path.exists():
            print("Missing image or label:", name)
            continue

        height, width = image.shape
        preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        content = label_path.read_text().strip()

        if content:
            for line in content.splitlines():
                values = list(map(float, line.split()))
                coords = np.array(values[1:]).reshape(-1, 2)

                coords[:, 0] *= width
                coords[:, 1] *= height

                polygon = np.round(coords).astype(np.int32)
                cv2.polylines(
                    preview, [polygon], True, (0, 255, 0), 2
                )

        status = "POSITIVE" if content else "NEGATIVE"

        cv2.putText(
            preview, f"{split.upper()} - {status}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )

        output_path = OUTPUT / f"{split}_{status}_{name}.png"
        cv2.imwrite(str(output_path), preview)

        print("Saved:", output_path)

print("\nPreview generation completed.")
