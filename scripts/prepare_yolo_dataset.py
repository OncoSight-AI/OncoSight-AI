
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

SOURCE = Path(r"D:\OncoSightData\processed\lung_ct_50")
SPLIT_FILE = Path("datasets/manifests/splits_50.json")
OUTPUT = Path(r"D:\OncoSightData\yolo_seg_50")

NEGATIVE_RATIO = 3
SEED = 42
PROGRESS_EVERY = 25

# =========================================================


def make_labels(mask_slice):
    """Convert a binary mask into YOLO segmentation polygons."""

    binary = (mask_slice > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = binary.shape
    labels = []
    skipped = 0

    for contour in contours:

        polygon = cv2.approxPolyDP(
            contour, 0.5, True
        )

        points = polygon.reshape(-1, 2)

        if len(points) < 3:
            skipped += 1
            continue

        coordinates = []

        for x, y in points:
            coordinates.extend([
                np.clip(x / width, 0, 1),
                np.clip(y / height, 0, 1)
            ])

        labels.append(
            "0 " + " ".join(
                f"{value:.6f}" for value in coordinates
            )
        )

    return labels, skipped


def normalize_volume(image):
    """Convert the volume to uint8 for PNG export."""

    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)

    finite = image[np.isfinite(image)]

    if finite.size == 0:
        raise ValueError("No finite image values found")

    low, high = np.percentile(finite, [1, 99])

    if high <= low:
        low = float(finite.min())
        high = float(finite.max())

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    image = np.clip(
        (image - low) / (high - low),
        0,
        1
    )

    image[~np.isfinite(image)] = 0

    return (image * 255).round().astype(np.uint8)


def save_image(path, image):
    """Write PNG safely using a temporary file."""

    temporary = path.with_name(
        path.stem + ".tmp.png"
    )

    if not cv2.imwrite(str(temporary), image):
        raise IOError(f"Image write failed: {path}")

    temporary.replace(path)


def save_label(path, labels):
    """Write YOLO label safely."""

    temporary = path.with_name(
        path.stem + ".tmp.txt"
    )

    content = "\n".join(labels)

    if content:
        content += "\n"

    temporary.write_text(
        content,
        encoding="utf-8"
    )

    temporary.replace(path)


def pair_is_complete(image_path, label_path):

    if not image_path.is_file():
        return False

    if not label_path.is_file():
        return False

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    return image is not None


def prepare_case(
    record,
    split_name,
    image_dir,
    label_dir,
    totals
):

    case_id = record["case_id"]

    npz_path = SOURCE / f"{case_id}.npz"

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    print(f"Loading {case_id}...", flush=True)

    started = time.time()

    with np.load(
        npz_path,
        allow_pickle=False
    ) as data:

        image = data["image"]
        mask = data["mask"]
        axis = int(data["slice_axis"])

        if image.ndim != 3:
            raise ValueError(
                f"{case_id}: image must be 3D"
            )

        if image.shape != mask.shape:
            raise ValueError(
                f"{case_id}: image/mask shape mismatch"
            )

        if axis not in (0, 1, 2):
            raise ValueError(
                f"{case_id}: invalid slice axis"
            )

        # Move slice axis to the first dimension.
        image = np.moveaxis(image, axis, 0)
        mask = np.moveaxis(mask, axis, 0)

        image = normalize_volume(image)

        positive = []
        negative = []

        for index in range(mask.shape[0]):

            if np.any(mask[index] > 0):
                positive.append(index)
            else:
                negative.append(index)

        # Select negative slices.
        if split_name == "train":

            rng = random.Random(
                f"{SEED}-{case_id}-{split_name}"
            )

            limit = NEGATIVE_RATIO * len(positive)

            selected_negative = rng.sample(
                negative,
                min(limit, len(negative))
            )

        else:
            selected_negative = negative

        selected = sorted(
            positive + selected_negative
        )

        print(
            f"  Positive: {len(positive)} | "
            f"Negative selected: {len(selected_negative)}",
            flush=True
        )

        for position, index in enumerate(
            selected, start=1
        ):

            filename = (
                f"{case_id}_slice_{index:04d}"
            )

            image_path = (
                image_dir / f"{filename}.png"
            )

            label_path = (
                label_dir / f"{filename}.txt"
            )

            # Preserve completed pairs.
            if pair_is_complete(
                image_path,
                label_path
            ):
                totals["existing_pairs"] += 1
                continue

            image_slice = image[index]
            mask_slice = mask[index]

            labels, skipped = make_labels(
                mask_slice
            )

            save_image(
                image_path,
                image_slice
            )

            save_label(
                label_path,
                labels
            )

            totals["created_images"] += 1
            totals["skipped_polygons"] += skipped

            if labels:
                totals["positive_slices"] += 1
            else:
                totals["negative_slices"] += 1

            if position % PROGRESS_EVERY == 0:
                print(
                    f"  Progress: {position}/{len(selected)}",
                    flush=True
                )

    print(
        f"Finished {case_id} in "
        f"{time.time() - started:.1f} seconds",
        flush=True
    )


def main():

    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"Source folder missing: {SOURCE}"
        )

    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(
            f"Split file missing: {SPLIT_FILE.resolve()}"
        )

    splits = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    for name in ("train", "val", "test"):
        if name not in splits:
            raise KeyError(
                f"Missing split: {name}"
            )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    totals = {
        "created_images": 0,
        "existing_pairs": 0,
        "positive_slices": 0,
        "negative_slices": 0,
        "skipped_polygons": 0
    }

    failed_cases = []

    for split_name in ("train", "val", "test"):

        records = splits[split_name]

        image_dir = (
            OUTPUT / split_name / "images"
        )

        label_dir = (
            OUTPUT / split_name / "labels"
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        label_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"\n===== {split_name.upper()} "
            f"({len(records)} cases) =====",
            flush=True
        )

        for number, record in enumerate(
            records, start=1
        ):

            case_id = record["case_id"]

            print(
                f"\n[{number}/{len(records)}] {case_id}",
                flush=True
            )

            try:
                prepare_case(
                    record,
                    split_name,
                    image_dir,
                    label_dir,
                    totals
                )

            except Exception as error:

                failed_cases.append({
                    "split": split_name,
                    "case_id": case_id,
                    "error": repr(error)
                })

                print(
                    f"ERROR: {case_id}: {error}",
                    flush=True
                )

    # Create YOLO configuration.
    yaml_text = f"""path: {OUTPUT.as_posix()}
train: train/images
val: val/images
test: test/images

names:
  0: lung_nodule
"""

    (OUTPUT / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8"
    )

    # Count generated files.
    counts = {}

    for split_name in ("train", "val", "test"):

        image_dir = (
            OUTPUT / split_name / "images"
        )

        label_dir = (
            OUTPUT / split_name / "labels"
        )

        counts[split_name] = {
            "images": len(
                list(image_dir.glob("*.png"))
            ),
            "labels": len(
                list(label_dir.glob("*.txt"))
            )
        }

    summary = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "negative_training_ratio": NEGATIVE_RATIO,
        "seed": SEED,
        "created_images": totals["created_images"],
        "existing_pairs_preserved": totals["existing_pairs"],
        "positive_slices": totals["positive_slices"],
        "negative_slices": totals["negative_slices"],
        "skipped_polygons": totals["skipped_polygons"],
        "files_per_split": counts,
        "failed_cases": failed_cases,
        "note": "2D YOLO segmentation dataset. Training not started."
    }

    (OUTPUT / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n===== FINAL RESULT =====", flush=True)

    print(
        f"New images: {totals['created_images']}"
    )

    print(
        f"Existing pairs preserved: "
        f"{totals['existing_pairs']}"
    )

    for split_name, value in counts.items():

        print(
            f"{split_name}: "
            f"{value['images']} images, "
            f"{value['labels']} labels"
        )

    print(
        f"Failed cases: {len(failed_cases)}"
    )

    print(
        f"Dataset location: {OUTPUT}"
    )

    print(
        "Training has NOT started.",
        flush=True
    )


if __name__ == "__main__":
    main()
