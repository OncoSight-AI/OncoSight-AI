
from pathlib import Path
import csv
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from unet_model import UNet
from lung_dataset import create_dataloaders


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = ROOT / "datasets/manifests/split_manifest.json"
CHECKPOINT = ROOT / "models/best_lung_unet_weighted.pth"

OUTPUT = ROOT / "outputs/evaluation"
VISUALS = OUTPUT / "visualizations"

IMAGE_SIZE = 256
BATCH_SIZE = 2
THRESHOLD = 0.50
MAX_VISUALS = 12

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    model = UNet(
        in_channels=1,
        out_channels=1
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint loaded:", CHECKPOINT.name)
    print("Epoch:", checkpoint.get("epoch", "Unknown"))
    print("Device:", DEVICE)

    return model, checkpoint.get("epoch", "Unknown")


# ==========================================
# METRIC CALCULATIONS
# ==========================================

def calculate_counts(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    tn = np.logical_and(~pred, ~target).sum()

    return int(tp), int(fp), int(fn), int(tn)


def divide(a, b, empty=0.0):

    return float(a / b) if b else empty


def calculate_metrics(tp, fp, fn, tn):

    return {
        "dice": divide(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": divide(tp, tp + fp + fn, 1.0),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }


# ==========================================
# VISUALIZATION
# ==========================================

def save_visualization(
    image,
    target,
    prediction,
    number,
    dice
):

    fig, axes = plt.subplots(
        1, 4,
        figsize=(16, 4)
    )

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original CT")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(
        target,
        cmap="Greens",
        alpha=0.55
    )
    axes[1].set_title("Ground Truth")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(
        prediction,
        cmap="Reds",
        alpha=0.55
    )
    axes[2].set_title("Prediction")

    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(
        target,
        cmap="Greens",
        alpha=0.45
    )
    axes[3].imshow(
        prediction,
        cmap="Reds",
        alpha=0.45
    )
    axes[3].set_title(
        "Green: Truth | Red: Prediction"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Test Sample {number} | Dice: {dice:.4f}"
    )

    fig.tight_layout()

    fig.savefig(
        VISUALS / f"test_sample_{number:04d}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# MAIN EVALUATION
# ==========================================

def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    VISUALS.mkdir(
        parents=True,
        exist_ok=True
    )

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    model, epoch = load_model()

    # Load the held-out test dataset.
    _, _, test_loader = create_dataloaders(
        str(MANIFEST),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    if len(test_loader.dataset) == 0:
        raise RuntimeError("Test dataset is empty.")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    rows = []

    saved_visuals = 0
    sample_number = 0

    # Run inference without calculating gradients.
    with torch.inference_mode():

        for images, masks in test_loader:

            images = images.to(
                DEVICE,
                dtype=torch.float32
            )

            masks = masks.to(
                DEVICE,
                dtype=torch.float32
            )

            probabilities = torch.sigmoid(
                model(images)
            )

            predictions = (
                probabilities >= THRESHOLD
            ).to(torch.uint8)

            targets = (
                masks >= 0.5
            ).to(torch.uint8)

            for i in range(images.shape[0]):

                sample_number += 1

                pred = predictions[i, 0].cpu().numpy()
                target = targets[i, 0].cpu().numpy()
                image = images[i, 0].cpu().numpy()

                tp, fp, fn, tn = calculate_counts(
                    pred,
                    target
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                metrics = calculate_metrics(
                    tp,
                    fp,
                    fn,
                    tn
                )

                is_positive = bool(target.any())

                rows.append({
                    "sample": sample_number,
                    "positive_slice": is_positive,
                    **metrics
                })

                # Save up to 12 positive slices.
                if (
                    is_positive
                    and saved_visuals < MAX_VISUALS
                ):

                    save_visualization(
                        image,
                        target,
                        pred,
                        sample_number,
                        metrics["dice"]
                    )

                    saved_visuals += 1

    # ======================================
    # FINAL METRICS
    # ======================================

    overall = calculate_metrics(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    positive_rows = [
        row for row in rows
        if row["positive_slice"]
    ]

    positive_mean_dice = (
        float(np.mean([
            row["dice"] for row in positive_rows
        ]))
        if positive_rows else None
    )

    all_slice_mean_dice = float(
        np.mean([
            row["dice"] for row in rows
        ])
    )

    summary = {

        "model": "2D U-Net",

        "checkpoint": CHECKPOINT.name,

        "checkpoint_epoch": epoch,

        "device": str(DEVICE),

        "threshold": THRESHOLD,

        "test_slice_count": len(rows),

        "positive_slice_count": len(positive_rows),

        "overall_pixel_metrics": overall,

        "mean_all_slice_dice": all_slice_mean_dice,

        "mean_positive_slice_dice": positive_mean_dice,

        "visualizations_saved": saved_visuals,

        "note": (
            "Preliminary research evaluation. "
            "Not clinical validation."
        )
    }

    # Save metrics.json.
    with open(
        OUTPUT / "metrics.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # Save per_slice_metrics.csv.
    with open(
        OUTPUT / "per_slice_metrics.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("\n========== EVALUATION RESULTS ==========")

    print("Test slices:", len(rows))

    print("Positive slices:", len(positive_rows))

    print("Threshold:", THRESHOLD)

    print("\nOverall Pixel Metrics:")

    for key in [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity"
    ]:

        print(
            f"{key}: {overall[key]:.4f}"
        )

    print(
        "\nMean all-slice Dice:",
        round(all_slice_mean_dice, 4)
    )

    print(
        "Mean positive-slice Dice:",
        positive_mean_dice
    )

    print("\nSaved outputs:", OUTPUT)

    print("========================================")


if __name__ == "__main__":
    main()
