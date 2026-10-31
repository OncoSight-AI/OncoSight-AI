
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from unet_model import UNet


# ---------------- CONFIGURATION ----------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "lung_ct"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_lung_unet_weighted.pth"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "validation" / "predictions"

IMAGE_SIZE = 256
THRESHOLD = 0.50
NUMBER_OF_SLICES = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD MODEL ----------------

def load_model():

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint loaded successfully.")
    print("Checkpoint epoch:", checkpoint.get("epoch", "Unknown"))
    print("Device:", DEVICE)

    return model


# ---------------- PREPARE IMAGE ----------------

def prepare_slice(image, mask, slice_index):

    image_slice = image[:, :, slice_index]
    mask_slice = mask[:, :, slice_index]

    # Resize image to 256 x 256
    image_tensor = torch.from_numpy(image_slice.copy()).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    image_tensor = torch.nn.functional.interpolate(
        image_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    # Resize mask using the same strategy as training
    mask_tensor = torch.from_numpy(mask_slice.copy()).float()
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

    mask_tensor = torch.nn.functional.adaptive_max_pool2d(
        mask_tensor,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    mask_tensor = (mask_tensor > 0).float()

    return image_tensor.to(DEVICE), mask_tensor.cpu().numpy()[0, 0]


# ---------------- DICE SCORE ----------------

def calculate_dice(prediction, target):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(prediction, target).sum()

    denominator = prediction.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return (2.0 * intersection) / denominator


# ---------------- VISUALIZATION ----------------

def save_visualization(
    image,
    actual_mask,
    predicted_mask,
    case_id,
    slice_index,
    dice
):

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original CT")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(actual_mask, cmap="Greens", alpha=0.55)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(predicted_mask, cmap="Reds", alpha=0.55)
    axes[2].set_title("Prediction")

    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(actual_mask, cmap="Greens", alpha=0.45)
    axes[3].imshow(predicted_mask, cmap="Reds", alpha=0.45)
    axes[3].set_title("Green: Actual | Red: Predicted")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{case_id} | Slice {slice_index} | Dice: {dice:.4f}",
        fontsize=12
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{case_id}_slice_{slice_index}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path.name, "| Dice:", round(dice, 4))


# ---------------- MAIN PREDICTION ----------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    model = load_model()

    npz_files = sorted(DATA_DIR.glob("*.npz"))

    if not npz_files:
        raise FileNotFoundError("No NPZ files found.")

    # Prefer the held-out test patient
    test_files = [
        file for file in npz_files
        if file.stem == "LIDC-IDRI-0980"
    ]

    if not test_files:
        print("Test patient not found by full ID; using available files.")
        test_files = npz_files

    results = []

    for file_path in test_files:

        print("\nProcessing:", file_path.name)

        with np.load(file_path) as data:

            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.uint8)

        if image.shape != mask.shape:
            print("Skipping: image and mask shapes differ.")
            continue

        total_slices = image.shape[2]

        # Select slices containing actual annotated pixels
        positive_slices = [
            i for i in range(total_slices)
            if np.any(mask[:, :, i] > 0)
        ]

        if not positive_slices:
            print("No positive slices found.")
            continue

        selected = positive_slices[:NUMBER_OF_SLICES]

        for slice_index in selected:

            image_tensor, actual_mask = prepare_slice(
                image, mask, slice_index
            )

            with torch.no_grad():

                logits = model(image_tensor)
                probabilities = torch.sigmoid(logits)

                predicted_mask = (
                    probabilities[0, 0].cpu().numpy() >= THRESHOLD
                ).astype(np.uint8)

            dice = calculate_dice(predicted_mask, actual_mask)

            save_visualization(
                image_tensor[0, 0].cpu().numpy(),
                actual_mask,
                predicted_mask,
                file_path.stem,
                slice_index,
                dice
            )

            results.append(dice)

    if results:

        print("\n========== PREDICTION SUMMARY ==========")
        print("Slices processed:", len(results))
        print("Mean Dice:", round(float(np.mean(results)), 4))
        print("Threshold:", THRESHOLD)
        print("Output folder:", OUTPUT_DIR)
        print("========================================")

    else:
        print("No slices were processed.")


if __name__ == "__main__":
    main()
