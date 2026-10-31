
from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from lung_dataset import create_dataloaders
from unet_model import UNet


# =====================================================
# 1. CONFIGURATION
# =====================================================

MANIFEST = "datasets/manifests/split_manifest.json"

IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
PATIENCE = 3

POSITIVE_WEIGHT = 20.0

THRESHOLDS = [0.25, 0.30, 0.40, 0.50]

# Original checkpoint: only used to initialize weights.
OLD_CHECKPOINT = Path("models/best_lung_unet.pth")

# New experiment files.
CHECKPOINT_PATH = Path("models/best_lung_unet_weighted.pth")
HISTORY_PATH = Path("models/training_history_weighted.csv")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 2. GPU INFORMATION
# =====================================================

def show_device_info():

    print("\n========== DEVICE INFORMATION ==========")

    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        memory = torch.cuda.get_device_properties(
            0
        ).total_memory

        print(
            "Total GPU memory:",
            round(memory / (1024 ** 3), 2),
            "GB"
        )

    print("Mixed precision:", USE_AMP)

    print("========================================")


def get_gpu_memory():

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated() / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved() / (1024 ** 3)
        )

        return f"{allocated:.2f}/{reserved:.2f} GB"

    return "CPU"


# =====================================================
# 3. WEIGHTED BCE + DICE LOSS
# =====================================================

class WeightedDiceBCELoss(nn.Module):

    def __init__(self, positive_weight=20.0):

        super().__init__()

        self.register_buffer(
            "positive_weight",
            torch.tensor([positive_weight], dtype=torch.float32)
        )

    def forward(self, logits, targets):

        logits = logits.float()
        targets = targets.float()

        # Weighted binary cross entropy.
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.positive_weight
        )

        probabilities = torch.sigmoid(logits)

        # Calculate Dice separately for every image.
        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1) +
            targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            denominator + 1e-6
        )

        dice_loss = 1.0 - dice.mean()

        return bce_loss + dice_loss


# =====================================================
# 4. CHECKPOINT INITIALIZATION
# =====================================================

def initialize_training(model, optimizer):

    start_epoch = 1
    best_score = -1.0

    # Resume the new weighted experiment if it exists.
    if CHECKPOINT_PATH.exists():

        print("\nLoading weighted checkpoint...")

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        best_score = float(
            checkpoint["best_positive_dice"]
        )

        print("Previous epoch:", checkpoint["epoch"])
        print("Best positive Dice:", best_score)
        print("Resuming from epoch:", start_epoch)

    # Otherwise initialize from the old model.
    elif OLD_CHECKPOINT.exists():

        print("\nLoading previous model weights...")

        checkpoint = torch.load(
            OLD_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        old_epoch = checkpoint["epoch"]

        start_epoch = old_epoch + 1

        print("Previous model epoch:", old_epoch)
        print("Starting weighted training from:", start_epoch)
        print("Using a fresh optimizer.")

        # Do not load the old optimizer state:
        # the loss function has changed.

    else:

        print("\nNo previous checkpoint found.")
        print("Training from scratch.")

    return start_epoch, best_score


# =====================================================
# 5. TRAINING FUNCTION
# =====================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler
):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, masks in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):

            outputs = model(images)

        # Loss calculation in float32.
        loss = criterion(outputs, masks)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "GPU": get_gpu_memory()
        })

    return running_loss / max(len(loader), 1)


# =====================================================
# 6. VALIDATION FUNCTION
# =====================================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0

    # Intersection, predicted pixels, actual pixels.
    threshold_counts = {
        t: [0, 0, 0]
        for t in THRESHOLDS
    }

    positive_slice_dice = {
        t: []
        for t in THRESHOLDS
    }

    positive_slices = 0
    negative_slices = 0

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Validation",
            leave=False
        )

        for images, masks in progress:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            with autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):

                outputs = model(images)

            loss = criterion(outputs, masks)

            running_loss += loss.item()

            probabilities = torch.sigmoid(
                outputs.float()
            )

            targets = masks > 0.5

            for batch_index in range(images.size(0)):

                target = targets[batch_index]

                actual_pixels = target.sum().item()

                if actual_pixels > 0:
                    positive_slices += 1
                else:
                    negative_slices += 1

                for threshold in THRESHOLDS:

                    prediction = (
                        probabilities[batch_index] > threshold
                    )

                    intersection = (
                        prediction & target
                    ).sum().item()

                    predicted_pixels = (
                        prediction.sum().item()
                    )

                    counts = threshold_counts[threshold]

                    counts[0] += intersection
                    counts[1] += predicted_pixels
                    counts[2] += actual_pixels

                    # Evaluate only slices containing foreground.
                    if actual_pixels > 0:

                        dice = (
                            2.0 * intersection + 1e-6
                        ) / (
                            predicted_pixels +
                            actual_pixels +
                            1e-6
                        )

                        positive_slice_dice[
                            threshold
                        ].append(dice)

            progress.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

    val_loss = (
        running_loss / max(len(loader), 1)
    )

    results = {}

    for threshold in THRESHOLDS:

        intersection, predicted, actual = (
            threshold_counts[threshold]
        )

        global_dice = (
            2.0 * intersection + 1e-6
        ) / (
            predicted + actual + 1e-6
        )

        global_iou = (
            intersection + 1e-6
        ) / (
            predicted + actual - intersection + 1e-6
        )

        values = positive_slice_dice[threshold]

        mean_positive_dice = (
            sum(values) / len(values)
            if values else 0.0
        )

        results[threshold] = {
            "global_dice": global_dice,
            "global_iou": global_iou,
            "positive_dice": mean_positive_dice,
            "predicted_pixels": predicted,
            "actual_pixels": actual
        }

    return (
        val_loss,
        results,
        positive_slices,
        negative_slices
    )


# =====================================================
# 7. SAVE CHECKPOINT
# =====================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_score
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_positive_dice": best_score,
            "image_size": IMAGE_SIZE,
            "positive_weight": POSITIVE_WEIGHT
        },
        CHECKPOINT_PATH
    )


# =====================================================
# 8. MAIN FUNCTION
# =====================================================

def main():

    show_device_info()

    print("\nLoading dataset...")

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            MANIFEST,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
    )

    print(
        "Training samples:",
        len(train_loader.dataset)
    )

    print(
        "Validation samples:",
        len(val_loader.dataset)
    )

    print(
        "Testing samples:",
        len(test_loader.dataset)
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # Initialize model.
    model = UNet().to(DEVICE)

    # New optimizer.
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = WeightedDiceBCELoss(
        positive_weight=POSITIVE_WEIGHT
    ).to(DEVICE)

    scaler = GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    start_epoch, best_score = initialize_training(
        model,
        optimizer
    )

    if start_epoch > NUM_EPOCHS:

        print("\nTraining already reached the epoch limit.")
        return

    patience_counter = 0

    history_exists = HISTORY_PATH.exists()

    with open(
        HISTORY_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as history_file:

        writer = csv.writer(history_file)

        if (
            not history_exists
            or HISTORY_PATH.stat().st_size == 0
        ):

            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "positive_dice_0.25",
                "positive_dice_0.30",
                "positive_dice_0.40",
                "positive_dice_0.50",
                "global_dice_0.30",
                "global_iou_0.30"
            ])

        print(
            "\n========== WEIGHTED TRAINING STARTED =========="
        )

        for epoch in range(
            start_epoch,
            NUM_EPOCHS + 1
        ):

            print(
                f"\nEpoch {epoch}/{NUM_EPOCHS}"
            )

            # Training.
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler
            )

            # Validation.
            (
                val_loss,
                results,
                positive_slices,
                negative_slices
            ) = validate(
                model,
                val_loader,
                criterion
            )

            # Display results.
            print("\n========== EPOCH RESULTS ==========")

            print(
                f"Training Loss: {train_loss:.4f}"
            )

            print(
                f"Validation Loss: {val_loss:.4f}"
            )

            print(
                "Positive validation slices:",
                positive_slices
            )

            print(
                "Negative validation slices:",
                negative_slices
            )

            for threshold in THRESHOLDS:

                result = results[threshold]

                print(
                    f"Threshold {threshold:.2f} | "
                    f"Positive Dice: "
                    f"{result['positive_dice']:.6f} | "
                    f"Global Dice: "
                    f"{result['global_dice']:.6f} | "
                    f"Predicted pixels: "
                    f"{result['predicted_pixels']}"
                )

            print("===================================")

            # Save history.
            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                results[0.25]["positive_dice"],
                results[0.30]["positive_dice"],
                results[0.40]["positive_dice"],
                results[0.50]["positive_dice"],
                results[0.30]["global_dice"],
                results[0.30]["global_iou"]
            ])

            history_file.flush()

            # Select checkpoint using positive-slice Dice.
            current_score = results[0.30]["positive_dice"]

            if current_score > best_score:

                best_score = current_score

                patience_counter = 0

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_score
                )

                print(
                    "\nBest weighted checkpoint updated."
                )

            else:

                patience_counter += 1

                print(
                    f"\nNo improvement: "
                    f"{patience_counter}/{PATIENCE}"
                )

            print(
                "Best positive-slice Dice:",
                f"{best_score:.6f}"
            )

            # Early stopping.
            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping triggered."
                )

                break

    print(
        "\n========== TRAINING FINISHED =========="
    )

    print(
        "Best positive-slice Dice:",
        f"{best_score:.6f}"
    )

    print(
        "New checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Training history:",
        HISTORY_PATH
    )


if __name__ == "__main__":

    main()
