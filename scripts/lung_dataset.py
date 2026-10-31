
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class LungCTDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        split="train",
        image_size=256,
        cache_size=2
    ):

        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parents[2]
        self.image_size = image_size
        self.cache_size = cache_size

        self.samples = []
        self.volume_cache = OrderedDict()

        rng = np.random.default_rng(42)

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for case in manifest["cases"]:

            if case["split"] != split:
                continue

            volume_path = self.root / case["volume"]

            with np.load(volume_path) as data:

                axis = int(data["slice_axis"].item())
                number_of_slices = data["image"].shape[axis]

                mask = np.moveaxis(data["mask"], axis, 0)

                positive = np.where(
                    mask.reshape(number_of_slices, -1).any(axis=1)
                )[0]

                negative = np.where(
                    ~mask.reshape(number_of_slices, -1).any(axis=1)
                )[0]

            if split == "train":

                # Select equal numbers of positive and negative slices.
                if len(negative) > len(positive):
                    negative = rng.choice(
                        negative,
                        size=len(positive),
                        replace=False
                    )

                selected = np.concatenate([positive, negative])
                rng.shuffle(selected)

            else:
                selected = np.arange(number_of_slices)

            for index in selected:
                self.samples.append(
                    (str(volume_path), int(index))
                )

        print(
            f"{split.capitalize()} dataset initialized: "
            f"{len(self.samples)} slices"
        )

    def __len__(self):
        return len(self.samples)

    def load_volume(self, volume_path):

        # Return cached volume if already loaded.
        if volume_path in self.volume_cache:

            self.volume_cache.move_to_end(volume_path)
            return self.volume_cache[volume_path]

        with np.load(volume_path) as data:

            axis = int(data["slice_axis"].item())

            image = np.moveaxis(data["image"], axis, 0)
            mask = np.moveaxis(data["mask"], axis, 0)

            image = np.asarray(image, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.uint8)

        self.volume_cache[volume_path] = (image, mask)

        # Remove the oldest cached volume when the limit is reached.
        while len(self.volume_cache) > self.cache_size:
            self.volume_cache.popitem(last=False)

        return image, mask

    def __getitem__(self, idx):

        volume_path, slice_index = self.samples[idx]

        image, mask = self.load_volume(volume_path)

        x = torch.from_numpy(
            np.array(
                image[slice_index],
                dtype=np.float32,
                copy=True
            )
        ).unsqueeze(0)

        y = torch.from_numpy(
            np.array(
                mask[slice_index],
                dtype=np.float32,
                copy=True
            )
        ).unsqueeze(0)

        # Resize CT image.
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Resize segmentation mask.
        if (
            y.shape[-2] > self.image_size
            or y.shape[-1] > self.image_size
        ):

            y = F.adaptive_max_pool2d(
                y.unsqueeze(0),
                output_size=(
                    self.image_size,
                    self.image_size
                )
            ).squeeze(0)

        else:

            y = F.interpolate(
                y.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest"
            ).squeeze(0)

        y = (y > 0).float()

        return x, y


def create_dataloaders(
    manifest_path,
    image_size=256,
    batch_size=2
):

    train_ds = LungCTDataset(
        manifest_path,
        split="train",
        image_size=image_size,
        cache_size=2
    )

    val_ds = LungCTDataset(
        manifest_path,
        split="validation",
        image_size=image_size,
        cache_size=2
    )

    test_ds = LungCTDataset(
        manifest_path,
        split="test",
        image_size=image_size,
        cache_size=1
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


# ==========================================
# DATASET TEST
# ==========================================

if __name__ == "__main__":

    manifest = "datasets/manifests/split_manifest.json"

    train_loader, val_loader, test_loader = create_dataloaders(
        manifest,
        image_size=256,
        batch_size=2
    )

    print("\n========== DATASET VALIDATION ==========")

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Testing samples:", len(test_loader.dataset))

    images, masks = next(iter(train_loader))

    print("\nBATCH INFORMATION")

    print("Image shape:", images.shape)
    print("Mask shape:", masks.shape)
    print("Image range:", images.min().item(), images.max().item())
    print("Mask values:", torch.unique(masks))
    print("Positive pixels:", (masks > 0).sum().item())

    print("\nDataset loading successful!")
