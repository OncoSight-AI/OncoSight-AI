
import torch
from lung_dataset import create_dataloaders

_, val_loader, _ = create_dataloaders(
    "datasets/manifests/split_manifest.json",
    image_size=256,
    batch_size=2
)

positive_counts = []

for images, masks in val_loader:
    for mask in masks:
        count = (mask >= 0.5).sum().item()

        if count > 0:
            positive_counts.append(count)

print("Positive slices:", len(positive_counts))

if positive_counts:
    print("Smallest positive mask:", min(positive_counts))
    print("Largest positive mask:", max(positive_counts))
    print("Average positive pixels:", sum(positive_counts) / len(positive_counts))
