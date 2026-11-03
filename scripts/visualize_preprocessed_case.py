from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path("datasets/processed/lung_ct")
file = folder / "LIDC-IDRI-0001_9720603192.npz"

data = np.load(file)

image = data["image"]
mask = data["mask"]

positive_slices = [
    i for i in range(mask.shape[2])
    if np.any(mask[:, :, i])
]

if not positive_slices:
    raise ValueError("No positive slices found")

indices = [
    positive_slices[0],
    positive_slices[len(positive_slices) // 2],
    positive_slices[-1]
]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row, index in enumerate(indices):
    ct = image[:, :, index]
    gt = mask[:, :, index]

    axes[row, 0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"CT - Slice {index}")

    axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title("Ground-truth mask")

    axes[row, 2].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].imshow(gt, cmap="Reds", alpha=0.5)
    axes[row, 2].set_title("CT + mask overlay")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()

output = Path("outputs/evaluation/preprocessing_check.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=150, bbox_inches="tight")
plt.close()

print("Visualization saved:", output.resolve())
print("Selected slices:", indices)