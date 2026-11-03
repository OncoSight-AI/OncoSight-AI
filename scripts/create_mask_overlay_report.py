from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_dir = Path("datasets/processed/lung_ct")
output_dir = Path("outputs/evaluation/mask_overlay_report")
output_dir.mkdir(parents=True, exist_ok=True)

for path in sorted(data_dir.glob("*.npz")):
    data = np.load(path)

    image = data["image"].astype(np.float32)
    mask = data["mask"].astype(bool)

    # Current pilot data uses slice axis 2.
    positive_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]

    if len(positive_slices) == 0:
        continue

    selected = np.unique(
        np.linspace(
            0,
            len(positive_slices) - 1,
            min(6, len(positive_slices)),
            dtype=int
        )
    )

    slices = positive_slices[selected]

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 8)
    )
    axes = axes.ravel()

    for ax, slice_index in zip(axes, slices):
        ax.imshow(image[:, :, slice_index], cmap="gray")
        ax.imshow(
            np.ma.masked_where(
                ~mask[:, :, slice_index],
                mask[:, :, slice_index]
            ),
            cmap="autumn",
            alpha=0.55,
            interpolation="nearest"
        )
        ax.set_title(
            f"Slice {slice_index} | "
            f"Pixels: {int(mask[:, :, slice_index].sum())}"
        )
        ax.axis("off")

    for ax in axes[len(slices):]:
        ax.axis("off")

    fig.suptitle(path.stem)
    fig.tight_layout()

    output_path = output_dir / f"{path.stem}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Saved:", output_path)

print("\nOverlay report completed.")