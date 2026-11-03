from pathlib import Path
import numpy as np
from scipy import ndimage

folder = Path("datasets/processed/lung_ct")

for path in sorted(folder.glob("*.npz")):
    data = np.load(path)
    mask = data["mask"].astype(bool)

    labels, count = ndimage.label(
        mask,
        structure=ndimage.generate_binary_structure(3, 1)
    )

    sizes = np.bincount(labels.ravel())[1:]
    sizes = np.sort(sizes)[::-1]

    print(f"\nCASE: {path.stem}")
    print("Connected components:", count)
    print("Largest component sizes:", sizes[:10].tolist())
    print("Total mask voxels:", int(mask.sum()))