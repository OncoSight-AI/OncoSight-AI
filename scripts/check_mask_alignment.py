from pathlib import Path
import numpy as np
import nibabel as nib

root = Path(".")
manifest = root / "datasets/manifests/pilot_cases.json"

import json
cases = json.loads(manifest.read_text())

for case in cases:
    name = case["case_id"]
    data = np.load(root / "datasets/processed/lung_ct" / f"{name}.npz")

    image = data["image"]
    mask = data["mask"]
    axis = int(data["slice_axis"])

    positive = np.where(np.any(mask > 0, axis=(0, 1)))[0] if axis == 2 else None

    print(f"\nCASE: {name}")
    print("Image/mask shape match:", image.shape == mask.shape)
    print("Positive slices:", len(positive) if positive is not None else "Check other axis")

    if positive is not None and len(positive):
        for index in [positive[0], positive[len(positive)//2], positive[-1]]:
            m = mask[:, :, index]
            coords = np.argwhere(m > 0)

            print(
                f"Slice {index}: "
                f"mask pixels={len(coords)}, "
                f"bounding box="
                f"x({coords[:, 0].min()}-{coords[:, 0].max()}), "
                f"y({coords[:, 1].min()}-{coords[:, 1].max()})"
            )