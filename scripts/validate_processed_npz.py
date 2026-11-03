import numpy as np
from pathlib import Path

folder = Path(r"D:\OncoSightData\processed\lung_ct_50")
files = sorted(folder.glob("*.npz"))

passed = 0
failed = []

print(f"Found {len(files)} processed cases\n")

for i, path in enumerate(files, 1):
    try:
        with np.load(path, allow_pickle=False) as data:
            image = data["image"]
            mask = data["mask"]
            axis = int(data["slice_axis"])

            if image.ndim != 3 or image.shape != mask.shape:
                raise ValueError("Invalid or mismatched shapes")

            if axis not in (0, 1, 2):
                raise ValueError("Invalid slice axis")

            if not np.isfinite(image).all():
                raise ValueError("Image contains NaN or Inf")

            if not np.isin(np.unique(mask), [0, 1]).all():
                raise ValueError("Mask is not binary")

            if not np.any(mask):
                raise ValueError("Empty mask")

        passed += 1
        print(f"[{i}/50] PASS: {path.stem}")

    except Exception as e:
        failed.append((path.name, str(e)))
        print(f"[{i}/50] FAIL: {path.name} - {e}")

print("\n========== VALIDATION RESULT ==========")
print(f"Passed: {passed}")
print(f"Failed: {len(failed)}")

for name, error in failed:
    print(name, error)