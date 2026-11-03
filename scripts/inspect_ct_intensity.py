import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads(
    (ROOT / "datasets/manifests/pilot_cases.json").read_text()
)

print("\n========== CT INTENSITY INSPECTION ==========")

for case in manifest:
    path = ROOT / case["image_path"]
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)

    percentiles = np.percentile(
        data, [0, 1, 5, 25, 50, 75, 95, 99, 100]
    )

    print(f"\n{case['case_id']}")
    print("Percentiles:", np.round(percentiles, 2))
    print("Slope/intercept:", nii.header.get_slope_inter())
    print("Units:", nii.header.get_xyzt_units())

print("\n========== COMPLETE ==========")