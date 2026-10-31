
from pathlib import Path
import json
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "datasets/raw/lidc_hf/images"
MASK_DIR = ROOT / "datasets/raw/lidc_hf/masks"
REPORT = ROOT / "datasets/manifests/batch_validation.json"

results = []

images = sorted(IMAGE_DIR.glob("*.nii.gz"))

for image_path in images:
    case_id = image_path.name.replace(".nii.gz", "")
    mask_path = MASK_DIR / image_path.name

    result = {"case_id": case_id, "status": "FAIL"}

    try:
        if not mask_path.exists():
            raise FileNotFoundError("Matching mask missing")

        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))

        image_data = np.asanyarray(image.dataobj)
        mask_data = np.asanyarray(mask.dataobj)

        checks = {
            "3d": image_data.ndim == 3 and mask_data.ndim == 3,
            "shape_match": image_data.shape == mask_data.shape,
            "affine_match": np.allclose(image.affine, mask.affine, atol=1e-4),
            "finite_image": np.isfinite(image_data).all(),
            "binary_mask": set(np.unique(mask_data).tolist()).issubset({0, 1}),
            "nonempty_mask": np.count_nonzero(mask_data) > 0,
        }

        result.update({
            "checks": checks,
            "shape": list(image_data.shape),
            "mask_voxels": int(np.count_nonzero(mask_data)),
            "status": "PASS" if all(checks.values()) else "FAIL"
        })

    except Exception as error:
        result["error"] = str(error)

    results.append(result)
    print(case_id, result["status"], result.get("checks", result.get("error")))

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(results, indent=2, default=bool), encoding="utf-8")

passed = sum(r["status"] == "PASS" for r in results)
print(f"\nRESULT: {passed}/{len(results)} cases passed")
print(f"Report saved: {REPORT}")
