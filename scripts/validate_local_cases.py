
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Validate local LIDC-IDRI CT images and masks."
    )
    parser.add_argument(
        "--manifest",
        default="datasets/manifests/cases_50.json",
        help="Manifest path relative to project root."
    )
    args = parser.parse_args()

    manifest = ROOT / args.manifest

    if not manifest.exists():
        print(f"[ERROR] Manifest not found: {manifest}")
        return 1

    cases = json.loads(manifest.read_text(encoding="utf-8"))

    if not isinstance(cases, list) or not cases:
        print("[ERROR] Manifest is empty or has an invalid format.")
        return 1

    total = len(cases)
    passed_cases = 0
    failed_cases = 0

    print("\n========== LOCAL DATASET VALIDATION ==========")
    print(f"Manifest: {manifest.name}")
    print(f"Total cases: {total}")

    for case in cases:
        name = case["case_id"]
        image_path = ROOT / case["image_path"]
        mask_path = ROOT / case["mask_path"]

        print(f"\nCASE: {name}")

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image missing: {image_path}")

            if not mask_path.exists():
                raise FileNotFoundError(f"Mask missing: {mask_path}")

            image = nib.load(str(image_path))
            mask = nib.load(str(mask_path))

            image_data = image.get_fdata(dtype=np.float32)
            mask_data = np.asanyarray(mask.dataobj)

            checks = {
                "Image is 3D": image_data.ndim == 3,
                "Mask is 3D": mask_data.ndim == 3,
                "Shapes match": image_data.shape == mask_data.shape,
                "Affines match": np.allclose(
                    image.affine, mask.affine, atol=1e-4
                ),
                "Image finite": np.isfinite(image_data).all(),
                "Mask finite": np.isfinite(mask_data).all(),
                "Mask binary": set(np.unique(mask_data)).issubset({0, 1}),
                "Mask non-empty": np.count_nonzero(mask_data) > 0,
            }

            for label, passed in checks.items():
                print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

            print("  Image shape:", image_data.shape)
            print("  Image range:", image_data.min(), image_data.max())
            print("  Mask voxels:", np.count_nonzero(mask_data))

            if all(checks.values()):
                passed_cases += 1
            else:
                failed_cases += 1

        except Exception as error:
            print("  [FAIL] Error:", error)
            failed_cases += 1

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"Total cases: {total}")
    print(f"Passed: {passed_cases}")
    print(f"Failed: {failed_cases}")

    if failed_cases == 0:
        print("\n[SUCCESS] ALL CASES VALIDATED!")
        return 0

    print("\n[WARNING] SOME CASES FAILED VALIDATION.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
