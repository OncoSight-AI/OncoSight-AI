
#!/usr/bin/env python3
"""Preprocess LIDC NIfTI CT volumes and masks for 2D U-Net training."""

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = logging.getLogger("preprocess_lung_ct")


def load_case_records(root):
    path = root / "datasets/manifests/pilot_cases.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a non-empty list")
    return records


def slice_axis_from_affine(affine):
    return int(np.argmax(np.abs(affine[2, :3])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--window-low", type=float, default=-1000)
    parser.add_argument("--window-high", type=float, default=400)
    parser.add_argument("--max-overlays", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    root = args.repo_root.resolve()

    if args.window_high <= args.window_low:
        parser.error("--window-high must exceed --window-low")

    records = load_case_records(root)

    output_dir = root / "datasets/processed/lung_ct"
    overlay_dir = root / "datasets/validation/pilot_overlays"
    report_path = root / "datasets/manifests/preprocessing_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    overlay_count = 0

    for record in records:
        case_id = record["case_id"]
        patient_id = record.get("patient_id")

        image_path = root / record["image_path"]
        mask_path = root / record["mask_path"]

        LOG.info("Processing %s", case_id)

        try:
            if not image_path.exists() or not mask_path.exists():
                raise FileNotFoundError("Image or mask file missing")

            image_nii = nib.load(str(image_path))
            mask_nii = nib.load(str(mask_path))

            if len(image_nii.shape) != 3 or len(mask_nii.shape) != 3:
                raise ValueError("Both files must be 3D")

            if image_nii.shape != mask_nii.shape:
                raise ValueError("Image and mask shapes differ")

            if not np.allclose(
                image_nii.affine, mask_nii.affine, atol=1e-4
            ):
                raise ValueError("Image and mask affine differ")

            image = image_nii.get_fdata(dtype=np.float32)
            mask_raw = np.asanyarray(mask_nii.dataobj)

            if not np.isfinite(image).all():
                raise ValueError("Image contains NaN or Inf")

            if not np.isin(np.unique(mask_raw), [0, 1]).all():
                raise ValueError("Mask is not binary {0,1}")

            mask = (mask_raw > 0).astype(np.uint8)

            mask_voxels = int(np.count_nonzero(mask))
            if mask_voxels == 0:
                raise ValueError("Mask contains no positive voxels")

            # CT intensity windowing and normalization.
            clipped = np.clip(
                image, args.window_low, args.window_high
            )

            normalized = (
                (clipped - args.window_low) /
                (args.window_high - args.window_low)
            ).astype(np.float16)

            # Identify the superior/inferior slice axis.
            axis = slice_axis_from_affine(image_nii.affine)
            n_slices = image.shape[axis]

            positive_indices = []
            for index in range(n_slices):
                mask_slice = np.take(mask, index, axis=axis)
                if np.any(mask_slice):
                    positive_indices.append(index)

            if not positive_indices:
                raise ValueError("No positive nodule slices found")

            # Save one compressed volume per case.
            output_file = output_dir / f"{case_id}.npz"

            np.savez_compressed(
                output_file,
                image=normalized,
                mask=mask,
                slice_axis=np.int8(axis)
            )

            # Generate a limited number of visual overlays.
            for index in positive_indices:
                if overlay_count >= args.max_overlays:
                    break

                image_slice = np.take(
                    normalized, index, axis=axis
                )
                mask_slice = np.take(
                    mask, index, axis=axis
                )

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image_slice, cmap="gray", vmin=0, vmax=1)

                if np.any(mask_slice):
                    ax.contour(
                        mask_slice,
                        levels=[0.5],
                        colors="red",
                        linewidths=0.7
                    )

                ax.set_title(f"{case_id} - Slice {index}")
                ax.axis("off")

                fig.savefig(
                    overlay_dir / f"{case_id}_slice_{index}.png",
                    dpi=120,
                    bbox_inches="tight"
                )
                plt.close(fig)
                overlay_count += 1

            summaries.append({
                "case_id": case_id,
                "patient_id": patient_id,
                "series_uid": record.get("series_uid"),
                "positive_slice_count": len(positive_indices),
                "mask_voxels": mask_voxels,
                "output": str(output_file.relative_to(root)),
                "n_slices": n_slices,
                "slice_axis": axis
            })

            LOG.info(
                "PASS %s | slices=%d | positive=%d | mask voxels=%d",
                case_id, n_slices,
                len(positive_indices), mask_voxels
            )

        except Exception as error:
            LOG.exception("FAILED %s: %s", case_id, error)

    report = {
        "window": {
            "low_hu": args.window_low,
            "high_hu": args.window_high
        },
        "total_cases": len(records),
        "processed_cases": len(summaries),
        "failed_cases": len(records) - len(summaries),
        "overlay_count": overlay_count,
        "cases": summaries
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nPREPROCESSING RESULT")
    print(f"Processed: {len(summaries)}/{len(records)}")
    print(f"Overlays: {overlay_count}")
    print(f"Report: {report_path}")

    return 0 if len(summaries) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
