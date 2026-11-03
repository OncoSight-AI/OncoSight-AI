
#!/usr/bin/env python3

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


def load_case_records(path):
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    records = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a non-empty list")

    return records


def slice_axis_from_affine(affine):
    return int(np.argmax(np.abs(affine[2, :3])))


def validate_existing_npz(path):
    """Check whether an existing NPZ can safely be reused."""
    try:
        with np.load(path, allow_pickle=False) as data:
            image = data["image"]
            mask = data["mask"]

            if image.shape != mask.shape or image.ndim != 3:
                return False

            if not np.isfinite(image).all():
                return False

            if not np.isin(np.unique(mask), [0, 1]).all():
                return False

            if not np.any(mask):
                return False

        return True

    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", default="datasets/manifests/cases_50.json")
    parser.add_argument("--output-dir", default="datasets/processed/lung_ct_50")
    parser.add_argument("--overlay-dir", default="datasets/validation/overlays_50")
    parser.add_argument("--report", default="datasets/manifests/preprocessing_report_50.json")
    parser.add_argument("--window-low", type=float, default=-1000)
    parser.add_argument("--window-high", type=float, default=400)
    parser.add_argument("--max-overlays", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    root = args.repo_root.resolve()

    if args.window_high <= args.window_low:
        parser.error("--window-high must exceed --window-low")

    records = load_case_records(root / args.manifest)

    # Absolute D: paths are supported; relative paths use project root.
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    overlay_dir = Path(args.overlay_dir)
    if not overlay_dir.is_absolute():
        overlay_dir = root / overlay_dir

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = root / report_path

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    failures = []
    overlay_count = 0

    for i, record in enumerate(records, start=1):
        case_id = record["case_id"]
        patient_id = record.get("patient_id")

        image_path = root / record["image_path"]
        mask_path = root / record["mask_path"]
        output_file = output_dir / f"{case_id}.npz"

        LOG.info("[%d/%d] Processing %s", i, len(records), case_id)

        try:
            # Reuse an existing complete file instead of processing it again.
            if output_file.exists() and not args.overwrite:
                if validate_existing_npz(output_file):
                    LOG.info("SKIP: Valid output already exists: %s", case_id)
                    summaries.append({
                        "case_id": case_id,
                        "patient_id": patient_id,
                        "output": str(output_file),
                        "status": "already_valid"
                    })
                    continue
                else:
                    LOG.warning("Existing output is invalid; regenerating %s", case_id)

            if not image_path.exists() or not mask_path.exists():
                raise FileNotFoundError("Image or mask file missing")

            image_nii = nib.load(str(image_path))
            mask_nii = nib.load(str(mask_path))

            if len(image_nii.shape) != 3 or len(mask_nii.shape) != 3:
                raise ValueError("Both files must be 3D")

            if image_nii.shape != mask_nii.shape:
                raise ValueError("Image and mask shapes differ")

            if not np.allclose(
                image_nii.affine,
                mask_nii.affine,
                atol=1e-4
            ):
                raise ValueError("Image and mask affine differ")

            image = image_nii.get_fdata(dtype=np.float32)
            mask_raw = np.asanyarray(mask_nii.dataobj)

            if not np.isfinite(image).all():
                raise ValueError("Image contains NaN or Inf")

            if not np.isfinite(mask_raw).all():
                raise ValueError("Mask contains NaN or Inf")

            if not np.isin(np.unique(mask_raw), [0, 1]).all():
                raise ValueError("Mask is not binary {0,1}")

            mask = (mask_raw > 0).astype(np.uint8)
            mask_voxels = int(np.count_nonzero(mask))

            if mask_voxels == 0:
                raise ValueError("Mask contains no positive voxels")

            axis = slice_axis_from_affine(image_nii.affine)
            n_slices = image.shape[axis]

            # Chunk-based normalization along the third array dimension.
            normalized = np.empty(image.shape, dtype=np.float16)

            low = args.window_low
            value_range = args.window_high - args.window_low
            chunk_size = 8

            for start in range(0, image.shape[2], chunk_size):
                end = min(start + chunk_size, image.shape[2])

                chunk = image[:, :, start:end].copy()
                np.clip(chunk, low, args.window_high, out=chunk)
                chunk -= low
                chunk /= value_range

                normalized[:, :, start:end] = chunk.astype(np.float16)
                del chunk

            positive_indices = [
                index
                for index in range(n_slices)
                if np.any(np.take(mask, index, axis=axis))
            ]

            if not positive_indices:
                raise ValueError("No positive nodule slices found")

            # Save to a temporary file first to avoid leaving partial NPZs.
            temporary_file = output_file.with_name(
                output_file.stem + ".tmp.npz"
            )

            try:
                np.savez_compressed(
                    temporary_file,
                    image=normalized,
                    mask=mask,
                    slice_axis=np.int8(axis)
                )
                temporary_file.replace(output_file)
            finally:
                if temporary_file.exists():
                    temporary_file.unlink()

            del image, mask_raw, mask, normalized

            case_overlay_count = 0

            for index in positive_indices:
                if overlay_count >= args.max_overlays:
                    break

                with np.load(output_file, allow_pickle=False) as saved:
                    image_slice = np.take(
                        saved["image"], index, axis=axis
                    )
                    mask_slice = np.take(
                        saved["mask"], index, axis=axis
                    )

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image_slice, cmap="gray", vmin=0, vmax=1)
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
                case_overlay_count += 1

            summaries.append({
                "case_id": case_id,
                "patient_id": patient_id,
                "series_uid": record.get("series_uid"),
                "positive_slice_count": len(positive_indices),
                "mask_voxels": mask_voxels,
                "output": str(output_file),
                "n_slices": n_slices,
                "slice_axis": axis,
                "overlays_saved": case_overlay_count,
                "status": "processed"
            })

            LOG.info(
                "PASS %s | slices=%d | positive=%d | voxels=%d",
                case_id,
                n_slices,
                len(positive_indices),
                mask_voxels
            )

        except Exception as error:
            failures.append({
                "case_id": case_id,
                "patient_id": patient_id,
                "error": f"{type(error).__name__}: {error}"
            })

            LOG.exception("FAILED %s", case_id)

    report = {
        "manifest": str(args.manifest),
        "window": {
            "low": args.window_low,
            "high": args.window_high
        },
        "total_cases": len(records),
        "processed_cases": len(summaries),
        "failed_cases": len(failures),
        "overlay_count": overlay_count,
        "cases": summaries,
        "failures": failures
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nPREPROCESSING RESULT")
    print(f"Successful/skipped: {len(summaries)}/{len(records)}")
    print(f"Failed: {len(failures)}")
    print(f"New overlays: {overlay_count}")
    print(f"Report: {report_path}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
