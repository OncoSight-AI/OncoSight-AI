#!/usr/bin/env python3
"""Select a small LIDC-IDRI NIfTI pilot; dry-run does not download or modify files."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

LOG = logging.getLogger("select_pilot_cases")
DEFAULT_REPO = "MedOtter/LIDC-IDRI"


def read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except Exception as exc:
        LOG.warning("Could not parse existing pilot manifest: %s", exc)
        return []


def validate_pair(image_path: Path, mask_path: Path) -> tuple[bool, str]:
    if not image_path.is_file() or not mask_path.is_file():
        return False, "image or mask file missing"
    try:
        image_nii = nib.load(str(image_path))
        mask_nii = nib.load(str(mask_path))
        if image_nii.shape != mask_nii.shape:
            return False, f"shape mismatch: {image_nii.shape} vs {mask_nii.shape}"
        if not np.allclose(image_nii.affine, mask_nii.affine, atol=1e-4):
            return False, "affine mismatch"
        mask_values = np.unique(np.asanyarray(mask_nii.dataobj))
        if not np.isin(mask_values, [0, 1]).all():
            return False, f"mask not binary: {mask_values[:10].tolist()}"
        if not np.any(mask_values == 1):
            return False, "mask is empty"
        return True, "valid"
    except Exception as exc:
        return False, f"NIfTI validation error: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.n_cases < 1:
        parser.error("--n-cases must be at least 1")

    root = args.repo_root.resolve()
    csv_path = root / "datasets" / "manifests" / "hf_metadata.csv"
    pilot_path = root / "datasets" / "manifests" / "pilot_cases.json"
    image_dir = root / "datasets" / "raw" / "lidc_hf" / "images"
    mask_dir = root / "datasets" / "raw" / "lidc_hf" / "masks"

    if not csv_path.is_file():
        LOG.error("Metadata CSV not found: %s", csv_path)
        return 2

    df = pd.read_csv(csv_path)
    required = {
        "case_id", "patient_id", "series_uid", "manufacturer",
        "has_seg", "n_nodules_default",
    }
    missing = required.difference(df.columns)
    if missing:
        LOG.error("Metadata CSV missing required columns: %s", sorted(missing))
        return 2

    df["has_seg"] = (
        df["has_seg"].astype(str).str.strip().str.lower()
        .isin({"true", "1", "yes"})
    )
    df["n_nodules_default"] = pd.to_numeric(
        df["n_nodules_default"], errors="coerce"
    ).fillna(0)

    candidates = df.loc[
        df["has_seg"] & (df["n_nodules_default"] >= 1)
    ].copy()
    candidates = candidates.drop_duplicates(
        subset=["patient_id", "series_uid"], keep="first"
    )

    old_ids = {
        item.get("case_id") for item in read_existing(pilot_path)
        if isinstance(item, dict) and item.get("case_id")
    }
    candidates["_old_pilot_priority"] = candidates["case_id"].isin(old_ids).map(
        {True: 0, False: 1}
    )
    candidates = candidates.sort_values(
        ["_old_pilot_priority", "patient_id", "case_id"]
    )

    # Round-robin manufacturer groups to improve representation.
    groups: dict[str, list[dict]] = {}
    for manufacturer, group in candidates.groupby("manufacturer", dropna=False, sort=True):
        label = "UNKNOWN" if pd.isna(manufacturer) else str(manufacturer)
        groups[label] = group.to_dict("records")

    ordered: list[dict] = []
    while any(groups.values()):
        for manufacturer in sorted(groups):
            if groups[manufacturer]:
                ordered.append(groups[manufacturer].pop(0))

    selected: list[dict] = []
    seen_patients: set[str] = set()
    seen_series: set[str] = set()
    for row in ordered:
        patient_id = str(row["patient_id"])
        series_uid = str(row["series_uid"])
        if patient_id in seen_patients or series_uid in seen_series:
            continue
        selected.append(row)
        seen_patients.add(patient_id)
        seen_series.add(series_uid)
        if len(selected) >= args.n_cases:
            break

    print(f"Metadata rows: {len(df)}", flush=True)
    print(f"Eligible rows (has_seg and n_nodules_default >= 1): {len(candidates)}", flush=True)
    print(f"Selected unique patients/series: {len(selected)}", flush=True)

    output_records: list[dict] = []
    for index, row in enumerate(selected, start=1):
        case_id = str(row["case_id"])
        image_path = image_dir / f"{case_id}.nii.gz"
        mask_path = mask_dir / f"{case_id}.nii.gz"
        valid, reason = validate_pair(image_path, mask_path)
        record = {
            "case_id": case_id,
            "patient_id": str(row["patient_id"]),
            "series_uid": str(row["series_uid"]),
            "manufacturer": str(row["manufacturer"]),
            "n_slices": (
                int(row["n_slices"])
                if "n_slices" in row and pd.notna(row["n_slices"]) else None
            ),
            "n_nodules_default": int(row["n_nodules_default"]),
            "image_path": str(image_path.relative_to(root)),
            "mask_path": str(mask_path.relative_to(root)),
            "already_valid": valid,
            "validation": reason,
        }
        output_records.append(record)
        print(
            f"{index:02d}. {case_id} | patient={record['patient_id']} | "
            f"manufacturer={record['manufacturer']} | local_pair={reason}",
            flush=True,
        )

    print(
        f"Unique patients: {len({x['patient_id'] for x in output_records})}; "
        f"unique series: {len({x['series_uid'] for x in output_records})}",
        flush=True,
    )

    if args.dry_run:
        print("DRY RUN complete: no downloads and no files changed.", flush=True)
        return 0

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    validated_records: list[dict] = []

    for record in output_records:
        image_path = root / record["image_path"]
        mask_path = root / record["mask_path"]
        valid, _ = validate_pair(image_path, mask_path)

        if not valid:
            for subfolder, target in (("images", image_path), ("masks", mask_path)):
                # Download each missing/invalid member to a temporary file first.
                if target.is_file():
                    target.unlink()
                remote_name = f"{subfolder}/{record['case_id']}.nii.gz"
                LOG.info("Downloading %s", remote_name)
                cached_path = Path(hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename=remote_name,
                    revision=args.revision,
                ))
                target.parent.mkdir(parents=True, exist_ok=True)
                temp_path = target.with_name(target.name + ".tmp")
                shutil.copy2(cached_path, temp_path)
                temp_path.replace(target)

        valid, reason = validate_pair(image_path, mask_path)
        if valid:
            record["already_valid"] = True
            record["validation"] = "valid"
            validated_records.append(record)
        else:
            LOG.error("Case %s failed validation after download: %s",
                      record["case_id"], reason)

    if pilot_path.exists():
        backup = pilot_path.with_suffix(pilot_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(pilot_path, backup)
            LOG.info("Backed up existing manifest to %s", backup)

    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_path.write_text(
        json.dumps(validated_records, indent=2), encoding="utf-8"
    )
    LOG.info("Validated %d of %d selected cases.", len(validated_records), len(output_records))


if __name__ == '__main__':
    raise SystemExit(main())
