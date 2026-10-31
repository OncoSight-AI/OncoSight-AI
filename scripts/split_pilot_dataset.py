#!/usr/bin/env python3
"""Create patient-level train/validation/test splits for a small CT pilot."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("split_pilot_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not np.isclose(args.train + args.val + args.test, 1.0):
        parser.error("--train + --val + --test must equal 1.0")

    root = args.repo_root.resolve()
    report_file = root / "datasets/manifests/preprocessing_report.json"
    if not report_file.exists():
        LOG.error("Preprocessing report not found. Run preprocess_lung_ct.py first.")
        return 2
    report = json.loads(report_file.read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    if not cases:
        LOG.error("No processed cases found.")
        return 2

    # One record per patient: duplicate patient entries are rejected to prevent leakage.
    patients = {}
    for case in cases:
        patient = case.get("patient_id")
        if not patient:
            LOG.error("Missing patient_id for case %s", case.get("case_id"))
            return 2
        if patient in patients:
            LOG.error("Multiple scans for patient %s. Resolve to one scan per patient before splitting.",
                      patient)
            return 2
        patients[patient] = case

    rng = np.random.default_rng(args.seed)
    patient_ids = np.array(sorted(patients), dtype=object)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_train = max(1, int(round(n * args.train)))
    n_val = max(1, int(round(n * args.val))) if n >= 3 else 0
    # Keep at least one test patient where possible.
    if n >= 3:
        n_val = max(1, min(n_val, n - 2))
        n_train = max(1, min(n_train, n - n_val - 1))
        n_test = n - n_train - n_val
    elif n == 2:
        n_train, n_val, n_test = 1, 0, 1
    else:
        n_train, n_val, n_test = 1, 0, 0

    train_ids = patient_ids[:n_train].tolist()
    val_ids = patient_ids[n_train:n_train + n_val].tolist()
    test_ids = patient_ids[n_train + n_val:n_train + n_val + n_test].tolist()
    assignments = {}
    for split, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
        for patient in ids:
            assignments[patient] = split

    split_cases = []
    for patient, case in patients.items():
        split_cases.append({
            "case_id": case["case_id"],
            "patient_id": patient,
            "series_uid": case.get("series_uid"),
            "split": assignments[patient],
            "positive_slice_count": int(case.get("positive_slice_count", 0)),
            "mask_voxels": int(case.get("mask_voxels", 0)),
            "volume": case.get("output"),
        })

    # Explicit leakage assertion.
    sets = [set(train_ids), set(val_ids), set(test_ids)]
    if (sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]):
        LOG.error("Patient leakage detected.")
        return 3

    summary = {
        "seed": args.seed,
        "requested_ratios": {"train": args.train, "validation": args.val, "test": args.test},
        "actual_counts": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
    }
    summary["patient_ids"] = {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }

    summary["total_cases"] = len(split_cases)
    summary["cases"] = split_cases

    output_file = root / "datasets/manifests/split_manifest.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    LOG.info("Dataset splitting completed successfully.")
    LOG.info("Training patients: %d", len(train_ids))
    LOG.info("Validation patients: %d", len(val_ids))
    LOG.info("Testing patients: %d", len(test_ids))
    LOG.info("Total patients: %d", n)
    LOG.info("Manifest saved: %s", output_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())