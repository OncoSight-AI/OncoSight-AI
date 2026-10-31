#!/usr/bin/env python
"""Pilot preprocessing test for ~10-20 LIDC-IDRI patients (reads datasets/manifests/pilot_patients.json).

For each patient (resumable: finished patients are skipped unless --force):
  * load the CT series, sort by z, de-duplicate, convert to Hounsfield units
  * rasterise reader contours into per-annotation masks (inclusion / exclusion handled)
  * group annotations of the same physical nodule across readers (simple centroid heuristic, PILOT ONLY)
  * save  volume (int16 HU), agreement-count mask, 50%-consensus mask, metadata.json, overlay PNGs

Original dataset files are only ever opened for reading. Outputs go to datasets/processed/pilot/.
No model is trained and no malignancy label is produced here.
"""
import argparse, json, math, os, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidc_xml import parse_lidc_xml  # noqa: E402

HU_WINDOW = (-1000.0, 400.0)   # lung-oriented window used for normalisation / overlays


def normalize_hu(vol, lo=HU_WINDOW[0], hi=HU_WINDOW[1]):
    """Clip to [lo, hi] HU and scale to [0, 1]. Applied at train time, not stored."""
    return (np.clip(vol.astype(np.float32), lo, hi) - lo) / (hi - lo)


def load_series(root, series_dir):
    files = sorted(Path(root, series_dir).glob("*"))
    slices, seen, dropped = [], set(), 0
    for f in files:
        if f.suffix.lower() not in (".dcm", ""):
            continue
        try:
            ds = pydicom.dcmread(str(f))
            _ = ds.ImagePositionPatient
        except Exception:
            continue
        if ds.SOPInstanceUID in seen:
            dropped += 1
            continue
        seen.add(ds.SOPInstanceUID)
        slices.append(ds)
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    shapes = {(int(d.Rows), int(d.Columns)) for d in slices}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent slice shapes: {shapes}")
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1) or 1)
                    + float(getattr(d, "RescaleIntercept", 0) or 0) for d in slices])
    vol = np.clip(np.rint(vol), -2048, 4095).astype(np.int16)
    zs = np.array([float(d.ImagePositionPatient[2]) for d in slices])
    ps = [float(x) for x in slices[0].PixelSpacing]
    meta = {
        "n_slices": len(slices), "dropped_duplicate_slices": dropped, "shape_zyx": list(vol.shape),
        "spacing_mm_zyx": [float(np.median(np.diff(zs))) if len(zs) > 1 else None, ps[0], ps[1]],
        "z_positions": zs.tolist(), "sop_uids": [str(d.SOPInstanceUID) for d in slices],
        "instance_numbers": [int(d.InstanceNumber) if "InstanceNumber" in d else None for d in slices],
        "patient_id": str(slices[0].PatientID), "study_uid": str(slices[0].StudyInstanceUID),
        "series_uid": str(slices[0].SeriesInstanceUID), "modality": str(slices[0].Modality),
        "manufacturer": str(getattr(slices[0], "Manufacturer", "")),
        "convolution_kernel": str(getattr(slices[0], "ConvolutionKernel", "")),
        "rescale_slope": float(getattr(slices[0], "RescaleSlope", 1) or 1),
        "rescale_intercept": float(getattr(slices[0], "RescaleIntercept", 0) or 0),
    }
    return vol, zs, meta


def rasterise(shape_zyx, zs, sop_to_idx, rois):
    """One annotation -> uint8 mask. Inclusion polygons are filled, exclusion polygons are subtracted."""
    D, H, W = shape_zyx
    mask = np.zeros(shape_zyx, np.uint8)
    markers, unresolved = [], 0
    by_slice = {}
    for r in rois:
        idx = sop_to_idx.get(r["sop_uid"])
        if idx is None and r["z"] is not None:           # fallback: nearest z within 0.5 mm
            j = int(np.argmin(np.abs(zs - r["z"])))
            idx = j if abs(zs[j] - r["z"]) < 0.5 else None
        if idx is None:
            unresolved += 1
            continue
        by_slice.setdefault(idx, []).append(r)
    for idx, rs in by_slice.items():
        inc = Image.new("L", (W, H), 0)
        exc = Image.new("L", (W, H), 0)
        for r in rs:
            pts = r["points"]
            if len(pts) == 1:                              # single-point marker, no contour
                markers.append((idx, pts[0][1], pts[0][0]))
            elif len(pts) >= 3:
                ImageDraw.Draw(inc if r["inclusion"] else exc).polygon(pts, fill=1, outline=1)
        m = (np.array(inc) > 0) & ~(np.array(exc) > 0)
        mask[idx] = m.astype(np.uint8)
    return mask, markers, unresolved


def centroid_mm(mask, spacing):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    return (idx.mean(0) * np.array(spacing)).tolist()


def group_annotations(anns, match_mm):
    """Greedy grouping by centroid distance. PILOT heuristic only (documented in the report)."""
    groups = []
    for a in anns:
        c = np.array(a["centroid_mm"])
        for g in groups:
            if a["reader"] not in g["readers"] and np.linalg.norm(np.array(g["centroid_mm"]) - c) <= match_mm:
                g["members"].append(a)
                g["readers"].add(a["reader"])
                g["centroid_mm"] = np.mean([m["centroid_mm"] for m in g["members"]], 0).tolist()
                break
        else:
            groups.append({"members": [a], "readers": {a["reader"]}, "centroid_mm": list(c)})
    return groups


def overlay_png(vol, mask, z, path):
    sl = (normalize_hu(vol[z]) * 255).astype(np.uint8)
    rgb = np.stack([sl] * 3, -1)
    m = mask[z] > 0
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    rgb[edge] = [255, 0, 0]
    Image.fromarray(rgb).save(path)


def process(root, entry, out_dir, match_mm, force):
    pid = entry["patient_id"]
    pdir = out_dir / pid
    if (pdir / "DONE.json").exists() and not force:
        return json.loads((pdir / "DONE.json").read_text()), True
    t0 = time.time()
    pdir.mkdir(parents=True, exist_ok=True)
    vol, zs, meta = load_series(root, entry["series_dir"])
    sop_to_idx = {s: i for i, s in enumerate(meta["sop_uids"])}
    xml = parse_lidc_xml(Path(root, entry["xml"]))
    if xml["error"]:
        raise ValueError(f"XML error: {xml['error']}")
    if xml["series_uid"] != meta["series_uid"]:
        raise ValueError("XML series UID does not match DICOM series")
    sp = meta["spacing_mm_zyx"]
    anns, marker_only, unresolved, uncharacterised = [], [], 0, 0
    for ri, s in enumerate(xml["sessions"]):
        for n in s["nodules"]:
            mask, markers, un = rasterise(vol.shape, zs, sop_to_idx, n["rois"])
            unresolved += un
            if mask.any():
                anns.append({"reader": ri, "nodule_id": n["id"], "mask": mask, "centroid_mm": centroid_mm(mask, sp),
                             "characteristics": n["characteristics"], "has_characteristics": n["has_characteristics"]})
            elif markers:
                marker_only.append({"reader": ri, "nodule_id": n["id"], "markers_zyx": markers,
                                    "characteristics": n["characteristics"]})
            uncharacterised += not n["has_characteristics"]
    groups = group_annotations(anns, match_mm)
    agreement = np.zeros(vol.shape, np.uint8)
    consensus = np.zeros(vol.shape, np.uint8)
    group_meta = []
    for gi, g in enumerate(groups, 1):
        stack = np.sum([m["mask"] for m in g["members"]], 0).astype(np.uint8)
        agreement = np.maximum(agreement, stack)
        cons = stack >= max(1, math.ceil(0.5 * len(g["members"])))
        consensus[cons] = np.maximum(consensus[cons], gi if gi < 255 else 255)
        vox = int(cons.sum())
        vol_mm3 = vox * sp[0] * sp[1] * sp[2]
        zz = np.argwhere(cons)
        group_meta.append({
            "group_id": gi, "n_annotations": len(g["members"]), "readers": sorted(g["readers"]),
            "consensus_voxels": vox, "consensus_volume_mm3": round(vol_mm3, 2),
            "equivalent_diameter_mm": round((6 * vol_mm3 / math.pi) ** (1 / 3), 2) if vol_mm3 > 0 else None,
            "bbox_zyx_min": zz.min(0).tolist() if len(zz) else None, "bbox_zyx_max": zz.max(0).tolist() if len(zz) else None,
            "centroid_mm_zyx": [round(x, 2) for x in g["centroid_mm"]],
            "per_annotation_characteristics": [m["characteristics"] for m in g["members"]],
            "malignancy_ratings_are_subjective_radiologist_impressions": True})
        if gi <= 3 and len(zz):
            zmid = int(np.median(zz[:, 0]))
            overlay_png(vol, cons, zmid, pdir / f"overlay_group{gi}_z{zmid}.png")
    np.savez_compressed(pdir / "data.npz", volume_hu=vol, agreement_count=agreement, consensus_mask=consensus)
    meta_out = {**{k: v for k, v in meta.items() if k not in ("sop_uids",)}, "sop_uids_in_order": meta["sop_uids"],
                "source_series_dir": entry["series_dir"], "source_xml": entry["xml"],
                "hu_window_for_normalisation": HU_WINDOW, "group_match_mm": match_mm,
                "group_heuristic": "greedy centroid proximity, one annotation per reader per group (pilot only)",
                "consensus_rule": "voxel kept if marked by >=50% of the annotations in the group",
                "nodule_groups": group_meta, "marker_only_annotations": marker_only,
                "n_annotations_with_contours": len(anns), "n_uncharacterised_annotations": uncharacterised,
                "unresolved_roi_slices": unresolved, "n_reader_sessions": len(xml["sessions"]),
                "non_nodule_marks": sum(len(s["non_nodules"]) for s in xml["sessions"])}
    (pdir / "metadata.json").write_text(json.dumps(meta_out, indent=2, default=str))
    summary = {"patient_id": pid, "status": "ok", "shape_zyx": meta["shape_zyx"], "spacing_mm_zyx": sp,
               "hu_min": int(vol.min()), "hu_max": int(vol.max()), "n_groups": len(groups),
               "n_annotations_with_contours": len(anns), "n_marker_only": len(marker_only),
               "unresolved_roi_slices": unresolved, "dropped_duplicate_slices": meta["dropped_duplicate_slices"],
               "seconds": round(time.time() - t0, 1)}
    (pdir / "DONE.json").write_text(json.dumps(summary, indent=2))
    return summary, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="same dataset folder given to inspect_lidc.py")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--match-mm", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    repo = Path(a.repo_root).resolve()
    root = Path(a.root).resolve()
    out_dir = repo / "datasets" / "processed" / "pilot"
    try:
        out_dir.relative_to(root)
        sys.exit("Refusing to write inside the dataset folder.")
    except ValueError:
        pass
    pilot = json.loads((repo / "datasets" / "manifests" / "pilot_patients.json").read_text())
    if a.limit:
        pilot = pilot[:a.limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, e in enumerate(pilot, 1):
        try:
            res, skipped = process(str(root), e, out_dir, a.match_mm, a.force)
            print(f"[{i}/{len(pilot)}] {e['patient_id']}: {'skipped (done)' if skipped else 'ok'} "
                  f"groups={res['n_groups']} shape={res['shape_zyx']}", flush=True)
        except Exception as ex:
            res = {"patient_id": e["patient_id"], "status": "failed", "error": f"{type(ex).__name__}: {ex}"[:300]}
            print(f"[{i}/{len(pilot)}] {e['patient_id']}: FAILED {res['error']}", flush=True)
        results.append(res)
    (repo / "datasets" / "manifests" / "pilot_preprocess_report.json").write_text(json.dumps(results, indent=2))
    ok = sum(r["status"] == "ok" for r in results)
    print(f"\n{ok}/{len(results)} patients processed. Report: datasets/manifests/pilot_preprocess_report.json")


if __name__ == "__main__":
    main()