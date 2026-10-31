#!/usr/bin/env python
"""Decode ONE real CT volume + its segmentation mask from MedOtter/LIDC-IDRI and verify them.

Uses the existing Hugging Face loader (streaming), stops at the first row that has a mask,
so only that one case is downloaded - never the full ~71 GB.

Checks: shape, dtype, affine/orientation, voxel spacing, HU plausibility, mask label values,
foreground voxels + connected components, image/mask alignment (shape, affine, and an
intensity-contrast test against axis-flipped masks), slice count vs `n_slices`.
Produces: CT / mask / overlay figures and a PASS/WARN/FAIL report with the numbers actually measured.

Writes (never touches existing files):
  datasets/raw/lidc_hf_peek/<case_id>/{image,mask}.nii.gz     untouched copy of the decoded case
  datasets/validation/<case_id>_slice_view.png, <case_id>_orthogonal_view.png
  datasets/manifests/validation_report.json
  datasets/IMAGE_MASK_VALIDATION.md

  python scripts/validate_image_mask.py --repo-root .
  python scripts/validate_image_mask.py --repo-root . --case-id LIDC-IDRI-0078     # a specific case
  python scripts/validate_image_mask.py --repo-root . --image-field X --mask-field Y   # if auto-detect is ambiguous
"""
import argparse, gzip, io, json, os, re, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_NAMES = ("image", "img", "ct", "volume", "scan")
MASK_NAMES = ("mask", "seg", "label", "annotation", "gt")


# ----------------------------------------------------------------------------- field handling
def is_empty(v):
    if v is None:
        return True
    if isinstance(v, dict):
        return not v.get("bytes") and not v.get("path")
    if isinstance(v, (str, bytes)):
        return len(v) == 0
    return False


def niftiish(v):
    if isinstance(v, nib.spatialimages.SpatialImage):
        return True
    if isinstance(v, dict) and ("bytes" in v or "path" in v):
        return True
    if isinstance(v, (bytes, bytearray)):
        return True
    if isinstance(v, str) and re.search(r"\.nii(\.gz)?$", v.lower()):
        return True
    return isinstance(v, np.ndarray)


def summarise(v):
    if isinstance(v, nib.spatialimages.SpatialImage):
        return f"nibabel image shape={v.shape}"
    if isinstance(v, dict):
        b = v.get("bytes")
        return f"dict(bytes={len(b) if b else b}, path={v.get('path')!r})"
    if isinstance(v, (bytes, bytearray)):
        return f"bytes[{len(v)}]"
    if isinstance(v, np.ndarray):
        return f"ndarray shape={v.shape} dtype={v.dtype}"
    return repr(v)[:100]


def detect_fields(sample, image_field, mask_field):
    keys = list(sample.keys())
    cand = [k for k in keys if niftiish(sample[k]) or is_empty(sample[k]) and any(n in k.lower() for n in MASK_NAMES + IMG_NAMES)]
    img = image_field or next((k for k in cand if any(n in k.lower() for n in IMG_NAMES) and not any(m in k.lower() for m in MASK_NAMES)), None)
    msk = mask_field or next((k for k in cand if any(n in k.lower() for n in MASK_NAMES)), None)
    return img, msk, cand


def materialise(value, repo, out_path):
    """Write the value as a .nii.gz file at out_path (untouched bytes whenever possible) and return how."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, nib.spatialimages.SpatialImage):
        nib.save(value, str(out_path))
        return "nibabel image re-saved (datasets decoded it; bytes not available)"
    raw, how = None, ""
    if isinstance(value, dict):
        if value.get("bytes"):
            raw, how = value["bytes"], f"bytes from column (path={value.get('path')!r})"
        elif value.get("path"):
            value = value["path"]
    if raw is None and isinstance(value, (bytes, bytearray)):
        raw, how = bytes(value), "raw bytes"
    if raw is None and isinstance(value, str):
        if os.path.exists(value):
            raw, how = Path(value).read_bytes(), f"local file {value}"
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=repo, filename=value, repo_type="dataset")
            raw, how = Path(p).read_bytes(), f"downloaded single file {value} from the Hub"
    if raw is None and isinstance(value, np.ndarray):
        nib.save(nib.Nifti1Image(value, np.eye(4)), str(out_path))
        return "ndarray wrapped with IDENTITY affine (no spacing/orientation info in source!)"
    if raw is None:
        raise ValueError(f"cannot interpret value of type {type(value)}")
    if raw[:2] != b"\x1f\x8b":                      # plain .nii -> store as-is but keep a .nii.gz name via recompress? keep exact bytes instead
        out_path = out_path.with_suffix("")          # image.nii
    out_path.write_bytes(raw)
    return how + (" [uncompressed NIfTI saved as .nii]" if out_path.suffix == ".nii" else "")


def find_saved(d, stem):
    for ext in (".nii.gz", ".nii"):
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    return d / f"{stem}.nii.gz"


# ----------------------------------------------------------------------------- analysis helpers
def contrast(hu, mask):
    """mean HU inside mask minus mean HU in a 3-voxel ring around it (crop for speed)."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    lo = np.maximum(idx.min(0) - 10, 0)
    hi = np.minimum(idx.max(0) + 11, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    m, h = mask[sl], hu[sl]
    ring = ndi.binary_dilation(m, iterations=4) & ~ndi.binary_dilation(m, iterations=1)
    if not ring.any():
        return None
    return float(h[m].mean() - h[ring].mean())


def status(ok, warn=False):
    return "PASS" if ok else ("WARN" if warn else "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="MedOtter/LIDC-IDRI")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-files", default=None, help="load local parquet files instead of the Hub (offline test)")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--max-scan", type=int, default=60, help="max streamed rows to look through for a row with a mask")
    ap.add_argument("--image-field", default=None)
    ap.add_argument("--mask-field", default=None)
    ap.add_argument("--hu-window", default="-1000,400")
    a = ap.parse_args()
    lo_w, hi_w = (float(x) for x in a.hu_window.split(","))

    from datasets import load_dataset
    root = Path(a.repo_root).resolve()
    t0 = time.time()
    print("Connecting to Hugging Face dataset (streaming) ...", flush=True)
    if a.data_files:
        ds = load_dataset("parquet", data_files=a.data_files, split="train", streaming=True)
    else:
        ds = load_dataset(a.repo, split=a.split, streaming=True)

    # ---- find a row that has a mask (and optionally a given case)
    sample, skipped = None, []
    for i, row in enumerate(ds):
        if i == 0:
            first_keys = {k: summarise(v) for k, v in row.items()}
            print("Fields in first row:")
            for k, v in first_keys.items():
                print(f"  {k}: {v}")
            img_f, msk_f, cand = detect_fields(row, a.image_field, a.mask_field)
            if not img_f or not msk_f or img_f == msk_f:
                sys.exit(f"Could not tell image/mask fields apart. Candidate fields: {cand}. Re-run with --image-field/--mask-field.")
            print(f"Using image field '{img_f}' and mask field '{msk_f}'")
        if i >= a.max_scan:
            break
        cid = row.get("case_id")
        if a.case_id and cid != a.case_id:
            continue
        if is_empty(row[msk_f]):
            skipped.append(cid)
            continue
        sample = row
        break
    if sample is None:
        sys.exit(f"No row with a mask found (looked at up to {a.max_scan} rows; rows without mask: {skipped[:10]}).")

    case = str(sample.get("case_id", "unknown_case"))
    scalar = {k: v for k, v in sample.items() if k not in (img_f, msk_f) and not niftiish(v)}
    print(f"\nSelected case {case}; scalar fields: {scalar}", flush=True)

    # ---- save untouched copies, then load them from disk
    peek_dir = root / "datasets" / "raw" / "lidc_hf_peek" / case
    how_img = materialise(sample[img_f], a.repo, peek_dir / "image.nii.gz") if not find_saved(peek_dir, "image").exists() else "already saved earlier"
    how_msk = materialise(sample[msk_f], a.repo, peek_dir / "mask.nii.gz") if not find_saved(peek_dir, "mask").exists() else "already saved earlier"
    img_path, msk_path = find_saved(peek_dir, "image"), find_saved(peek_dir, "mask")

    issues, checks = [], []
    def add(name, st, detail):
        checks.append({"check": name, "status": st, "detail": detail})
        if st != "PASS":
            issues.append(f"{st}: {name} - {detail}")

    try:
        img = nib.load(str(img_path))
        add("Decode CT image (NIfTI)", "PASS", f"{img_path.name}, {img_path.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        add("Decode CT image (NIfTI)", "FAIL", f"{type(e).__name__}: {e}")
        img = None
    try:
        msk = nib.load(str(msk_path))
        add("Decode segmentation mask (NIfTI)", "PASS", f"{msk_path.name}, {msk_path.stat().st_size / 1e6:.2f} MB")
    except Exception as e:
        add("Decode segmentation mask (NIfTI)", "FAIL", f"{type(e).__name__}: {e}")
        msk = None
    R = {"generated": datetime.now().isoformat(timespec="seconds"), "case_id": case, "scalar_fields": scalar,
         "first_row_fields": first_keys, "image_field": img_f, "mask_field": msk_f, "image_source": how_img,
         "mask_source": how_msk, "rows_without_mask_skipped_first": skipped}

    if img is not None and msk is not None:
        ih, mh = img.header, msk.header
        R["image"] = {"shape": list(img.shape), "stored_dtype": str(img.get_data_dtype()), "zooms_mm": [round(float(z), 4) for z in img.header.get_zooms()[:3]],
                      "axcodes": "".join(nib.aff2axcodes(img.affine)), "affine": np.round(img.affine, 4).tolist(),
                      "scl_slope": None if np.isnan(ih["scl_slope"]) else float(ih["scl_slope"]),
                      "scl_inter": None if np.isnan(ih["scl_inter"]) else float(ih["scl_inter"])}
        R["mask"] = {"shape": list(msk.shape), "stored_dtype": str(msk.get_data_dtype()), "zooms_mm": [round(float(z), 4) for z in msk.header.get_zooms()[:3]],
                     "axcodes": "".join(nib.aff2axcodes(msk.affine)), "affine": np.round(msk.affine, 4).tolist()}
        hu = img.get_fdata(dtype=np.float32)
        m_raw = np.asanyarray(msk.dataobj)
        if hu.ndim != 3 or m_raw.ndim != 3:
            add("Dimensionality", "FAIL", f"image ndim={hu.ndim}, mask ndim={m_raw.ndim} (expected 3)")
        else:
            add("Dimensionality", "PASS", f"image {hu.shape}, mask {m_raw.shape}")
        finite = bool(np.isfinite(hu).all())
        add("Image values finite", status(finite), "no NaN/Inf" if finite else "contains NaN/Inf")
        hmin, hmax = float(np.nanmin(hu)), float(np.nanmax(hu))
        R["image"].update(min=hmin, max=hmax, mean=float(np.nanmean(hu)), median=float(np.nanmedian(hu[::4, ::4, ::2])))
        hu_ok = (-3100 <= hmin <= -500) and (200 <= hmax <= 4100)
        add("Image intensities look like Hounsfield units", status(hu_ok, warn=True),
            f"min {hmin:.0f}, max {hmax:.0f} (expect roughly min -1024..-2000, max >200). If WARN, values may be normalised or scaled - check before choosing a window")
        zm = R["image"]["zooms_mm"]
        add("Voxel spacing positive and plausible", status(all(0.2 < z < 10 for z in zm), warn=True), f"{zm} mm")
        slice_axis = next((i for i, c in enumerate(R["image"]["axcodes"]) if c in "SI"), 2)
        add("Axial (S/I) axis identified", "PASS" if any(c in "SI" for c in R["image"]["axcodes"]) else "WARN",
            f"axcodes {R['image']['axcodes']}; slice axis = array axis {slice_axis}" + ("" if R['image']['axcodes'] == 'RAS' else "; not canonical RAS -> reorient with nib.as_closest_canonical before use"))
        # image vs mask geometry
        same_shape = tuple(img.shape) == tuple(msk.shape)
        add("Image and mask have identical shape", status(same_shape), f"{img.shape} vs {msk.shape}")
        same_aff = bool(np.allclose(img.affine, msk.affine, atol=1e-3))
        add("Image and mask affines match (same physical space)", status(same_aff), "match within 1e-3" if same_aff else f"image {np.round(img.affine, 3).tolist()} vs mask {np.round(msk.affine, 3).tolist()}")
        ns = scalar.get("n_slices")
        if ns is not None:
            hit = [ax for ax, n in enumerate(img.shape) if n == ns]
            add("Slice count equals `n_slices` metadata", status(bool(hit), warn=True), f"n_slices={ns}, image shape={img.shape}" + (f", matches array axis {hit}" if hit else ""))
        # mask labels
        vals, counts = np.unique(m_raw, return_counts=True)
        R["mask"]["unique_values"] = {str(v): int(c) for v, c in zip(vals[:30], counts[:30])}
        integral = bool(np.all(np.equal(np.mod(vals, 1), 0)))
        add("Mask values are integer labels", status(integral), f"unique values: {R['mask']['unique_values']}" + (" (more not shown)" if len(vals) > 30 else ""))
        binary = set(vals.tolist()) <= {0, 1}
        add("Mask is binary {0,1}", "PASS" if binary else "WARN", "binary" if binary else f"labels {vals[:10].tolist()} - per-nodule ids or multi-class? decide how to binarise (foreground = >0)")
        fg = m_raw > 0
        nfg = int(fg.sum())
        R["mask"]["foreground_voxels"] = nfg
        R["mask"]["foreground_fraction"] = nfg / fg.size
        add("Foreground voxels exist", status(nfg > 0), f"{nfg} voxels ({100 * nfg / fg.size:.5f}% of volume)")
        if nfg and same_shape:
            vox_mm3 = float(np.prod(zm))
            lab, ncomp = ndi.label(fg, structure=np.ones((3, 3, 3)))
            sizes = np.bincount(lab.ravel())[1:]
            comps = []
            objs = ndi.find_objects(lab)
            for ci, sz in enumerate(sizes, 1):
                bb = objs[ci - 1]
                vmm = float(sz) * vox_mm3
                comps.append({"component": ci, "voxels": int(sz), "volume_mm3": round(vmm, 1),
                              "equiv_diameter_mm": round((6 * vmm / np.pi) ** (1 / 3), 2),
                              "bbox": None if bb is None else [[s.start, s.stop] for s in bb]})
            R["mask"]["n_components_26conn"] = int(ncomp)
            R["mask"]["components"] = comps[:40]
            R["mask"]["slices_with_foreground"] = int((fg.sum(axis=tuple(i for i in range(3) if i != slice_axis)) > 0).sum())
            tiny = [c for c in comps if c["voxels"] < 8]
            add("Foreground components", "PASS" if not tiny else "WARN", f"{ncomp} component(s); diameters mm: {[c['equiv_diameter_mm'] for c in comps[:12]]}" + (f"; {len(tiny)} tiny (<8 voxels)" if tiny else ""))
            # contrast / flip test
            c0 = contrast(hu, fg)
            flips = {}
            for ax, nm in enumerate("012"):
                flips[f"flip_axis{ax}"] = contrast(hu, np.flip(fg, axis=ax))
            R["alignment_contrast_HU"] = {"original": c0, **flips}
            if c0 is not None:
                best_flip = max((v for v in flips.values() if v is not None), default=None)
                ok = best_flip is None or c0 > best_flip
                add("Alignment plausibility: mask-vs-ring HU contrast beats all axis-flipped masks", status(ok, warn=True),
                    f"original {c0:.0f} HU vs flipped {{{', '.join(f'{k}: {v:.0f}' for k, v in flips.items() if v is not None)}}}. Supporting evidence only (nodules are usually denser than surrounding lung; ground-glass/juxta-pleural nodules can weaken it)")
            inside = hu[fg]
            R["mask"]["hu_inside"] = {"mean": float(inside.mean()), "min": float(inside.min()), "max": float(inside.max())}
        # same-case evidence
        names_ok = None
        pi, pm = str(sample[img_f].get("path") if isinstance(sample[img_f], dict) else ""), str(sample[msk_f].get("path") if isinstance(sample[msk_f], dict) else "")
        num = re.search(r"(\d{4})", case)
        if pi and pm and num:
            names_ok = num.group(1) in pi and num.group(1) in pm
        add("Image and mask belong to the same case", "PASS" if (same_shape and same_aff) else "FAIL",
            "both fields come from the SAME dataset row (case_id=%s, patient_id=%s); geometry agrees (shape+affine)" % (case, scalar.get("patient_id")) +
            (f"; file names carry the case number: {names_ok}" if names_ok is not None else "; file names not available in the row"))

        # ---- figures
        ci, cm = nib.as_closest_canonical(img), nib.as_closest_canonical(msk)
        I, M = ci.get_fdata(dtype=np.float32), (np.asanyarray(cm.dataobj) > 0)
        if M.any() and I.shape == M.shape:
            k = int(np.argmax(M.sum(axis=(0, 1))))
            idx = np.argwhere(M[:, :, k])
            x0, x1 = max(idx[:, 0].min() - 24, 0), min(idx[:, 0].max() + 25, I.shape[0])
            y0, y1 = max(idx[:, 1].min() - 24, 0), min(idx[:, 1].max() + 25, I.shape[1])

            def disp(a):
                return np.rot90(a)                                    # anterior up
            def show(ax, ct, m, kind, title):
                if kind == "ct":
                    ax.imshow(disp(ct), cmap="gray", vmin=lo_w, vmax=hi_w)
                elif kind == "mask":
                    ax.imshow(disp(m), cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(disp(ct), cmap="gray", vmin=lo_w, vmax=hi_w)
                    ov = np.ma.masked_where(~disp(m), np.ones_like(disp(m), float))
                    ax.imshow(ov, cmap="autumn", alpha=0.55)
                    ax.contour(disp(m).astype(float), levels=[0.5], colors="yellow", linewidths=0.7)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            fig, axs = plt.subplots(2, 3, figsize=(13, 9))
            for r, (sl, ttl) in enumerate([((slice(None), slice(None)), "full slice"), ((slice(x0, x1), slice(y0, y1)), "zoom")]):
                ct2, m2 = I[sl + (k,)], M[sl + (k,)]
                show(axs[r, 0], ct2, m2, "ct", f"CT ({ttl}) z={k}")
                show(axs[r, 1], ct2, m2, "mask", f"Mask ({ttl}) foreground voxels={int(m2.sum())}")
                show(axs[r, 2], ct2, m2, "overlay", f"Overlay ({ttl})")
            fig.suptitle(f"{case}  |  axial slice with largest mask area (canonical RAS, anterior up)  |  window [{lo_w:.0f},{hi_w:.0f}] HU", fontsize=10)
            fig.tight_layout()
            vdir = root / "datasets" / "validation"
            vdir.mkdir(parents=True, exist_ok=True)
            p1 = vdir / f"{case}_slice_view.png"
            fig.savefig(p1, dpi=130)
            plt.close(fig)
            cx, cy, cz = [int(round(v)) for v in np.argwhere(M).mean(0)]
            fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
            for ax, (ct2, m2, t) in zip(axs, [(I[:, :, cz], M[:, :, cz], "axial"), (I[:, cy, :], M[:, cy, :], "coronal"), (I[cx, :, :], M[cx, :, :], "sagittal")]):
                ax.imshow(np.rot90(ct2), cmap="gray", vmin=lo_w, vmax=hi_w, aspect="auto" if t == "axial" else None)
                ov = np.ma.masked_where(~np.rot90(m2), np.ones_like(np.rot90(m2), float))
                ax.imshow(ov, cmap="autumn", alpha=0.6, aspect="auto" if t == "axial" else None)
                ax.set_title(f"{t} through mask centroid", fontsize=9)
                ax.axis("off")
            fig.suptitle("Orthogonal check: mask should sit on a nodule in all three planes", fontsize=10)
            fig.tight_layout()
            p2 = vdir / f"{case}_orthogonal_view.png"
            fig.savefig(p2, dpi=130)
            plt.close(fig)
            R["figures"] = [str(p1.relative_to(root)), str(p2.relative_to(root))]

    R["checks"], R["issues"] = checks, issues
    (root / "datasets" / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "datasets" / "manifests" / "validation_report.json").write_text(json.dumps(R, indent=2, default=str))
    L = [f"# Image / Mask Validation — {case}\n",
         f"_Auto-generated by `scripts/validate_image_mask.py` on {R['generated']} ({time.time() - t0:.0f} s). Numbers are measured from the decoded case; nothing assumed._\n",
         f"- Loader: `load_dataset(\"{a.repo}\", split=\"{a.split}\", streaming=True)`, first row with a mask. Rows without a mask skipped before it: {len(skipped)}.",
         f"- Image field `{img_f}` ({how_img}); mask field `{msk_f}` ({how_msk}).",
         f"- Row scalar fields: `{json.dumps(scalar, default=str)}`",
         f"- Untouched copy saved in `datasets/raw/lidc_hf_peek/{case}/`.\n",
         "## Results\n", "| check | status | detail |", "|---|---|---|"]
    for c in checks:
        L.append(f"| {c['check']} | **{c['status']}** | {c['detail'].replace('|', '/')} |")
    if "image" in R:
        L += ["\n## Image\n```\n" + json.dumps({k: v for k, v in R["image"].items() if k != "affine"}, indent=1) + "\n```",
              "Affine:\n```\n" + json.dumps(R["image"]["affine"]) + "\n```",
              "\n## Mask\n```\n" + json.dumps({k: v for k, v in R["mask"].items() if k not in ("affine", "components")}, indent=1) + "\n```"]
        if R["mask"].get("components"):
            L.append("Components: " + json.dumps(R["mask"]["components"][:12]))
    if "figures" in R:
        L += ["\n## Figures\n"] + [f"- `{f}`" for f in R["figures"]]
    L.append("\n## Issues\n" + ("\n".join(f"- {i}" for i in issues) if issues else "None detected by the automated checks."))
    L.append("\n## Limits of this validation\n- One case only; it says nothing about the other 1,000+ cases.\n- Visual inspection of the figures is still required; the contrast test is supporting evidence, not proof of alignment.\n- Nothing here is a diagnosis; masks are research nodule segmentations.\n")
    (root / "datasets" / "IMAGE_MASK_VALIDATION.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(f"[{c['status']}] {c['check']}: {c['detail']}" for c in checks))
    print(f"\nReport: datasets/IMAGE_MASK_VALIDATION.md\nFigures: {R.get('figures')}")


if __name__ == "__main__":
    main()