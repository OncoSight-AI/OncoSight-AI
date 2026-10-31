#!/usr/bin/env python
"""Read-only inspection of the Hugging Face dataset  MedOtter/LIDC-IDRI  (or a local snapshot of it).

Why not just stream it?  Streaming decodes every CT volume (~71 GB).  Parquet keeps a footer with the
schema and row counts, and columns can be read separately, so this script gets the full metadata table
(case_id, patient_id, manufacturer, n_slices, ...) WITHOUT downloading the image/mask columns.

Writes only inside <repo-root>/datasets/ :
  DATASET_INVENTORY.md            human-readable report (every number computed from the data)
  manifests/hf_metadata.csv       one row per case (metadata columns only)
  manifests/inventory_hf.json     machine-readable summary
  manifests/pilot_cases.json      10-20 pilot cases (patient-level, stratified by manufacturer)
  raw/lidc_hf_pilot/*.parquet     ONLY with --export-pilot: untouched copies of the pilot rows

Examples
  python scripts/inspect_lidc_hf.py --repo-root .                      # metadata-only, small download
  python scripts/inspect_lidc_hf.py --repo-root . --peek 1             # also download ONE row group to look at image/mask formats
  python scripts/inspect_lidc_hf.py --repo-root . --export-pilot       # save the pilot rows locally (~1 GB)
  python scripts/inspect_lidc_hf.py --local-dir D:\\lidc_snapshot      # if you used snapshot_download()
"""
import argparse, gzip, io, json, random, re, struct, sys, time, hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec

PID_RE = re.compile(r"^LIDC-IDRI-\d{4}$")
# Reference numbers published by TCIA for the ORIGINAL collection (for comparison only).
TCIA = {"patients": 1010, "ct_series": 1018, "images": 244527}
KEYWORDS = {
    "mask / segmentation": ["mask", "seg"], "nodule info": ["nodule", "lesion", "annot", "contour", "bbox", "box"],
    "malignancy / diagnosis": ["malig", "diagn", "cancer", "benign", "label", "target"],
    "geometry": ["spacing", "thickness", "origin", "affine", "shape", "size"],
    "raw formats": ["dicom", "dcm", "xml", "nii"],
}


# ----------------------------------------------------------------------------- filesystem
def get_fs(a):
    if a.local_dir:
        base = str(Path(a.local_dir).resolve())
        return fsspec.filesystem("file"), base, True
    from huggingface_hub import HfFileSystem
    base = f"datasets/{a.repo}" + (f"@{a.revision}" if a.revision else "")
    return HfFileSystem(), base, False


def list_files(fs, base, local):
    out = []
    for p, info in fs.find(base, detail=True).items():
        if info.get("type") == "directory":
            continue
        rel = (str(Path(p).relative_to(base)).replace("\\", "/") if local
               else re.sub(r"^datasets/[^/]+/[^/]+(@[^/]+)?/", "", p))
        out.append((p, rel, int(info.get("size") or 0)))
    return sorted(out, key=lambda t: t[1])


def ext_of(rel):
    n = rel.lower()
    for e in (".nii.gz", ".tar.gz"):
        if n.endswith(e):
            return e
    return Path(n).suffix or "<none>"


def tree_text(files, depth=2, maxkids=12):
    counts, sizes = Counter(), Counter()
    for _, rel, sz in files:
        parts = Path(rel).parts[:-1]
        for k in range(0, min(len(parts), depth) + 1):
            counts[parts[:k]] += 1
            sizes[parts[:k]] += sz
    kids = defaultdict(list)
    for k in counts:
        if k:
            kids[k[:-1]].append(k)
    lines = [f"<repo>  ({counts[()]} files, {sizes[()] / 1e9:.2f} GB)"]

    def rec(key, lvl):
        ks = sorted(kids.get(key, []))
        for k in ks[:maxkids]:
            lines.append("  " * lvl + f"{k[-1]}/  ({counts[k]} files, {sizes[k] / 1e9:.2f} GB)")
            if lvl <= depth:
                rec(k, lvl + 1)
        if len(ks) > maxkids:
            lines.append("  " * lvl + f"... +{len(ks) - maxkids} more folders")
    rec((), 1)
    top_files = [(rel, sz) for _, rel, sz in files if len(Path(rel).parts) == 1]
    for rel, sz in top_files:
        lines.append(f"  {rel}  ({sz / 1e6:.2f} MB)")
    return "\n".join(lines)


def split_of(rel):
    b = rel.lower()
    for key, name in (("train", "train"), ("test", "test"), ("validation", "val"), ("valid", "val"), ("val", "val")):
        if re.search(rf"(^|[/_\-.]){key}([/_\-.]|$)", b):
            return name
    return "unspecified"


def md_table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


# ----------------------------------------------------------------------------- card
def parse_card(text):
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    meta = {}
    if m:
        try:
            import yaml
            meta = yaml.safe_load(m.group(1)) or {}
        except Exception as e:
            meta = {"_yaml_error": str(e)}
    return meta, text


# ----------------------------------------------------------------------------- parquet
def pq_info(fs, path, rel):
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        col_bytes = Counter()
        for rg in range(md.num_row_groups):
            for ci in range(md.num_columns):
                cc = md.row_group(rg).column(ci)
                col_bytes[cc.path_in_schema.split(".")[0]] += cc.total_compressed_size
        schema = pf.schema_arrow
        hf = None
        if schema.metadata and b"huggingface" in schema.metadata:
            try:
                hf = json.loads(schema.metadata[b"huggingface"])
            except Exception:
                pass
        return {"path": path, "rel": rel, "num_rows": md.num_rows, "num_row_groups": md.num_row_groups,
                "rows_per_group": [md.row_group(i).num_rows for i in range(md.num_row_groups)],
                "schema": [(fl.name, str(fl.type)) for fl in schema], "col_bytes": dict(col_bytes), "hf_features": hf}


def read_meta(fs, info, cols):
    with fs.open(info["path"], "rb") as f:
        names = set(pq.ParquetFile(f).schema_arrow.names)
        t = pq.read_table(f, columns=[c for c in cols if c in names])
    df = t.to_pandas()
    df["_file"] = info["rel"]
    df["_row"] = np.arange(len(df))
    return df


def is_heavy(arrow_type, avg_bytes):
    t = arrow_type
    if pa.types.is_struct(t) or pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_fixed_size_binary(t):
        return True
    return avg_bytes >= 4096


# ----------------------------------------------------------------------------- sniffing heavy values
def sniff_bytes(b):
    """Identify what a binary blob is and report shape/dtype where cheaply possible."""
    r = {"n_bytes": len(b)}
    try:
        if b[:2] == b"\x1f\x8b":
            r["format"] = "gzip"
            h = gzip.GzipFile(fileobj=io.BytesIO(b)).read(352)
            r.update(_nifti_header(h) or {})
        elif b[:6] == b"\x93NUMPY":
            arr = np.load(io.BytesIO(b), allow_pickle=False)
            r.update(format="npy", shape=list(arr.shape), dtype=str(arr.dtype))
            if arr.size:
                r["min"], r["max"] = float(arr.min()), float(arr.max())
                if arr.dtype.kind in "iub" and arr.size < 3e8:
                    u = np.unique(arr)
                    r["unique_values"] = u[:20].tolist() if len(u) <= 20 else f"{len(u)} distinct"
        elif b[:8] == b"\x89PNG\r\n\x1a\n" or b[:4] in (b"II*\x00", b"MM\x00*") or b[:3] == b"\xff\xd8\xff":
            from PIL import Image
            im = Image.open(io.BytesIO(b))
            r.update(format=im.format, size_wh=list(im.size), mode=im.mode)
        elif b[128:132] == b"DICM":
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(b), stop_before_pixels=True)
            r.update(format="dicom", modality=str(ds.get("Modality")), rows=int(ds.get("Rows", 0)), cols=int(ds.get("Columns", 0)))
        else:
            h = b[:352]
            r.update(_nifti_header(h) or {"format": "unknown", "first_bytes": b[:16].hex()})
    except Exception as e:
        r["decode_error"] = f"{type(e).__name__}: {e}"[:150]
    return r


def _nifti_header(h):
    if len(h) < 348:
        return None
    for en in ("<", ">"):
        if struct.unpack(en + "i", h[:4])[0] == 348:
            dim = struct.unpack(en + "8h", h[40:56])
            pix = struct.unpack(en + "8f", h[76:108])
            return {"format": "nifti-1", "dim": list(dim[1:1 + dim[0]]), "pixdim_mm": [round(x, 4) for x in pix[1:1 + dim[0]]],
                    "datatype_code": struct.unpack(en + "h", h[70:72])[0]}
    return None


def describe_value(v):
    if isinstance(v, dict) and "bytes" in v:
        d = {"path": v.get("path")}
        d.update(sniff_bytes(v["bytes"]) if v["bytes"] else {"empty_bytes": True})
        return d
    if isinstance(v, (bytes, bytearray)):
        return sniff_bytes(bytes(v))
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            a = np.asarray(v)
            return {"type": "array", "shape": list(a.shape), "dtype": str(a.dtype)}
        except Exception:
            return {"type": "list", "len": len(v)}
    return {"type": type(v).__name__, "value": str(v)[:80]}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="MedOtter/LIDC-IDRI")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--local-dir", default=None, help="inspect a local snapshot instead of the Hub")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--peek", type=int, default=0, help="download N row groups (all columns) to sniff image/mask formats")
    ap.add_argument("--verify", type=int, default=0, help="fully read N random row groups to detect corruption (-1 = all; downloads everything)")
    ap.add_argument("--pilot-n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export-pilot", action="store_true", help="copy the pilot rows (untouched) to datasets/raw/lidc_hf_pilot/")
    a = ap.parse_args()

    repo = Path(a.repo_root).resolve()
    man, raw_dir, inv_md = repo / "datasets" / "manifests", repo / "datasets" / "raw" / "lidc_hf_pilot", repo / "datasets" / "DATASET_INVENTORY.md"
    man.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fs, base, local = get_fs(a)
    if local:
        try:
            man.relative_to(Path(base))
            sys.exit("Refusing to write inside the dataset snapshot folder.")
        except ValueError:
            pass
    L = []

    print("1/6 listing repository files ...", flush=True)
    files = list_files(fs, base, local)
    if not files:
        sys.exit("No files found. Check --repo / --local-dir / network / login (huggingface-cli login).")
    ext_counts = Counter(ext_of(r) for _, r, _ in files)
    ext_sizes = Counter()
    for _, r, s in files:
        ext_sizes[ext_of(r)] += s
    raw_hits = {e: ext_counts.get(e, 0) for e in (".dcm", ".xml", ".csv", ".xls", ".xlsx", ".nii.gz", ".nii", ".npy", ".npz", ".parquet", ".json", ".jsonl")}

    print("2/6 reading dataset card ...", flush=True)
    card_meta, card_text = {}, ""
    readme = next((p for p, r, _ in files if r == "README.md"), None)
    if readme:
        with fs.open(readme, "rb") as f:
            card_meta, card_text = parse_card(f.read().decode("utf-8", "replace"))
    card_features, card_splits = [], []
    di = card_meta.get("dataset_info")
    for d in (di if isinstance(di, list) else [di] if isinstance(di, dict) else []):
        card_features += [(x.get("name"), str(x.get("dtype", x))) for x in d.get("features", []) if isinstance(x, dict)]
        card_splits += d.get("splits", []) or []

    print("3/6 reading parquet footers ...", flush=True)
    pq_files = [(p, r) for p, r, _ in files if r.lower().endswith(".parquet")]
    infos, pq_errors = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [(r, ex.submit(pq_info, fs, p, r)) for p, r in pq_files]
        for r, fu in futs:
            try:
                infos.append(fu.result())
            except Exception as e:
                pq_errors.append((r, f"{type(e).__name__}: {e}"[:200]))
    infos.sort(key=lambda i: i["rel"])

    df = pd.DataFrame()
    schema, meta_cols, heavy_cols, col_avg = [], [], [], {}
    if infos:
        schema = infos[0]["schema"]
        total_rows = sum(i["num_rows"] for i in infos)
        agg = Counter()
        for i in infos:
            for c, b in i["col_bytes"].items():
                agg[c] += b
        col_avg = {c: agg[c] / max(total_rows, 1) for c in agg}
        ftype = dict((n, None) for n, _ in schema)
        with fs.open(infos[0]["path"], "rb") as f:
            sch = pq.ParquetFile(f).schema_arrow
        for n in sch.names:
            (heavy_cols if is_heavy(sch.field(n).type, col_avg.get(n, 0)) else meta_cols).append(n)
        print(f"4/6 reading metadata columns {meta_cols} from {len(infos)} files ...", flush=True)
        frames = []
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for fr in ex.map(lambda i: read_meta(fs, i, meta_cols), infos):
                frames.append(fr)
        df = pd.concat(frames, ignore_index=True)
        df["_split"] = df["_file"].map(split_of)
        df.to_csv(man / "hf_metadata.csv", index=False)

    # ---- analysis
    def pick(*names, kw=()):
        for n in names:
            if n in df.columns:
                return n
        for c in df.columns:
            if any(k in c.lower() for k in kw):
                return c
        return None
    c_case = pick("case_id", kw=("case", "series", "scan"))
    c_pat = pick("patient_id", kw=("patient",))
    c_sl = pick("n_slices", kw=("slice",))
    c_man = pick("manufacturer", kw=("manufact", "vendor", "scanner"))

    R = {"generated": datetime.now().isoformat(timespec="seconds"), "source": a.local_dir or a.repo,
         "n_files": len(files), "total_gb": round(sum(s for _, _, s in files) / 1e9, 3), "n_parquet": len(infos),
         "n_rows": int(len(df)), "columns": schema, "meta_cols": meta_cols, "heavy_cols": heavy_cols}
    issues = []
    if len(df):
        R["n_unique_case"] = int(df[c_case].nunique()) if c_case else None
        R["n_unique_patient"] = int(df[c_pat].nunique()) if c_pat else None
        if c_sl:
            s = pd.to_numeric(df[c_sl], errors="coerce")
            R["slices"] = {"sum": int(s.sum()), "min": float(s.min()), "median": float(s.median()), "max": float(s.max()),
                           "null_or_nonnumeric": int(s.isna().sum()), "le_zero": int((s <= 0).sum())}

    # ---- peek
    peeks = []
    if a.peek and infos:
        print(f"5/6 peeking at {a.peek} row group(s) (downloads them) ...", flush=True)
        for i in infos[:a.peek]:
            with fs.open(i["path"], "rb") as f:
                pf = pq.ParquetFile(f)
                tbl = pf.read_row_group(0)
            row = tbl.slice(0, 1).to_pylist()[0]
            peeks.append({"file": i["rel"], "fields": {k: describe_value(v) for k, v in row.items()}})
    R["peek"] = peeks

    verify_fail = []
    if a.verify and infos:
        groups = [(i, g) for i in infos for g in range(i["num_row_groups"])]
        sel = groups if a.verify < 0 else random.Random(a.seed).sample(groups, min(a.verify, len(groups)))
        print(f"  verifying {len(sel)} row groups ...", flush=True)

        def chk(t):
            i, g = t
            try:
                with fs.open(i["path"], "rb") as f:
                    pq.ParquetFile(f).read_row_group(g)
                return None
            except Exception as e:
                return (i["rel"], g, f"{type(e).__name__}: {e}"[:150])
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            verify_fail = [r for r in ex.map(chk, sel) if r]
        R["verify"] = {"checked": len(sel), "failed": len(verify_fail)}

    # ---- pilot
    pilot = []
    if len(df) and c_case and c_pat:
        elig = df.dropna(subset=[c_case, c_pat]).copy()
        elig = elig[~elig[c_case].duplicated(keep=False)]
        if c_sl:
            elig = elig[pd.to_numeric(elig[c_sl], errors="coerce") > 0]
        elig = elig[~elig[c_pat].duplicated(keep=False)]               # patients with exactly one case
        rng = random.Random(a.seed)
        buckets = defaultdict(list)
        for _, r in elig.iterrows():
            buckets[str(r[c_man]) if c_man else "all"].append(r)
        for b in buckets.values():
            rng.shuffle(b)
        keys = sorted(buckets)
        while len(pilot) < a.pilot_n and any(buckets.values()):
            for k in keys:
                if buckets[k] and len(pilot) < a.pilot_n:
                    r = buckets[k].pop()
                    pilot.append({"case_id": str(r[c_case]), "patient_id": str(r[c_pat]),
                                  "manufacturer": str(r[c_man]) if c_man else None,
                                  "n_slices": int(r[c_sl]) if c_sl and pd.notna(r[c_sl]) else None,
                                  "file": r["_file"], "row_in_file": int(r["_row"]), "split": r["_split"]})
        (man / "pilot_cases.json").write_text(json.dumps(pilot, indent=2))
        R["n_eligible_for_pilot"] = int(len(elig))

    exported = []
    if a.export_pilot and pilot:
        print("  exporting pilot rows (untouched copies) ...", flush=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        info_by_rel = {i["rel"]: i for i in infos}
        for pc in pilot:
            out = raw_dir / f"{pc['case_id']}.parquet"
            if out.exists():
                exported.append({"case_id": pc["case_id"], "status": "already_exported"})
                continue
            i = info_by_rel[pc["file"]]
            start = 0
            for g, n in enumerate(i["rows_per_group"]):
                if pc["row_in_file"] < start + n:
                    with fs.open(i["path"], "rb") as f:
                        tbl = pq.ParquetFile(f).read_row_group(g)
                    pq.write_table(tbl.slice(pc["row_in_file"] - start, 1), out)
                    exported.append({"case_id": pc["case_id"], "status": "ok", "bytes": out.stat().st_size,
                                     "sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
                    break
                start += n
        (man / "pilot_export.json").write_text(json.dumps(exported, indent=2))

    # ================= markdown
    L.append("# LIDC-IDRI (Hugging Face: `%s`) — Dataset Inventory\n" % a.repo)
    L.append(f"_Auto-generated by `scripts/inspect_lidc_hf.py` on {R['generated']} ({time.time() - t0:.0f} s). "
             "Every number was computed from the repository; nothing is assumed._\n")
    L.append(f"- Source inspected: `{R['source']}`" + (f" (revision `{a.revision}`)" if a.revision else "") +
             "\n- Access mode: read-only (footers + metadata columns; images/masks were not downloaded"
             + (f" except {a.peek} row group(s) for `--peek`" if a.peek else "") + ").\n")
    lic = card_meta.get("license")
    L.append(f"- License stated on the dataset card: **{lic if lic else 'not found in card front-matter'}**. "
             "The original TCIA collection is CC BY 3.0; this is a *redistribution*, so also read its card for conversion details and cite the original authors.\n")
    L.append("## 1. Repository structure\n")
    L.append("```\n" + tree_text(files) + "\n```\n")
    L.append(md_table(["extension", "files", "size (GB)"], [[e, c, f"{ext_sizes[e] / 1e9:.3f}"] for e, c in ext_counts.most_common()]))
    L.append("\n**Raw-format check** (the things you asked me to look for):\n")
    L.append(md_table(["file type", "count"], [[k, v] for k, v in raw_hits.items()]))
    if not raw_hits[".dcm"] and not raw_hits[".xml"]:
        L.append("\n> This repository contains **no DICOM (.dcm) and no annotation XML files**. It is a preprocessed redistribution, so "
                 "scan metadata and the per-reader XML structure were already converted into whatever columns are listed below. "
                 "Anything not in those columns (per-reader contours, nodule characteristics, malignancy ratings, diagnosis) is **not available from this repo**.\n")
    L.append("\n## 2. Schema\n")
    if schema:
        rows = []
        for n, t in schema:
            rows.append([n, t, "heavy (not downloaded)" if n in heavy_cols else "metadata (read)", f"{col_avg.get(n, 0):,.0f}"])
        L.append(md_table(["column", "arrow type", "handling", "avg compressed bytes / row"], rows))
        hf = infos[0].get("hf_features")
        if hf:
            feats = hf.get("info", {}).get("features", {})
            L.append("\nHugging Face feature types stored in the parquet schema metadata:\n\n```\n" + json.dumps(feats, indent=1)[:3500] + "\n```\n")
        mism = [i["rel"] for i in infos if [s[0] for s in i["schema"]] != [s[0] for s in schema]]
        L.append(f"Files whose column names differ from the first file: **{len(mism)}**" + (f" (e.g. `{mism[0]}`)" if mism else "") + ".\n")
    else:
        L.append("No parquet files found; see Section 8 for how to inspect via `datasets`.\n")
    if card_features:
        L.append("\nFeatures declared in the dataset card: " + ", ".join(f"`{n}`: {t}" for n, t in card_features) + "\n")
    if card_splits:
        L.append("\nSplits declared in the card: " + ", ".join(f"{s.get('name')}: {s.get('num_examples')} examples" for s in card_splits) + "\n")

    if len(df):
        L.append("## 3. Counts\n")
        rows = [["Parquet files", len(infos)], ["Rows (cases/scans) in all files", len(df)]]
        if c_case:
            rows.append([f"Unique `{c_case}`", R["n_unique_case"]])
        if c_pat:
            rows.append([f"Unique `{c_pat}` (patients)", R["n_unique_patient"]])
        if c_pat:
            multi = df.groupby(c_pat).size()
            rows.append(["Patients with >1 row", int((multi > 1).sum())])
        if c_sl:
            rows += [[f"Total CT slices (sum of `{c_sl}`)", f"{R['slices']['sum']:,}"],
                     ["Slices per scan: min / median / max", f"{R['slices']['min']:.0f} / {R['slices']['median']:.0f} / {R['slices']['max']:.0f}"]]
        L.append(md_table(["item", "value"], rows))
        L.append(f"\nFor comparison, TCIA lists the **original** collection as {TCIA['patients']:,} patients, {TCIA['ct_series']:,} CT series and "
                 f"{TCIA['images']:,} images (TCIA wiki). Differences here are expected if the redistribution excluded or re-sliced series — "
                 "the card should say which; check it.\n")
        L.append("\nRows per split (inferred from file names — verify against the card):\n")
        sp = df.groupby("_split").agg(rows=("_split", "size"), **({"patients": (c_pat, "nunique")} if c_pat else {})).reset_index()
        L.append(md_table(list(sp.columns), sp.values.tolist()))
        if c_man:
            L.append(f"\nScanner manufacturer (`{c_man}`):\n")
            L.append(md_table(["manufacturer", "rows"], list(df[c_man].fillna("<missing>").value_counts().items())))
        others = [c for c in meta_cols if c not in (c_case, c_pat, c_sl, c_man)]
        for c in others:
            if df[c].nunique(dropna=False) <= 25:
                L.append(f"\nValue counts for `{c}`: {df[c].fillna('<missing>').value_counts().to_dict()}\n")
            elif pd.api.types.is_numeric_dtype(df[c]):
                d = df[c].describe()
                L.append(f"\n`{c}`: min {d['min']:.4g}, median {df[c].median():.4g}, max {d['max']:.4g}\n")

        L.append("## 4. Integrity checks\n")
        chk = [["Null values per metadata column", {c: int(df[c].isna().sum()) for c in meta_cols if df[c].isna().any()} or "none"]]
        if c_case:
            dups = df[df[c_case].duplicated(keep=False)]
            chk.append([f"Duplicate `{c_case}` values", int(dups[c_case].nunique())])
            if len(dups):
                issues.append(f"{dups[c_case].nunique()} duplicated {c_case} value(s), e.g. {dups[c_case].iloc[0]}")
        if c_pat:
            bad = df[~df[c_pat].astype(str).str.match(PID_RE)]
            chk.append([f"`{c_pat}` not matching LIDC-IDRI-NNNN", int(len(bad))])
            if len(bad):
                issues.append(f"{len(bad)} patient_id values not in LIDC-IDRI-NNNN format, e.g. {bad[c_pat].iloc[0]}")
            sp_per_pat = df.groupby(c_pat)["_split"].nunique()
            leak = sp_per_pat[sp_per_pat > 1]
            chk.append(["Patients appearing in >1 split (LEAKAGE)", int(len(leak))])
            if len(leak):
                issues.append(f"{len(leak)} patient(s) appear in more than one split, e.g. {leak.index[0]} — patient-level re-splitting required")
        if c_sl:
            s = pd.to_numeric(df[c_sl], errors="coerce")
            chk.append([f"`{c_sl}` null / non-numeric / <=0", f"{int(s.isna().sum())} / {int((s <= 0).sum())}"])
            q1, q3 = s.quantile(.25), s.quantile(.75)
            out = df[(s < q1 - 3 * (q3 - q1)) | (s > q3 + 3 * (q3 - q1))]
            chk.append([f"`{c_sl}` extreme outliers (3×IQR)", int(len(out))])
        chk.append(["Parquet footers that failed to read", len(pq_errors)])
        chk.append(["Full read-back of row groups (--verify)", f"{R['verify']['checked']} checked, {R['verify']['failed']} failed" if "verify" in R else "not run (needs a full download)"])
        L.append(md_table(["check", "result"], chk))
        if issues:
            L.append("\n**Issues to decide on:**\n" + "\n".join(f"- {i}" for i in issues) + "\n")
        for r, e in pq_errors[:10]:
            L.append(f"- footer error `{r}`: {e}\n")
        for r, g, e in verify_fail[:10]:
            L.append(f"- corrupted row group `{r}` #{g}: {e}\n")
        L.append("\n> Footer checks prove the files are structurally readable; they do not decode every image. "
                 "Detecting corrupt volumes needs `--verify -1` (downloads everything) or a decode pass during preprocessing.\n")

        L.append("## 5. Annotation / label information actually present\n")
        found = {}
        for cat, kws in KEYWORDS.items():
            cols = [c for c, _ in schema if any(k in c.lower() for k in kws)]
            if cols:
                found[cat] = cols
        L.append(md_table(["category", "columns found by name"], [[k, ", ".join(v)] for k, v in found.items()] or [["(none)", "-"]]))
        for pk in peeks:
            L.append(f"\n**Peek at `{pk['file']}` row 0:**\n\n```\n{json.dumps(pk['fields'], indent=1, default=str)[:4500]}\n```\n")
        if not peeks:
            L.append("\nColumn *contents* (image/mask format, shape, mask label values, voxel spacing) are unknown until a `--peek 1` run; "
                     "column names alone don't prove what is inside.\n")
        L.append("\nBackground (from the LIDC-IDRI papers; verify against this repo's card): the original annotations are per radiologist (up to four), "
                 "nodules <3 mm have only a centroid, and the 1–5 malignancy value is a subjective impression. "
                 "Patient-level pathology-type diagnosis exists for only a subset (reported as 113 patients in one analysis) and is a separate TCIA file, "
                 "not part of the images.\n")
        L.append("## 6. Which cases have reliable diagnosis information\n")
        dcols = found.get("malignancy / diagnosis", [])
        if dcols:
            L.append(f"Columns that may carry labels: {', '.join(dcols)}. Their meaning and provenance must be confirmed from the card / peek before use.\n")
        else:
            L.append("**None identifiable from the schema.** No column looks like a diagnosis or malignancy label, so this dataset alone supports "
                     "**nodule detection/segmentation only**; malignancy classification would need the TCIA diagnosis table matched by `patient_id`.\n")
        L.append("## 7. Task separation (proposal for approval)\n")
        L.append("1. **Nodule detection / segmentation** — images + (consensus) masks only, no malignancy claim.\n"
                 "2. **Malignancy classification** — separate task, only if trustworthy labels are added and matched by patient; radiologist ratings, if ever used, are labelled *subjective*.\n"
                 "3. All splitting **by patient**.\n")
        L.append("## 8. Pilot subset\n")
        L.append(f"{len(pilot)} cases chosen (seed {a.seed}) from {R.get('n_eligible_for_pilot', 0)} eligible (unique case, patient with exactly one case, `{c_sl}` > 0), "
                 "round-robin across manufacturers. Manifest: `datasets/manifests/pilot_cases.json`.\n")
        L.append(md_table(["case", "patient", "manufacturer", "slices", "split"], [[p["case_id"], p["patient_id"], p["manufacturer"], p["n_slices"], p["split"]] for p in pilot]))
        if exported:
            L.append(f"\nExported untouched rows to `datasets/raw/lidc_hf_pilot/`: {sum(e['status'] in ('ok', 'already_exported') for e in exported)} of {len(pilot)}.\n")
        L.append("## 9. Open items for your decision before any training\n")
        L.append("- [ ] Confirm counts vs. what the card says were excluded/changed.\n- [ ] Review schema + `--peek` output: volume format, mask format/label values, spacing, orientation.\n"
                 "- [ ] Decide on any integrity issues in Section 4.\n- [ ] Confirm patient-level split policy (repo splits may leak patients, if flagged).\n"
                 "- [ ] Decide if/where malignancy labels come from (Section 6).\n- [ ] Approve the pilot cases, then I write the pilot preprocessing for the confirmed schema.\n")
    if card_text:
        L.append("\n## Appendix: dataset card (verbatim, first 6000 characters)\n\n```\n" + card_text[:6000] + "\n```\n")
    inv_md.write_text("\n".join(L), encoding="utf-8")
    (man / "inventory_hf.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"\nDone in {time.time() - t0:.0f} s.\n  {inv_md}\n  {man}")


if __name__ == "__main__":
    main()