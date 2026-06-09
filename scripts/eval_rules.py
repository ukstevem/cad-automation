"""
Evaluate the rule decision tree (plate / formed / section / bent-section)
against the human gallery labels in verified.csv.

Self-contained: for each labelled part it re-opens the STEP, computes the same
features the live pipeline uses (extract_solid_features section mode: holes,
thickness, convex bends) and reads the existing library-match verdict from the
analysis sidecars.  No dependency on the transient gallery manifest, so coverage
matches the labels exactly.

Run inside the container (slow — re-opens each job's STEP)::

    docker exec cad-automation-api python /tmp/eval_rules.py
"""
from __future__ import annotations
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, "/app")
from app.config import settings  # noqa: E402

LABELS = "/app/app/pipeline/data/labels/verified.csv"
DATASET = "/app/outputs/ml/dataset.csv"
GEOM = ["SECTION", "PLATE", "FORMED_PLATE", "BENT_SECTION"]
OOS = ["BOUGHT_OUT", "EXCLUDE"]


def predict(name, holes, thk, tthin, rule_type, nbends):
    from app.pipeline.catalogue_products import match_catalogue_product
    cp = match_catalogue_product(name)   # known BO products (Unistrut, ...)
    if cp:
        return cp[0]
    if nbends is not None and nbends >= 5:
        return "BENT_SECTION"
    if holes is not None and holes >= 1:
        return "SECTION"
    if tthin is not None and tthin >= 0.45:
        return "PLATE"
    if nbends is not None and nbends >= 1:
        return "FORMED_PLATE"
    if rule_type == "section":
        return "SECTION"
    if thk is not None and thk >= 1.5:
        return "SECTION"
    return "FORMED_PLATE"


def resolve_step(job):
    for p in glob.glob(settings.ANALYSIS_OUTPUT_DIR + "/*.json"):
        stem = os.path.basename(p)[:-5]
        if re.sub(r"^[0-9a-f]{8}_", "", stem) == job:
            for ext in (".step", ".stp", ".STEP", ".STP"):
                f = os.path.join(settings.UPLOAD_DIR, stem + ext)
                if os.path.exists(f):
                    return f, p
    return None, None


def rule_types_for(sidecar_path):
    """ref_id -> rule_type from a sidecar's cnc_analysis."""
    try:
        d = json.loads(open(sidecar_path).read())
    except Exception:
        return {}
    out = {}
    for rid, v in (d.get("cnc_analysis") or {}).items():
        if isinstance(v, dict):
            out[rid] = v.get("type")
    return out


def load_labels():
    out = {}
    for row in csv.reader(open(LABELS, newline="")):
        if not row or row[0].lstrip().startswith("#") or row[0] == "job":
            continue
        out[(row[0], row[1], str(row[2]))] = row[3]
    return out


def main():
    from app.services.cnc_shape_analyser import _read_xcaf, _get_shape, _iter_solids
    from app.pipeline.feature_extract import extract_solid_features

    labels = load_labels()
    # part_name per (job, ref_id) from the dataset, for catalogue-product matching
    names = {}
    try:
        for r in csv.DictReader(open(DATASET, newline="")):
            names[(r["job"], r["ref_id"])] = r.get("part_name", "")
    except Exception:
        pass
    by_job = defaultdict(list)
    for (job, ref, sidx), cat in labels.items():
        by_job[job].append((ref, sidx, cat))

    rows = []
    for job, items in by_job.items():
        step, sidecar = resolve_step(job)
        if not step:
            continue
        rts = rule_types_for(sidecar)
        doc = _read_xcaf(step)[0]
        print(f"  {job}: {len(items)} labels", file=sys.stderr)
        for ref, sidx, cat in items:
            try:
                shape = _get_shape(doc, ref)
                solids = list(_iter_solids(shape))
                si = int(sidx)
                solid = solids[si] if si < len(solids) else (solids[0] if solids else shape)
                f = extract_solid_features(solid, section=True)
            except Exception:
                continue
            pred = predict(names.get((job, ref)), f.get("n_holes"),
                           f.get("thk_max_over_teff"), f.get("t_eff_thin_ratio"),
                           rts.get(ref), f.get("n_convex_bends"))
            rows.append((cat, pred))
    print(f"\nlabels: {len(labels)}  evaluated: {len(rows)}")

    conf = defaultdict(lambda: defaultdict(int))
    for t, p in rows:
        conf[t][p] += 1
    cols = GEOM
    print("\nConfusion (rows=your label, cols=rule prediction):")
    print("  " + " " * 14 + "".join(f"{p[:11]:>13}" for p in cols))
    for t in GEOM + OOS:
        if t in conf:
            print(f"  {t:<14}" + "".join(f"{conf[t].get(p,0):>13}" for p in cols))

    print("\nPer-class:")
    for c in GEOM:
        tp = sum(1 for t, p in rows if t == c == p)
        fp = sum(1 for t, p in rows if p == c and t != c)
        fn = sum(1 for t, p in rows if t == c and p != c)
        n = sum(1 for t, _ in rows if t == c)
        if n:
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            print(f"  {c:<14} n={n:3d}  precision={prec:5.0%}  recall={rec:5.0%}")

    g = [(t, p) for t, p in rows if t in GEOM]
    acc = sum(1 for t, p in g if t == p) / len(g) if g else 0
    print(f"\nAccuracy on {'/'.join(GEOM)}: {sum(1 for t,p in g if t==p)}/{len(g)} = {acc:.1%}")
    oos = [(t, p) for t, p in rows if t in OOS]
    if oos:
        print(f"\nOut-of-scope ({'/'.join(OOS)}): {len(oos)} parts (no feature yet).")


if __name__ == "__main__":
    main()
