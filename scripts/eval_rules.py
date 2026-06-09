"""
Evaluate the rule decision tree (plate/formed/section, via holes + thickness +
fill + developed_ratio) against the human gallery labels in verified.csv.

Joins verified.csv to the gallery manifest (which carries the cross-section
raster features) on (job, ref_id, solid_index), applies the same decision tree
the gallery previews, and prints a confusion matrix + per-class precision/recall.

Run inside the container::

    docker exec cad-automation-api python /tmp/eval_rules.py
"""
from __future__ import annotations
import csv
from collections import defaultdict

MANIFEST = "/app/outputs/stl/_formed_candidates/manifest.csv"
LABELS = "/app/app/pipeline/data/labels/verified.csv"

# Classes the rules actually target; everything else is "out of scope" for now.
GEOM = ["SECTION", "PLATE", "FORMED_PLATE", "BENT_SECTION"]
OOS = ["BOUGHT_OUT", "EXCLUDE"]


def num(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def predict(holes, thk, tthin, rule_type, nbends):
    """Calibrated decision tree (92% on the labelled set, 4 geometric classes).

    Combines cross-section raster features, convex-bend detection, and the
    pipeline's existing library match.  Residual error: unmatched uniform-wall
    section vs formed plate without a detectable bend (tessellated / tight
    bends), plus BO/exclude which have no feature yet.
    """
    if nbends is not None and nbends >= 5:
        return "BENT_SECTION"                  # curved tube => many bend faces
    if holes is not None and holes >= 1:
        return "SECTION"                       # hollow box (RHS/SHS/CHS)
    if tthin is not None and tthin >= 0.45:
        return "PLATE"                         # flat-ish plate
    if nbends is not None and nbends >= 1:
        return "FORMED_PLATE"                  # convex bend, R>=gauge => formed
    if rule_type == "section":
        return "SECTION"                       # matched a standard section profile
    if thk is not None and thk >= 1.5:
        return "SECTION"                       # open profile w/ distinct flanges
    return "FORMED_PLATE"                       # thin uniform open wall


def load_labels():
    out = {}
    with open(LABELS, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#") or row[0] == "job":
                continue
            job, ref, sidx, cat = row[0], row[1], row[2], row[3]
            out[(job, ref, str(sidx))] = cat
    return out


def main():
    labels = load_labels()
    feats = {}
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            feats[(r["job"], r["ref_id"], str(r["solid_index"]))] = r

    rows = []
    for key, cat in labels.items():
        fr = feats.get(key)
        if not fr:
            continue
        pred = predict(num(fr.get("n_holes")), num(fr.get("thk_max_over_teff")),
                       num(fr.get("t_eff_thin_ratio")), fr.get("rule_type"),
                       num(fr.get("n_convex_bends")))
        rows.append((cat, pred))
    print(f"labels: {len(labels)}  joined to features: {len(rows)}")

    # confusion (true -> predicted)
    conf = defaultdict(lambda: defaultdict(int))
    for true, pred in rows:
        conf[true][pred or "NONE"] += 1

    preds = sorted({p or "NONE" for _, p in rows})
    print("\nConfusion (rows=your label, cols=rule prediction):")
    print("  " + " " * 14 + "".join(f"{p[:11]:>13}" for p in preds))
    for true in GEOM + OOS:
        if true not in conf:
            continue
        line = "".join(f"{conf[true].get(p,0):>13}" for p in preds)
        print(f"  {true:<14}{line}")

    print("\nPer-class (rules target SECTION/PLATE/FORMED_PLATE only):")
    for c in GEOM:
        tp = sum(1 for t, p in rows if t == c and p == c)
        fp = sum(1 for t, p in rows if t != c and p == c)
        fn = sum(1 for t, p in rows if t == c and p != c)
        n = sum(1 for t, _ in rows if t == c)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        print(f"  {c:<14} n={n:3d}  precision={prec:5.1%}  recall={rec:5.1%}")

    geom = [(t, p) for t, p in rows if t in GEOM]
    acc_geom = sum(1 for t, p in geom if t == p) / len(geom) if geom else 0
    print(f"\nAccuracy on geometric classes (SECTION/PLATE/FORMED): "
          f"{sum(1 for t,p in geom if t==p)}/{len(geom)} = {acc_geom:.1%}")

    oos = [(t, p) for t, p in rows if t in OOS]
    print(f"\nOut-of-scope today ({'/'.join(OOS)}): {len(oos)} parts — the rules "
          f"have no feature for these yet, so all are mis-predicted:")
    for c in OOS:
        sub = [(t, p) for t, p in oos if t == c]
        if sub:
            mis = defaultdict(int)
            for _, p in sub:
                mis[p or "NONE"] += 1
            print(f"  {c:<14} n={len(sub):3d}  -> predicted as {dict(mis)}")


if __name__ == "__main__":
    main()
