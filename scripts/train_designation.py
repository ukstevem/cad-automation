"""
Baseline trainer / evaluator for fine-grained section designation ID.

Consumes the dataset produced by ``app.pipeline.feature_export`` and reports,
with honest split-by-job / dedup-by-geometry guards:

  1. Dataset summary (rows by label source, type, job).
  2. Agreement matrix: name-derived labels vs the existing rule classifier.
  3. Library-NN designation baseline — predict designation by nearest neighbour
     in the steel library's (height, width, csa) space using *measured* dims
     (the "de-noise then lookup" target). Category top-1, designation top-1/3.
  4. kNN category classifier, leave-one-job-out, deduped by fingerprint_key.

Runs on numpy + pandas alone (both already in the cad-automation image).
scikit-learn is used automatically if present but is not required.

Usage (inside the container)::

    docker cp scripts/train_designation.py cad-automation-api:/tmp/train_designation.py
    docker exec cad-automation-api python /tmp/train_designation.py \
        --data /outputs/ml/dataset.csv \
        --library /app/app/pipeline/data/Shape_classifier_info.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Numeric features used by the kNN category classifier. Kept in sync with
# app/pipeline/feature_extract.FEATURE_KEYS but listed explicitly so the trainer
# is robust to dataset column reordering.
NUM_FEATURES = [
    "obb_length", "obb_height", "obb_width",
    "dim_long", "dim_mid", "dim_thin",
    "section_area", "cs_H", "cs_W",
    "fill_ratio", "elongation", "flatness", "compactness",
    "csa_from_vol", "t_eff", "t_eff_thin_ratio", "developed_ratio",
    "volume_mm3", "surface_area_mm2",
    "inertia0", "inertia1", "inertia2",
    "inertia_ratio_10", "inertia_ratio_21",
    "n_holes", "thk_max", "thk_cov", "thk_max_over_teff",
]


def _hr(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def summarise(df: pd.DataFrame) -> None:
    _hr("1. DATASET SUMMARY")
    print(f"Total rows (solids): {len(df)}")
    print(f"Distinct jobs:       {df['job'].nunique()}")
    print(f"Distinct fingerprints: {df['fingerprint_key'].nunique()}")
    print(f"features_ok rows:    {int(df['features_ok'].fillna(False).astype(bool).sum())}")
    print("\nBy label_source:")
    print(df["label_source"].value_counts().to_string())
    print("\nBy y_type:")
    print(df["y_type"].value_counts(dropna=False).to_string())
    print("\nTop categories (trusted labels):")
    trusted = df[df["label_source"].isin(["verified", "name"])]
    if len(trusted):
        print(trusted["y_category"].value_counts().head(15).to_string())
    else:
        print("  (none — no name/verified labels yet)")


def agreement(df: pd.DataFrame) -> None:
    _hr("2. NAME vs RULE AGREEMENT")
    both = df[df["name_designation"].notna() & df["rule_designation"].notna()].copy()
    if not len(both):
        print("No rows have both a name designation and a rule designation.")
        return
    both["cat_match"] = (
        both["name_category"].str.upper() == both["rule_category"].astype(str).str.upper()
    )
    both["des_match"] = (
        both["name_designation"].astype(str).str.lower().str.replace("-", "x")
        == both["rule_designation"].astype(str).str.lower().str.replace("-", "x")
    )
    n = len(both)
    print(f"Rows with both name + rule designation: {n}")
    print(f"  category agreement:    {both['cat_match'].mean():.1%}")
    print(f"  designation agreement: {both['des_match'].mean():.1%}")
    disagree = both[~both["des_match"]]
    if len(disagree):
        print(f"\nTop designation disagreements (candidates for verified.csv): "
              f"{len(disagree)} rows")
        cols = ["job", "ref_id", "part_name", "name_designation",
                "rule_category", "rule_designation"]
        print(disagree[cols].head(20).to_string(index=False))


# --------------------------------------------------------------------------
# Library nearest-neighbour designation baseline
# --------------------------------------------------------------------------
def load_library(path: Path):
    lib = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for cat, ents in lib.items():
        for des, info in ents.items():
            h, w, a = info.get("height"), info.get("width"), info.get("csa")
            if h and w and a:
                rows.append((cat, des, float(h), float(w), float(a)))
    arr = np.array([[r[2], r[3], r[4]] for r in rows], dtype=float)
    meta = [(r[0], r[1]) for r in rows]
    # z-score normalise each column by library spread
    mu, sd = arr.mean(0), arr.std(0)
    sd[sd == 0] = 1.0
    return arr, meta, mu, sd


def library_nn_eval(df: pd.DataFrame, lib_path: Path) -> None:
    _hr("3. LIBRARY-NN DESIGNATION BASELINE (measured dims -> nearest library)")
    if not lib_path.exists():
        print(f"Library not found: {lib_path}")
        return
    arr, meta, mu, sd = load_library(lib_path)
    libn = (arr - mu) / sd

    # Evaluate on rows that have a trusted (name) designation and measured dims.
    ev = df[df["name_designation"].notna()
            & df["section_area"].notna()
            & df["cs_H"].notna() & df["cs_W"].notna()].copy()
    ev = ev[ev["section_area"] > 0]
    if not len(ev):
        print("No evaluable rows (need name designation + measured cross-section).")
        return

    top1_des = top3_des = top1_cat = 0
    n = 0
    for _, r in ev.iterrows():
        h, w, a = float(r["cs_H"]), float(r["cs_W"]), float(r["section_area"])
        # try both orientations (profile may be measured rotated 90 deg)
        best = None
        for hh, ww in ((h, w), (w, h)):
            q = (np.array([hh, ww, a]) - mu) / sd
            d = np.linalg.norm(libn - q, axis=1)
            order = np.argsort(d)
            if best is None or d[order[0]] < best[0]:
                best = (d[order[0]], order)
        order = best[1]
        true_des = str(r["name_designation"]).lower().replace("-", "x")
        true_cat = str(r["name_category"]).upper()
        top3 = [meta[i] for i in order[:3]]
        if top3 and top3[0][1].lower().replace("-", "x") == true_des:
            top1_des += 1
        if any(m[1].lower().replace("-", "x") == true_des for m in top3):
            top3_des += 1
        if top3 and top3[0][0].upper() == true_cat:
            top1_cat += 1
        n += 1

    print(f"Evaluated on {n} name-labelled section rows:")
    print(f"  category   top-1: {top1_cat / n:.1%}")
    print(f"  designation top-1: {top1_des / n:.1%}")
    print(f"  designation top-3: {top3_des / n:.1%}")
    print("\nNote: this is RAW nearest-neighbour (no learning). It is the floor a "
          "trained de-noiser must beat.")


# --------------------------------------------------------------------------
# kNN category classifier, leave-one-job-out
# --------------------------------------------------------------------------
def _knn_predict(Xtr, ytr, Xte, k=5):
    preds = []
    for q in Xte:
        d = np.linalg.norm(Xtr - q, axis=1)
        idx = np.argsort(d)[:k]
        votes = defaultdict(float)
        for j in idx:
            votes[ytr[j]] += 1.0 / (1.0 + d[j])
        preds.append(max(votes.items(), key=lambda kv: kv[1])[0])
    return np.array(preds)


def loo_category(df: pd.DataFrame) -> None:
    _hr("4. kNN CATEGORY CLASSIFIER (leave-one-job-out, dedup by geometry)")
    data = df[df["label_source"].isin(["verified", "name"])
             & df["features_ok"].fillna(False).astype(bool)
             & df["y_category"].notna()].copy()
    if data["job"].nunique() < 2:
        print(f"Need >=2 jobs with trusted labels; have "
              f"{data['job'].nunique()}. Skipping LOO.")
        return

    feats = [c for c in NUM_FEATURES if c in data.columns]
    X = data[feats].astype(float)
    med = X.median()
    X = X.fillna(med).to_numpy()
    y = data["y_category"].str.upper().to_numpy()
    jobs = data["job"].to_numpy()
    fps = data["fingerprint_key"].to_numpy()

    # standardise on the full set (cheap, avoids per-fold drift for a baseline)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd

    correct = total = 0
    per_cat = defaultdict(lambda: [0, 0])  # cat -> [correct, total]
    for held in np.unique(jobs):
        te = jobs == held
        tr = ~te
        # dedup: drop train rows sharing a fingerprint with the test fold
        test_fps = set(fps[te])
        tr = tr & np.array([fp not in test_fps for fp in fps])
        if tr.sum() < 5 or te.sum() == 0:
            continue
        preds = _knn_predict(Xn[tr], y[tr], Xn[te], k=5)
        for p, t in zip(preds, y[te]):
            ok = p == t
            correct += ok
            total += 1
            per_cat[t][1] += 1
            per_cat[t][0] += int(ok)

    if total == 0:
        print("No evaluable folds after dedup.")
        return
    print(f"Overall LOO category accuracy: {correct}/{total} = {correct/total:.1%}")
    print(f"({len(feats)} features, k=5, distance-weighted vote)")
    print("\nPer-category recall:")
    for cat in sorted(per_cat, key=lambda c: -per_cat[c][1]):
        c, t = per_cat[cat]
        print(f"  {cat:<8} {c:>4}/{t:<4} = {c/t:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/outputs/ml/dataset.csv")
    ap.add_argument("--library",
                    default="/app/app/pipeline/data/Shape_classifier_info.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")
    summarise(df)
    agreement(df)
    library_nn_eval(df, Path(args.library))
    loo_category(df)


if __name__ == "__main__":
    main()
