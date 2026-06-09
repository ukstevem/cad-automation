"""
Test whether geometry predicts the CNC / Bought-out / Exclude routing decision.

Uses the aligned ``cnc_class`` column emitted by feature_export (each part's
human classification from its own sidecar's project_state).  Evaluates a kNN
classifier leave-one-job-out, deduped by geometry, and reports per-class
precision/recall + balanced accuracy + confusion matrix (accuracy alone is
misleading on imbalanced routing data).

Numpy + pandas only.  Run inside the container::

    docker cp scripts/classify_cnc_bo.py cad-automation-api:/tmp/classify_cnc_bo.py
    docker exec cad-automation-api python /tmp/classify_cnc_bo.py --data /outputs/ml/dataset.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

FEATURES = [
    "dim_long", "dim_mid", "dim_thin",
    "fill_ratio", "elongation", "flatness", "compactness",
    "t_eff", "t_eff_thin_ratio", "developed_ratio",
    "csa_from_vol", "volume_mm3", "surface_area_mm2",
    "inertia_ratio_10", "inertia_ratio_21",
    "n_solids", "instance_count",
]
CLASSMAP = {"postprocess": "CNC", "bought-out": "BO", "exclude": "EXCL"}


def _knn(Xtr, ytr, Xte, k=7):
    preds = []
    for q in Xte:
        d = np.linalg.norm(Xtr - q, axis=1)
        idx = np.argsort(d)[:k]
        votes = defaultdict(float)
        for j in idx:
            votes[ytr[j]] += 1.0 / (1.0 + d[j])
        preds.append(max(votes.items(), key=lambda kv: kv[1])[0])
    return np.array(preds)


def _report(y_true, y_pred, labels, title):
    print(f"\n{title}")
    n = len(y_true)
    acc = (y_true == y_pred).mean()
    print(f"  n={n}  accuracy={acc:.1%}")
    # confusion + per-class
    recalls = []
    print(f"  {'class':6s} {'n':>5} {'precision':>10} {'recall':>8}")
    for c in labels:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        recalls.append(rec)
        print(f"  {c:6s} {int((y_true==c).sum()):5d} {prec:10.1%} {rec:8.1%}")
    print(f"  balanced accuracy (mean recall): {np.mean(recalls):.1%}")


def run(data, binary):
    df = pd.read_csv(data)
    df = df[df["cnc_class"].isin(CLASSMAP) & df["features_ok"].fillna(False).astype(bool)].copy()
    df["cls"] = df["cnc_class"].map(CLASSMAP)
    # one row per (job, ref_id) — part-level routing
    df = df.sort_values("solid_index").drop_duplicates(["job", "ref_id"], keep="first")
    if binary:
        df = df[df["cls"].isin(["CNC", "BO"])]
    print(f"parts: {len(df)}  jobs: {df['job'].nunique()}")
    print("class balance:", df["cls"].value_counts().to_dict())

    feats = [f for f in FEATURES if f in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).to_numpy()
    y = df["cls"].to_numpy()
    jobs = df["job"].to_numpy()
    fps = df["fingerprint_key"].to_numpy()
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xn = (X - mu) / sd

    base = pd.Series(y).value_counts(normalize=True).max()
    print(f"base rate (always-majority): {base:.1%}")

    yt, yp = [], []
    for h in np.unique(jobs):
        te = jobs == h
        tr = ~te
        tfps = set(fps[te])
        tr = tr & np.array([f not in tfps for f in fps])
        if tr.sum() < 20 or te.sum() == 0:
            continue
        pred = _knn(Xn[tr], y[tr], Xn[te])
        yt.extend(y[te]); yp.extend(pred)
    yt, yp = np.array(yt), np.array(yp)
    labels = ["CNC", "BO"] if binary else ["CNC", "BO", "EXCL"]
    _report(yt, yp, labels, "Leave-one-job-out kNN (deduped by geometry):")
    print(f"\n=> base rate {base:.1%}; lift = {(yt==yp).mean()-base:+.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/outputs/ml/dataset.csv")
    ap.add_argument("--binary", action="store_true", help="CNC vs BO only (drop exclude)")
    args = ap.parse_args()
    run(args.data, args.binary)


if __name__ == "__main__":
    main()
