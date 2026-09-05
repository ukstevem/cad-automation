#!/usr/bin/env python3
"""
Search the labelled survey for a CAD-only property that predicts orientation recovery.

This is the step that was impossible until now. Two screens were built and both were rejected by
the single physical part available; one part can reject a screen but cannot calibrate one. The
survey supplies what they lacked - dozens of parts, each labelled with how often its orientation
was actually recovered - so candidate metrics can be scored against ground truth instead of
against intuition.

The candidates below encode different guesses about WHY a part is determinable:

  end_asymmetry     do the two ENDS look different? The Main Frame is determinable because one
                    end carries a bolted plate and the other is an open box. This compares the
                    outer quarter at each end after mirroring, which is the end-for-end question
                    asked directly rather than as a global symmetry test.
  flip_residual     the global 180-degree flip test, kept as a control - it has already failed
                    twice and should fail again if the labels are meaningful.
  end_mass_ratio    crude but cheap: how much more material sits at one end than the other.
  aspect, edges     shape and richness, both already checked band-wise with no trend.
  extent_ratio      mid dimension over thin - distinguishes a flat plate from a box section.

Correlation here is a screen, not a mechanism. Anything that survives needs testing on parts held
out of the fitting, because with several candidates and dozens of parts, one will look good by
chance.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_to_model import load_stl  # noqa: E402


def metrics(path: str, sample: int = 4000) -> dict | None:
    try:
        tris = load_stl(path)
    except Exception:
        return None
    pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    if len(pts) > sample:
        pts = pts[np.random.default_rng(0).choice(len(pts), sample, replace=False)]
    if len(pts) < 50:
        return None
    from scipy.spatial import cKDTree

    c = pts.mean(axis=0)
    d = pts - c
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    evecs = evecs[:, np.argsort(evals)[::-1]]
    t = d @ evecs[:, 0]
    ext = np.array([float(np.ptp(d @ evecs[:, i])) for i in range(3)])
    longest = float(ext[0]) or 1.0
    tol = 0.01 * longest

    lo, hi = float(t.min()), float(t.max())
    span = hi - lo
    a = pts[t < lo + 0.25 * span]                 # outer quarter, one end
    b = pts[t > hi - 0.25 * span]                 # outer quarter, the other end

    end_asym = float("nan")
    if len(a) > 20 and len(b) > 20:
        # Mirror end B onto end A through the centroid along the long axis, then ask how far each
        # mirrored point lands from anything in A. If the ends match, this is ~0 and no camera can
        # tell them apart; if one end carries a plate the other lacks, it is large.
        ax = evecs[:, 0]
        mb = b - 2.0 * np.outer((b - c) @ ax, ax)
        dist, _ = cKDTree(a).query(mb)
        end_asym = float((dist > tol).mean())

    # Global flip, as a control.
    moved = c + 2.0 * np.outer(d @ evecs[:, 0], evecs[:, 0]) - d
    dist2, _ = cKDTree(pts).query(moved)
    flip_residual = float((dist2 > tol).mean())

    return {
        "end_asymmetry": round(end_asym, 4),
        "flip_residual": round(flip_residual, 4),
        "end_mass_ratio": round(float(max(len(a), len(b)) / max(min(len(a), len(b)), 1)), 3),
        "aspect": round(longest / max(float(ext[2]), 1e-6), 2),
        "extent_ratio": round(float(ext[1]) / max(float(ext[2]), 1e-6), 3),
        "edges": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", default="outputs/ar_fits/survey_night.csv")
    ap.add_argument("--models", default="outputs/ar_models/_survey")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.survey)) if int(r["trials"]) > 0]
    data = []
    for r in rows:
        stl = os.path.join(args.models, r["part"] + ".stl")
        if not os.path.exists(stl):
            continue
        m = metrics(stl)
        if not m:
            continue
        m["edges"] = float(r["edges"])
        m["rate"] = int(r["recovered"]) / int(r["trials"])
        m["part"] = r["part"]
        m["source"] = r["source"]
        data.append(m)
    if not data:
        print("no parts scored - are the survey meshes still in %s?" % args.models,
              file=sys.stderr)
        return 1

    print("scored %d labelled parts\n" % len(data))
    keys = ["end_asymmetry", "flip_residual", "end_mass_ratio", "aspect", "extent_ratio", "edges"]
    y = np.array([d["rate"] for d in data])
    print("%-16s %8s   %s" % ("metric", "corr", "mean recovery in each tercile (low/mid/high)"))
    for k in keys:
        x = np.array([d[k] if d[k] == d[k] else np.nan for d in data], float)
        ok = ~np.isnan(x)
        if ok.sum() < 10 or np.std(x[ok]) < 1e-9:
            continue
        corr = float(np.corrcoef(x[ok], y[ok])[0, 1])
        order = np.argsort(x[ok])
        thirds = np.array_split(order, 3)
        means = [float(y[ok][t].mean()) for t in thirds]
        print("%-16s %8.3f   %.0f%%  %.0f%%  %.0f%%"
              % (k, corr, 100 * means[0], 100 * means[1], 100 * means[2]))

    print("\nthe 9 parts recovered EVERY time:")
    for d in sorted(data, key=lambda d: -d["rate"])[:9]:
        if d["rate"] < 1.0:
            break
        print("   end_asym %.2f  flip %.2f  aspect %6.1f  edges %5.0f  %s"
              % (d["end_asymmetry"], d["flip_residual"], d["aspect"], d["edges"],
                 d["source"][:40]))
    print("\nthe parts recovered NEVER:")
    for d in sorted(data, key=lambda d: d["rate"])[:9]:
        if d["rate"] > 0.0:
            break
        print("   end_asym %.2f  flip %.2f  aspect %6.1f  edges %5.0f  %s"
              % (d["end_asymmetry"], d["flip_residual"], d["aspect"], d["edges"],
                 d["source"][:40]))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print("\nwrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
