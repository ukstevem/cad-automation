#!/usr/bin/env python3
"""
NOT VALIDATED. Do not use these numbers to decide whether a part will work.

The intent was a pre-flight check: given a job's CAD, predict which parts have an orientation a
camera can recover, before committing the cell to them. The idea is sound - if a part is
near-symmetric under a 180 degree flip then NO camera settles which way round it is, because the
information is absent from the shape rather than lost by the sensor.

Two metrics were tried and BOTH failed the only calibration available, so this file is kept as a
record of what does not work rather than as a working tool.

    ground truth (1:5 Main Frame, four controlled tests):
        end-for-end   recovered correctly 3/3 seen broadside, 0/1 seen near end-on
        roll          never recovered, in any test

    metric 1, 3D symmetry (`flip_asymmetry` below):
        roll scores 0.618 - i.e. "62% of points move, easily distinguishable".
        Wrong. It ignores that the solver re-fits position and yaw to compensate, and it counts
        points on the far side of the part that no camera can see. On this rig only ~18% of
        sampled points are visible at a typical pose (see app/services/visibility.py).

    metric 2, projected silhouette after re-fitting position and yaw:
        roll 0.724 against end-for-end 0.788 - almost identical, when one works and one never
        does. Better founded, still wrong. It too ignores visibility, and it compares the flipped
        part floating in free space rather than RE-SEATED on the board, which is the only way it
        can actually sit.

There is also an unresolved inconsistency to settle before trusting anything here. An earlier
measurement (app/services/model_symmetry.py, using CAD EDGE points and OBJECT-frame axes) put the
roll at 3.4% discriminating, which matches the observed behaviour. The measurements above use MESH
points and PRINCIPAL axes and disagree by a factor of twenty. At least one is measuring a
different rotation than it claims - object Y and the principal long axis need not coincide.

THE DEEPER PROBLEM IS n=1. One physical part with ground truth cannot calibrate a screen; it can
only reject one. Both metrics above were rejected by it, which is worth something, but the next
metric to pass it may pass by luck. Ground truth on several parts of genuinely different shape has
to come first - see the bead on screening.

Kept runnable so the measurements can be reproduced. The RAG verdict was REMOVED deliberately: a
confident label that is wrong is worse than no label, and this one was wrong on the one case where
the answer is known.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import struct
import sys

import numpy as np


class MeshError(Exception):
    """Raised when a mesh cannot be read."""


def load_stl(path: str) -> np.ndarray:
    """Binary STL -> (n, 3, 3) triangle vertices. Standalone: numpy and scipy only."""
    with open(path, "rb") as fh:
        if len(fh.read(80)) < 80:
            raise MeshError("%s: too short to be a binary STL" % path)
        count = struct.unpack("<I", fh.read(4))[0]
        raw = fh.read(count * 50)
    if len(raw) < count * 50:
        raise MeshError("%s: truncated - expected %d triangles" % (path, count))
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(count, 50)
    return buf[:, 12:48].copy().view(np.float32).reshape(count, 3, 3).astype(np.float64)


def sample_points(tris: np.ndarray, n: int = 4000, seed: int = 0) -> np.ndarray:
    pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    if len(pts) > n:
        pts = pts[np.random.default_rng(seed).choice(len(pts), n, replace=False)]
    return pts


def flip_asymmetry(pts: np.ndarray, axis: np.ndarray, tol: float) -> float:
    """METRIC 1 - known to overestimate. Fraction of points landing >tol from the shape."""
    from scipy.spatial import cKDTree

    c = pts.mean(axis=0)
    a = axis / np.linalg.norm(axis)
    d = pts - c
    moved = c + 2.0 * np.outer(d @ a, a) - d
    dist, _ = cKDTree(pts).query(moved)
    return float((dist > tol).mean())


def measure(path: str, sample: int = 4000) -> dict | None:
    """Raw numbers only. No verdict - see the module docstring for why."""
    try:
        tris = load_stl(path)
    except (MeshError, OSError, ValueError):
        return None
    if len(tris) < 8:
        return None
    pts = sample_points(tris, sample)
    if len(pts) < 50:
        return None

    c = pts.mean(axis=0)
    d = pts - c
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    evecs = evecs[:, np.argsort(evals)[::-1]]
    extent = np.array([float(np.ptp(d @ evecs[:, i])) for i in range(3)])
    longest = float(extent.max())
    if longest < 1e-6:
        return None
    tol = 0.01 * longest

    fr = [flip_asymmetry(pts, evecs[:, i], tol) for i in range(3)]
    return {
        "part": os.path.basename(path),
        "project": os.path.basename(os.path.dirname(path)),
        "len_mm": round(longest, 1),
        "mid_mm": round(float(extent[1]), 1),
        "thin_mm": round(float(extent[2]), 1),
        "aspect": round(longest / max(float(extent[2]), 1e-6), 1),
        "m1_roll_long_axis": round(fr[0], 3),
        "m1_flip_mid_axis": round(fr[1], 3),
        "m1_flip_short_axis": round(fr[2], 3),
        "m1_worst": round(float(min(fr)), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--min-mm", type=float, default=50.0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    print("*** METRIC NOT VALIDATED - it scores the Main Frame's roll as easily")
    print("*** distinguishable when roll has never once been recovered. Numbers")
    print("*** below are for investigation only. See the module docstring.\n")

    files = []
    for r in args.roots:
        for pat in ("*.stl", "*.STL"):
            files.extend(glob.glob(os.path.join(r, "**", pat), recursive=True))
    files = sorted(set(files))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("no STL files found", file=sys.stderr)
        return 2

    rows = []
    for f in files:
        m = measure(f, args.sample)
        if m and m["len_mm"] >= args.min_mm:
            rows.append(m)
    if not rows:
        print("nothing measured", file=sys.stderr)
        return 1

    w = np.array([r["m1_worst"] for r in rows])
    print("measured %d parts.  metric-1 worst-flip score:" % len(rows))
    for q in (5, 25, 50, 75, 95):
        print("    %2dth percentile  %.3f" % (q, float(np.percentile(w, q))))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(sorted(rows, key=lambda r: r["m1_worst"]))
        print("\nwrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
