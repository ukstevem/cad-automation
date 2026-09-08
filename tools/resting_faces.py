#!/usr/bin/env python3
"""
Enumerate the ways a part can be set down, and record the ones an operator may choose from.

WHY THIS IS CURATED RATHER THAN DERIVED. A pose search cannot represent discrete placements - a
plate face-down, a frame turned over - so when one of those is the truth it returns the best of a
set of wrong answers with a plausible score. Handing that choice to the operator fixes it in one
click, but only if the choices offered are the right ones.

Deriving them from the convex hull works for a box or a frame, where the part rests on a face and
there are a handful of them. It fails for a cylinder, which rests on a line and has a continuum of
orientations, and it would offer nonsense for anything that cannot physically be stood on end. So
this proposes candidates and writes the ones you keep into the model JSON: geometry does the
tedious part, a human keeps the veto.

A candidate is a hull face large enough to stand on, with the centre of mass projecting inside it -
the actual criterion for stability, not merely a big face. Orientations that differ only by a turn
about the vertical are the same resting pose, because yaw is solved from the clicks; they are
merged so the operator is not offered the same placement four times.

    python tools/resting_faces.py --mesh outputs/ar_models/mainframe_default_1to5.stl \\
        --model outputs/ar_models/mainframe_default_1to5.json --out outputs/ar_models/thumbs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import visibility as VIS  # noqa: E402


def rotation_putting_down(n):
    """Rotation taking the unit vector *n* to -z, i.e. laying that face on the table."""
    n = np.asarray(n, np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    target = np.array([0.0, 0.0, -1.0])
    v = np.cross(n, target)
    c = float(np.dot(n, target))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def candidates(tris, min_area_frac=0.02):
    """
    Hull faces the part could actually stand on, as (rotation, area, stable) tuples.

    Grouping coplanar hull triangles matters: a rectangular base arrives as two triangles, and
    judged separately neither may contain the centre of mass while the base plainly does.
    """
    from scipy.spatial import ConvexHull

    pts = tris.reshape(-1, 3)
    hull = ConvexHull(pts)
    com = pts.mean(axis=0)                      # centroid stands in for the centre of mass
    total = hull.area

    groups = {}
    for eq, simplex in zip(hull.equations, hull.simplices):
        n = eq[:3]
        key = tuple(np.round(n, 2)) + (round(float(eq[3]), 1),)
        tri = pts[simplex]
        a = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
        g = groups.setdefault(key, {"n": n, "d": float(eq[3]), "area": 0.0, "pts": []})
        g["area"] += a
        g["pts"].append(tri)

    out = []
    for g in groups.values():
        if g["area"] < min_area_frac * total:
            continue
        n = g["n"]
        P = np.vstack(g["pts"]).reshape(-1, 3)
        # does the centre of mass fall inside the footprint? project both onto the face plane
        u = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(u, n)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        u = u - np.dot(u, n) * n
        u /= max(np.linalg.norm(u), 1e-12)
        w = np.cross(n, u)
        flat = np.stack([P @ u, P @ w], axis=1).astype(np.float32)
        c2 = np.array([float(com @ u), float(com @ w)], np.float32)
        hull2 = cv2.convexHull(flat)
        stable = cv2.pointPolygonTest(hull2, (float(c2[0]), float(c2[1])), False) >= 0
        out.append({"R": rotation_putting_down(n), "area": g["area"] / total, "stable": bool(stable),
                    "normal": n.tolist()})
    return sorted(out, key=lambda r: -r["area"])


def merge_by_yaw(cands, tris, tol_mm=2.0, samples=400, step_deg=5.0):
    """
    Drop candidates that a turn about the vertical maps onto one another.

    Tested DIRECTLY - rotate one point set through every yaw and see whether it lands on the other
    - rather than by comparing rotation-invariant signatures. A signature built from radius and
    height cannot see handedness at all: every point on a plate's outline exists at both the top
    and the bottom face, so flipping the plate produces an identical set of (radius, height) pairs
    and the two orientations merge. Those are exactly the face-up and face-down pair a handedness
    test exists to tell apart, so losing them is worse than doing the work.

    A reflection cannot be undone by a rotation, which is the whole point: the direct test keeps
    the pair and a signature test cannot.
    """
    from scipy.spatial import cKDTree

    P = tris.reshape(-1, 3)
    P = P - P.mean(axis=0)
    if len(P) > samples:
        P = P[np.linspace(0, len(P) - 1, samples).astype(int)]
    yaws = [cv2.Rodrigues(np.array([0.0, 0.0, np.radians(a)]).reshape(3, 1))[0]
            for a in np.arange(0, 360, step_deg)]

    keep = []
    for c in cands:
        rp = (c["R"] @ P.T).T
        dup = False
        for k in keep:
            tree = k["_tree"]
            for Ry in yaws:
                d, _ = tree.query((Ry @ rp.T).T)
                if float(np.percentile(d, 95)) < tol_mm:
                    dup = True
                    break
            if dup:
                break
        if not dup:
            c["_tree"] = cKDTree(rp)
            keep.append(c)
    return keep


def thumbnail(tris, R, size=220):
    """A small shaded view of the part in this orientation, for the operator to recognise."""
    P = (R @ tris.reshape(-1, 3).T).T.reshape(-1, 3, 3)
    P = P - P.reshape(-1, 3).min(axis=0)
    # a fixed three-quarter view, so every thumbnail is comparable
    look, _ = cv2.Rodrigues(np.array([-1.05, 0.0, 0.0]))
    spin, _ = cv2.Rodrigues(np.array([0.0, 0.0, -0.6]))
    V = (look @ spin @ P.reshape(-1, 3).T).T.reshape(-1, 3, 3)
    flat = V.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) or 1.0
    s = (size - 24) / span
    img = np.full((size, size, 3), 250, np.uint8)
    order = np.argsort(V[:, :, 2].mean(axis=1))          # painter's, far first
    n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    shade = np.clip(0.35 + 0.65 * np.abs(n[:, 2]), 0, 1)
    for i in order:
        q = ((V[i, :, :2] - lo[:2]) * s + 12).astype(np.int32)
        q[:, 1] = size - q[:, 1]
        g = int(70 + 150 * shade[i])
        cv2.fillConvexPoly(img, q, (g, g, g), lineType=cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--model", default=None, help="model JSON to write the chosen set into")
    ap.add_argument("--out", default=None, help="directory for the thumbnails")
    ap.add_argument("--keep", default=None,
                    help="comma-separated indices to record, from a previous listing. Omit to "
                         "list candidates without writing anything.")
    ap.add_argument("--min-area", type=float, default=0.02)
    ap.add_argument("--unstable", action="store_true",
                    help="include faces the part would topple off. Occasionally wanted for a part "
                         "held in a fixture rather than free-standing.")
    args = ap.parse_args()

    tris = VIS.load_stl(args.mesh)
    cands = candidates(tris, min_area_frac=args.min_area)
    if not args.unstable:
        cands = [c for c in cands if c["stable"]]
    cands = merge_by_yaw(cands, tris)
    if not cands:
        print("no candidate resting faces. A part that rests on a line or a point - a cylinder, a "
              "cone - has no discrete set to choose from, and needs its orientations defined by "
              "hand.", file=sys.stderr)
        return 1

    print("%3s %9s %8s  %s" % ("idx", "footprint", "stable", "face normal in model coords"))
    for i, c in enumerate(cands):
        print("%3d %8.1f%% %8s  [%6.2f %6.2f %6.2f]"
              % (i, 100 * c["area"], "yes" if c["stable"] else "NO", *c["normal"]))
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            name = "%s_rest%d.png" % (os.path.splitext(os.path.basename(args.mesh))[0], i)
            cv2.imwrite(os.path.join(args.out, name), thumbnail(tris, c["R"]))
    if args.out:
        print("\nthumbnails in %s - look at them before keeping any" % args.out)

    if args.keep is None:
        print("\nNothing written. Re-run with --keep to record the ones an operator may choose,")
        print("e.g. --keep 0,2  (a cylinder or a part that only ever sits one way may want just one)")
        return 0

    idx = [int(x) for x in args.keep.split(",") if x.strip() != ""]
    chosen = [{"index": i, "footprint": round(float(cands[i]["area"]), 4),
               "stable": cands[i]["stable"],
               "normal": [round(float(x), 6) for x in cands[i]["normal"]],
               "rvec": [round(float(x), 6)
                        for x in cv2.Rodrigues(cands[i]["R"])[0].ravel()]}
              for i in idx]
    if args.model:
        with open(args.model, "r", encoding="utf-8") as fh:
            model = json.load(fh)
        model["resting_faces"] = chosen
        with open(args.model, "w", encoding="utf-8") as fh:
            json.dump(model, fh, indent=2)
        print("\nrecorded %d resting faces in %s" % (len(chosen), args.model))
        print("The operator picks one of these; yaw and position come from the clicks.")
    else:
        print(json.dumps(chosen, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
