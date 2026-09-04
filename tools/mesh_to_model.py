#!/usr/bin/env python3
"""
Build an AR model (edge polylines) from a mesh, so any part can be tested without its CAD.

The fit consumes CAD edge polylines, which normally come from the STEP file via
app/workers/extract_edges.py. That limits testing to parts whose CAD has been through the
pipeline - one part, in practice. But ~4000 part meshes already exist under outputs/stl/ from
real jobs, and the question "would THIS part's orientation be recoverable" is exactly what needs
answering at that scale.

Feature edges recover what matters. An edge shared by two faces whose normals differ by more than
a threshold is a real crease in the surface; one shared by a single face is a boundary. Together
they are very nearly the edge set a CAD extract produces, because both are picking out the same
physical creases - a tessellated cylinder loses its silhouette edges, but on fabricated steelwork
almost every edge is a genuine crease between planar faces.

No chaining is needed: the fit densifies each polyline with `sample_polylines`, so emitting one
two-point polyline per edge gives an identical point cloud to a chained version, with much less
code to go wrong.

VALIDATE BEFORE TRUSTING. Run this on a mesh whose CAD model already exists, fit both against the
same real captures, and check they agree. If a mesh-derived model fits worse than its CAD
original, every survey result built on it inherits that gap.

    docker compose run --rm --no-deps api python tools/mesh_to_model.py \
        outputs/ar_models/mainframe_default_1to5.stl --out outputs/ar_models/mainframe_mesh.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np


class MeshError(Exception):
    """Raised when a mesh cannot be read."""


def load_stl(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        if len(fh.read(80)) < 80:
            raise MeshError("%s: too short to be a binary STL" % path)
        count = struct.unpack("<I", fh.read(4))[0]
        raw = fh.read(count * 50)
    if len(raw) < count * 50:
        raise MeshError("%s: truncated - expected %d triangles" % (path, count))
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(count, 50)
    return buf[:, 12:48].copy().view(np.float32).reshape(count, 3, 3).astype(np.float64)


def feature_edges(tris: np.ndarray, angle_deg: float = 25.0, weld: float = 1e-3):
    """
    Edges that are creases or boundaries, as an (m, 2, 3) array of segment endpoints.

    STL stores every triangle independently, so shared vertices are duplicated and there is no
    topology at all. Welding by rounded position rebuilds it: vertices closer than *weld* become
    the same index, which is what makes "how many faces share this edge" answerable.
    """
    v = tris.reshape(-1, 3)
    q = np.round(v / weld).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)

    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ok = ln.ravel() > 1e-12
    n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), 0.0)

    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    owner = np.tile(np.arange(len(faces)), 3)
    key = np.sort(pairs, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner, pairs = key[order], owner[order], pairs[order]

    keep = []
    i, m = 0, len(key)
    cos_thr = np.cos(np.radians(angle_deg))
    while i < m:
        j = i + 1
        while j < m and key[j, 0] == key[i, 0] and key[j, 1] == key[i, 1]:
            j += 1
        share = j - i
        if share == 1:                                   # boundary edge - always a real edge
            keep.append(i)
        elif share == 2:
            a, b = owner[i], owner[i + 1]
            if ok[a] and ok[b] and float(np.dot(n[a], n[b])) < cos_thr:
                keep.append(i)                           # crease sharper than the threshold
        # share > 2: non-manifold, skip - it is a mesh defect, not a feature
        i = j

    if not keep:
        return np.zeros((0, 2, 3))
    idx = np.asarray(keep)
    # Map welded indices back to a representative position.
    rep = np.zeros((int(inv.max()) + 1, 3))
    rep[inv] = v
    return np.stack([rep[pairs[idx, 0]], rep[pairs[idx, 1]]], axis=1)


def build(path: str, angle_deg: float = 25.0, scale: float = 1.0, name: str | None = None) -> dict:
    tris = load_stl(path)
    if scale != 1.0:
        tris = tris * scale
    segs = feature_edges(tris, angle_deg=angle_deg)
    if len(segs) == 0:
        raise MeshError("%s: no feature edges at %.0f degrees" % (path, angle_deg))
    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    dims = sorted((hi - lo).tolist(), reverse=True)
    return {
        "name": name or os.path.splitext(os.path.basename(path))[0],
        "source_mesh": os.path.basename(path),
        "scale": scale,
        "units": "mm",
        "note": ("edges derived from the MESH by feature-edge extraction at %.0f degrees, not "
                 "from CAD - see tools/mesh_to_model.py" % angle_deg),
        "bbox": {"min": [round(float(x), 3) for x in lo],
                 "max": [round(float(x), 3) for x in hi]},
        "dims_sorted": [round(float(d), 1) for d in dims],
        "edges": [[[round(float(c), 3) for c in p] for p in s] for s in segs],
        "summary": {"edges": int(len(segs)), "triangles": int(len(tris))},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--angle", type=float, default=25.0,
                    help="dihedral threshold in degrees (default 25)")
    ap.add_argument("--scale", type=float, default=1.0, help="scale the mesh, e.g. 0.2 for 1:5")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    m = build(args.mesh, angle_deg=args.angle, scale=args.scale, name=args.name)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(m, fh)
    print("%s -> %s   %d feature edges from %d triangles, dims %s"
          % (os.path.basename(args.mesh), args.out, m["summary"]["edges"],
             m["summary"]["triangles"], m["dims_sorted"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
