#!/usr/bin/env python3
"""
How THIS article departs from its drawing, kept apart from the drawing (bd 0sb).

The IFC weld sidecar says where the welds are on the DRAWING, and it stays that way. When a fitted
article turns out not to match it - a sub-assembly glued or welded on turned, say - the departure is
recorded as a separate, measured deviation, and the
AR view applies it at runtime to both the outline model and the welds. The page can then show the
article as built and flag that it is not as drawn, without the drawing ever being edited to fit a
bad part.

    # an as-built mesh, for pose_refine
    python tools/deviation.py mesh --deviation outputs/ar_fits/tower01_asbuilt/deviation.json \\
        --out outputs/ar_models/mainframe_default_1to5_asbuilt.stl

    # the page
    python tools/ar_view.py ... --deviation outputs/ar_fits/tower01_asbuilt/deviation.json

ONE KIND SO FAR: half_roll. Everything on one side of a cut across the length is turned about the
length axis, in quarter turns. The axis runs through the SECTION centre - the middle of the box round
the cross-section - not the area centroid. A half glued on turned keeps its rails in line with the
other half's; on the tower the centroid sits 0.6 mm off the section centre, and turning about it
would push the half about a millimetre out of line.

NO DEVIATION HAS BEEN VALIDATED WITH THIS YET. The glued tower was the motivating case, and turning
one half first appeared to lift its silhouette confirmation from 56% to 85%. That was an artefact of
splitting halves by triangle centroid (see as_built_mesh) and of a stale edge cache (bd qhf). With
both fixed, no turn of either half changes the tower's score - and neither does turning the WHOLE
model, so this scoring cannot see a quarter turn of a four-fold symmetric frame at all. Only write a
record from evidence where turning the whole model does change the score.

A weld belongs to the half its centroid is in and turns with it - face codes included, since a weld
on the +W face of a half turned a quarter turn is on +D afterwards.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import visibility as VIS  # noqa: E402
import weld_faces as WF  # noqa: E402

SCHEMA = "PSS-ArticleDeviation/0.1"
KINDS = ("half_roll",)


def load(path):
    """Read a deviation record, refusing anything this module does not know how to apply."""
    with open(path, "r", encoding="utf-8") as fh:
        dev = json.load(fh)
    if dev.get("schema") != SCHEMA or dev.get("kind") not in KINDS:
        raise ValueError("%s is not a %s record of a known kind %s" % (path, SCHEMA, KINDS))
    if dev.get("turned_half") not in ("-L", "+L"):
        raise ValueError("turned_half must be -L or +L, not %r" % dev.get("turned_half"))
    if float(dev["angle_deg"]) % 90.0:
        raise ValueError("a half can only be turned in quarter turns, not %s degrees" % dev["angle_deg"])
    return dev


def check_article(dev, frame, tol_mm=0.5):
    """A deviation was measured against one drawing; refuse to apply it to another."""
    want = np.asarray(dev["article"]["frame_extent_mm"], np.float64)
    got = frame["hi"] - frame["lo"]
    if np.any(np.abs(want - got) > tol_mm):
        raise ValueError("the deviation was measured on an article of %s mm, this mesh is %s mm"
                         % (np.round(want, 1).tolist(), np.round(got, 1).tolist()))


def turn(frame, dev):
    """The rotation, and the point on the length axis it turns about, in model coordinates."""
    mid = (frame["lo"] + frame["hi"]) / 2.0
    pivot = frame["centre"] + frame["axes"] @ np.array([float(dev["split_mm"]), mid[1], mid[2]])
    R, _ = cv2.Rodrigues((frame["axes"][:, 0] * np.radians(float(dev["angle_deg"]))).reshape(3, 1))
    return R, pivot


def on_turned_half(points, frame, dev):
    """True for each point on the turned side of the cut."""
    s = ((np.asarray(points, np.float64).reshape(-1, 3) - frame["centre"]) @ frame["axes"][:, 0]
         - float(dev["split_mm"]))
    return s < 0 if dev["turned_half"] == "-L" else s >= 0


def _clip(tri, d):
    """Split one triangle by the plane ``d == 0`` into triangles wholly on each side.

    Sutherland-Hodgman against each side of the plane, then a fan, so the winding is kept."""
    below, above = [], []
    for i in range(3):
        p, q, dp, dq = tri[i], tri[(i + 1) % 3], d[i], d[(i + 1) % 3]
        if dp <= 0:
            below.append(p)
        if dp >= 0:
            above.append(p)
        if dp * dq < 0:
            x = p + (q - p) * (dp / (dp - dq))
            below.append(x)
            above.append(x)
    pieces = []
    for poly in (below, above):
        for k in range(1, len(poly) - 1):
            t = np.array([poly[0], poly[k], poly[k + 1]])
            if np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-9:
                pieces.append(t)
    return pieces


def as_built_mesh(mesh, frame, dev):
    """The drawn mesh with the turned half turned. Returns ``(mesh, mask of turned triangles)``.

    Triangles that cross the cut are SPLIT at it first. Deciding by centroid alone is wrong on real
    models: on the tower 394 triangles are rail faces running the whole 427 mm length, so each
    'half' took long faces belonging to the other one with it - which alone made a half turn look
    like an 85% fit on an article where, split properly, no turn changes the score. Split at the
    cut, each half is exactly its half."""
    tris = np.asarray(mesh, np.float64).reshape(-1, 3, 3)
    d = (tris - frame["centre"]) @ frame["axes"][:, 0] - float(dev["split_mm"])
    crosses = (d.min(axis=1) < 0) & (d.max(axis=1) > 0)
    pieces = [t for tri, dt in zip(tris[crosses], d[crosses]) for t in _clip(tri, dt)]
    out = np.concatenate([tris[~crosses], np.asarray(pieces, np.float64).reshape(-1, 3, 3)])
    mask = on_turned_half(out.mean(axis=1), frame, dev)
    R, pivot = turn(frame, dev)
    out[mask] = ((out[mask].reshape(-1, 3) - pivot) @ R.T + pivot).reshape(-1, 3, 3)
    return out, mask


def remap_faces(codes, frame, dev):
    """Face codes after the turn, in ``weld_faces.FACES`` order. Ends stay ends; a long face goes to
    the long face its outward normal is turned onto."""
    R, _ = turn(frame, dev)
    out = set()
    for code in codes:
        n = R @ WF._normal(frame, code)
        best = max(WF.FACES, key=lambda c: float(WF._normal(frame, c) @ n))
        if float(WF._normal(frame, best) @ n) < 0.9:
            raise ValueError("a %s degree turn does not take face %s onto a face"
                             % (dev["angle_deg"], code))
        out.add(best)
    return [c for c in WF.FACES if c in out]


def as_built_welds(welds, frame, dev, scale):
    """The welds as they sit on this article. The input is left untouched - the sidecar is the
    drawing. Returns ``(welds, names of the welds that turned)``."""
    R, pivot = turn(frame, dev)
    out, turned = copy.deepcopy(list(welds)), []
    for w in out:
        segs = [np.asarray(s, np.float64).reshape(-1, 3) * scale
                for s in w["Representation"]["segments"]]
        if not on_turned_half(np.vstack(segs).mean(axis=0), frame, dev)[0]:
            continue
        w["Representation"]["segments"] = [(((p - pivot) @ R.T + pivot) / scale).tolist()
                                           for p in segs]
        geometry = w.get("Pset_PSS_WeldGeometry") or {}
        if geometry.get("ArticleFaces") is not None:
            geometry["ArticleFaces"] = remap_faces(geometry["ArticleFaces"], frame, dev)
        turned.append(w["Name"])
    return out, turned


def summary(dev, turned, path=None):
    """What the AR page needs to say about the deviation."""
    return {"description": dev["description"], "turned_half": dev["turned_half"],
            "angle_deg": dev["angle_deg"], "joint": dev.get("joint_note"),
            "turned_welds": sorted(turned), "file": os.path.basename(path) if path else None,
            "evidence": dev.get("evidence", [])}


def write_stl(path, tris):
    """Binary STL, normals from the winding."""
    tris = np.asarray(tris, np.float32).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tris
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(np.uint32(len(tris)).tobytes())
        fh.write(rec.tobytes())


def cmd_mesh(args) -> int:
    dev = load(args.deviation)
    mesh_path = args.mesh or os.path.join("outputs/ar_models", dev["article"]["mesh"])
    mesh = VIS.load_stl(mesh_path)
    frame = WF.article_frame(mesh)
    check_article(dev, frame)
    built, mask = as_built_mesh(mesh, frame, dev)
    write_stl(args.out, built)
    print("%s: %d of %d triangles turned %s degrees -> %s"
          % (dev["description"], int(mask.sum()), len(mask), dev["angle_deg"], args.out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mesh", help="write the article's mesh as built")
    m.add_argument("--deviation", required=True)
    m.add_argument("--mesh", default=None, help="defaults to the mesh the deviation was measured on")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"mesh": cmd_mesh}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
