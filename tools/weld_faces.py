#!/usr/bin/env python3
"""
Which face of the article each weld is on, and which faces a camera is looking at (bd bn5).

WHY FACES, NOT A HULL. The rule is "number only the welds on the faces of the article presented to
this camera". The first implementation approximated "presented" with the part's CONVEX HULL: a weld
counted if it sat within 25 mm of the hull's near surface. On the tower that inverted the result -
each camera showed welds at its own end only, although both look down the same top face. Measured
on 2026-09-14, the hull removed all 14 line-of-sight joints at camera B's end plate: joints on the
end plate's inner face, plainly visible over the open top, but far behind the envelope. On an open
frame "near the hull" is not "on a face". So a weld gets FACE CODES from the geometry, once, and a
camera shows the welds on the faces it looks at - a rule a fabricator would recognise.

THE ARTICLE'S OWN FRAME. Length is the principal axis of the mesh. Width and depth are NOT the other
two principal axes: on a square section they are degenerate, and on the tower (116 x 116 mm) they
came out 23 degrees off the rails, which made every side face tilted and put almost no weld on one.
They come from the minimum-area rectangle of the cross-section instead, which lies along the rails.

FACE CODES. +L / -L are the ends, +W / -W and +D / -D the four long faces, all in the MODEL frame so
they describe the article whatever way up it is lying. Which of them is "top" is a question about a
placement, answered by `face_name` from the pose.

A WELD'S FACES. Every face whose plane the weld's centroid lies within ``band`` of - so a corner weld
belongs to two faces and is shown by either. Members have depth: on the tower the welds sit on the
rails' inner faces, 16 to 20 mm in from the envelope. The default band is 20% of the smaller
cross-section extent (23 mm there), which put all 64 tower welds on at least one face.

PRESENTED. A camera is presented a face when it is on the outer side of that face by more than a
grazing margin: cos(outward normal, direction to the camera) above ``min_cos``. On the tower both
cameras score the top at 0.66 and the same long side at 0.53, and the ends only at 0.06 and 0.14 -
grazing, so excluded by the 0.2 default.

VISIBLE. On a presented face a weld can still be blocked. Occlusion compares against the FARTHEST
depth in a 3x3 neighbourhood, not the nearest: welds sit in concave corners, where every neighbouring
pixel belongs to a face coming towards the camera, and at a grazing view one pixel spans more than
the tolerance - so a nearest-depth test hid visible corners, worse with distance from the camera.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services import visibility as VIS  # noqa: E402

FACES = ("+L", "-L", "+W", "-W", "+D", "-D")
_AXIS = {"L": 0, "W": 1, "D": 2}


def _positive(a):
    """Sign an axis so its largest component is positive: the same article always gets the same
    frame, rather than one that flips between runs."""
    return a if a[int(np.argmax(np.abs(a)))] >= 0 else -a


def _section_axes(L, pts, c):
    """Axes with ``L`` as length and the cross axes along the sides of the tightest rectangle round
    the section - the rails, not the principal axes, which are arbitrary on a square section."""
    L = np.asarray(L, np.float64) / np.linalg.norm(L)
    ref = np.array([1.0, 0.0, 0.0]) if abs(L[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(L, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(L, e1)
    flat = np.stack([(pts - c) @ e1, (pts - c) @ e2], axis=1).astype(np.float32)
    _ctr, _size, angle = cv2.minAreaRect(flat)
    a = np.radians(angle)
    W = np.cos(a) * e1 + np.sin(a) * e2
    return np.stack([L, W, np.cross(L, W)], axis=1)


def article_frame(mesh):
    """The article's own axes (columns L, W, D, in model coordinates), centre and extents.

    The length axis is NOT taken from raw vertices. A mesh is densest where it is curved - holes,
    fillets - so a vertex average pulls the axis towards wherever the tessellation happens to be
    fine, rather than along the steel. Triangles are weighted by their area instead, which measures
    the part's surface and not its mesh.

    Even area-weighted principal axes can sit a little off a box's sides, and on a square section
    the cross axes are arbitrary. So the frame is chosen as the TIGHTEST box among a few candidates:
    the principal axis and each model axis as the length, each with its section squared up to the
    sides. CAD parts are usually modelled square to their axes, and a tilted candidate always loses
    on volume. The longest side of the winning box is the length."""
    tris = np.asarray(mesh, np.float64).reshape(-1, 3, 3)
    pts = tris.reshape(-1, 3)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    centres = tris.mean(axis=1)
    total = float(area.sum()) or 1.0
    c = (centres * area[:, None]).sum(axis=0) / total
    d = centres - c
    evals, evecs = np.linalg.eigh((d * area[:, None]).T @ d / total)

    best = None
    for L0 in (evecs[:, int(np.argmax(evals))], np.eye(3)[:, 0], np.eye(3)[:, 1], np.eye(3)[:, 2]):
        axes = _section_axes(L0, pts, c)
        q = (pts - c) @ axes
        vol = float(np.prod(q.max(axis=0) - q.min(axis=0)))
        if best is None or vol < best[0] * (1.0 - 1e-9):
            best = (vol, axes)

    axes = best[1]
    q = (pts - c) @ axes
    order = np.argsort(-(q.max(axis=0) - q.min(axis=0)), kind="stable")
    L = _positive(axes[:, order[0]])
    W = _positive(axes[:, order[1]])
    axes = np.stack([L, W, np.cross(L, W)], axis=1)          # right-handed, whatever W's sign
    q = (pts - c) @ axes
    return {"centre": c, "axes": axes, "lo": q.min(axis=0), "hi": q.max(axis=0)}


def default_band(frame, fraction=0.2):
    """How far in from a face plane a weld may sit and still be on that face."""
    ext = frame["hi"] - frame["lo"]
    return float(fraction * min(ext[1], ext[2]))


def faces_of(points_model, frame, band):
    """The faces a weld is on, from its centroid's distance to each face plane, in FACES order."""
    x = ((np.asarray(points_model, np.float64).reshape(-1, 3) - frame["centre"])
         @ frame["axes"]).mean(axis=0)
    lo, hi = frame["lo"], frame["hi"]
    dist = {"+L": hi[0] - x[0], "-L": x[0] - lo[0], "+W": hi[1] - x[1], "-W": x[1] - lo[1],
            "+D": hi[2] - x[2], "-D": x[2] - lo[2]}
    return [k for k in FACES if dist[k] <= band]


def faces_for_welds(welds, frame, scale, band):
    """``{weld name: [face codes]}`` - from the sidecar where extract recorded them, else computed
    here from the same frame. Returns the mapping and how many had to be computed."""
    out, computed = {}, 0
    for w in welds:
        codes = (w.get("PSS_WeldGeometry") or {}).get("ArticleFaces")
        if codes is None:
            pts = np.vstack([np.asarray(s, np.float64).reshape(-1, 3)
                             for s in w["Representation"]["segments"]]) * scale
            codes = faces_of(pts, frame, band)
            computed += 1
        out[w["Name"]] = list(codes)
    return out, computed


def frame_to_json(frame, band, mesh_name, scale):
    """What the sidecar records, so a reader can see exactly how the face codes were assigned."""
    ext = frame["hi"] - frame["lo"]
    return {
        "basis": "length = principal axis of the mesh; width and depth = the minimum-area rectangle "
                 "of the cross-section, so they lie along the members",
        "mesh": mesh_name,
        "scale_applied_to_welds": scale,
        "axes_in_model": {k: [round(float(v), 4) for v in frame["axes"][:, i]]
                          for k, i in _AXIS.items()},
        "centre_mm": [round(float(v), 2) for v in frame["centre"]],
        "extent_mm": {k: round(float(ext[i]), 1) for k, i in _AXIS.items()},
        "face_band_mm": round(float(band), 1),
        "codes": {"+L": "end, positive length", "-L": "end, negative length",
                  "+W": "long face, positive width", "-W": "long face, negative width",
                  "+D": "long face, positive depth", "-D": "long face, negative depth"},
    }


def _normal(frame, code):
    return frame["axes"][:, _AXIS[code[1]]] * (1.0 if code[0] == "+" else -1.0)


def presented_faces(frame, rvec, tvec, view, min_cos=0.2):
    """The faces this camera is on the outer side of, by more than a grazing margin.

    Returns ``(set of codes, {code: cosine})`` so a caller can show how close a call was."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t = np.asarray(tvec, np.float64).ravel()
    Rc, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    eye = (-Rc.T @ np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)).ravel()
    mid = (frame["hi"] + frame["lo"]) / 2.0
    cos = {}
    for code in FACES:
        i = _AXIS[code[1]]
        local = mid.copy()
        local[i] = frame["hi"][i] if code[0] == "+" else frame["lo"][i]
        centre = R @ (frame["centre"] + frame["axes"] @ local) + t
        n = R @ _normal(frame, code)
        to_cam = eye - centre
        cos[code] = float(n @ to_cam / max(np.linalg.norm(to_cam), 1e-9))
    return {c for c in FACES if cos[c] > min_cos}, cos


def face_name(frame, rvec, code):
    """A face's name for THIS placement. The board frame has the part at negative Z, so up is -Z."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    z = float((R @ _normal(frame, code))[2])
    if z < -0.7:
        return "top"
    if z > 0.7:
        return "bottom"
    return ("end " if code[1] == "L" else "side ") + code


def visible_weld_points(placed, faces_by_weld, mesh, rvec, tvec, view, frame, min_cos=0.2,
                        occlusion_mm=8.0):
    """The welds this camera should show, as image polylines.

    Returns ``(lines, status, presented, cosines)``: ``lines`` is ``{name: [[(x, y), ...]]}``,
    ``status`` gives every weld one of 'shown', 'face not presented', 'hidden' (blocked or out of
    frame) or 'no face'. A polyline breaks wherever a point is hidden, so a partly blocked run is
    drawn as the pieces that can be seen rather than bridged across what cannot."""
    presented, cosines = presented_faces(frame, rvec, tvec, view, min_cos)
    rvec = np.asarray(rvec, np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)
    depth, _ = VIS.depth_buffer(mesh, rvec, tvec, view, downscale=1)
    farthest = cv2.dilate(depth, np.ones((3, 3), np.uint8))
    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    dist = np.asarray(view["dist"], np.float64)
    Rc, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    tc = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    h, w = depth.shape
    lines_out, status = {}, {}
    for name, worlds, _row in placed:
        codes = faces_by_weld.get(name) or []
        if not codes:
            status[name] = "no face"
            continue
        if not presented.intersection(codes):
            status[name] = "face not presented"
            continue
        lines = []
        for world in worlds:
            p2, _ = cv2.projectPoints(world.reshape(-1, 1, 3), view["rvec_cam"], view["tvec_cam"],
                                      K, dist)
            cz = (Rc @ world.T + tc)[2]
            cur = []
            for (x, y), z in zip(p2.reshape(-1, 2), cz):
                xi, yi = int(round(x)), int(round(y))
                seen = (0 <= xi < w and 0 <= yi < h and (z - farthest[yi, xi]) < occlusion_mm)
                if seen:
                    cur.append((float(x), float(y)))
                elif cur:
                    lines.append(cur)
                    cur = []
            if cur:
                lines.append(cur)
        if lines:
            lines_out[name] = lines
            status[name] = "shown"
        else:
            status[name] = "hidden"
    return lines_out, status, presented, cosines
