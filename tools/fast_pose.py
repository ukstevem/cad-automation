#!/usr/bin/env python3
"""
Fast pose search for placement guidance (bd 6et) - orientation locked, position searched.

The accurate scorer (pose_refine.score) renders a full-resolution depth image and projects every model
edge separately for each camera and each candidate pose: one to two seconds a pose. A search over a few
hundred poses takes minutes, which rules out live guidance. Measured on the tower (2026-09-17) the fine
refiner also has a narrow basin - it came back from 0 of 24 starts only 12-44 mm off - so position has to
come from a SEARCH, and the search has to be cheap.

What makes it cheap is that almost nothing changes between candidate poses:

- per photograph, once: undistort, find edges, and build distance maps to them in orientation bins, so a
  model edge only matches photographed edges running the same way;
- per target, once: the model's visible edge points and their image direction, taken from the accurate
  visibility code at the target pose. Over a few centimetres and degrees the set of visible edges barely
  changes, so it is reused for every candidate;
- per candidate pose: move those points, project them with a pinhole model, and read the distance maps.

Orientation is NOT searched. The score cannot tell this tower end-for-end or a quarter roll apart (bd q1h,
6et), so which face is down and which end is where come from the operator; only position and a small turn
are searched here.

Measured on tower09 (2026-09-17, both cameras, stereo rig): edge maps 0.5 s for both photographs, edge
points 0.4 s per target, search 2.1 s (441 grid poses in 1.1 s, then the fine stage) - against minutes for
the same search on the accurate score. From the placement target and from the 24 random starts it lands on
one of TWO poses the accurate score also rates equally (60/56% and 61/60%): the clicked pose (5 of 24
starts, 0.8-1.7 mm) or one 7 mm across and 2.25 deg turned from it (most of the rest). Camera A confirms
86% of lines at the first and 70% at the second, camera B the reverse - the cameras disagree about the
part, by 14 mm sideways to both their rays at the open end, with the stereo rig and without it. That is an
accuracy finding, not a search defect; see bd 6et.
"""
from __future__ import annotations

import time

import numpy as np

import cv2

from critical_edges import visible_feature_edges

N_BINS = 6                      # 30-degree bins over the undirected 180 degrees of an edge line


def _rot(v):
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def _vec(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def photo_edges(image, blur=1.2, contrast=3.0, floor=8.0, window_px=41):
    """Edge pixels by LOCAL contrast, with the gradient each was found from.

    One global threshold is the wrong rule for these photographs, and it failed on the first real one
    (tower09): set from the whole frame, it is decided by the white paper and the black-and-white board,
    and the dark part on the dark mat falls under it - 5% of the model's edge points had a photo edge
    within 4 px at the confirmed pose. The accurate check (line_check.search_along_normals) accepts a line
    that is ``contrast`` times its local background and above an absolute ``floor``; this is the same rule,
    applied through Canny by dividing the gradient by its local mean before thinning."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    g = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), blur)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    local = cv2.blur(cv2.magnitude(gx, gy), (window_px, window_px))
    # below floor / contrast the background is too quiet to judge against, and the floor decides instead
    scale = 10.0 / np.maximum(local, floor / contrast)
    nx = np.clip(gx * scale, -32767, 32767).astype(np.int16)
    ny = np.clip(gy * scale, -32767, 32767).astype(np.int16)
    hi = 10.0 * contrast
    return cv2.Canny(nx, ny, 0.67 * hi, hi, L2gradient=True), gx, gy


def edge_maps(view, cap_px=40.0):
    """Distance maps to the photograph's edges, one per orientation bin, on the UNDISTORTED image."""
    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    und = cv2.undistort(view["image"], K, np.asarray(view["dist"], np.float64))
    edges, gx, gy = photo_edges(und)
    line_deg = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
    bins = (line_deg // (180.0 / N_BINS)).astype(np.int16) % N_BINS
    h, w = edges.shape
    dt = np.empty((N_BINS, h, w), np.float32)
    for b in range(N_BINS):
        src = np.where((edges > 0) & (bins == b), 0, 255).astype(np.uint8)
        dt[b] = np.minimum(cv2.distanceTransform(src, cv2.DIST_L2, 3), cap_px)
    Rc = _rot(view["rvec_cam"])
    return {"dt": dt, "K": K, "Rc": Rc, "tc": np.asarray(view["tvec_cam"], np.float64).ravel(), "w": w, "h": h,
            "edge_px": int((edges > 0).sum())}


def prepare_samples(mesh, rvec, tvec, views, step_px=3.0):
    """The model's visible edge points at a pose, in MODEL coordinates, with their image direction bin."""
    R0, t0 = _rot(rvec), np.asarray(tvec, np.float64).ravel()
    out = []
    for v in views:
        _pts, tan, _z, world = visible_feature_edges(mesh, np.asarray(rvec, np.float64).reshape(3, 1),
                                                      t0.reshape(3, 1), v, step_px=step_px, with_world=True)
        model = (R0.T @ (world - t0).T).T
        deg = np.degrees(np.arctan2(tan[:, 1], tan[:, 0])) % 180.0
        out.append({"model": model, "bin": (deg // (180.0 / N_BINS)).astype(np.int64) % N_BINS})
    return out


def _bilinear(img, x, y):
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def chamfer(R, t, samples, maps, cap_px):
    """Mean distance, truncated at ``cap_px``, from the model's edge points to same-direction photo edges."""
    total, n = 0.0, 0
    for s, m in zip(samples, maps):
        c = (s["model"] @ R.T + t) @ m["Rc"].T + m["tc"]
        z = c[:, 2]
        K = m["K"]
        u = K[0, 0] * c[:, 0] / z + K[0, 2]
        v = K[1, 1] * c[:, 1] / z + K[1, 2]
        inside = (z > 1.0) & (u >= 0) & (v >= 0) & (u < m["w"] - 1) & (v < m["h"] - 1)
        d = np.full(len(z), cap_px, np.float64)
        if inside.any():
            b = s["bin"][inside]
            ui, vi = u[inside], v[inside]
            best = np.full(len(b), cap_px)
            for shift in (-1, 0, 1):
                bb = (b + shift) % N_BINS
                for k in range(N_BINS):
                    sel = bb == k
                    if sel.any():
                        best[sel] = np.minimum(best[sel], _bilinear(m["dt"][k], ui[sel], vi[sel]))
            d[inside] = np.minimum(best, cap_px)
        total += float(d.sum())
        n += len(d)
    return total / max(n, 1)


class LockedPose:
    """Poses near a base pose that keep its orientation: a slide along and across the part's length on the
    table, a turn about the vertical through the part's centre, and a height change."""

    def __init__(self, rvec, tvec, model_centre, model_length_axis):
        self.R, self.t = _rot(rvec), np.asarray(tvec, np.float64).ravel()
        self.c_w = self.R @ np.asarray(model_centre, np.float64) + self.t
        L = self.R @ np.asarray(model_length_axis, np.float64)
        L[2] = 0.0
        self.L = L / np.linalg.norm(L)
        self.A = np.array([-self.L[1], self.L[0], 0.0])

    def at(self, along, across, yaw_deg, dz=0.0):
        Rz = _rot([0.0, 0.0, np.radians(yaw_deg)])
        return Rz @ self.R, Rz @ (self.t - self.c_w) + self.c_w + along * self.L + across * self.A + np.array([0.0, 0.0, dz])


def search(base, samples, maps, along=40.0, across=30.0, yaw=6.0, dz=10.0, coarse=(10.0, 10.0, 2.0),
           cap_coarse=25.0, cap_fine=8.0, starts=3):
    """Coarse grid, then Nelder-Mead from the best few. Returns (R, t, params, chamfer_px, timings)."""
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    grid = []
    for a in np.arange(-along, along + 1e-9, coarse[0]):
        for c in np.arange(-across, across + 1e-9, coarse[1]):
            for y in np.arange(-yaw, yaw + 1e-9, coarse[2]):
                R, t = base.at(a, c, y)
                grid.append((chamfer(R, t, samples, maps, cap_coarse), a, c, y))
    grid.sort(key=lambda g: g[0])
    t1 = time.perf_counter()
    best = None
    for _score, a, c, y in grid[:starts]:
        f = lambda p: chamfer(*base.at(p[0], p[1], p[2], p[3]), samples, maps, cap_fine)
        simplex = np.array([[a, c, y, 0.0], [a + 5, c, y, 0.0], [a, c + 5, y, 0.0], [a, c, y + 1.0, 0.0], [a, c, y, 4.0]])
        # bounded to the searched range: past it the answer is "not in the envelope", not a pose
        res = minimize(f, simplex[0], method="Nelder-Mead",
                       bounds=[(-along, along), (-across, across), (-yaw, yaw), (-dz, dz)],
                       options={"initial_simplex": np.clip(simplex, [-along, -across, -yaw, -dz], [along, across, yaw, dz]),
                                "xatol": 0.05, "fatol": 1e-4, "maxfev": 600})
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x)
    t2 = time.perf_counter()
    R, t = base.at(*best[1])
    return R, t, best[1], best[0], {"grid_s": t1 - t0, "grid_poses": len(grid), "fine_s": t2 - t1}
