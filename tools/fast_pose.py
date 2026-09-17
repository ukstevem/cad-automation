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
6et), so which face is down and which end is where come from the operator. Searched: a slide along and
across, a turn about the vertical, and - in the fine stage - height and a small TILT.

The tilt is not optional. Every pose so far was "seated": resting on a stored hull face, flat on the board's
plane. On tower09 (2026-09-17) no seated pose satisfied both cameras - camera A confirmed 86% of lines at one
pose and camera B 87% at another 2.25 deg away, which read as the cameras disagreeing. They do not: with the
board raised onto the tower they agree to 2-4 mm and 0.4 deg. The stored rests were upside down
(resting_faces.py, fixed the same day), pitching the tower 2.19 deg the wrong way, and even the corrected hull
rest is not how the real part lay - the photographs say level. With height and tilt free, five of six starts
reach one pose, 88% / 89% confirmed by the two cameras, accurate score 94 / 76% against 60 / 56% seated; on
tower01-04 silhouette 47-62% -> 70-85%. A real part sits on dunnage, a mat or a fixture, not on its convex
hull, so the rest is a starting point and the photographs decide the last few degrees.

Timing on tower09, both cameras: edge maps 0.5 s for both photographs, edge points 0.4 s per target, search
about 2 s.
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
    """Distance maps to the photograph's edges on the UNDISTORTED image, one per orientation bin. Each map
    holds the distance to an edge in its own bin OR either neighbouring bin, so an edge 20 deg off still
    matches and a crossing edge does not."""
    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    und = cv2.undistort(view["image"], K, np.asarray(view["dist"], np.float64))
    edges, gx, gy = photo_edges(und)
    line_deg = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
    bins = (line_deg // (180.0 / N_BINS)).astype(np.int16) % N_BINS
    h, w = edges.shape
    own = np.empty((N_BINS, h, w), np.float32)
    for b in range(N_BINS):
        src = np.where((edges > 0) & (bins == b), 0, 255).astype(np.uint8)
        own[b] = np.minimum(cv2.distanceTransform(src, cv2.DIST_L2, 3), cap_px)
    dt = np.minimum(np.minimum(own, np.roll(own, 1, axis=0)), np.roll(own, -1, axis=0))
    return {"dt": dt, "K": K, "Rc": _rot(view["rvec_cam"]), "tc": np.asarray(view["tvec_cam"], np.float64).ravel(),
            "w": w, "h": h, "edge_px": int((edges > 0).sum())}


def prepare_samples(mesh, rvec, tvec, views, step_px=3.0):
    """The model's visible edge points at a pose, in MODEL coordinates, grouped by image direction bin."""
    R0, t0 = _rot(rvec), np.asarray(tvec, np.float64).ravel()
    out = []
    for v in views:
        _pts, tan, _z, world = visible_feature_edges(mesh, np.asarray(rvec, np.float64).reshape(3, 1),
                                                      t0.reshape(3, 1), v, step_px=step_px, with_world=True)
        model = (R0.T @ (world - t0).T).T
        deg = np.degrees(np.arctan2(tan[:, 1], tan[:, 0])) % 180.0
        b = (deg // (180.0 / N_BINS)).astype(np.int64) % N_BINS
        order = np.argsort(b, kind="stable")
        out.append({"model": model[order], "bin": b[order],
                    "start": np.searchsorted(b[order], np.arange(N_BINS + 1))})
    return out


def subset(samples, keep):
    """The same samples restricted by a boolean mask per view (e.g. one part of the model)."""
    out = []
    for s, k in zip(samples, keep):
        m, b = s["model"][k], s["bin"][k]
        out.append({"model": m, "bin": b, "start": np.searchsorted(b, np.arange(N_BINS + 1))})
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
        for k in range(N_BINS):
            a, b = s["start"][k], s["start"][k + 1]
            sel = np.nonzero(inside[a:b])[0] + a
            if len(sel):
                d[sel] = np.minimum(_bilinear(m["dt"][k], u[sel], v[sel]), cap_px)
        total += float(d.sum())
        n += len(d)
    return total / max(n, 1)


class LockedPose:
    """Poses near a base pose that keep its orientation up to small changes: a slide along and across the part's
    length on the table, a turn about the vertical, a height change, and a tilt about the length and about the
    across direction - all about the part's centre."""

    def __init__(self, rvec, tvec, model_centre, model_length_axis):
        self.R, self.t = _rot(rvec), np.asarray(tvec, np.float64).ravel()
        self.c_w = self.R @ np.asarray(model_centre, np.float64) + self.t
        L = self.R @ np.asarray(model_length_axis, np.float64)
        L[2] = 0.0
        self.L = L / np.linalg.norm(L)
        self.A = np.array([-self.L[1], self.L[0], 0.0])

    def at(self, along, across, yaw_deg, dz=0.0, tilt_length_deg=0.0, tilt_across_deg=0.0):
        Rm = (_rot(np.radians(tilt_length_deg) * self.L) @ _rot(np.radians(tilt_across_deg) * self.A)
              @ _rot([0.0, 0.0, np.radians(yaw_deg)]))
        c_new = self.c_w + along * self.L + across * self.A + np.array([0.0, 0.0, dz])
        return Rm @ self.R, Rm @ (self.t - self.c_w) + c_new


def search(base, samples, maps, along=40.0, across=30.0, yaw=6.0, dz=15.0, tilt=3.0, coarse=(10.0, 10.0, 2.0),
           cap_coarse=25.0, cap_fine=8.0, starts=3):
    """Coarse grid over slide and turn, a 4-parameter polish (adding height) from the best few, then all six
    parameters (adding tilt) from the best of those. Returns (R, t, params, chamfer_px, timings)."""
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    grid = []
    for a in np.arange(-along, along + 1e-9, coarse[0]):
        for c in np.arange(-across, across + 1e-9, coarse[1]):
            for y in np.arange(-yaw, yaw + 1e-9, coarse[2]):
                grid.append((chamfer(*base.at(a, c, y), samples, maps, cap_coarse), a, c, y))
    grid.sort(key=lambda g: g[0])
    t1 = time.perf_counter()

    lo = np.array([-along, -across, -yaw, -dz, -tilt, -tilt])
    hi = -lo
    f = lambda p: chamfer(*base.at(*p), samples, maps, cap_fine)

    def polish(x0, n, steps, maxfev):
        simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(n)])
        # bounded to the searched range: past it the answer is "not in the envelope", not a pose
        return minimize(f if n == 6 else (lambda p: f(np.concatenate([p, [0.0, 0.0]]))), x0, method="Nelder-Mead",
                        bounds=list(zip(lo[:n], hi[:n])),
                        options={"initial_simplex": np.clip(simplex, lo[:n], hi[:n]), "xatol": 0.05, "fatol": 1e-4,
                                 "maxfev": maxfev})

    best = None
    for _score, a, c, y in grid[:starts]:
        r = polish(np.array([a, c, y, 0.0]), 4, (5.0, 5.0, 1.0, 4.0), 300)
        if best is None or r.fun < best.fun:
            best = r
    t2 = time.perf_counter()
    r6 = polish(np.concatenate([best.x, [0.0, 0.0]]), 6, (2.0, 2.0, 0.5, 2.0, 0.7, 0.7), 1200)
    t3 = time.perf_counter()
    R, t = base.at(*r6.x)
    return R, t, r6.x, r6.fun, {"grid_s": t1 - t0, "grid_poses": len(grid), "fine_s": t2 - t1, "tilt_s": t3 - t2,
                                "evals": len(grid) + r6.nfev}


def locate(mesh, views, maps, rvec, tvec, model_centre, model_length_axis, cap_px=8.0, **search_kw):
    """Search from a start pose, then take the model's visible edges again AT the answer and polish once more.

    The edge points come from the start pose, and a start 30 mm and a few degrees out sees a slightly different
    set of edges from the true pose. A slide or a turn is pinned hard enough not to care; roll about a long
    member's length is not, and absorbs the difference - 1.75 deg of invented roll, 2.1 mm, on the synthetic
    bar. Re-taking the points at the answer removes the bias."""
    from scipy.optimize import minimize

    samples = prepare_samples(mesh, rvec, tvec, views)
    base = LockedPose(rvec, tvec, model_centre, model_length_axis)
    R, t, _p, _c, timing = search(base, samples, maps, cap_fine=cap_px, **search_kw)
    t0 = time.perf_counter()
    samples = prepare_samples(mesh, _vec(R), t, views)
    near = LockedPose(_vec(R), t, model_centre, model_length_axis)
    f = lambda p: chamfer(*near.at(*p), samples, maps, cap_px)
    x0 = np.zeros(6)
    lim = np.array([5.0, 5.0, 1.0, 5.0, 1.0, 1.0])
    r = minimize(f, x0, method="Nelder-Mead", bounds=list(zip(-lim, lim)),
                 options={"initial_simplex": np.vstack([x0, np.diag([1.0, 1.0, 0.25, 1.0, 0.25, 0.25])]),
                          "xatol": 0.02, "fatol": 1e-5, "maxfev": 1200})
    R, t = near.at(*r.x)
    timing["again_s"] = time.perf_counter() - t0
    return R, t, r.fun, timing
