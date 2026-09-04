"""
Multi-view object-pose fitting — the AR robotic-cell pose spine (ADR 0001).

Given several **calibrated** views (each: intrinsics + a KNOWN world→camera pose, from a
shared ChArUco board or pre-calibrated rig extrinsics) and the 2D edge pixels detected in
each, recover the 6DoF pose of a **known** CAD object by minimising the multi-view
edge-reprojection distance. We already have the CAD, so this is *registration of a known
model*, not reconstruction — far more forgiving than 3D scanning.

A CAD point ``X`` (object frame) maps to view ``v`` by::

    X_world = R_obj·X + t_obj                 # object pose — what we solve for
    X_cam   = R_cam_v·X_world + t_cam_v       # known calibrated view pose
    x_pix   = project(X_cam, K_v, dist_v)

Cost = distance from each projected CAD-edge sample to the nearest detected edge pixel in
that view (nearest-neighbour via KD-tree), summed over all views, optimised with a robust
loss (``scipy.least_squares``, Huber). Fusing views disambiguates poses that are degenerate
from a single viewpoint (e.g. a near-symmetric section) — the exact failure that sank
single-image markerless. See ``app/services/pose.py`` for the single-view projection core.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # import guard — keep module importable if a wheel is missing
    import cv2

    _CV2_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = str(exc)


class MultiViewError(Exception):
    """Raised when a multi-view fit cannot be performed (bad inputs, no edges)."""


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise MultiViewError(f"OpenCV is not available: {_CV2_IMPORT_ERROR}")


def sample_polylines(polylines: List[Sequence], max_step: float = 5.0) -> np.ndarray:
    """
    Densify a list of 3D polylines into an Nx3 point cloud, inserting points so no gap
    exceeds *max_step* (mm). Even sampling keeps the edge cost from being dominated by
    a few long segments. Returns object-frame points.
    """
    out: List[np.ndarray] = []
    for pl in polylines:
        pts = np.asarray(pl, dtype=np.float64).reshape(-1, 3)
        if len(pts) < 2:
            if len(pts) == 1:
                out.append(pts)
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            seg = b - a
            n = max(1, int(np.ceil(np.linalg.norm(seg) / max_step)))
            for k in range(n):
                out.append((a + seg * (k / n))[None, :])
        out.append(pts[-1][None, :])
    if not out:
        raise MultiViewError("no polyline points to sample")
    return np.vstack(out)


def _compose_object_to_cam(rvec_obj, tvec_obj, R_cam, t_cam):
    """object→camera = (R_cam·R_obj, R_cam·t_obj + t_cam) → return (rvec_oc, tvec_oc)."""
    R_obj, _ = cv2.Rodrigues(np.asarray(rvec_obj, np.float64).reshape(3, 1))
    t_obj = np.asarray(tvec_obj, np.float64).reshape(3, 1)
    R_oc = R_cam @ R_obj
    t_oc = R_cam @ t_obj + t_cam
    rvec_oc, _ = cv2.Rodrigues(R_oc)
    return rvec_oc, t_oc


class _View:
    """A calibrated view + a KD-tree over its detected edge pixels for NN distance."""

    def __init__(self, K, dist, rvec_cam, tvec_cam, edge_pixels, width=None, height=None,
                 reverse_weight=0.0, reverse_max=3000, reverse_cap=40.0, seed=0,
                 point_mask=None):
        from scipy.spatial import cKDTree

        self.K = np.asarray(K, np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, np.float64).reshape(-1, 1)
        self.R_cam, _ = cv2.Rodrigues(np.asarray(rvec_cam, np.float64).reshape(3, 1))
        self.t_cam = np.asarray(tvec_cam, np.float64).reshape(3, 1)
        ep = np.asarray(edge_pixels, np.float64).reshape(-1, 2)
        if len(ep) == 0:
            raise MultiViewError("a view has no detected edge pixels")
        self.tree = cKDTree(ep)
        # Fall back to the observed edge extent when the frame size is not supplied; only used
        # for the visibility diagnostic, never for the cost itself.
        self.width = float(width) if width else float(ep[:, 0].max() + 1)
        self.height = float(height) if height else float(ep[:, 1].max() + 1)

        # Subsample the observed edges for the reverse term. Every one of them carries the same
        # message ("explain me"), so a few thousand is plenty and keeps the residual vector -
        # and therefore the Jacobian - a sensible size.
        # Which of the shared CAD samples this view can actually see. Visibility is per-view
        # and pose-dependent, so it is supplied by the caller's outer loop rather than
        # recomputed inside the optimiser.
        self.point_mask = None if point_mask is None else np.asarray(point_mask, bool)
        self.reverse_weight = float(reverse_weight)
        self.reverse_cap = float(reverse_cap)
        self.rev_pixels = None
        if reverse_weight > 0 and len(ep):
            idx = np.arange(len(ep))
            if len(ep) > reverse_max:
                idx = np.random.default_rng(seed).choice(len(ep), reverse_max, replace=False)
            pts = np.round(ep[np.sort(idx)]).astype(np.int64)
            inb = ((pts[:, 0] >= 0) & (pts[:, 0] < int(self.width)) &
                   (pts[:, 1] >= 0) & (pts[:, 1] < int(self.height)))
            self.rev_pixels = pts[inb]

    def points(self, obj_pts):
        if self.point_mask is None or len(self.point_mask) != len(obj_pts):
            return obj_pts
        sub = obj_pts[self.point_mask]
        return sub if len(sub) else obj_pts

    def project(self, rvec_obj, tvec_obj, obj_pts):
        rvec_oc, t_oc = _compose_object_to_cam(rvec_obj, tvec_obj, self.R_cam, self.t_cam)
        pts = self.points(obj_pts)
        proj, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec_oc, t_oc, self.K, self.dist)
        return proj.reshape(-1, 2)

    def residuals(self, rvec_obj, tvec_obj, obj_pts):
        """
        Symmetric edge distance for this view.

        FORWARD: for each projected CAD point, the distance to the nearest detected edge.
        REVERSE: for each detected edge pixel, the distance to the nearest projected CAD point.

        The forward term alone is not enough, and on a latticed object it is barely a constraint
        at all. It only asks "is each CAD point near an edge?", and a wireframe lattice laid over
        a real lattice finds *a* neighbour almost anywhere — measured on this rig, the cost moved
        only ~2.5 px across a 2x change in model scale. Nothing penalised a pose that covered the
        part while leaving the actual edges unexplained.

        The reverse term supplies exactly that missing question. It is computed by rasterising
        the projected model and distance-transforming it, so it costs one transform per
        evaluation rather than a KD-tree rebuild over ~50k points.

        DEFAULT OFF, on the evidence. It did not help: on real captures it moved the cost's
        sensitivity to model scale from 5.8 px to 5.0 px across a 2x range - i.e. nothing - and
        it made the four-view synthetic case *worse* (translation error 6.6 mm against a 5 mm
        bar it used to clear). The reason is that the reverse question is just as easy to
        satisfy as the forward one while the projection still contains every edge on the far
        side of the object. Revisit once visible-edge extraction lands and the projected model
        is only what a camera could actually see; the formulation is right, its input is not.
        """
        proj = self.project(rvec_obj, tvec_obj, obj_pts)
        forward, _ = self.tree.query(proj)
        if self.reverse_weight <= 0 or self.rev_pixels is None:
            return forward

        h, w = int(self.height), int(self.width)
        canvas = np.zeros((h, w), np.uint8)
        p = np.round(proj).astype(np.int64)
        ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        if not np.any(ok):
            # Model entirely off-frame: every observed edge is unexplained.
            reverse = np.full(len(self.rev_pixels), float(self.reverse_cap))
        else:
            canvas[p[ok, 1], p[ok, 0]] = 255
            dt = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 3)
            xi = self.rev_pixels[:, 0].astype(np.int32)
            yi = self.rev_pixels[:, 1].astype(np.int32)
            reverse = np.minimum(dt[yi, xi], self.reverse_cap)
        return np.concatenate([forward, reverse * self.reverse_weight])


def _visible_fraction(views, obj_pts, rvec_obj, tvec_obj) -> float:
    """Fraction of projected CAD points that land inside any view's image bounds."""
    inside = 0
    total = 0
    for v in views:
        rvec_oc, t_oc = _compose_object_to_cam(rvec_obj, tvec_obj, v.R_cam, v.t_cam)
        proj, _ = cv2.projectPoints(obj_pts.reshape(-1, 1, 3), rvec_oc, t_oc, v.K, v.dist)
        p = proj.reshape(-1, 2)
        w, h = v.width, v.height
        inside += int(np.count_nonzero((p[:, 0] >= 0) & (p[:, 0] < w) &
                                       (p[:, 1] >= 0) & (p[:, 1] < h)))
        total += len(p)
    return float(inside) / float(total) if total else 0.0


def fit_object_pose_planar(
    views: List[dict],
    cad_edges: List[Sequence],
    init_rvec,
    init_tvec,
    *,
    max_step: float = 2.0,
    huber_delta: float = 10.0,
    max_nfev: int = 200,
    xy_bounds: Optional[Tuple[Sequence, Sequence]] = None,
    reverse_weight: float = 0.0,
    obj_points: Optional[np.ndarray] = None,
    visibility: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit only ``(x, y, yaw)``, holding the object on the board plane at the initial height.

    The part rests on the same surface as the board, so its out-of-plane rotation and its height
    are *known*, not unknown. Solving all six degrees of freedom lets the optimiser trade a real
    misfit for an unphysical tilt, and on this rig it did exactly that every time: with six DOF
    the height ran to its upper bound on every single run, putting the part through the table,
    while the rotation reached ~1.4 rad. Constraining to the plane removes the escape routes and
    shrinks the search to the three parameters that are genuinely free.

    Yaw is taken about the board normal, applied on top of *init_rvec*.
    """
    _require_cv2()
    from scipy.optimize import least_squares

    if not views:
        raise MultiViewError("need at least one view")
    obj_pts = sample_polylines(cad_edges, max_step=max_step) if obj_points is None else obj_points
    vs = [_View(v["K"], v["dist"], v["rvec_cam"], v["tvec_cam"], v["edge_pixels"],
                v.get("width"), v.get("height"), reverse_weight=reverse_weight,
                point_mask=(visibility[i] if visibility else None))
          for i, v in enumerate(views)]

    R_base, _ = cv2.Rodrigues(np.asarray(init_rvec, np.float64).reshape(3, 1))
    t_init = np.asarray(init_tvec, np.float64).reshape(3)
    z_fixed = float(t_init[2])

    def pose_from(p):
        Rz, _ = cv2.Rodrigues(np.array([[0.0], [0.0], [float(p[2])]]))
        R_obj = Rz @ R_base
        rvec, _ = cv2.Rodrigues(R_obj)
        return rvec, np.array([[p[0]], [p[1]], [z_fixed]])

    def residuals(p):
        rvec, tvec = pose_from(p)
        return np.concatenate([v.residuals(rvec, tvec, obj_pts) for v in vs])

    p0 = np.array([t_init[0], t_init[1], 0.0])
    rms_before = float(np.sqrt(np.mean(residuals(p0) ** 2)))
    kwargs = {}
    if xy_bounds is not None:
        lo = np.array([xy_bounds[0][0], xy_bounds[0][1], -np.pi])
        hi = np.array([xy_bounds[1][0], xy_bounds[1][1], np.pi])
        p0 = np.clip(p0, lo, hi)
        kwargs["bounds"] = (lo, hi)
    sol = least_squares(residuals, p0, loss="huber", f_scale=huber_delta, max_nfev=max_nfev,
                        **kwargs)
    rvec_obj, tvec_obj = pose_from(sol.x)
    per_view = [float(np.sqrt(np.mean(v.residuals(rvec_obj, tvec_obj, obj_pts) ** 2))) for v in vs]
    info = {
        "mode": "planar",
        "visible_fraction": round(_visible_fraction(vs, obj_pts, rvec_obj, tvec_obj), 4),
        "rms_before_px": round(rms_before, 3),
        "rms_after_px": round(float(np.sqrt(np.mean(sol.fun ** 2))), 3),
        "per_view_rms_px": [round(x, 3) for x in per_view],
        "n_points": int(len(obj_pts)),
        "n_views": len(vs),
        # TOTAL yaw, not the increment this solve applied. sol.x[2] is measured relative to
        # init_rvec, so when the init came from the coarse scan it reads as a couple of degrees
        # while the object is actually rotated 90 - which is exactly the number a UI must not
        # show. Both rotations are about the board normal, so the composed rvec's Z is the total.
        "yaw_deg": round(float(np.degrees(np.asarray(rvec_obj).reshape(3)[2])), 2),
        "yaw_increment_deg": round(float(np.degrees(sol.x[2])), 2),
        "success": bool(sol.success),
    }
    return rvec_obj, tvec_obj, info


def fit_object_pose(
    views: List[dict],
    cad_edges: List[Sequence],
    init_rvec,
    init_tvec,
    *,
    max_step: float = 2.0,
    huber_delta: float = 10.0,
    max_nfev: int = 400,
    tvec_bounds: Optional[Tuple[Sequence, Sequence]] = None,
    reverse_weight: float = 0.0,
    obj_points: Optional[np.ndarray] = None,
    visibility: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit a known CAD object's 6DoF pose across several calibrated views by minimising the
    edge-reprojection NN distance.

    *views*: list of ``{K, dist, rvec_cam, tvec_cam, edge_pixels}`` — each view's intrinsics,
    its KNOWN world→camera pose, and the Mx2 detected edge pixels in that view.
    *cad_edges*: CAD edge polylines in the OBJECT frame. *init_rvec/init_tvec*: starting
    object→world pose (e.g. the nominal jig placement).

    Returns ``(rvec_obj, tvec_obj, info)`` where info carries per-view RMS + cost.
    """
    _require_cv2()
    from scipy.optimize import least_squares

    if not views:
        raise MultiViewError("need at least one view")
    obj_pts = sample_polylines(cad_edges, max_step=max_step) if obj_points is None else obj_points
    vs = [_View(v["K"], v["dist"], v["rvec_cam"], v["tvec_cam"], v["edge_pixels"],
                v.get("width"), v.get("height"), reverse_weight=reverse_weight,
                point_mask=(visibility[i] if visibility else None))
          for i, v in enumerate(views)]

    def residuals(p):
        rvec_obj, tvec_obj = p[:3], p[3:]
        return np.concatenate([v.residuals(rvec_obj, tvec_obj, obj_pts) for v in vs])

    p0 = np.concatenate([
        np.asarray(init_rvec, np.float64).reshape(3),
        np.asarray(init_tvec, np.float64).reshape(3),
    ])
    rms_before = float(np.sqrt(np.mean(residuals(p0) ** 2)))

    # Bound the translation, or the cost has a degenerate global minimum: push the object far
    # enough away and it projects into a handful of pixels, every one of which lands near SOME
    # detected edge, so the mean NN distance collapses. Observed on the first real capture —
    # the solver "improved" 318 -> 5.2 px by moving a 433 mm part to 4.4 metres and shrinking
    # it to a dot. The residual is genuinely low; the pose is nonsense.
    kwargs = {}
    if tvec_bounds is not None:
        lo = np.concatenate([np.full(3, -np.inf), np.asarray(tvec_bounds[0], np.float64)])
        hi = np.concatenate([np.full(3, np.inf), np.asarray(tvec_bounds[1], np.float64)])
        p0 = np.clip(p0, lo, hi)                       # least_squares requires p0 within bounds
        kwargs["bounds"] = (lo, hi)
    sol = least_squares(residuals, p0, loss="huber", f_scale=huber_delta, max_nfev=max_nfev,
                        **kwargs)
    rvec_obj = sol.x[:3].reshape(3, 1)
    tvec_obj = sol.x[3:].reshape(3, 1)

    per_view = [float(np.sqrt(np.mean(v.residuals(rvec_obj, tvec_obj, obj_pts) ** 2))) for v in vs]
    visible = _visible_fraction(vs, obj_pts, rvec_obj, tvec_obj)
    info = {
        "visible_fraction": round(visible, 4),
        "rms_before_px": round(rms_before, 3),
        "rms_after_px": round(float(np.sqrt(np.mean(sol.fun ** 2))), 3),
        "per_view_rms_px": [round(x, 3) for x in per_view],
        "n_points": int(len(obj_pts)),
        "n_views": len(vs),
        "success": bool(sol.success),
    }
    # A collapsed pose scores a beautiful RMS while projecting almost nothing into frame, so
    # report it rather than leaving a low residual to speak for itself.
    if visible < 0.2:
        info["degenerate"] = (
            f"only {visible * 100:.1f}% of CAD points project inside the images - the fit has "
            f"collapsed away from the cameras. The low RMS is meaningless; constrain "
            f"tvec_bounds or improve the initial pose."
        )
    return rvec_obj, tvec_obj, info
