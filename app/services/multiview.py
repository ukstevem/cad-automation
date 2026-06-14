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

    def __init__(self, K, dist, rvec_cam, tvec_cam, edge_pixels):
        from scipy.spatial import cKDTree

        self.K = np.asarray(K, np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, np.float64).reshape(-1, 1)
        self.R_cam, _ = cv2.Rodrigues(np.asarray(rvec_cam, np.float64).reshape(3, 1))
        self.t_cam = np.asarray(tvec_cam, np.float64).reshape(3, 1)
        ep = np.asarray(edge_pixels, np.float64).reshape(-1, 2)
        if len(ep) == 0:
            raise MultiViewError("a view has no detected edge pixels")
        self.tree = cKDTree(ep)

    def residuals(self, rvec_obj, tvec_obj, obj_pts):
        """NN pixel distance for each projected CAD sample point in this view."""
        rvec_oc, t_oc = _compose_object_to_cam(rvec_obj, tvec_obj, self.R_cam, self.t_cam)
        proj, _ = cv2.projectPoints(obj_pts.reshape(-1, 1, 3), rvec_oc, t_oc, self.K, self.dist)
        proj = proj.reshape(-1, 2)
        d, _ = self.tree.query(proj)
        return d


def fit_object_pose(
    views: List[dict],
    cad_edges: List[Sequence],
    init_rvec,
    init_tvec,
    *,
    max_step: float = 2.0,
    huber_delta: float = 10.0,
    max_nfev: int = 400,
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
    obj_pts = sample_polylines(cad_edges, max_step=max_step)
    vs = [_View(v["K"], v["dist"], v["rvec_cam"], v["tvec_cam"], v["edge_pixels"]) for v in views]

    def residuals(p):
        rvec_obj, tvec_obj = p[:3], p[3:]
        return np.concatenate([v.residuals(rvec_obj, tvec_obj, obj_pts) for v in vs])

    p0 = np.concatenate([
        np.asarray(init_rvec, np.float64).reshape(3),
        np.asarray(init_tvec, np.float64).reshape(3),
    ])
    rms_before = float(np.sqrt(np.mean(residuals(p0) ** 2)))
    sol = least_squares(residuals, p0, loss="huber", f_scale=huber_delta, max_nfev=max_nfev)
    rvec_obj = sol.x[:3].reshape(3, 1)
    tvec_obj = sol.x[3:].reshape(3, 1)

    per_view = [float(np.sqrt(np.mean(v.residuals(rvec_obj, tvec_obj, obj_pts) ** 2))) for v in vs]
    info = {
        "rms_before_px": round(rms_before, 3),
        "rms_after_px": round(float(np.sqrt(np.mean(sol.fun ** 2))), 3),
        "per_view_rms_px": [round(x, 3) for x in per_view],
        "n_points": int(len(obj_pts)),
        "n_views": len(vs),
        "success": bool(sol.success),
    }
    return rvec_obj, tvec_obj, info
