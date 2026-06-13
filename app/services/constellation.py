"""
Marker pose-graph ("constellation") calibration.

On a long fabrication no single photo gives a clean multi-marker registration —
wide shots see the whole part but only one marker decodes; close shots see two
markers but not enough of the part to click correspondences. The fix:

1. Detect markers across MANY photos. Each photo where two markers are both visible
   pins their *relative* pose (no CAD needed).
2. Chain those relative poses through co-visibility into one rigid "constellation"
   frame (this module).
3. One manual solve on a wide shot ties the constellation to the CAD model — because
   all markers are locked together, anchoring one anchors them all.

Builds a spanning tree over the co-visibility graph; the result is each marker's
pose in a chosen reference marker's frame.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services.pose import rt_compose, rt_invert

try:
    import cv2
    _CV2_ERR = None
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = str(exc)


def build_constellation(photo_markers: List[Dict[int, Tuple]]) -> Optional[dict]:
    """
    *photo_markers*: one dict per photo, ``{marker_id: (R 3x3, t 3x1)}`` giving each
    marker's pose in that photo's CAMERA frame (marker→camera, e.g. from solvePnP).

    Returns ``{"reference": ref_id, "poses": {mid: {"R": 3x3, "t": [3]}}, "linked":
    [ids], "unlinked": [ids]}`` where each pose is marker→reference (constellation)
    frame, or None if no markers were seen.
    """
    appears: Dict[int, int] = defaultdict(int)
    # edges[i][j] = T(j→i): pose of marker j expressed in marker i's frame.
    edges: Dict[int, Dict[int, Tuple]] = defaultdict(dict)

    for pm in photo_markers:
        ids = list(pm.keys())
        for mid in ids:
            appears[mid] += 1
        for i in ids:
            Ti = pm[i]
            inv_i = rt_invert(*Ti)
            for j in ids:
                if i == j:
                    continue
                # T(j→i) = T(cam→i) ∘ T(j→cam)
                edges[i][j] = rt_compose(inv_i, pm[j])

    if not appears:
        return None

    # Reference = most-seen marker (best connected anchor for the spanning tree).
    ref = max(appears, key=lambda k: appears[k])

    poses: Dict[int, Tuple] = {ref: (np.eye(3), np.zeros((3, 1)))}
    q = deque([ref])
    while q:
        cur = q.popleft()
        for nb, T_nb_cur in edges[cur].items():       # T(nb→cur)
            if nb in poses:
                continue
            # T(nb→ref) = T(cur→ref) ∘ T(nb→cur)
            poses[nb] = rt_compose(poses[cur], T_nb_cur)
            q.append(nb)

    all_ids = set(appears)
    linked = set(poses)
    return {
        "reference": int(ref),
        "poses": {
            int(m): {"R": np.asarray(R).tolist(), "t": np.asarray(t).reshape(3).tolist()}
            for m, (R, t) in poses.items()
        },
        "linked": sorted(int(x) for x in linked),
        "unlinked": sorted(int(x) for x in (all_ids - linked)),
    }


def _marker_objp(marker_size_mm: float) -> np.ndarray:
    h = marker_size_mm / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def bundle_adjust(photo_corners: List[Dict[int, np.ndarray]], init_poses: Dict[int, Tuple],
                  reference: int, marker_size_mm: float, K, dist):
    """
    Refine the constellation by jointly optimising ALL marker poses (in the reference
    frame, reference fixed at identity) and ALL camera poses against every marker-corner
    observation — minimising total reprojection error with a robust loss.

    Fixes the spanning-tree's accumulated chaining error by using direct co-visibility
    constraints. *photo_corners*: per photo ``{marker_id: 4x2 image corners}``.
    *init_poses*: spanning-tree ``{marker_id: (R 3x3, t 3x1)}`` (marker→ref). Returns
    ``(refined_poses, info)`` with before/after RMS.
    """
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(f"OpenCV unavailable: {_CV2_ERR}")
    from scipy.optimize import least_squares

    K = np.asarray(K, float).reshape(3, 3)
    dist = np.asarray(dist, float).reshape(-1, 1)
    objp = _marker_objp(marker_size_mm)

    opt = [m for m in sorted(init_poses) if m != reference]
    m_off = {m: 6 * i for i, m in enumerate(opt)}
    n_m = len(opt)

    def world_corners(rt):
        R = np.asarray(rt[0], float).reshape(3, 3)
        t = np.asarray(rt[1], float).reshape(3, 1)
        return (R @ objp.T + t).T

    # Per-photo camera-pose init via solvePnP from the spanning-tree marker poses.
    photos = []   # (obs=[(mid, corners)],)
    cam0 = []
    for pc in photo_corners:
        obs = [(m, np.asarray(c, float)) for m, c in pc.items() if m in init_poses]
        if len(obs) < 1:
            continue
        op = np.vstack([world_corners(init_poses[m]) for m, _ in obs])
        ip = np.vstack([c for _, c in obs])
        ok, rv, tv = cv2.solvePnP(op.reshape(-1, 1, 3), ip.reshape(-1, 1, 2), K, dist,
                                  flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        photos.append(obs)
        cam0.append((rv.ravel(), tv.ravel()))

    n_p = len(photos)
    cam_base = 6 * n_m
    x0 = np.zeros(6 * n_m + 6 * n_p)
    for m in opt:
        o = m_off[m]
        x0[o:o + 3] = cv2.Rodrigues(np.asarray(init_poses[m][0], float).reshape(3, 3))[0].ravel()
        x0[o + 3:o + 6] = np.asarray(init_poses[m][1], float).ravel()
    for j, (rv, tv) in enumerate(cam0):
        o = cam_base + 6 * j
        x0[o:o + 3] = rv
        x0[o + 3:o + 6] = tv

    def m_pose(m, x):
        if m == reference:
            return np.zeros(3), np.zeros(3)
        o = m_off[m]
        return x[o:o + 3], x[o + 3:o + 6]

    def residuals(x):
        res = []
        for j, obs in enumerate(photos):
            o = cam_base + 6 * j
            crv, ctv = x[o:o + 3], x[o + 3:o + 6]
            for m, corners in obs:
                mrv, mtv = m_pose(m, x)
                world = (cv2.Rodrigues(mrv)[0] @ objp.T + mtv.reshape(3, 1)).T
                proj, _ = cv2.projectPoints(world.reshape(-1, 1, 3), crv, ctv, K, dist)
                res.append((proj.reshape(-1, 2) - corners).ravel())
        return np.concatenate(res) if res else np.zeros(0)

    rms_before = float(np.sqrt(np.mean(residuals(x0) ** 2)))
    sol = least_squares(residuals, x0, method="trf", loss="huber", f_scale=5.0, max_nfev=400)
    rms_after = float(np.sqrt(np.mean(residuals(sol.x) ** 2)))

    refined = {int(reference): (np.eye(3), np.zeros((3, 1)))}
    for m in opt:
        mrv, mtv = m_pose(m, sol.x)
        refined[int(m)] = (cv2.Rodrigues(mrv)[0], np.asarray(mtv).reshape(3, 1))
    return refined, {"rms_before_px": round(rms_before, 2), "rms_after_px": round(rms_after, 2),
                     "photos": n_p, "markers_optimised": n_m}


def anchor_to_model(constellation: dict, anchor_marker_id: int,
                    R_marker_cam, t_marker_cam, rvec_model, tvec_model) -> Dict[int, dict]:
    """
    Tie a calibrated constellation to the CAD model using one anchor photo where the
    full part was clicked (model pose) and an anchor marker is visible.

    Returns ``{marker_id: {R_wm, t_wm, ...}}`` — each constellation marker's pose in
    the MODEL/world frame (same shape as a manual registration), so the existing
    auto-solve can fuse any visible subset.
    """
    import cv2

    R2, _ = cv2.Rodrigues(np.asarray(rvec_model, np.float64).reshape(3, 1))
    t2 = np.asarray(tvec_model, np.float64).reshape(3, 1)

    # T(anchor marker → world) from the anchor photo.
    T_K_world = rt_compose(rt_invert(R2, t2), (R_marker_cam, t_marker_cam))

    poses = constellation["poses"]
    if str(anchor_marker_id) in poses:
        Kp = poses[str(anchor_marker_id)]
    elif anchor_marker_id in poses:
        Kp = poses[anchor_marker_id]
    else:
        raise ValueError(f"Anchor marker {anchor_marker_id} is not in the constellation")
    T_K_C = (np.asarray(Kp["R"]), np.asarray(Kp["t"]))

    # T(constellation → world) = T(K→world) ∘ T(C→K)
    T_C_world = rt_compose(T_K_world, rt_invert(*T_K_C))

    out: Dict[int, dict] = {}
    for mid, mp in poses.items():
        T_M_C = (np.asarray(mp["R"]), np.asarray(mp["t"]))
        R_wm, t_wm = rt_compose(T_C_world, T_M_C)     # T(marker → world)
        out[int(mid)] = {
            "R_wm": np.asarray(R_wm).tolist(),
            "t_wm": np.asarray(t_wm).reshape(3).tolist(),
        }
    return out
