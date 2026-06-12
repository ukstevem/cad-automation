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
