"""
Headless verification of the multi-view object-pose fit (app/services/multiview.py) —
the robotic-cell pose spine (ADR 0001).

Strategy (mirrors test_pose.py): build a synthetic CAD wireframe, place virtual
*calibrated* cameras at known world→camera poses, render the object's edges at a known
TRUE object pose to get the "detected" edge pixels per view, then check that
fit_object_pose() recovers the true object pose. The headline test proves the whole point
of going multi-view: two views resolve a near-planar tilt ambiguity that a single frontal
view cannot.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from app.services import multiview as MV


# ── Intrinsics (shared 4000x3000 camera, no distortion) ─────────────────────
K = np.array([[3000.0, 0, 2000.0], [0, 3000.0, 1500.0], [0, 0, 1.0]])
DIST = np.zeros((5, 1))


def _look_at(cam_pos, target, world_up=(0.0, 0.0, 1.0)):
    """OpenCV-convention world→camera pose (x right, y down, z forward)."""
    cam_pos = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    up = np.asarray(world_up, float)
    if abs(np.dot(z, up)) > 0.95:          # avoid degenerate up
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = (-R @ cam_pos).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t


def _rt(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return R, np.asarray(tvec, float).reshape(3, 1)


def _detected_edges(cad_edges, true_rvec, true_tvec, rvec_cam, tvec_cam, step=3.0):
    """Project CAD edges at the TRUE object pose into a view → Nx2 'detected' pixels."""
    pts = MV.sample_polylines(cad_edges, max_step=step)              # object frame
    R_obj, t_obj = _rt(true_rvec, true_tvec)
    world = (R_obj @ pts.T + t_obj).T                                # object → world
    R_cam, t_cam = _rt(rvec_cam, tvec_cam)
    rvec_wc, _ = cv2.Rodrigues(R_cam)
    proj, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rvec_wc, t_cam, K, DIST)
    return proj.reshape(-1, 2)


def _pose_err(rv_a, tv_a, rv_b, tv_b):
    Ra, ta = _rt(rv_a, tv_a)
    Rb, tb = _rt(rv_b, tv_b)
    ang = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
    return ang, float(np.linalg.norm(ta - tb))


# ── A distinctive, asymmetric "L-extrusion" wireframe (object frame, mm) ─────
def _l_section_edges():
    # L cross-section in XY, extruded along Z. Asymmetric → no rotational symmetry.
    prof = np.array([[0, 0], [80, 0], [80, 20], [20, 20], [20, 60], [0, 60]], float)
    z0, z1 = -100.0, 100.0
    prof = prof - prof.mean(axis=0)                                  # centre it
    edges = []
    for z in (z0, z1):                                              # two end loops
        loop = [[x, y, z] for x, y in prof] + [[prof[0, 0], prof[0, 1], z]]
        edges.append(np.array(loop, float))
    for x, y in prof:                                              # connecting edges
        edges.append(np.array([[x, y, z0], [x, y, z1]], float))
    return edges


# ── Tests ───────────────────────────────────────────────────────────────────

# Four cameras ringing the part (the planned rig). Edges sampled fine (1 mm) so the
# synthetic 'detected' pixel sets are dense — mirrors a real edge map's density.
_TRUE_RVEC = np.array([0.15, -0.2, 0.1]).reshape(3, 1)
_TRUE_TVEC = np.array([30.0, -10.0, 600.0]).reshape(3, 1)
_CAMS = [
    _look_at([400, -300, 500], [0, 0, 0]),
    _look_at([-450, 250, 550], [0, 0, 0]),
    _look_at([450, 400, 400], [0, 0, 0]),
    _look_at([-400, -350, 600], [0, 0, 0]),
]


def _views(cams, edges):
    return [{"K": K, "dist": DIST, "rvec_cam": rc, "tvec_cam": tc,
             "edge_pixels": _detected_edges(edges, _TRUE_RVEC, _TRUE_TVEC, rc, tc, step=1.0)}
            for rc, tc in cams]


def test_four_views_recover_pose_from_jig_nominal_init():
    """
    Four calibrated views recover the object pose from a tight init (≈3°/~12 mm — the kind
    of start a jig-nominal placement gives in a real cell). Headline de-risk for the rig.
    """
    edges = _l_section_edges()
    views = _views(_CAMS, edges)

    rvec0 = _TRUE_RVEC + np.array([0.035, -0.025, 0.03]).reshape(3, 1)   # ~3 deg off
    tvec0 = _TRUE_TVEC + np.array([10.0, -8.0, 12.0]).reshape(3, 1)

    rv, tv, info = MV.fit_object_pose(views, edges, rvec0, tvec0)
    ang, dist = _pose_err(rv, tv, _TRUE_RVEC, _TRUE_TVEC)

    assert info["rms_after_px"] < info["rms_before_px"]   # the fit improved things
    assert ang < 1.0                                       # sub-degree rotation
    assert dist < 5.0                                      # few-mm translation on a 0.6 m shot
    assert len(info["per_view_rms_px"]) == 4


def test_multiview_beats_single_view():
    """
    The justification for the rig: from the SAME init, a single view's edge-fit lands at a
    wrong pose (the chamfer cost is riddled with flipped minima), while four views lock on.
    This is the core reason single-image markerless failed and multi-view is the spine.
    """
    edges = _l_section_edges()
    rvec0 = _TRUE_RVEC + np.array([0.035, -0.025, 0.03]).reshape(3, 1)
    tvec0 = _TRUE_TVEC + np.array([10.0, -8.0, 12.0]).reshape(3, 1)

    rv1, tv1, _ = MV.fit_object_pose(_views(_CAMS[:1], edges), edges, rvec0, tvec0)
    rv4, tv4, _ = MV.fit_object_pose(_views(_CAMS, edges), edges, rvec0, tvec0)

    ang1, _ = _pose_err(rv1, tv1, _TRUE_RVEC, _TRUE_TVEC)
    ang4, _ = _pose_err(rv4, tv4, _TRUE_RVEC, _TRUE_TVEC)

    assert ang1 > 3.0          # one view drifts to a wrong pose...
    assert ang4 < 1.0          # ...four views recover truth
    assert ang4 < ang1 - 2.0
