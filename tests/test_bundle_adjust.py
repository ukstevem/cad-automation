"""
Bundle adjustment should beat naive spanning-tree chaining when marker poses are
noisy (small/distant tags) but observations are redundant — by using ALL co-visibility
jointly. Compared via a gauge-invariant relative pose T(3→0).
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from app.services import constellation as C
from app.services.pose import rt_compose, rt_invert


def _R(rv):
    return cv2.Rodrigues(np.array(rv, float).reshape(3, 1))[0]


K = np.array([[2500.0, 0, 600.0], [0, 2500.0, 450.0], [0, 0, 1.0]])
DIST = np.zeros((5, 1))
SIZE = 80.0
OBJP = C._marker_objp(SIZE)

TRUTH = {
    0: (np.eye(3), np.zeros((3, 1))),
    1: (_R([0.05, 0.10, 0]), np.array([[400.0], [0], [20]])),
    2: (_R([0, -0.10, 0.05]), np.array([[800.0], [30], [-10]])),
    3: (_R([0.10, 0, 0]), np.array([[1200.0], [0], [40]])),
}


def _world(m):
    R, t = TRUTH[m]
    return (R @ OBJP.T + t).T


def _look_at(cam, tgt, up=(0, 1, 0.0)):
    cam = np.asarray(cam, float)
    z = np.asarray(tgt, float) - cam; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 0)
    return cv2.Rodrigues(R)[0], (-R @ cam).reshape(3, 1)


def _rel(poses, a, b):                      # T(a→b), gauge-invariant for fixed a,b
    return rt_compose(rt_invert(*poses[b]), poses[a])


def test_bundle_adjust_beats_spanning_tree():
    cams = [_look_at([200, -200, -1600], [600, 0, 0]),
            _look_at([100, -150, -1400], [300, 0, 0]),
            _look_at([700, -150, -1400], [900, 0, 0]),
            _look_at([400, -250, -1500], [600, 0, 0])]
    vis = [[0, 1, 2, 3], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3]]
    rng = np.random.default_rng(0)

    photo_markers, photo_corners = [], []
    for (rv, tv), mids in zip(cams, vis):
        pm, pc = {}, {}
        for m in mids:
            proj, _ = cv2.projectPoints(_world(m).reshape(-1, 1, 3), rv, tv, K, DIST)
            c = proj.reshape(-1, 2) + rng.normal(0, 1.0, (4, 2))      # 1 px corner noise
            pc[m] = c
            ok, r2, t2 = cv2.solvePnP(OBJP, c, K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            pm[m] = (cv2.Rodrigues(r2)[0], t2)
        photo_markers.append(pm)
        photo_corners.append(pc)

    con = C.build_constellation(photo_markers)
    init = {int(m): (np.array(p["R"]), np.array(p["t"]).reshape(3, 1)) for m, p in con["poses"].items()}
    ref = con["reference"]

    truth_30_t = _rel(TRUTH, 3, 0)[1].ravel()
    st_err = np.linalg.norm(_rel(init, 3, 0)[1].ravel() - truth_30_t)

    refined, info = C.bundle_adjust(photo_corners, init, ref, SIZE, K, DIST)
    ba_err = np.linalg.norm(_rel(refined, 3, 0)[1].ravel() - truth_30_t)

    # BA should reduce both the marker-pose error and the reprojection RMS.
    assert ba_err < st_err, f"BA {ba_err:.1f}mm did not beat spanning-tree {st_err:.1f}mm"
    assert info["rms_after_px"] <= info["rms_before_px"] + 1e-6
