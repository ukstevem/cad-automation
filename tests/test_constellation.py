"""
Verify the marker-SLAM constellation: link markers from close (2-marker) shots, then
anchor the whole constellation to the model from one wide shot — recovering every
marker's model-frame pose exactly (synthetic, noise-free).
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.services import constellation as C
from app.services.pose import rt_compose


def _R(rv):
    return cv2.Rodrigues(np.array(rv, float).reshape(3, 1))[0]


# True marker→world (model) poses.
MARKERS = {
    0: (_R([0, 0, 0]), np.array([[0.0], [0], [0]])),
    1: (_R([0.1, 0.2, 0.0]), np.array([[300.0], [0], [50]])),
    2: (_R([0.0, -0.15, 0.1]), np.array([[600.0], [40], [-20]])),
}
# Camera poses (world→cam).
CAMS = {
    "A": (_R([0.2, -0.3, 0.05]), np.array([[-200.0], [100], [800]])),   # sees 0,1
    "B": (_R([0.1, 0.4, -0.1]), np.array([[-450.0], [120], [820]])),    # sees 1,2
    "C": (_R([0.0, 0.1, 0.0]), np.array([[-300.0], [110], [1500]])),    # anchor, sees all
}


def _marker_in_cam(cam, m):
    return rt_compose(CAMS[cam], MARKERS[m])           # marker→cam


def test_constellation_links_and_anchors_exactly():
    # Close shots: each links a co-visible pair (no full part needed).
    visibility = {"A": [0, 1], "B": [1, 2]}
    photo_markers = []
    for cam, mids in visibility.items():
        photo_markers.append({m: _marker_in_cam(cam, m) for m in mids})

    con = C.build_constellation(photo_markers)
    assert con is not None
    assert set(con["linked"]) == {0, 1, 2}            # 0-1 and 1-2 chain into one frame
    assert con["unlinked"] == []

    # Anchor with the wide shot C (its world→cam IS the model pose), marker 0 visible.
    Rwc, twc = CAMS["C"]
    rvec_model = cv2.Rodrigues(Rwc)[0]
    R_m0_cam, t_m0_cam = _marker_in_cam("C", 0)
    regs = C.anchor_to_model(con, 0, R_m0_cam, t_m0_cam, rvec_model, twc)

    # Every marker's recovered model pose matches truth.
    for m in (0, 1, 2):
        R_wm = np.array(regs[m]["R_wm"])
        t_wm = np.array(regs[m]["t_wm"])
        R_true, t_true = MARKERS[m]
        assert np.allclose(R_wm, R_true, atol=1e-6), f"marker {m} rotation"
        assert np.allclose(t_wm.reshape(3), t_true.reshape(3), atol=1e-3), f"marker {m} translation"


def test_unlinked_markers_reported():
    # A marker seen only alone (no co-visibility) can't be linked.
    photo_markers = [{0: _marker_in_cam("A", 0), 1: _marker_in_cam("A", 1)},
                     {2: _marker_in_cam("B", 2)}]
    con = C.build_constellation(photo_markers)
    assert 2 in con["unlinked"]
    assert set(con["linked"]) == {0, 1}
