"""
Tests for app/services/stereo_rig.py - the two rig cameras as one instrument (bd ghn).

Synthetic cameras with a known transform between them. The point of the module is that a capture gets
ONE board pose that both views agree on, so these pin: the transform is recovered, a pair shot while
the board moved is thrown out, the joint pose is right and both views are consistent with it, and a
rig solved against one lens calibration is refused against another.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services import stereo_rig as SR  # noqa: E402

K = np.array([[1440.0, 0, 950], [0, 1442.0, 540], [0, 0, 1]])
D = np.zeros((5, 1))
R_AB = cv2.Rodrigues(np.array([0.05, -0.62, 0.10]))[0]          # about 36 degrees, mostly about y
T_AB = np.array([520.0, 20.0, 230.0])
GRID = np.array([[x * 40.0, y * 40.0, 0.0] for y in range(5) for x in range(8)])
SIZE = (1920, 1080)


def _board_pose(i):
    """A spread of board poses about 700 mm in front of camera A, tilted differently each time."""
    rng = np.random.default_rng(i)
    rv = np.array([np.pi, 0, 0]) + rng.uniform(-0.45, 0.45, 3)
    tv = np.array([-140 + rng.uniform(-120, 120), -80 + rng.uniform(-80, 80), 700 + rng.uniform(-60, 60)])
    return rv, tv


def _observe(rv, tv, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    ia, _ = cv2.projectPoints(GRID, rv, tv, K, D)
    rb, tb = SR.camera_b_pose(rv, tv, R_AB, T_AB)
    ib, _ = cv2.projectPoints(GRID, rb, tb, K, D)
    return (ia.reshape(-1, 2) + rng.normal(0, noise, (len(GRID), 2)),
            ib.reshape(-1, 2) + rng.normal(0, noise, (len(GRID), 2)))


def test_the_transform_between_the_cameras_is_recovered():
    pairs = []
    for i in range(12):
        ia, ib = _observe(*_board_pose(i), seed=i)
        pairs.append(("p%02d" % i, GRID, ia, ib))
    res = SR.calibrate(pairs, K, D, K, D, SIZE)
    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(res["R"] @ R_AB.T)[0]))
    assert angle < 0.05
    assert np.linalg.norm(res["T"] - T_AB) < 1.0
    assert res["rms_px"] < 0.3


def test_a_pair_shot_while_the_board_moved_is_dropped():
    """The rig exposes its cameras a second or two apart; a board nudged in between fits no transform."""
    pairs = []
    for i in range(12):
        rv, tv = _board_pose(i)
        ia, ib = _observe(rv, tv, seed=i)
        if i == 5:
            _ia2, ib = _observe(rv, tv + np.array([6.0, 0, 0]), seed=i)
        pairs.append(("p%02d" % i, GRID, ia, ib))
    res = SR.calibrate(pairs, K, D, K, D, SIZE)
    assert "p05" not in res["pairs_used"]
    assert np.linalg.norm(res["T"] - T_AB) < 1.0


def test_one_board_pose_from_both_cameras_is_right_and_both_views_agree_with_it():
    rv, tv = _board_pose(40)
    ia, ib = _observe(rv, tv, noise=0.2, seed=40)
    ra, ta, rms_a, rms_b = SR.joint_board_pose((GRID, ia), (GRID, ib), K, D, K, D, R_AB, T_AB)
    assert np.linalg.norm(ta - tv) < 1.0
    assert rms_a < 0.4 and rms_b < 0.4
    rb, tb = SR.camera_b_pose(ra, ta, R_AB, T_AB)
    board_in_a = (SR._rot(ra) @ GRID.T).T + ta
    board_in_b = (SR._rot(rb) @ GRID.T).T + tb
    assert np.allclose((R_AB @ board_in_a.T).T + T_AB, board_in_b, atol=1e-6)


def test_views_get_the_joint_pose_and_keep_their_own_for_diagnostics():
    rv, tv = _board_pose(41)
    ia, ib = _observe(rv, tv, seed=41)
    # each camera's own estimate carries its own error - 2 mm apart, like the rig
    va = {"tag": "x_AAAA_1.png", "K": K, "dist": D, "rvec_cam": rv, "tvec_cam": tv + [1.0, 0, 0]}
    rb, tb = SR.camera_b_pose(rv, tv, R_AB, T_AB)
    vb = {"tag": "x_BBBB_1.png", "K": K, "dist": D, "rvec_cam": rb, "tvec_cam": tb - [1.0, 0, 0]}
    rig = {"cameras": {"A": "AAAA", "B": "BBBB"}, "R": R_AB, "T": T_AB}
    info = SR.apply_observations(va, vb, (GRID, ia), (GRID, ib), rig)
    assert info["applied"]
    assert np.allclose(np.ravel(va["tvec_cam_independent"]), tv + [1.0, 0, 0])
    board_a = (SR._rot(va["rvec_cam"]) @ GRID.T).T + np.ravel(va["tvec_cam"])
    board_b = (SR._rot(vb["rvec_cam"]) @ GRID.T).T + np.ravel(vb["tvec_cam"])
    assert np.allclose((R_AB @ board_a.T).T + T_AB, board_b, atol=1e-6)
    assert np.linalg.norm(np.ravel(va["tvec_cam"]) - tv) < 1.0


def test_a_rig_is_refused_against_a_different_lens_calibration():
    rig = {"cameras": {"A": "AAAA", "B": "BBBB"}, "path": "RigStereo_AAAA_BBBB.json",
           "intrinsics": {"A": {"fx": 1440.0, "fy": 1442.0, "cx": 950.0, "cy": 540.0}}}
    SR.check_intrinsics(rig, "A", K)
    stale = K.copy()
    stale[0, 0] = 1415.6
    with pytest.raises(ValueError):
        SR.check_intrinsics(rig, "A", stale)


def test_a_capture_without_one_photo_per_camera_is_left_alone():
    rig = {"cameras": {"A": "AAAA", "B": "BBBB"}}
    views = [{"tag": "x_AAAA_1.png"}, {"tag": "y_AAAA_2.png"}]
    info = SR.apply(views, rig, {})
    assert not info["applied"]
