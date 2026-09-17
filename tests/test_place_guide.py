"""
Tests for the placement guide's geometry (tools/place_guide.py, bd 6et): what it tells the operator has to be
right in the operator's world - which way to turn, seen from above the table - not just in the board's frame.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

import fast_pose as FP  # noqa: E402
import place_guide as PG  # noqa: E402
from test_fast_pose import PART, _camera  # noqa: E402
from weld_faces import article_frame  # noqa: E402


def test_a_positive_turn_about_board_z_looks_clockwise_from_above():
    """Board z points INTO the table. Photograph the table from straight above and follow +x turning toward +y:
    on the picture that sweep must go clockwise, or every turn instruction is backwards."""
    # above the table is negative z; a hair off vertical, because the helper builds its image axes from the vertical
    view = _camera((0.0, -1.0, -1000.0), (0.0, 0.0, 0.0))
    K = np.asarray(view["K"], float)
    pts = np.array([[0.0, 0, 0], [100.0, 0, 0], [0, 100.0, 0]])
    uv, _ = cv2.projectPoints(pts, np.asarray(view["rvec_cam"], float), np.asarray(view["tvec_cam"], float), K, None)
    o, x, y = uv.reshape(-1, 2)
    a, b = x - o, y - o
    # image v runs DOWN, so a clockwise sweep from a to b has a positive cross product in image coordinates
    assert a[0] * b[1] - a[1] * b[0] > 0
    assert "clockwise" in PG.turn_words(10.0) and "anticlockwise" not in PG.turn_words(10.0)
    assert "anticlockwise" in PG.turn_words(-10.0)


def test_the_target_lies_on_the_board_along_the_axis_with_the_master_end_where_asked():
    fr = article_frame(PART)
    end, in_master = PG.master_region(PART, fr)
    assert end == "lo"                                        # the plate is on the -x end of the synthetic part
    for sign in (1, -1):
        R, t = PG.target_pose(PART, fr, [0.0, 0.0, 0.0], sign, (120.0, -40.0), 30.0)
        P = PART.reshape(-1, 3) @ R.T + t
        assert np.isclose(P[:, 2].max(), 0.0)                 # resting on the board plane (z down)
        c = R @ fr["centre"] + t
        assert np.allclose(c[:2], (120.0, -40.0))
        L = R @ fr["axes"][:, 0]
        axis = np.array([np.cos(np.radians(30.0)), np.sin(np.radians(30.0)), 0.0])
        assert np.isclose(L @ axis, sign, atol=1e-9)


def test_a_target_carried_through_a_fixed_camera_lands_where_that_camera_saw_it():
    """The board moved between the reference shot and this one; the camera did not."""
    home = _camera((-500.0, -700.0, -650.0), (0.0, 0.0, 0.0))
    # the same camera, described from a board that has since been turned 25 deg and slid 80 mm
    Rb, tb = FP._rot([0.0, 0.0, np.radians(25.0)]), np.array([80.0, -30.0, 0.0])   # x_home = Rb x_live + tb
    Rh, th = FP._rot(home["rvec_cam"]), np.asarray(home["tvec_cam"], float)
    live = {"rvec_cam": FP._vec(Rh @ Rb), "tvec_cam": Rh @ tb + th}
    M, m = PG.home_to_capture(home, live)
    x_home = np.array([150.0, 60.0, -20.0])
    x_live = M @ x_home + m
    assert np.allclose(Rb @ x_live + tb, x_home)


def test_turning_end_for_end_keeps_the_centre_and_reverses_the_length():
    fr = article_frame(PART)
    R, t = PG.target_pose(PART, fr, [0.0, 0.0, 0.0], 1, (0.0, 0.0), 0.0)
    R2, t2 = PG.turned_end_for_end(R, t, fr["centre"])
    assert np.allclose(R2 @ fr["centre"] + t2, R @ fr["centre"] + t)
    assert np.isclose((R2 @ fr["axes"][:, 0]) @ (R @ fr["axes"][:, 0]), -1.0)
    assert np.isclose(abs(PG.yaw_between(R, R2, fr["axes"][:, 0])), 180.0)


def test_the_zone_test_uses_the_whole_outline_not_the_centre():
    fr = article_frame(PART)
    R, t = PG.target_pose(PART, fr, [0.0, 0.0, 0.0], 1, (0.0, 0.0), 0.0)       # 420 mm long along x
    big = [[-300, -100], [300, -100], [300, 100], [-300, 100]]
    short = [[-150, -100], [150, -100], [150, 100], [-150, 100]]              # holds the centre, not the ends
    assert PG.footprint_inside(PART, R, t, big)[0]
    assert not PG.footprint_inside(PART, R, t, short)[0]
