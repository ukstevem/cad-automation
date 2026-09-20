"""
Tests for projecting the outline onto the table (tools/projector.py, bd c5o).

The whole idea rests on one claim: what the projector draws lies on the table, so a homography is enough and no
projector calibration is needed. These build a synthetic projector, light the table with it, photograph that with
a synthetic camera, and check the recovered mapping puts millimetres back where they came from.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

import projector as PJ  # noqa: E402
from test_fast_pose import _camera  # noqa: E402


def _light_the_table(proj, points_px):
    """Where a projector's pixels land on the table (z = 0), by the same ray-plane step the camera uses."""
    return PJ.ray_to_table(points_px, proj)


def test_pixels_land_on_the_table_where_the_geometry_says():
    """The pixel at the principal point must land where the camera's own axis meets the table, and any table point
    must survive the round trip through the lens and back."""
    eye, aim = np.array([-300.0, 200.0, -800.0]), np.array([120.0, -60.0, 0.0])
    view = _camera(eye, aim)
    K = np.asarray(view["K"], float)
    hit = PJ.ray_to_table([[K[0, 2], K[1, 2]]], view)[0]
    assert np.allclose(hit, aim, atol=1e-6)                            # the axis was aimed there

    table = np.array([[0.0, 0.0, 0.0], [250.0, -120.0, 0.0], [-80.0, 90.0, 0.0]])
    uv, _ = cv2.projectPoints(table, np.asarray(view["rvec_cam"], float), np.asarray(view["tvec_cam"], float),
                              K, np.zeros(5))
    assert np.abs(PJ.ray_to_table(uv.reshape(-1, 2), view) - table).max() < 1e-6


def test_the_mapping_recovers_millimetres_from_one_photograph():
    """Projector lights the table, camera photographs it, homography fitted: a fresh point must come back within
    a tenth of a millimetre."""
    proj = _camera((-200.0, 150.0, -900.0), (150.0, 0.0, 0.0))        # the projector, as a camera run backwards
    cam = _camera((-500.0, -700.0, -650.0), (150.0, 0.0, 0.0))
    pattern = PJ.dot_grid(1920, 1080, 7, 5, 18)
    on_table = _light_the_table(proj, pattern)

    # the camera sees those lit dots; its pixels ray back onto the table
    uv, _ = cv2.projectPoints(on_table, np.asarray(cam["rvec_cam"], float), np.asarray(cam["tvec_cam"], float),
                              np.asarray(cam["K"], float), np.zeros(5))
    seen = PJ.ray_to_table(uv.reshape(-1, 2), cam)
    assert np.abs(seen - on_table).max() < 1e-6                        # the camera step alone is exact

    H, med, worst, inliers = PJ.fit_homography(pattern, seen)
    assert inliers == len(pattern) and med < 0.1 and worst < 0.5

    fresh = PJ.dot_grid(1920, 1080, 3, 3, 10)[1:]                      # points the fit never saw
    want = _light_the_table(proj, fresh)
    got = cv2.perspectiveTransform(fresh.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.abs(got - want[:, :2]).max() < 0.1


def test_millimetres_go_back_to_pixels():
    proj = _camera((-200.0, 150.0, -900.0), (150.0, 0.0, 0.0))
    pattern = PJ.dot_grid(1920, 1080, 7, 5, 18)
    H, _med, _worst, _n = PJ.fit_homography(pattern, _light_the_table(proj, pattern))
    where = np.array([[100.0, -50.0], [220.0, 40.0], [0.0, 0.0]])
    back = PJ.to_projector(H, where)
    assert np.abs(_light_the_table(proj, back)[:, :2] - where).max() < 0.1


def test_the_dots_are_found_and_put_back_in_pattern_order():
    """The photograph is taken from an angle and the blobs arrive in any order; a wrong pairing would fit a
    plausible, wrong mapping rather than fail."""
    spec = {"width": 1920, "height": 1080, "cols": 7, "rows": 5, "radius": 18}
    pattern = PJ.dot_grid(spec["width"], spec["height"], spec["cols"], spec["rows"], spec["radius"])
    cam = _camera((-500.0, -700.0, -650.0), (150.0, 0.0, 0.0))
    proj = _camera((-200.0, 150.0, -900.0), (150.0, 0.0, 0.0))
    on_table = _light_the_table(proj, pattern)
    uv, _ = cv2.projectPoints(on_table, np.asarray(cam["rvec_cam"], float), np.asarray(cam["tvec_cam"], float),
                              np.asarray(cam["K"], float), np.zeros(5))
    uv = uv.reshape(-1, 2)
    img = np.zeros((int(cam["height"]), int(cam["width"])), np.uint8)
    for i, (x, y) in enumerate(uv):
        cv2.circle(img, (int(round(x)), int(round(y))), 22 if i == 0 else 13, 255, -1, cv2.LINE_AA)
    found, why = PJ.find_dots(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), spec)
    assert found is not None, why
    assert np.abs(found - uv).max() < 2.0                              # same dots, same order
