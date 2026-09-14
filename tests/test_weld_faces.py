"""
Tests for tools/weld_faces.py - which face of the article a weld is on, and which faces a camera sees.

Synthetic boxes only, no photographs. These pin the two decisions that were wrong before (bd bn5):
a square cross-section must not tilt the frame, and a camera must be presented the faces it actually
looks at, not whatever lies near a convex hull.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import weld_faces as WF  # noqa: E402

_TRIS = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def _box(lx, ly, lz, R=None):
    """A closed cuboid as (N, 3, 3) triangles, centred on the origin, optionally rotated."""
    v = np.array([[x, y, z] for x in (0, lx) for y in (0, ly) for z in (0, lz)], float)
    v -= v.mean(axis=0)
    if R is not None:
        v = (R @ v.T).T
    return np.array([[v[a], v[b], v[c]] for a, b, c in _TRIS])


def _rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _aligned(axis, target):
    return abs(abs(float(np.dot(axis, target))) - 1.0) < 1e-6


def test_frame_follows_the_long_axis_with_its_extents():
    f = WF.article_frame(_box(400, 100, 60))
    assert _aligned(f["axes"][:, 0], [1, 0, 0])
    ext = f["hi"] - f["lo"]
    assert abs(ext[0] - 400) < 1e-6
    assert sorted([round(ext[1]), round(ext[2])]) == [60, 100]


def test_a_square_section_does_not_tilt_the_frame():
    """The tower's failure: with equal width and depth the principal axes are arbitrary, and they
    came out 23 degrees off the rails. The cross axes must lie along the section's own sides."""
    f = WF.article_frame(_box(400, 100, 100))
    W, D = f["axes"][:, 1], f["axes"][:, 2]
    assert _aligned(W, [0, 1, 0]) or _aligned(W, [0, 0, 1])
    assert _aligned(D, [0, 1, 0]) or _aligned(D, [0, 0, 1])


def test_a_rotated_square_section_follows_its_rails():
    R = _rot_x(30)
    f = WF.article_frame(_box(400, 100, 100, R))
    W = f["axes"][:, 1]
    assert _aligned(W, R @ [0, 1, 0]) or _aligned(W, R @ [0, 0, 1])


def test_the_frame_is_right_handed():
    f = WF.article_frame(_box(400, 100, 60))
    assert abs(np.linalg.det(f["axes"]) - 1.0) < 1e-6


def test_default_band_is_a_fifth_of_the_smaller_section():
    assert abs(WF.default_band(WF.article_frame(_box(400, 100, 60))) - 12.0) < 1e-6


def test_a_corner_weld_is_on_two_faces_and_a_mid_face_weld_on_one():
    f = WF.article_frame(_box(400, 100, 100))
    # both cross axes point along +/- y and z in some order; build points from the frame itself
    c, ax, hi = f["centre"], f["axes"], f["hi"]
    corner = c + ax @ np.array([0.0, hi[1] - 3, hi[2] - 3])
    mid_face = c + ax @ np.array([0.0, 0.0, hi[2] - 3])
    assert WF.faces_of([corner], f, band=10) == ["+W", "+D"]
    assert WF.faces_of([mid_face], f, band=10) == ["+D"]
    assert WF.faces_of([c], f, band=10) == []


def test_a_camera_overhead_is_presented_the_top_only():
    """The board frame puts the part at negative Z, so a camera above it sits further negative."""
    f = WF.article_frame(_box(400, 100, 100))
    eye = np.array([0.0, 0.0, -2000.0])
    view = {"rvec_cam": [0.0, 0.0, 0.0], "tvec_cam": list(-eye)}
    presented, cos = WF.presented_faces(f, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], view)
    assert len(presented) == 1
    (top,) = presented
    assert WF.face_name(f, [0.0, 0.0, 0.0], top) == "top"
    assert all(cos[k] <= 0.2 for k in WF.FACES if k != top)


def test_a_camera_off_to_one_side_is_presented_that_side_as_well():
    f = WF.article_frame(_box(400, 100, 100))
    eye = np.array([0.0, 1500.0, -1500.0])
    view = {"rvec_cam": [0.0, 0.0, 0.0], "tvec_cam": list(-eye)}
    presented, _cos = WF.presented_faces(f, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], view)
    names = {WF.face_name(f, [0.0, 0.0, 0.0], k) for k in presented}
    assert "top" in names and len(presented) == 2


def test_faces_recorded_in_the_sidecar_win_over_computed_ones():
    f = WF.article_frame(_box(400, 100, 100))
    welds = [
        {"Name": "A-W001", "PSS_WeldGeometry": {"ArticleFaces": ["-L"]},
         "Representation": {"segments": [[[0, 0, 0], [1, 0, 0]]]}},
        {"Name": "A-W002", "PSS_WeldGeometry": {},
         "Representation": {"segments": [[list(f["centre"]), list(f["centre"])]]}},
    ]
    faces, computed = WF.faces_for_welds(welds, f, scale=1.0, band=10)
    assert faces["A-W001"] == ["-L"]
    assert faces["A-W002"] == [] and computed == 1
