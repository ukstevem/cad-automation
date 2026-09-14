"""
Tests for the clicking page's model render (tools/click_pose.py, bd vge).

Two defects Steve hit on the first real use, 2026-09-14: the model was drawn orthographically, so it
did not look like the photograph, and a click was recorded at the centroid of the triangle under it,
which on a long rail face is nowhere near the click.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cv2  # noqa: E402

import click_pose as CP  # noqa: E402

_TRIS = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def _box(lo, hi):
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                 float)
    return np.array([[v[a], v[b], v[c]] for a, b, c in _TRIS])


def _camera(eye, target):
    """An OpenCV camera at ``eye`` looking at ``target``, image y running towards board +Z (down)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = (target - eye) / np.linalg.norm(target - eye)
    y = np.array([0.0, 0.0, 1.0]) - f * f[2]
    y /= np.linalg.norm(y)
    x = np.cross(y, f)
    Rc = np.stack([x, y, f])
    rvec, _ = cv2.Rodrigues(Rc)
    return {"rvec_cam": rvec.ravel().tolist(), "tvec_cam": (-Rc @ eye).tolist()}


K = [[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]]
DIST = [0.0, 0, 0, 0, 0]
# a 400 mm bar running towards and away from a camera that looks down on it at 45 degrees
BAR = _box((-20, -200, -20), (20, 200, 20))
VIEW = _camera(eye=(0, -600, -600), target=(0, 0, -20))


def _render():
    return CP.render_perspective_with_lookup(BAR, np.eye(3), VIEW, K, DIST)


def test_a_click_is_recorded_where_it_lands_not_at_the_triangle_centroid():
    """The bar's long faces are two triangles each. A centroid lookup can only ever return a dozen
    distinct points; the surface under the pixels runs the whole 400 mm."""
    _img, lut = _render()
    y = lut[:, :, 1][~np.isnan(lut[:, :, 1])]
    assert y.max() - y.min() > 350
    assert len(np.unique(np.round(y))) > 100


def test_the_near_end_is_drawn_larger_than_the_far_end():
    """Perspective: the end nearer the camera (y = -200) must span more pixels across the bar than
    the far end. An orthographic view draws them the same."""
    _img, lut = _render()
    y = lut[:, :, 1]
    ok = ~np.isnan(y)

    def width(sel):
        xs = np.nonzero(sel & ok)[1]
        return xs.max() - xs.min()

    near, far = width(y < -185), width(y > 185)
    assert near > 1.25 * far


def test_the_lookup_is_on_the_model_surface():
    _img, lut = _render()
    p = lut[~np.isnan(lut[:, :, 0])]
    outside = np.maximum(np.abs(p) - [20, 200, 20], 0)
    assert outside.max() < 0.5
    on_a_face = np.min(np.abs(np.abs(p) - [20, 200, 20]), axis=1)
    assert np.percentile(on_a_face, 95) < 0.5
