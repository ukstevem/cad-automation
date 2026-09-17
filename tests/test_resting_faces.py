"""
Tests for the resting-face poses (tools/resting_faces.py).

The pose is in the ChArUco board frame, where z points into the table. The rotation used to take the
chosen face to -z - on top - and a part with an end plate proud of its frame came out pitched the wrong
way, resting on the plate edge alone with the other end in the air (bd 6et, 2026-09-17).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import resting_faces as RF  # noqa: E402

_TRIS = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def _box(lo, hi):
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    return np.array([[v[a], v[b], v[c]] for a, b, c in _TRIS])


# a 400 mm frame, 80 x 80, with a 120 x 120 plate on its -x end: the plate stands 20 mm proud all round
PART = np.concatenate([_box((-190, -40, -40), (210, 40, 40)), _box((-210, -60, -60), (-190, 60, 60))])


def test_the_chosen_face_ends_up_underneath():
    V = PART.reshape(-1, 3)
    for c in RF.candidates(PART):
        n = np.asarray(c["normal"])
        on_face = np.isclose(V @ n, (V @ n).max(), atol=1e-6)
        z = (V @ c["R"].T)[:, 2]
        assert np.allclose(z[on_face], z.max(), atol=1e-6), c["normal"]      # the face is the lowest thing (z down)
        assert z.mean() < z.max() - 1.0                                       # and the part is above it


def test_a_proud_end_plate_pitches_the_far_end_down_to_the_table():
    """Lying on its side the part touches the table at the plate edge AND the far end of the frame."""
    V = PART.reshape(-1, 3)
    side = [c for c in RF.candidates(PART) if abs(c["normal"][0]) < 0.2 and c["area"] > 0.05]
    assert side
    for c in side:
        z = (V @ c["R"].T)[:, 2]
        bottom = z.max()
        # the hull meets the plate along its INNER edge (x = -190): the outer edge sits 1 mm inside the slope
        plate_end, far_end = V[:, 0] < -185, V[:, 0] > 205
        assert np.isclose(z[plate_end].max(), bottom, atol=1e-6)
        assert np.isclose(z[far_end].max(), bottom, atol=1e-6)
