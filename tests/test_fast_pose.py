"""
Tests for the fast locked pose search (tools/fast_pose.py, bd 6et).

The photographs are synthetic: a bar with a plate on one end, flat-shaded and drawn by two cameras
looking down on it from different sides, so the edges the search matches are real intensity steps.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cv2  # noqa: E402

import fast_pose as FP  # noqa: E402

_TRIS = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def _box(lo, hi):
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    return np.array([[v[a], v[b], v[c]] for a, b, c in _TRIS])


# model frame: a 300 mm bar along x, 40 x 40, with an 80 x 80 x 10 plate on its -x end
PART = np.concatenate([_box((-150, -20, -20), (150, 20, 20)), _box((-160, -40, -40), (-150, 40, 40))])
K = [[1100.0, 0, 640], [0, 1100.0, 400], [0, 0, 1]]
W, H = 1280, 800


def _camera(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = (target - eye) / np.linalg.norm(target - eye)
    y = np.array([0.0, 0.0, 1.0]) - f * f[2]
    y /= np.linalg.norm(y)
    x = np.cross(y, f)
    Rc = np.stack([x, y, f])
    return {"rvec_cam": cv2.Rodrigues(Rc)[0].ravel(), "tvec_cam": -Rc @ eye, "K": K, "dist": [0.0] * 5,
            "width": W, "height": H}


def _render(tris, R, t, view):
    """Flat shading, far triangles first: each face a different grey on a mid-grey table."""
    Rc, tc = FP._rot(view["rvec_cam"]), np.asarray(view["tvec_cam"], float)
    Kc = np.asarray(K)
    img = np.full((H, W), 90, np.uint8)
    world = tris @ R.T + t
    cam = world @ Rc.T + tc
    normals = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    light = np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8])
    for i in np.argsort(-cam[:, :, 2].mean(axis=1)):
        uv = (cam[i, :, :2] / cam[i, :, 2:3]) * [Kc[0, 0], Kc[1, 1]] + [Kc[0, 2], Kc[1, 2]]
        shade = int(40 + 200 * abs(normals[i] @ light))
        cv2.fillConvexPoly(img, np.round(uv * 16).astype(np.int32), shade, lineType=cv2.LINE_AA, shift=4)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _on_table(yaw_deg, xy):
    """Lying on the table (board z = 0, the part above at negative z), turned about the vertical."""
    R = FP._rot([0, 0, np.radians(yaw_deg)])
    lowest = (PART.reshape(-1, 3) @ R.T)[:, 2].max()
    return R, np.array([xy[0], xy[1], -lowest])


TRUE_R, TRUE_T = _on_table(20.0, (30.0, -10.0))
VIEWS = [_camera((-500, -700, -650), (0, 0, 0)), _camera((600, 450, -700), (0, 0, 0))]
for _v in VIEWS:
    _v["image"] = _render(PART, TRUE_R, TRUE_T, _v)
CENTRE, LENGTH = np.zeros(3), np.array([1.0, 0, 0])


def _mean_gap(R, t):
    p = PART.reshape(-1, 3)
    return float(np.linalg.norm(p @ R.T + t - (p @ TRUE_R.T + TRUE_T), axis=1).mean())


def test_locked_poses_only_slide_and_turn_about_the_vertical():
    base = FP.LockedPose(FP._vec(TRUE_R), TRUE_T, CENTRE, LENGTH)
    R0, t0 = base.at(0, 0, 0)
    assert np.allclose(R0, TRUE_R) and np.allclose(t0, TRUE_T)
    R, t = base.at(25.0, -10.0, 4.0)
    turn = R @ TRUE_R.T
    assert np.allclose(turn[2], [0, 0, 1], atol=1e-12)                     # about the vertical only
    assert np.isclose(np.degrees(np.arctan2(turn[1, 0], turn[0, 0])), 4.0)
    moved = (R @ CENTRE + t) - (TRUE_R @ CENTRE + TRUE_T)                  # the centre slides, the turn doesn't move it
    along = TRUE_R @ LENGTH
    assert np.isclose(moved @ along, 25.0) and np.isclose(moved[2], 0.0)


def test_the_chamfer_is_lowest_at_the_true_pose():
    maps = [FP.edge_maps(v) for v in VIEWS]
    samples = FP.prepare_samples(PART, FP._vec(TRUE_R), TRUE_T, VIEWS)
    base = FP.LockedPose(FP._vec(TRUE_R), TRUE_T, CENTRE, LENGTH)
    at_true = FP.chamfer(*base.at(0, 0, 0), samples, maps, 25.0)
    # not zero: the bar's end edges lie flat on the plate's face, so nothing in the picture marks them
    assert at_true < 2.5
    for off in ((10, 0, 0), (0, 10, 0), (0, 0, 2), (-10, 5, -2)):
        assert FP.chamfer(*base.at(*off), samples, maps, 25.0) > at_true + 1.0
    # small moves still cost, least along the length: a bar slid along itself only moves its end edges
    for off in ((5, 0, 0), (-5, 0, 0), (0, 5, 0), (0, 0, 1)):
        assert FP.chamfer(*base.at(*off), samples, maps, 25.0) > at_true + 0.3


def test_the_search_finds_the_part_from_an_offset_start():
    """The operator put it about 30 mm and a few degrees from where the target said. The search locates;
    the accurate refinement that follows it measures, so the bar here is 'close enough to hand over', 2 mm."""
    start_R, start_t = FP.LockedPose(FP._vec(TRUE_R), TRUE_T, CENTRE, LENGTH).at(-24.0, 17.0, -3.5)
    maps = [FP.edge_maps(v) for v in VIEWS]
    samples = FP.prepare_samples(PART, FP._vec(start_R), start_t, VIEWS)
    base = FP.LockedPose(FP._vec(start_R), start_t, CENTRE, LENGTH)
    R, t, _p, cham, timing = FP.search(base, samples, maps)
    assert _mean_gap(start_R, start_t) > 25.0
    assert _mean_gap(R, t) < 2.0, (_mean_gap(R, t), cham)
    assert timing["grid_poses"] == 9 * 7 * 7
