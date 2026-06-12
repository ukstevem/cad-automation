"""
Headless verification of the AR pose-solve + projection core (app/services/pose.py).

Strategy: build a synthetic single "section" (a box) with known 3D corners, place a
virtual camera at a known pose, project the corners to get ground-truth image points,
then check that solve_pose() recovers the pose and that projection round-trips. No real
photo or OCC needed — this isolates and proves the maths.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.services import pose as P


# ── Synthetic scene ─────────────────────────────────────────────────────────

# A 100 x 50 x 1000 mm "section", corners in world (mm).
_X, _Y, _Z = 50.0, 25.0, 1000.0
CORNERS = np.array([
    [-_X, -_Y, 0.0], [_X, -_Y, 0.0], [_X, _Y, 0.0], [-_X, _Y, 0.0],
    [-_X, -_Y, _Z],  [_X, -_Y, _Z],  [_X, _Y, _Z],  [-_X, _Y, _Z],
], dtype=np.float64)

# 12 box edges as 3D polylines (pairs of corners).
_EDGE_IDX = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
EDGES = [np.array([CORNERS[a], CORNERS[b]]) for a, b in _EDGE_IDX]

# Camera intrinsics for a 4000 x 3000 image, no distortion.
K = np.array([[3000.0, 0, 2000.0],
              [0, 3000.0, 1500.0],
              [0, 0, 1.0]])
DIST = np.zeros((5, 1))
IMG_W, IMG_H = 4000, 3000


def _look_at(cam_pos, target, world_up=(0.0, 0.0, 1.0)):
    """OpenCV-convention pose (x right, y down, z forward) looking at target."""
    cam_pos = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(world_up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)          # rows = camera axes in world
    t = (-R @ cam_pos).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t


CAM_POS = np.array([1500.0, -1200.0, -800.0])   # where the camera "stands"
RVEC_TRUE, TVEC_TRUE = _look_at(CAM_POS, target=[0, 0, 500])


def _ground_truth_image_points():
    proj, _ = cv2.projectPoints(CORNERS, RVEC_TRUE, TVEC_TRUE, K, DIST)
    return proj.reshape(-1, 2)


# ── Tests ───────────────────────────────────────────────────────────────────

def test_scene_projects_in_frame():
    """Sanity: the synthetic box actually lands inside the image."""
    img = _ground_truth_image_points()
    assert img[:, 0].min() > 0 and img[:, 0].max() < IMG_W
    assert img[:, 1].min() > 0 and img[:, 1].max() < IMG_H


def test_solve_pose_recovers_camera_clean():
    img = _ground_truth_image_points()
    rvec, tvec, _ = P.solve_pose(CORNERS, img, K, DIST)

    # Camera centre recovered to sub-mm from clean correspondences.
    cam = P.camera_position_world(rvec, tvec)
    assert np.linalg.norm(cam - CAM_POS) < 0.05

    rms = P.reprojection_rms(CORNERS, img, rvec, tvec, K, DIST)
    assert rms < 1e-3


def test_projection_round_trips():
    """Edges projected under the solved pose match the ground-truth corner pixels."""
    img = _ground_truth_image_points()
    rvec, tvec, _ = P.solve_pose(CORNERS, img, K, DIST)

    polylines = P.project_polylines(EDGES, rvec, tvec, K, DIST)
    assert len(polylines) == 12
    for poly, (a, b) in zip(polylines, _EDGE_IDX):
        poly = np.asarray(poly)
        assert poly.shape == (2, 2)
        assert np.allclose(poly[0], img[a], atol=1e-2)
        assert np.allclose(poly[1], img[b], atol=1e-2)


def test_solve_pose_robust_to_noise():
    rng = np.random.default_rng(0)
    img = _ground_truth_image_points() + rng.normal(0, 0.5, size=(8, 2))

    rvec, tvec, _ = P.solve_pose(CORNERS, img, K, DIST)
    cam = P.camera_position_world(rvec, tvec)

    # 0.5 px noise on a ~2.3 m shot → camera within a few mm, RMS ~ the noise level.
    assert np.linalg.norm(cam - CAM_POS) < 10.0
    rms = P.reprojection_rms(CORNERS, img, rvec, tvec, K, DIST)
    assert 0.1 < rms < 2.0


def test_ransac_path_and_outlier_rejection():
    img = _ground_truth_image_points()
    img_noisy = img.copy()
    img_noisy[3] += np.array([400.0, -300.0])   # one gross outlier correspondence

    rvec, tvec, inliers = P.solve_pose(CORNERS, img_noisy, K, DIST,
                                       ransac=True, reproj_threshold=5.0)
    assert inliers is not None
    # The outlier should be rejected; the rest recover the camera well.
    assert 3 not in inliers.ravel().tolist()
    cam = P.camera_position_world(rvec, tvec)
    assert np.linalg.norm(cam - CAM_POS) < 5.0


def test_solve_pose_five_points():
    """5 correspondences must work — ITERATIVE's DLT init needs >=6 and threw on 5."""
    img = _ground_truth_image_points()
    rvec, tvec, _ = P.solve_pose(CORNERS[:5], img[:5], K, DIST)
    cam = P.camera_position_world(rvec, tvec)
    assert np.linalg.norm(cam - CAM_POS) < 1.0


def test_too_few_points_raises():
    img = _ground_truth_image_points()
    with pytest.raises(P.PoseError):
        P.solve_pose(CORNERS[:3], img[:3], K, DIST)


def test_marker_registration_roundtrip():
    """register_marker_to_model then model_pose_from_marker recovers the camera pose
    on a different shot — the bootstrap→auto-align chain, exact (no image noise)."""
    # True world→camera (reuse) and a true marker→model transform.
    R2, _ = cv2.Rodrigues(RVEC_TRUE)
    R_wm_true, _ = cv2.Rodrigues(np.array([[0.1], [0.2], [-0.15]]))
    t_wm_true = np.array([[30.0], [-20.0], [500.0]])

    # Forward: marker pose in the camera frame for the registration shot.
    R1 = R2 @ R_wm_true
    t1 = R2 @ t_wm_true + TVEC_TRUE
    rvec_m, _ = cv2.Rodrigues(R1)

    R_wm, t_wm = P.register_marker_to_model(rvec_m, t1, RVEC_TRUE, TVEC_TRUE)
    assert np.allclose(R_wm, R_wm_true, atol=1e-6)
    assert np.allclose(t_wm.ravel(), t_wm_true.ravel(), atol=1e-3)

    # A different camera pose; auto-recover world→camera from the marker alone.
    rvec2b, tvec2b = _look_at(CAM_POS + np.array([500.0, 300.0, -400.0]), target=[0, 0, 500])
    R2b, _ = cv2.Rodrigues(rvec2b)
    R1b = R2b @ R_wm_true
    t1b = R2b @ t_wm_true + tvec2b
    rvec_mb, _ = cv2.Rodrigues(R1b)

    rvec_rec, tvec_rec = P.model_pose_from_marker(rvec_mb, t1b, R_wm, t_wm)
    R_rec, _ = cv2.Rodrigues(rvec_rec)
    assert np.allclose(R_rec, R2b, atol=1e-6)
    assert np.allclose(tvec_rec.ravel(), tvec2b.ravel(), atol=1e-3)


def test_load_intrinsics_from_profile():
    profile = {"camera_matrix": K.tolist(), "dist_coeffs": DIST.ravel().tolist()}
    K2, dist2 = P.load_intrinsics(profile)
    assert np.allclose(K2, K)
    assert dist2.shape[0] == 5
