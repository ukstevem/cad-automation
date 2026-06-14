"""
Headless verification of ChArUco board pose + rig extrinsics (app/services/board_pose.py).

No real image needed: take the board's own 3D chessboard corners, place two virtual
calibrated cameras at known poses, project the corners to get synthetic 2D detections, then
check we recover each camera's board→camera pose and the camera-A→camera-B extrinsic.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.services import board_pose as BP


K = np.array([[3200.0, 0, 2000.0], [0, 3200.0, 1500.0], [0, 0, 1.0]])
DIST = np.zeros((5, 1))


def _board():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.CharucoBoard((7, 5), 30.0, 22.0, d)


def _look_at(cam_pos, target, world_up=(0.0, 0.0, 1.0)):
    cam_pos = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    up = np.asarray(world_up, float)
    if abs(np.dot(z, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = (-R @ cam_pos).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t


def _project_board(board, rvec, tvec):
    """Project all chessboard corners → (corners Nx1x2, ids Nx1) as a synthetic detection."""
    obj = np.asarray(board.getChessboardCorners(), np.float64)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, DIST)
    ids = np.arange(len(obj)).reshape(-1, 1)
    return proj.reshape(-1, 1, 2), ids


def test_recovers_board_pose():
    board = _board()
    # Board centred under a camera ~0.5 m up, looking down at it.
    rvec_true, tvec_true = _look_at([90, 60, 500], [90, 60, 0], world_up=(0, 1, 0))
    corners, ids = _project_board(board, rvec_true, tvec_true)

    rvec, tvec, n = BP.charuco_board_pose(corners, ids, board, K, DIST)
    assert n == len(board.getChessboardCorners())

    # Recover the camera centre to sub-mm from clean projections.
    R, _ = cv2.Rodrigues(rvec)
    cam = (-R.T @ tvec).reshape(3)
    R0, _ = cv2.Rodrigues(rvec_true)
    cam0 = (-R0.T @ tvec_true).reshape(3)
    assert np.linalg.norm(cam - cam0) < 0.05


def test_relative_extrinsics_between_two_cameras():
    board = _board()
    cam_a = _look_at([90, 60, 500], [90, 60, 0], world_up=(0, 1, 0))
    cam_b = _look_at([400, 60, 450], [90, 60, 0], world_up=(0, 1, 0))

    ca, ia = _project_board(board, *cam_a)
    cb, ib = _project_board(board, *cam_b)
    ra, ta, _ = BP.charuco_board_pose(ca, ia, board, K, DIST)
    rb, tb, _ = BP.charuco_board_pose(cb, ib, board, K, DIST)

    R_ab, t_ab, baseline = BP.relative_pose(ra, ta, rb, tb)

    # Ground-truth baseline = distance between the two camera centres.
    Ca = np.array([90, 60, 500.0]); Cb = np.array([400, 60, 450.0])
    assert abs(baseline - np.linalg.norm(Ca - Cb)) < 0.1

    # Applying T(A→B) to a point in A's frame must equal placing it via B directly.
    Ra, _ = cv2.Rodrigues(ra); Rb, _ = cv2.Rodrigues(rb)
    X = np.array([[120.0], [30.0], [0.0]])            # a board point
    Xa = Ra @ X + ta
    Xb_direct = Rb @ X + tb
    Xb_via = R_ab @ Xa + t_ab
    assert np.linalg.norm(Xb_via - Xb_direct) < 1e-3


def test_average_relative_pose_is_stable():
    board = _board()
    cam_a = _look_at([90, 60, 500], [90, 60, 0], world_up=(0, 1, 0))
    cam_b = _look_at([400, 60, 450], [90, 60, 0], world_up=(0, 1, 0))
    ca, ia = _project_board(board, *cam_a)
    cb, ib = _project_board(board, *cam_b)
    ra, ta, _ = BP.charuco_board_pose(ca, ia, board, K, DIST)
    rb, tb, _ = BP.charuco_board_pose(cb, ib, board, K, DIST)

    # Three identical clean views → tight spread, baseline recovered.
    R_ab, t_ab, info = BP.average_relative_pose([((ra, ta), (rb, tb))] * 3)
    assert info["views"] == 3
    assert info["rot_spread_deg"] < 0.05
    assert info["trans_spread_mm"] < 0.05
    assert abs(info["baseline_mm"] - np.linalg.norm(np.array([90, 60, 500.0]) - np.array([400, 60, 450.0]))) < 0.1
