"""
ChArUco board pose + multi-camera extrinsics — feeds the multi-view object fit (ADR 0001).

Two jobs for the test cell:

1. **Per-camera world pose (board-in-shot mode).** When several cameras photograph the same
   ChArUco board at once, each camera's pose *relative to the board* is its world→camera pose,
   with the board defining the shared world frame. That is exactly the ``rvec_cam/tvec_cam``
   each view needs in ``app/services/multiview.fit_object_pose``. No separate calibration step.

2. **Rig extrinsics (production mode).** The relative pose between two cameras (camera A → camera
   B), recovered from a shared board view (averaged over several). Once known, the board can be
   removed and the cameras' relative geometry is fixed.

Board 3D corners come from ``CharucoBoard.getChessboardCorners()`` (board frame, mm); detected
ChArUco corners give the 2D image points; ``solvePnP`` (SQPNP — see ``pose.py``) ties them.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2

    _CV2_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = str(exc)

from app.services.pose import rt_compose, rt_invert


class BoardPoseError(Exception):
    """Raised when a board pose cannot be solved (too few corners, degenerate)."""


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise BoardPoseError(f"OpenCV is not available: {_CV2_IMPORT_ERROR}")


def charuco_board_pose(
    charuco_corners,
    charuco_ids,
    board,
    K,
    dist,
    *,
    min_corners: int = 6,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Solve the board→camera pose from detected ChArUco corners.

    *charuco_corners*: Nx1x2 (or Nx2) detected sub-pixel corners. *charuco_ids*: Nx1 ids
    into the board's chessboard-corner array. Returns ``(rvec, tvec, n_corners)`` mapping a
    board point ``X`` to camera coords by ``R·X + t``. Raises on too few corners.
    """
    _require_cv2()
    ids = np.asarray(charuco_ids).reshape(-1)
    img = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    if len(ids) != len(img):
        raise BoardPoseError(f"corner/id count mismatch: {len(img)} vs {len(ids)}")
    if len(ids) < min_corners:
        raise BoardPoseError(f"need >={min_corners} ChArUco corners, got {len(ids)}")

    all_obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)  # Mx3, board frame
    obj = all_obj[ids]
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    ok, rvec, tvec = cv2.solvePnP(
        obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        raise BoardPoseError("solvePnP failed on the board corners")
    rvec, tvec = cv2.solvePnPRefineLM(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, dist, rvec, tvec)
    return rvec, tvec, int(len(ids))


def relative_pose(rvec_a, tvec_a, rvec_b, tvec_b) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Relative pose camera A → camera B from each camera's board→camera pose for the SAME board.

    A point in A's camera frame maps to B's by ``T(A→B) = T(board→B) ∘ T(A→board)``, i.e.
    compose B's pose with the inverse of A's. Returns ``(R_ab, t_ab, baseline_mm)`` where the
    baseline is the camera-centre separation.
    """
    _require_cv2()
    Ra, _ = cv2.Rodrigues(np.asarray(rvec_a, np.float64).reshape(3, 1))
    Rb, _ = cv2.Rodrigues(np.asarray(rvec_b, np.float64).reshape(3, 1))
    ta = np.asarray(tvec_a, np.float64).reshape(3, 1)
    tb = np.asarray(tvec_b, np.float64).reshape(3, 1)
    R_ab, t_ab = rt_compose((Rb, tb), rt_invert(Ra, ta))
    # Camera centres in board frame: C = -Rᵀ·t. Baseline = |C_a - C_b|.
    Ca = -Ra.T @ ta
    Cb = -Rb.T @ tb
    baseline = float(np.linalg.norm(Ca - Cb))
    return R_ab, t_ab, baseline


def average_relative_pose(pairs: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Average several A→B relative poses (one per shared board view) into a stable rig extrinsic.

    *pairs*: list of ``((rvec_a, tvec_a), (rvec_b, tvec_b))``. Rotations averaged via mean
    quaternion (eigenvector of the accumulated outer product); translations by mean. Returns
    ``(R_ab, t_ab, info)`` with spread diagnostics so a wobbly view can be spotted.
    """
    _require_cv2()
    if not pairs:
        raise BoardPoseError("no board-view pairs to average")
    quats, ts, baselines = [], [], []
    for (ra, ta), (rb, tb) in pairs:
        R_ab, t_ab, base = relative_pose(ra, ta, rb, tb)
        quats.append(_quat_from_R(R_ab))
        ts.append(np.asarray(t_ab).reshape(3))
        baselines.append(base)
    q = _average_quaternions(np.array(quats))
    R_ab = _R_from_quat(q)
    t_ab = np.mean(np.array(ts), axis=0).reshape(3, 1)
    # Spread: max angular deviation of any view from the mean rotation.
    angs = [_quat_angle_deg(qi, q) for qi in quats]
    info = {
        "views": len(pairs),
        "baseline_mm": round(float(np.mean(baselines)), 2),
        "rot_spread_deg": round(float(np.max(angs)), 3),
        "trans_spread_mm": round(float(np.max(np.linalg.norm(np.array(ts) - t_ab.reshape(3), axis=1))), 3),
    }
    return R_ab, t_ab, info


# ── small quaternion helpers (averaging rotations) ──────────────────────────

def _quat_from_R(R) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64).reshape(3, 3))
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = (rvec.reshape(3) / theta)
    return np.array([np.cos(theta / 2), *(np.sin(theta / 2) * axis)])


def _R_from_quat(q) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    theta = 2 * np.arccos(np.clip(w, -1, 1))
    s = np.sqrt(max(1 - w * w, 1e-18))
    axis = np.array([x, y, z]) / s if s > 1e-9 else np.array([1.0, 0.0, 0.0])
    R, _ = cv2.Rodrigues((axis * theta).reshape(3, 1))
    return R


def _average_quaternions(Q) -> np.ndarray:
    # Sign-align to the first, then take the principal eigenvector of Σqqᵀ.
    Q = Q.copy()
    for i in range(len(Q)):
        if np.dot(Q[i], Q[0]) < 0:
            Q[i] = -Q[i]
    A = Q.T @ Q
    w, v = np.linalg.eigh(A)
    return v[:, -1]


def _quat_angle_deg(qa, qb) -> float:
    d = abs(float(np.dot(qa, qb)))
    return float(np.degrees(2 * np.arccos(np.clip(d, -1, 1))))
