"""
Camera pose estimation + 3D→2D projection for the AR ratification overlay.

Given a calibrated camera (intrinsics from a saved calibration profile) and a set
of 2D↔3D correspondences between a photo and the CAD model, solve the camera pose
(``solvePnP``) and project arbitrary 3D model geometry — CAD edges, weld paths —
onto the photo as 2D overlay polylines.

This is the core of the Phase 0 alignment spike: prove the chain
``intrinsics + pose → project model onto photo``. It is pure and fast (small point
sets), so it runs inline in the FastAPI process — no subprocess worker needed
(unlike OCC mesh generation, which holds the GIL).

Conventions
-----------
- ``object_points`` are 3D points in the CAD **world** frame (same frame the weld
  paths / world-space STLs live in), units = mm.
- ``image_points`` are 2D pixels in the photo.
- Pose is returned as OpenCV ``rvec`` (Rodrigues) + ``tvec`` such that a world
  point ``X`` maps to camera coords by ``R·X + t`` (``R = Rodrigues(rvec)``).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # import guard — keep module importable if the wheel is missing
    import cv2

    _CV2_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = str(exc)


class PoseError(Exception):
    """Raised when a pose cannot be solved (too few / degenerate points)."""


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise PoseError(f"OpenCV is not available: {_CV2_IMPORT_ERROR}")


def load_intrinsics(profile: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Pull (camera_matrix 3x3, dist_coeffs Nx1) out of a saved calibration profile."""
    K = np.asarray(profile["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(profile["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return K, dist


def solve_pose(
    object_points: Sequence,
    image_points: Sequence,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    ransac: bool = False,
    reproj_threshold: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Solve camera pose from 3D↔2D correspondences.

    Returns ``(rvec, tvec, inliers)``. ``inliers`` is None unless ``ransac=True``.
    Refines the result with Levenberg–Marquardt for sub-pixel accuracy.
    """
    _require_cv2()
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if len(obj) != len(img):
        raise PoseError(f"point count mismatch: {len(obj)} object vs {len(img)} image")
    if len(obj) < 4:
        raise PoseError(f"need at least 4 correspondences, got {len(obj)}")

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    inliers: Optional[np.ndarray] = None
    if ransac:
        # EPnP init inside RANSAC works from 4 points (ITERATIVE's DLT needs >=6).
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, dist, reprojectionError=reproj_threshold,
            flags=cv2.SOLVEPNP_EPNP,
        )
    else:
        # SQPNP solves from >=4 points and handles planar AND non-planar configs.
        # SOLVEPNP_ITERATIVE's DLT initialisation needs >=6 non-coplanar points and
        # raised "DLT algorithm needs at least 6 points" on a 5-point click set.
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)

    if not ok:
        raise PoseError("solvePnP failed to find a pose")

    # LM refinement over the (inlier) correspondences for sub-pixel accuracy.
    if inliers is not None and len(inliers) >= 4:
        idx = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx], K, dist, rvec, tvec)
    else:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)

    return rvec, tvec, inliers


def reprojection_rms(
    object_points: Sequence,
    image_points: Sequence,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """RMS reprojection error (px) of the correspondences under the solved pose."""
    _require_cv2()
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, np.asarray(K, np.float64).reshape(3, 3),
                                np.asarray(dist, np.float64).reshape(-1, 1))
    diff = (proj - img).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def project_points(
    points3d: Sequence,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Project Nx3 world points → Nx2 pixels under the given pose + intrinsics."""
    _require_cv2()
    pts = np.asarray(points3d, dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, np.asarray(K, np.float64).reshape(3, 3),
                                np.asarray(dist, np.float64).reshape(-1, 1))
    return proj.reshape(-1, 2)


def _clip_polyline_to_front(
    pts_cam: np.ndarray, near: float
) -> List[np.ndarray]:
    """
    Split a polyline (Nx3, already in CAMERA coords) into sub-polylines that lie in
    front of the near plane (camera Z >= near). Segments crossing the plane are cut
    at the crossing point so the line stops cleanly at the plane instead of wrapping.

    Why this matters: cv2.projectPoints will happily project points with Z<=0 (behind
    the camera). Those produce wildly wrong 2D coords, and the distortion polynomial
    diverges far outside the image and curls them back toward the principal point —
    the tell-tale 'extra lines converging on the centre' artifact. Dropping the behind-
    camera parts removes it.
    """
    z = pts_cam[:, 2]
    front = z >= near
    segments: List[np.ndarray] = []
    current: List[np.ndarray] = []
    for i in range(len(pts_cam)):
        if front[i]:
            current.append(pts_cam[i])
        if i + 1 < len(pts_cam):
            a, b = pts_cam[i], pts_cam[i + 1]
            za, zb = a[2], b[2]
            crosses = (za >= near) != (zb >= near)
            if crosses:
                # linear interpolation to the point where Z == near
                tcr = (near - za) / (zb - za)
                cross_pt = a + tcr * (b - a)
                if za >= near:
                    # leaving the front region → end this segment at the plane
                    current.append(cross_pt)
                    if len(current) >= 2:
                        segments.append(np.asarray(current))
                    current = []
                else:
                    # entering the front region → start a new segment at the plane
                    current = [cross_pt]
    if len(current) >= 2:
        segments.append(np.asarray(current))
    return segments


def project_polylines(
    polylines3d: List[Sequence],
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    near: float = 1.0,
) -> List[list]:
    """
    Project a list of 3D polylines (each Nx3 world points) → list of 2D pixel
    polylines, ready to draw as the overlay. Empty polylines are skipped.

    Polylines are clipped to the part in front of the camera near plane (``near`` mm
    in front) before projection — see ``_clip_polyline_to_front``. This prevents the
    behind-camera distortion artifact (stray lines converging on the image centre)
    when the pose places some model edges behind the camera.
    """
    _require_cv2()
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t = np.asarray(tvec, np.float64).reshape(3, 1)
    out: List[list] = []
    for pl in polylines3d:
        pts = np.asarray(pl, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            continue
        pts_cam = (R @ pts.T + t).T  # world → camera coords
        for seg in _clip_polyline_to_front(pts_cam, near):
            # seg is already in camera coords → project with identity pose
            proj = project_points(seg, np.zeros(3), np.zeros(3), K, dist)
            out.append(proj.tolist())
    return out


def rt_compose(A, B):
    """Compose rigid transforms (apply B then A): X → A(B(X)). Each is (R 3x3, t 3x1)."""
    RA = np.asarray(A[0], np.float64).reshape(3, 3)
    tA = np.asarray(A[1], np.float64).reshape(3, 1)
    RB = np.asarray(B[0], np.float64).reshape(3, 3)
    tB = np.asarray(B[1], np.float64).reshape(3, 1)
    return RA @ RB, RA @ tB + tA


def rt_invert(R, t):
    """Invert a rigid transform (R, t) → (Rᵀ, -Rᵀ·t)."""
    R = np.asarray(R, np.float64).reshape(3, 3)
    t = np.asarray(t, np.float64).reshape(3, 1)
    return R.T, -R.T @ t


def register_marker_to_model(rvec_marker, tvec_marker, rvec_model, tvec_model):
    """
    Compute T(marker→model) from one photo where both are known.

    marker pose (rvec_marker,tvec_marker): X_cam = R1·X_marker + t1
    model pose  (rvec_model, tvec_model ): X_cam = R2·X_world  + t2
    → X_world = R_wm·X_marker + t_wm, with R_wm = R2ᵀ·R1, t_wm = R2ᵀ·(t1 - t2).

    Returns (R_wm 3x3, t_wm 3x1) as lists-ready numpy arrays.
    """
    _require_cv2()
    R1, _ = cv2.Rodrigues(np.asarray(rvec_marker, np.float64).reshape(3, 1))
    R2, _ = cv2.Rodrigues(np.asarray(rvec_model, np.float64).reshape(3, 1))
    t1 = np.asarray(tvec_marker, np.float64).reshape(3, 1)
    t2 = np.asarray(tvec_model, np.float64).reshape(3, 1)
    R_wm = R2.T @ R1
    t_wm = R2.T @ (t1 - t2)
    return R_wm, t_wm


def model_pose_from_marker(rvec_marker, tvec_marker, R_wm, t_wm):
    """
    Given a freshly-detected marker pose and the stored T(marker→model), recover the
    world→camera pose (rvec, tvec) for projecting CAD geometry. Inverse of the above:
    R2 = R1·R_wmᵀ, t2 = t1 - R2·t_wm.
    """
    _require_cv2()
    R1, _ = cv2.Rodrigues(np.asarray(rvec_marker, np.float64).reshape(3, 1))
    t1 = np.asarray(tvec_marker, np.float64).reshape(3, 1)
    R_wm = np.asarray(R_wm, np.float64).reshape(3, 3)
    t_wm = np.asarray(t_wm, np.float64).reshape(3, 1)
    R2 = R1 @ R_wm.T
    t2 = t1 - R2 @ t_wm
    rvec2, _ = cv2.Rodrigues(R2)
    return rvec2, t2


def camera_position_world(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Camera centre in world/model coordinates: ``C = -Rᵀ·t``. Useful for sanity
    checks ('is the camera roughly where I stood?') and for the 3D viewer.
    """
    _require_cv2()
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    return (-R.T @ np.asarray(tvec, np.float64).reshape(3, 1)).reshape(3)
