"""
Verify the marker detection + per-marker pose approach used by /ar/detect-markers:
detect an AprilTag, then solvePnP (IPPE_SQUARE) on its 4 corners against the marker's
object points. Synthetic fronto-parallel placement gives a known ground-truth pose.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

K = np.array([[3000, 0, 2000], [0, 3000, 1500], [0, 0, 1.0]])
DIST = np.zeros((5, 1))


def test_marker_detect_and_pose_fronto_parallel():
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    size_mm, z = 55.0, 800.0
    mpx = int(round(K[0, 0] * size_mm / z))           # marker px at that distance
    marker = cv2.aruco.generateImageMarker(adict, 7, mpx)

    canvas = np.full((3000, 4000), 255, np.uint8)
    y0, x0 = 1500 - mpx // 2, 2000 - mpx // 2          # centred on principal point
    canvas[y0:y0 + mpx, x0:x0 + mpx] = marker

    det = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(canvas)
    assert ids is not None and 7 in ids.ravel().tolist()

    half = size_mm / 2
    objp = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], float)
    pts = corners[0].reshape(4, 2).astype(float)
    ok, rvec, tvec = cv2.solvePnP(objp, pts, K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    assert ok
    t = tvec.ravel()
    assert abs(t[2] - z) < 10        # recovered distance
    assert abs(t[0]) < 5 and abs(t[1]) < 5   # centred → ~0 lateral
