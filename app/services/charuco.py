"""
ChArUco board construction + detection — shared by calibration and the AR fit path.

Extracted from ``app/routers/calibration.py`` so that headless callers (``tools/fit_multiview.py``)
build the board **the same way** the calibration that produced the profile did. A second copy of
this logic is precisely how a board/dictionary mismatch creeps in, and that failure mode is
*silent*: the detector simply returns zero corners, with no error. One definition, used everywhere.

Board geometry note: ``square_mm``/``marker_mm`` set only the metric scale of the board frame.
Detection depends on ``squares_x``/``squares_y`` and the **dictionary** alone — get the dictionary
wrong and a perfectly printed board scores nothing.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:  # import guard — keep module importable if a wheel is missing
    import cv2

    _CV2_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - import guard
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = str(exc)


class CharucoError(Exception):
    """Raised when a board cannot be built or a dictionary is unsupported."""


# Supported ArUco dictionaries (name → cv2 constant resolved lazily).
DICT_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_100",
    "DICT_6X6_250",
    "DICT_APRILTAG_36h11",
]

# Sensible defaults: a 5x7 board of 30 mm squares / 22 mm markers prints onto A4
# and gives plenty of corners. Marker length must be < square length.
DEFAULT_BOARD = {
    "squares_x": 5,
    "squares_y": 7,
    "square_mm": 30.0,
    "marker_mm": 22.0,
    "dictionary": "DICT_5X5_100",
}


def require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise CharucoError(
            "OpenCV is not available in this environment. Install/rebuild with "
            f"opencv-contrib-python-headless ({_CV2_IMPORT_ERROR})"
        )


def resolve_dictionary(name: str):
    """Look up a predefined ArUco dictionary by name."""
    require_cv2()
    if name not in DICT_NAMES:
        raise CharucoError(f"Unsupported dictionary '{name}'. Allowed: {DICT_NAMES}")
    const = getattr(cv2.aruco, name, None)
    if const is None:
        raise CharucoError(f"Dictionary '{name}' not present in this OpenCV build")
    return cv2.aruco.getPredefinedDictionary(const)


def build_board(squares_x: int, squares_y: int, square_mm: float, marker_mm: float, dictionary: str):
    """Build a CharucoBoard. Units are mm, so downstream poses are metric."""
    require_cv2()
    if squares_x < 3 or squares_y < 3:
        raise CharucoError("Board must be at least 3x3 squares")
    if marker_mm >= square_mm:
        raise CharucoError("marker_mm must be smaller than square_mm")
    aruco_dict = resolve_dictionary(dictionary)
    # (squares_x, squares_y) = (cols, rows).
    return cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)), float(square_mm), float(marker_mm), aruco_dict
    )


def build_board_from_config(cfg: Optional[dict]):
    """
    Build a board from a config mapping, falling back to DEFAULT_BOARD per key.

    Accepts the ``board`` block stored inside a calibration profile, so the fit path
    reconstructs the exact board the profile was calibrated against.
    """
    cfg = dict(cfg or {})
    merged = {**DEFAULT_BOARD, **{k: v for k, v in cfg.items() if k in DEFAULT_BOARD}}
    return build_board(
        int(merged["squares_x"]),
        int(merged["squares_y"]),
        float(merged["square_mm"]),
        float(merged["marker_mm"]),
        str(merged["dictionary"]),
    )


def detect_board(detector, gray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return ``(charuco_corners, charuco_ids)`` or ``(None, None)`` if nothing decoded."""
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) == 0:
        return None, None
    return charuco_corners, charuco_ids


def detect_board_detailed(detector, gray):
    """
    Full detector output: ``(charuco_corners, charuco_ids, marker_corners, marker_ids)``.

    The marker corners are what the AR path needs to mask the board out of the image before
    edge detection — a ChArUco board is a dense field of high-contrast edges sitting right
    next to the object, and feeding those to the fit gives it something wrong to latch onto.
    """
    return detector.detectBoard(gray)


def make_detector(board):
    """A CharucoDetector for *board*."""
    require_cv2()
    return cv2.aruco.CharucoDetector(board)
