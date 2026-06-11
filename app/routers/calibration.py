"""
Camera calibration (ChArUco) endpoints.

First building block of the AR weld-ratification tool: compute a camera's
intrinsic matrix + distortion coefficients from ChArUco board views, and persist
them as named camera profiles under ``outputs/calibration/`` for later use by the
pose/overlay pipeline.

Capture is done in the browser (live webcam via getUserMedia, or uploaded image
set); all detection + calibration runs here with OpenCV (contrib build, for the
aruco/charuco modules). Calibration is fast (small number of small images), so it
runs inline in the request — no subprocess worker needed (unlike OCC work).
"""
from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/calibration", tags=["calibration"])

# cv2 is imported lazily so the module still imports if the wheel is missing
# (e.g. a container built before opencv was added to requirements). Endpoints
# return a clear 503 in that case rather than crashing app startup.
try:  # pragma: no cover - import guard
    import cv2

    _CV2_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - import guard
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = str(exc)


# ── ChArUco board configuration ────────────────────────────────────────────

# Supported ArUco dictionaries (name → cv2 constant resolved lazily).
_DICT_NAMES = [
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

MIN_CORNERS_PER_VIEW = 6
MIN_VIEWS = 4
RECOMMENDED_VIEWS = 12

# Profile names: keep filesystem-safe.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")


def _require_cv2():
    if cv2 is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenCV is not available in this container. Rebuild the image so "
                f"opencv-contrib-python-headless is installed ({_CV2_IMPORT_ERROR})."
            ),
        )


def _resolve_dict(name: str):
    if name not in _DICT_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported dictionary '{name}'. Allowed: {_DICT_NAMES}")
    const = getattr(cv2.aruco, name, None)
    if const is None:
        raise HTTPException(status_code=400, detail=f"Dictionary '{name}' not present in this OpenCV build")
    return cv2.aruco.getPredefinedDictionary(const)


def _build_board(squares_x: int, squares_y: int, square_mm: float, marker_mm: float, dictionary: str):
    if squares_x < 3 or squares_y < 3:
        raise HTTPException(status_code=400, detail="Board must be at least 3x3 squares")
    if marker_mm >= square_mm:
        raise HTTPException(status_code=400, detail="marker_mm must be smaller than square_mm")
    aruco_dict = _resolve_dict(dictionary)
    # Board scale only affects extrinsics, not the intrinsic matrix; we keep mm so
    # any future pose use is metric. (squares_x, squares_y) = (cols, rows).
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), float(square_mm), float(marker_mm), aruco_dict
    )
    return board


def _decode_image(raw: bytes, field: str):
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Could not decode image '{field}'")
    return img


def _detect_charuco(detector, gray):
    """Return (charuco_corners, charuco_ids) or (None, None)."""
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) == 0:
        return None, None
    return charuco_corners, charuco_ids


# ── Profile persistence ─────────────────────────────────────────────────────

def _profiles_dir() -> str:
    d = settings.CALIBRATION_OUTPUT_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _profile_path(name: str) -> str:
    safe = name.replace(" ", "_")
    return os.path.join(_profiles_dir(), f"{safe}.json")


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Profile name must be 1-64 chars: letters, digits, space, _ . - (and start alphanumeric)",
        )
    return name


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/board-defaults")
async def board_defaults():
    """Recommended board parameters + supported dictionaries for the UI."""
    return {
        "default_board": DEFAULT_BOARD,
        "dictionaries": _DICT_NAMES,
        "min_views": MIN_VIEWS,
        "recommended_views": RECOMMENDED_VIEWS,
        "min_corners_per_view": MIN_CORNERS_PER_VIEW,
        "opencv_available": cv2 is not None,
    }


@router.get("/board")
async def board_image(
    squares_x: int = DEFAULT_BOARD["squares_x"],
    squares_y: int = DEFAULT_BOARD["squares_y"],
    square_mm: float = DEFAULT_BOARD["square_mm"],
    marker_mm: float = DEFAULT_BOARD["marker_mm"],
    dictionary: str = DEFAULT_BOARD["dictionary"],
    dpi: int = 300,
):
    """
    Render the ChArUco board as a PNG to print. IMPORTANT: print at 100% / actual
    size (no fit-to-page), then measure a printed square and use THAT measured
    value as square_mm when calibrating — printers rescale.
    """
    _require_cv2()
    board = _build_board(squares_x, squares_y, square_mm, marker_mm, dictionary)

    dpi = max(72, min(1200, int(dpi)))
    px_per_mm = dpi / 25.4
    width_px = int(round(squares_x * square_mm * px_per_mm))
    height_px = int(round(squares_y * square_mm * px_per_mm))
    margin_px = int(round(square_mm * px_per_mm * 0.25))  # quarter-square quiet border

    img = board.generateImage((width_px, height_px), marginSize=margin_px, borderBits=1)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode board image")

    board_w_mm = round(squares_x * square_mm, 1)
    board_h_mm = round(squares_y * square_mm, 1)
    headers = {
        "Content-Disposition": (
            f'inline; filename="charuco_{squares_x}x{squares_y}_{int(square_mm)}mm.png"'
        ),
        "X-Board-Width-mm": str(board_w_mm),
        "X-Board-Height-mm": str(board_h_mm),
    }
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png", headers=headers)


@router.post("/detect")
async def detect(
    frame: UploadFile = File(...),
    squares_x: int = Form(DEFAULT_BOARD["squares_x"]),
    squares_y: int = Form(DEFAULT_BOARD["squares_y"]),
    square_mm: float = Form(DEFAULT_BOARD["square_mm"]),
    marker_mm: float = Form(DEFAULT_BOARD["marker_mm"]),
    dictionary: str = Form(DEFAULT_BOARD["dictionary"]),
):
    """
    Lightweight single-frame detection for live capture feedback. Returns the
    number of ChArUco corners found and a coarse coverage metric (fraction of the
    board's interior corners seen), so the UI can guide the user before they keep
    a frame.
    """
    _require_cv2()
    board = _build_board(squares_x, squares_y, square_mm, marker_mm, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    raw = await frame.read()
    img = _decode_image(raw, frame.filename or "frame")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids = _detect_charuco(detector, gray)

    n = 0 if ids is None else int(len(ids))
    total_corners = (squares_x - 1) * (squares_y - 1)
    coverage = round(n / total_corners, 3) if total_corners else 0.0
    return {
        "detected": n >= MIN_CORNERS_PER_VIEW,
        "corners": n,
        "total_corners": total_corners,
        "coverage": coverage,
        "image_size": [int(img.shape[1]), int(img.shape[0])],
    }


@router.post("/compute")
async def compute(
    frames: List[UploadFile] = File(...),
    name: str = Form(...),
    squares_x: int = Form(DEFAULT_BOARD["squares_x"]),
    squares_y: int = Form(DEFAULT_BOARD["squares_y"]),
    square_mm: float = Form(DEFAULT_BOARD["square_mm"]),
    marker_mm: float = Form(DEFAULT_BOARD["marker_mm"]),
    dictionary: str = Form(DEFAULT_BOARD["dictionary"]),
    camera_label: str = Form(""),
):
    """
    Run the full calibration over a set of board images and persist a named
    profile. Detects ChArUco corners per frame, matches them to board object
    points, and calls cv2.calibrateCamera.
    """
    _require_cv2()
    name = _validate_name(name)
    board = _build_board(squares_x, squares_y, square_mm, marker_mm, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    obj_points: list = []
    img_points: list = []
    image_size = None
    per_view = []

    for f in frames:
        raw = await f.read()
        try:
            img = _decode_image(raw, f.filename or "frame")
        except HTTPException:
            per_view.append({"name": f.filename, "used": False, "reason": "undecodable"})
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            per_view.append({"name": f.filename, "used": False, "reason": "size-mismatch"})
            continue

        corners, ids = _detect_charuco(detector, gray)
        if ids is None or len(ids) < MIN_CORNERS_PER_VIEW:
            per_view.append({
                "name": f.filename, "used": False,
                "corners": 0 if ids is None else int(len(ids)),
                "reason": "too-few-corners",
            })
            continue

        op, ip = board.matchImagePoints(corners, ids)
        if op is None or len(op) < MIN_CORNERS_PER_VIEW:
            per_view.append({"name": f.filename, "used": False, "reason": "match-failed"})
            continue

        obj_points.append(op)
        img_points.append(ip)
        per_view.append({"name": f.filename, "used": True, "corners": int(len(ids))})

    if len(obj_points) < MIN_VIEWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(obj_points)} usable view(s); need at least {MIN_VIEWS} "
                f"(aim for ~{RECOMMENDED_VIEWS}). Capture the board from more angles "
                "with good coverage and lighting."
            ),
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # Per-view RMS reprojection error, to flag bad frames back to the user.
    # RMS over the view's points = ||observed - projected||_2 / sqrt(N), matching
    # the convention cv2.calibrateCamera uses for its overall RMS return value.
    per_used = [v for v in per_view if v.get("used")]
    for i, v in enumerate(per_used):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        err = float(cv2.norm(img_points[i], proj, cv2.NORM_L2) / np.sqrt(len(proj)))
        v["reproj_error_px"] = round(err, 3)

    K = np.asarray(K)
    dist = np.asarray(dist).ravel()
    profile = {
        "name": name,
        "camera_label": camera_label.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "rms_reproj_error_px": round(float(rms), 4),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "intrinsics": {
            "fx": round(float(K[0, 0]), 4),
            "fy": round(float(K[1, 1]), 4),
            "cx": round(float(K[0, 2]), 4),
            "cy": round(float(K[1, 2]), 4),
        },
        "board": {
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_mm": square_mm,
            "marker_mm": marker_mm,
            "dictionary": dictionary,
        },
        "views_used": len(obj_points),
        "views_total": len(frames),
        "per_view": per_view,
    }

    path = _profile_path(name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)

    logger.info(
        "calibration_computed",
        name=name, rms=profile["rms_reproj_error_px"],
        views_used=len(obj_points), views_total=len(frames),
    )
    return profile


@router.get("/profiles")
async def list_profiles():
    """List saved calibration profiles (summary only)."""
    d = _profiles_dir()
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), "r", encoding="utf-8") as fh:
                p = json.load(fh)
            out.append({
                "name": p.get("name", fn[:-5]),
                "camera_label": p.get("camera_label", ""),
                "created_at": p.get("created_at"),
                "rms_reproj_error_px": p.get("rms_reproj_error_px"),
                "image_size": p.get("image_size"),
                "views_used": p.get("views_used"),
            })
        except Exception:
            continue
    return {"profiles": out}


@router.get("/profiles/{name}")
async def get_profile(name: str):
    path = _profile_path(name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No calibration profile '{name}'")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@router.delete("/profiles/{name}")
async def delete_profile(name: str):
    path = _profile_path(name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No calibration profile '{name}'")
    os.remove(path)
    return {"deleted": name}
