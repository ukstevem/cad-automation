"""
AR alignment endpoints (Phase 0 spike).

Two pieces:
- GET  /ar/geometry/{filename}  → the part's CAD edges + corner anchors in world
  coords (runs the extract_edges worker, cached in the sidecar).
- POST /ar/solve/{filename}     → given a calibration profile + 2D↔3D correspondences
  (clicked photo points ↔ CAD corner anchors), solve the camera pose and project the
  CAD edges to 2D overlay polylines to draw on the photo.

This proves the alignment chain on a single uploaded section: does the CAD project
onto the real photo? (A lone section has no welds — this tests pose/overlay only.)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import base64

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import settings
from app.services import pose as P

try:  # cv2 for marker generation (already a dep)
    import cv2
    _CV2_ERR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_ERR = str(exc)

_MARKER_DICTS = {
    "DICT_APRILTAG_36h11": "AprilTag 36h11",
    "DICT_4X4_50": "ArUco 4x4",
    "DICT_5X5_100": "ArUco 5x5",
    "DICT_6X6_250": "ArUco 6x6",
}

router = APIRouter(prefix="/ar", tags=["ar"])
logger = structlog.get_logger()

_WORKER = Path(__file__).parent.parent / "workers" / "extract_edges.py"
_TIMEOUT = 300


def _analysis_json_path(filename: str) -> Path:
    return Path(settings.ANALYSIS_OUTPUT_DIR) / f"{Path(filename).stem}.json"


def _profile_path(name: str) -> Path:
    return Path(settings.CALIBRATION_OUTPUT_DIR) / f"{name.replace(' ', '_')}.json"


async def _run_edge_worker(file_path: Path, analysis_json: Path, node_id: Optional[str]) -> dict:
    args = [sys.executable, str(_WORKER), str(file_path), str(analysis_json), node_id or ""]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=float(_TIMEOUT))
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise HTTPException(504, "Edge extraction timed out")
    out = out_b.decode("utf-8", errors="replace")
    err = err_b.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        logger.error("ar_edge_worker_failed", returncode=proc.returncode, stderr=err[:1000])
        raise HTTPException(500, f"Edge extraction failed: {err or 'unknown error'}")
    line = next((l for l in out.splitlines() if l.strip().startswith("{")), None)
    if not line:
        raise HTTPException(500, "Edge worker produced no JSON output")
    return json.loads(line)


async def _get_geometry(filename: str, node_id: Optional[str], refresh: bool = False) -> dict:
    file_path = Path(settings.UPLOAD_DIR) / filename
    analysis_json = _analysis_json_path(filename)
    if not file_path.exists():
        raise HTTPException(404, f"Uploaded file not found: {filename}")
    if not analysis_json.exists():
        raise HTTPException(404, "No analysis for this file yet — run analysis first")

    sidecar = json.loads(analysis_json.read_text(encoding="utf-8"))
    cache_key = node_id or "_all"
    cached = (sidecar.get("ar_geometry") or {}).get(cache_key)
    if cached and not refresh:
        return cached

    result = await _run_edge_worker(file_path, analysis_json, node_id)
    sidecar.setdefault("ar_geometry", {})[cache_key] = result
    analysis_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return result


@router.get("/geometry/{filename}")
async def geometry(filename: str, node_id: Optional[str] = None, refresh: bool = False):
    """CAD edges + corner anchors (world coords) for the part. Cached in the sidecar."""
    return await _get_geometry(filename, node_id, refresh=refresh)


class Correspondence(BaseModel):
    image: List[float]   # [u, v] pixel
    world: List[float]   # [x, y, z] mm (a CAD corner anchor)


class SolveRequest(BaseModel):
    profile: str
    correspondences: List[Correspondence]
    node_id: Optional[str] = None
    ransac: bool = False


@router.post("/solve/{filename}")
async def solve(filename: str, req: SolveRequest):
    """
    Solve camera pose from clicked correspondences and return the CAD edges
    projected to 2D overlay polylines (plus pose + reprojection error).
    """
    prof_path = _profile_path(req.profile)
    if not prof_path.exists():
        raise HTTPException(404, f"No calibration profile '{req.profile}'")
    profile = json.loads(prof_path.read_text(encoding="utf-8"))
    K, dist = P.load_intrinsics(profile)

    obj = [c.world for c in req.correspondences]
    img = [c.image for c in req.correspondences]
    if len(obj) < 4:
        raise HTTPException(400, f"Need at least 4 correspondences, got {len(obj)}")

    try:
        rvec, tvec, inliers = P.solve_pose(obj, img, K, dist, ransac=req.ransac)
    except P.PoseError as exc:
        raise HTTPException(400, str(exc))

    rms = P.reprojection_rms(obj, img, rvec, tvec, K, dist)
    geom = await _get_geometry(filename, req.node_id)
    overlay = P.project_polylines(geom.get("edges", []), rvec, tvec, K, dist)
    cam = P.camera_position_world(rvec, tvec)

    return {
        "reproj_rms": round(rms, 3),
        "camera_position": [round(float(x), 1) for x in cam],
        "rvec": [round(float(x), 6) for x in rvec.ravel()],
        "tvec": [round(float(x), 3) for x in tvec.ravel()],
        "overlay": overlay,
        "inliers": inliers.ravel().tolist() if inliers is not None else None,
        "image_size": profile.get("image_size"),
        "n_correspondences": len(obj),
    }


@router.get("/markers.pdf")
def markers_pdf(
    dictionary: str = "DICT_APRILTAG_36h11",
    start: int = 0,
    count: int = 6,
    size_mm: float = 100.0,
    cols: int = 2,
):
    """
    Printable sheet of uniquely-numbered AR placement markers, each at an exact
    physical size. Print at 100% / actual size, then measure a printed marker —
    that measured edge length (black square) is what the pose solver uses.

    The marker ID is load-bearing: the placement plan maps ID → CAD datum, so each
    physical marker must keep its number.
    """
    if cv2 is None:
        raise HTTPException(503, f"OpenCV unavailable: {_CV2_ERR}")
    if dictionary not in _MARKER_DICTS:
        raise HTTPException(400, f"Unsupported dictionary. Allowed: {list(_MARKER_DICTS)}")
    const = getattr(cv2.aruco, dictionary, None)
    if const is None:
        raise HTTPException(400, f"Dictionary '{dictionary}' not in this OpenCV build")
    if count < 1 or count > 100:
        raise HTTPException(400, "count must be 1..100")

    adict = cv2.aruco.getPredefinedDictionary(const)
    label = _MARKER_DICTS[dictionary]

    cells = []
    for i in range(count):
        mid = start + i
        img = cv2.aruco.generateImageMarker(adict, mid, 1000)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise HTTPException(500, f"Failed to render marker {mid}")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        cells.append(
            f'<div class="cell">'
            f'<div class="mwrap"><img src="data:image/png;base64,{b64}"></div>'
            f'<div class="lbl">{label} &middot; ID {mid} &middot; {size_mm:g} mm</div>'
            f'</div>'
        )

    pad = size_mm * 0.07
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
        @page {{ size: A4; margin: 8mm; }}
        body {{ font-family: sans-serif; margin: 0; }}
        h1 {{ font-size: 12pt; margin: 0 0 2mm; }}
        .note {{ font-size: 9pt; color: #444; margin-bottom: 4mm; }}
        /* inline-block + break-inside:avoid paginates cleanly in weasyprint;
           flex does not (it split markers across page breaks). */
        .cell {{
            display: inline-block; vertical-align: top; text-align: center;
            width: {size_mm + 2 * pad:g}mm; margin: 2mm;
            break-inside: avoid; page-break-inside: avoid;
        }}
        .mwrap {{ background: #fff; padding: {pad:g}mm; display: inline-block;
            break-inside: avoid; page-break-inside: avoid; }}
        .mwrap img {{ width: {size_mm:g}mm; height: {size_mm:g}mm; image-rendering: pixelated; display: block; }}
        .lbl {{ font-size: 8pt; margin-top: 1.5mm; }}
    </style></head><body>
        <h1>AR placement markers — {label}</h1>
        <div class="note">Print at <b>100% / actual size</b> (no fit-to-page). Measure a printed
        black marker; it should be <b>{size_mm:g} mm</b>. Mount flat &amp; rigid. Keep each ID with its
        planned location.</div>
        <div>{''.join(cells)}</div>
    </body></html>"""

    from weasyprint import HTML
    pdf = HTML(string=html).write_pdf()
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="ar_markers_{dictionary}_{start}-{start+count-1}.pdf"'},
    )


def _run_id_of(filename: str) -> str:
    """8-hex run prefix from the stored upload filename."""
    return Path(filename).stem.split("_", 1)[0]


@router.post("/capture/{filename}")
async def save_capture(
    filename: str,
    photo: UploadFile = File(...),
    profile: str = Form(...),
    correspondences: str = Form("[]"),
    pose: str = Form("{}"),
    reproj_rms: float = Form(0.0),
    node_id: str = Form(""),
):
    """
    Persist a capture as a permanent, fully-described record (ML-readiness item 1
    = provenance, item 2 = label quality). Local storage for the spike:
    ``outputs/captures/<run_id>/<capture_id>/`` holding the original photo + a
    metadata JSON. Migrates to Supabase (table + Storage) later.
    """
    prof_path = _profile_path(profile)
    profile_data = json.loads(prof_path.read_text(encoding="utf-8")) if prof_path.exists() else {}

    raw = await photo.read()
    run_id = _run_id_of(filename)
    cap_id = hashlib.sha1(os.urandom(16)).hexdigest()[:12]
    out_dir = Path(settings.OUTPUT_DIR) / "captures" / run_id / cap_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = (Path(photo.filename or "photo.jpg").suffix or ".jpg").lower()
    photo_name = f"photo{ext}"
    (out_dir / photo_name).write_bytes(raw)

    record = {
        "capture_id": cap_id,
        "run_id": run_id,
        "source_filename": filename,
        "node_id": node_id or None,
        # ── provenance (item 1) ──
        "photo_file": photo_name,
        "photo_original_name": photo.filename,
        "photo_bytes": len(raw),
        "photo_sha256": hashlib.sha256(raw).hexdigest(),
        "calibration_profile": profile,
        "image_size": profile_data.get("image_size"),
        "intrinsics": profile_data.get("intrinsics"),
        "correspondences": json.loads(correspondences),
        "pose": json.loads(pose),
        # ── label quality (item 2) ──
        "reproj_rms_px": reproj_rms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    logger.info("ar_capture_saved", run_id=run_id, capture_id=cap_id, reproj_rms=reproj_rms)
    return {"capture_id": cap_id, "run_id": run_id, "stored": str(out_dir), "record": record}
