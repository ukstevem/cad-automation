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
import json
import sys
from pathlib import Path
from typing import List, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services import pose as P

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
