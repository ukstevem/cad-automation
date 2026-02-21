"""
CNC shape analysis router.

Detects whether each CNC-classified part is a plate (→ DXF) or standard
structural section (→ NC1/DSTV), and provides download endpoints for the
generated output files.

Endpoints
---------
GET  /cnc-analysis/result/{filename}
     Return cached CNC analysis results (from sidecar JSON "cnc_analysis"
     key).  Returns 404 if not yet run.

POST /cnc-analysis/analyse/{filename}
     Body: {"ref_ids": ["0:1:1:2", ...], "member_ids": {"0:1:1:2": "Name"}}
     Start a background worker subprocess that analyses each ref_id.
     Returns {"cnc_task_id": "...", "status": "pending"}.

GET  /cnc-analysis/status/{task_id}
     Poll a running analysis task.
     Returns {"status": "completed", "results": {...}} when done.

GET  /cnc-analysis/download/{filename}/{safe_ref}/{ext}
     Download a generated DXF (ext=dxf) or NC1 (ext=nc1) file.
     safe_ref is ref_id with colons replaced by hyphens, e.g. "0-1-1-2".
"""
import asyncio
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import FileResponse, Response
import structlog

from app.config import settings
from app.services.task_manager import task_manager, TaskStatus

# Path to the standalone worker script
_CNC_WORKER = Path(__file__).parent.parent / "workers" / "analyse_cnc_parts.py"

# Hard timeout for a single CNC analysis run (seconds)
_CNC_TIMEOUT = 600  # 10 minutes

router = APIRouter()
logger = structlog.get_logger()


# ---------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------

def _analysis_json_path(filename: str) -> Path:
    """Per-file JSON sidecar used to cache all analysis results."""
    stem = Path(filename).stem
    return Path(settings.ANALYSIS_OUTPUT_DIR) / f"{stem}.json"


def _cnc_out_dir(filename: str) -> Path:
    """Root output directory for CNC files generated for this STEP file."""
    # Use the 8-character hex prefix that identifies this upload uniquely.
    stem = filename[:8]
    return Path(settings.OUTPUT_DIR) / "cnc" / stem


# ---------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------

def _load_cache(filename: str) -> Optional[dict]:
    path = _analysis_json_path(filename)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("cnc_cache_read_failed", filename=filename, error=str(e))
    return None


def _save_cnc_analysis(
    filename: str,
    results: dict,
    *,
    member_ids: Optional[Dict[str, str]] = None,
    parent_names: Optional[Dict[str, str]] = None,
    project_number: str = "",
    steel_grade: str = "",
) -> None:
    """Merge CNC results into the sidecar JSON, preserving other sections."""
    path = _analysis_json_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    existing["cnc_analysis"] = results
    if member_ids:
        existing["cnc_member_names"] = member_ids
    if parent_names:
        existing["cnc_parent_names"] = parent_names
    if project_number:
        existing["cnc_project_number"] = project_number
    if steel_grade:
        existing["cnc_steel_grade"] = steel_grade

    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info("cnc_analysis_cached", filename=filename, n_refs=len(results))


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------

@router.get("/cnc-analysis/result/{filename}")
async def get_cnc_result(filename: str) -> Dict[str, Any]:
    """Return cached CNC analysis results, or 404 if not yet run."""
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    cache = _load_cache(filename)
    if not cache or "cnc_analysis" not in cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "CNC analysis not yet run for this file", "filename": filename},
        )

    return {"filename": filename, "results": cache["cnc_analysis"]}


@router.post("/cnc-analysis/analyse/{filename}")
async def start_cnc_analysis(
    filename: str,
    body: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    """
    Start background CNC shape analysis for a list of XCAF ref_ids.

    Body:
        {
            "ref_ids":        ["0:1:1:2", "0:1:1:5", ...],
            "member_ids":     {"0:1:1:2": "Gusset Plate", "0:1:1:5": "Rafter"},
            "project_number": "C25001",
            "steel_grade":    "S275"
        }

    Always starts a fresh run (or returns the existing in-progress task id if
    one is already running for this file).
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    ref_ids: List[str] = body.get("ref_ids", [])
    member_ids: Dict[str, str] = body.get("member_ids", {})
    parent_names: Dict[str, str] = body.get("parent_names", {})
    project_number: str = body.get("project_number", "")
    steel_grade: str = body.get("steel_grade", "")

    if not ref_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ref_ids must not be empty"},
        )

    # Check for an existing in-progress task for this file
    existing = task_manager.get_tasks_for_file(filename, task_type="cnc_analysis")
    for task in existing:
        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            logger.info(
                "cnc_task_already_running",
                filename=filename,
                task_id=task.task_id,
            )
            return {"cnc_task_id": task.task_id, "status": task.status.value}

    # Prepare paths and arguments
    analysis_json_path = str(_analysis_json_path(filename))
    out_dir = _cnc_out_dir(filename)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_ids_json = json.dumps(ref_ids)
    member_ids_json = json.dumps(member_ids)
    parent_names_json = json.dumps(parent_names)
    # project_number / steel_grade are plain strings — pass directly as argv

    async def run_cnc_async(_progress_callback):
        """Run the CNC worker in a subprocess."""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(_CNC_WORKER),
                str(file_path),
                analysis_json_path,
                ref_ids_json,
                str(out_dir),
                member_ids_json,
                parent_names_json,
                project_number,
                steel_grade,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to launch CNC worker: {exc}")

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(_CNC_TIMEOUT),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"CNC analysis timed out after {_CNC_TIMEOUT // 60} minutes"
            )

        returncode = proc.returncode
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        if returncode != 0:
            stdout_tail = stdout.strip()[-500:] if stdout.strip() else ""
            err = (
                f"CNC worker failed (exit {returncode}): {stderr}"
                if stderr
                else f"CNC worker crashed (exit {returncode}), last output: {stdout_tail}"
            )
            logger.error(
                "cnc_worker_failed",
                filename=filename,
                returncode=returncode,
                stderr=stderr[:1000] if stderr else "",
                stdout_tail=stdout_tail,
            )
            raise RuntimeError(err)

        # Worker redirects structlog to stderr; find the first JSON object in stdout
        json_line = next(
            (line for line in stdout.splitlines() if line.strip().startswith("{")),
            stdout,
        )
        payload = json.loads(json_line)
        cnc_results: dict = payload.get("results", {})
        _save_cnc_analysis(
            filename, cnc_results,
            member_ids=member_ids,
            parent_names=parent_names,
            project_number=project_number,
            steel_grade=steel_grade,
        )
        return cnc_results

    task_id = task_manager.submit_async("cnc_analysis", filename, run_cnc_async)
    logger.info(
        "cnc_task_submitted",
        filename=filename,
        task_id=task_id,
        n_refs=len(ref_ids),
    )
    return {"cnc_task_id": task_id, "status": "pending"}


@router.get("/cnc-analysis/status/{task_id}")
async def get_cnc_status(task_id: str) -> Dict[str, Any]:
    """
    Poll a CNC analysis background task.

    While pending/running: returns {"status": "pending"|"running"}.
    On failure:            returns {"status": "failed", "error": "..."}.
    On completion:         returns {"status": "completed", "results": {...}}.
    """
    info = task_manager.get_task(task_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Task not found", "task_id": task_id},
        )

    if info.status == TaskStatus.FAILED:
        return {"status": "failed", "error": info.error or "CNC analysis failed"}

    if info.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        return {"status": info.status.value, "cnc_task_id": task_id}

    # Completed
    return {
        "status": "completed",
        "results": info.results or {},
        "filename": info.filename,
    }


@router.get("/cnc-analysis/download-all/{filename}/{ext}")
async def download_all_cnc_files(filename: str, ext: str) -> Response:
    """
    Download a ZIP archive of all generated DXF or NC1 files for this STEP file,
    plus a manifest.json with provenance metadata.

    ext must be "dxf" or "nc1".
    """
    if ext not in ("dxf", "nc1"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ext must be 'dxf' or 'nc1'"},
        )

    cache = _load_cache(filename)
    if not cache or "cnc_analysis" not in cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "CNC analysis not yet run for this file", "filename": filename},
        )

    cnc_results: dict = cache["cnc_analysis"]
    member_names: dict = cache.get("cnc_member_names", {})
    parent_names_cache: dict = cache.get("cnc_parent_names", {})
    project_number: str = cache.get("cnc_project_number", "")
    steel_grade: str = cache.get("cnc_steel_grade", "")

    path_key = "dxf_path" if ext == "dxf" else "nc1_path"

    # Collect (arc_name, file_path, meta) for each output file
    entries: List[tuple] = []

    def _collect(ref_id: str, result: dict, solid_idx: Optional[int]) -> None:
        path_str = result.get(path_key)
        if not path_str:
            return
        fp = Path(path_str)
        if not fp.exists():
            logger.warning("cnc_zip_file_missing", path=str(fp))
            return
        meta = {
            "filename": fp.name,
            "ref_id": ref_id,
            "solid_idx": solid_idx,
            "part_name": member_names.get(ref_id, ""),
            "parent_assembly": parent_names_cache.get(ref_id, ""),
            "type": result.get("type"),
            "dims_mm": result.get("dims"),
            "volume_mm3": result.get("volume_mm3"),
            "mass_kg": result.get("mass_kg"),
            "category": result.get("category"),
            "designation": result.get("designation"),
            "holes": result.get("holes"),
            "end_cuts": result.get("end_cuts"),
            "analysed_at": result.get("analysed_at"),
        }
        entries.append((fp.name, fp, meta))

    for ref_id, result in cnc_results.items():
        if result.get("type") == "multi_solid":
            for idx, solid in enumerate(result.get("solids", [])):
                _collect(ref_id, solid, idx)
        else:
            _collect(ref_id, result, None)

    if not entries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"No {ext.upper()} files found in this analysis"},
        )

    # Build manifest
    manifest = {
        "project_number": project_number,
        "steel_grade": steel_grade,
        "step_file": filename,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "files": [meta for _, _, meta in entries],
    }

    # Create ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        seen_names: dict = {}
        for arc_name, fp, _ in entries:
            # Deduplicate: if arc_name already used, append a counter
            if arc_name in seen_names:
                seen_names[arc_name] += 1
                stem = fp.stem
                suffix = fp.suffix
                arc_name = f"{stem}_{seen_names[arc_name]}{suffix}"
            else:
                seen_names[arc_name] = 0
            zf.write(str(fp), arcname=arc_name)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    buf.seek(0)

    stem = project_number or Path(filename).stem[:8]
    zip_name = f"{stem}-{ext.upper()}-files.zip"

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@router.get("/cnc-analysis/download/{filename}/{safe_ref}/{ext}")
async def download_cnc_file(
    filename: str,
    safe_ref: str,
    ext: str,
) -> FileResponse:
    """
    Download a generated DXF or NC1 file.

    Parameters
    ----------
    filename : str
        The uploaded STEP filename (with its 8-char prefix).
    safe_ref : str
        ref_id with colons replaced by hyphens (e.g. "0-1-1-2").
        For solid N>0 of a multi-solid: "0-1-1-2-s1".
    ext : str
        "dxf" or "nc1".
    """
    if ext not in ("dxf", "nc1"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ext must be 'dxf' or 'nc1'"},
        )

    # Validate safe_ref to prevent path traversal
    if not re.match(r'^[\w\-\.]+$', safe_ref):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid safe_ref format"},
        )

    # Resolve the actual file path from the cached result rather than
    # reconstructing it from the URL.  safe_ref encodes the ref_id
    # (colons → hyphens) with an optional "-s{N}" solid-body suffix.
    cache = _load_cache(filename)
    cnc_results: dict = cache.get("cnc_analysis", {}) if cache else {}

    # Decode safe_ref → ref_id + optional solid_idx
    m = re.match(r'^(.+?)(?:-s(\d+))?$', safe_ref)
    ref_id_candidate = (m.group(1) if m else safe_ref).replace("-", ":")
    solid_idx = int(m.group(2)) if (m and m.group(2)) else None

    result = cnc_results.get(ref_id_candidate)
    if result and result.get("type") == "multi_solid" and solid_idx is not None:
        solids = result.get("solids", [])
        result = solids[solid_idx] if solid_idx < len(solids) else None

    path_key = "dxf_path" if ext == "dxf" else "nc1_path"
    stored_path = result.get(path_key) if result else None

    if stored_path:
        file_path = Path(stored_path)
    else:
        # Fallback: try the legacy safe_ref-based path
        out_dir = _cnc_out_dir(filename)
        sub = "plates" if ext == "dxf" else "nc1"
        file_path = out_dir / sub / f"{safe_ref}.{ext}"

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Output file not found", "path": str(file_path)},
        )

    media_type = "application/dxf" if ext == "dxf" else "text/plain"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )
