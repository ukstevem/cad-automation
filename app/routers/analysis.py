"""
Analysis router - assembly tree inspection, caching, and project state persistence
"""
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, status
import structlog

from app.config import settings
from app.services.task_manager import task_manager, TaskStatus

# Path to the standalone analysis worker script
_WORKER = Path(__file__).parent.parent / "workers" / "analyze_step.py"

# Hard timeout for a single XCAF parse (seconds)
_ANALYSIS_TIMEOUT = 600  # 10 minutes

router = APIRouter()
logger = structlog.get_logger()

STEP_EXTENSIONS = {".step", ".stp"}

# Uploaded files are stored as "<8-hex-chars>_<original_name>"
_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_")


def _original_name(filename: str) -> str:
    """Strip the unique-id prefix to recover the original filename."""
    return _PREFIX_RE.sub("", filename)


# ---------------------------------------------------------------
# Analysis cache helpers
# ---------------------------------------------------------------

def _analysis_json_path(filename: str) -> Path:
    """Per-file JSON path for cached analysis + project state."""
    stem = Path(filename).stem
    return Path(settings.ANALYSIS_OUTPUT_DIR) / f"{stem}.json"


def _load_cache(filename: str) -> Optional[dict]:
    path = _analysis_json_path(filename)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("cache_read_failed", filename=filename, error=str(e))
    return None


def _save_analysis(filename: str, analysis_result: dict):
    """Save analysis result to cache, preserving any existing project_state."""
    path = _analysis_json_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    existing["analysis"] = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        **analysis_result,
    }
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info("analysis_cached", filename=filename, path=str(path))


def _save_project_state(filename: str, state: dict):
    """Update the project_state section of the cache, preserving analysis."""
    path = _analysis_json_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    existing["project_state"] = state
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _build_analysis_response(
    filename: str,
    analysis: dict,
    project_state: Optional[dict],
) -> dict:
    """Assemble the full successful analysis response dict."""
    response: dict = {
        "success": True,
        "filename": filename,
        **analysis,
    }
    if project_state:
        response["project_state"] = project_state
    return response


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------

@router.get("/analysis/files")
async def list_uploaded_files() -> Dict[str, Any]:
    """List STEP files available for analysis, sorted newest-first."""
    upload_dir = Path(settings.UPLOAD_DIR)
    if not upload_dir.exists():
        return {"files": []}

    files: List[Dict[str, Any]] = []
    for entry in upload_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in STEP_EXTENSIONS:
            stat = entry.stat()
            uploaded_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            files.append({
                "filename": entry.name,
                "display_name": _original_name(entry.name),
                "size_bytes": stat.st_size,
                "uploaded_at": uploaded_at.isoformat(),
            })

    # Newest first
    files.sort(key=lambda f: f["uploaded_at"], reverse=True)

    return {"files": files}


@router.get("/analysis/assembly/{filename}")
async def get_assembly_tree(filename: str) -> Dict[str, Any]:
    """
    Return the assembly tree for a STEP file.

    Cache hit  → returns full assembly data immediately.
    Cache miss → submits background XCAF parse task and returns
                 {"analysis_task_id": "...", "status": "pending"} immediately.
                 The frontend should poll GET /analysis/status/{task_id}.
    """
    file_path = Path(settings.UPLOAD_DIR) / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    # --- Cache hit: return immediately ---
    cache = _load_cache(filename)
    if cache and "analysis" in cache:
        logger.info("analysis_cache_hit", filename=filename)
        return _build_analysis_response(
            filename, cache["analysis"], cache.get("project_state")
        )

    # --- Cache miss: check for an existing in-progress task ---
    logger.info("analysis_cache_miss", filename=filename)
    existing = task_manager.get_tasks_for_file(filename, task_type="assembly_analysis")
    for task in existing:
        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            logger.info("analysis_task_already_running", filename=filename, task_id=task.task_id)
            return {"analysis_task_id": task.task_id, "status": task.status.value}

    # --- Submit a new background analysis task (truly async, no GIL blocking) ---
    async def run_analysis_async(_progress_callback):
        """
        Run XCAF analysis in a subprocess using asyncio.create_subprocess_exec().
        await proc.communicate() yields to the event loop while waiting, so
        uvicorn remains fully responsive to poll requests during the parse.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(_WORKER), str(file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to launch analysis worker: {exc}")

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(_ANALYSIS_TIMEOUT),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"Assembly analysis timed out after {_ANALYSIS_TIMEOUT // 60} minutes"
            )

        returncode = proc.returncode
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        if returncode != 0:
            stdout_tail = stdout.strip()[-500:] if stdout.strip() else ""
            if stderr:
                err = f"Worker failed (exit {returncode}): {stderr}"
            elif stdout_tail:
                err = f"Worker crashed (exit {returncode}), last output: {stdout_tail}"
            else:
                err = f"Worker crashed with no output (exit {returncode})"
            logger.error(
                "analysis_worker_failed",
                filename=filename,
                returncode=returncode,
                stderr=stderr[:1000] if stderr else "",
                stdout_tail=stdout_tail,
            )
            raise RuntimeError(err)

        # The worker redirects structlog to stderr; find the first JSON line in stdout
        json_line = next(
            (line for line in stdout.splitlines() if line.strip().startswith("{")),
            stdout,
        )
        result = json.loads(json_line)
        _save_analysis(filename, result)
        return result

    task_id = task_manager.submit_async("assembly_analysis", filename, run_analysis_async)
    logger.info("analysis_task_submitted", filename=filename, task_id=task_id)
    return {"analysis_task_id": task_id, "status": "pending"}


@router.get("/analysis/status/{task_id}")
async def get_analysis_status(task_id: str) -> Dict[str, Any]:
    """
    Poll the status of a background assembly analysis task.

    While pending/running: returns {"status": "pending"|"running"}.
    On failure:            returns {"status": "failed", "error": "..."}.
    On completion:         returns the full assembly tree response
                           (same shape as a cache-hit from get_assembly_tree),
                           including auto-triggered STL task_id.
    """
    info = task_manager.get_task(task_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Task not found", "task_id": task_id},
        )

    if info.status == TaskStatus.FAILED:
        return {"status": "failed", "error": info.error or "Analysis failed"}

    if info.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        return {"status": info.status.value, "analysis_task_id": task_id}

    # Completed — build and return the full response
    analysis = info.results  # dict returned by AssemblyAnalyzer.analyze()
    cache = _load_cache(info.filename)
    project_state = cache.get("project_state") if cache else None

    response = _build_analysis_response(info.filename, analysis, project_state)
    # The frontend polls on status.status === 'completed' — make sure the field is present
    response["status"] = "completed"
    return response


@router.put("/analysis/project-state/{filename}")
async def save_project_state(
    filename: str,
    body: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    """
    Save project state (classifications, exploded nodes, STL map, solid children) for a file.

    Body:
        {
            "classifications": {"0:1:1:2": "postprocess", ...},
            "exploded_nodes": ["0:1:1:1", ...],
            "stl_map": {"0:1:1:2": "/outputs/stl/<runid>/Part.stl", ...},
            "solid_children": {"0:1:2": [{"name": "Part - Solid 1", "nodeId": "0:1:2:s0"}, ...]}
        }
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    state = {
        "classifications": body.get("classifications", {}),
        "exploded_nodes": body.get("exploded_nodes", []),
        "stl_map": body.get("stl_map", {}),
        "solid_children": body.get("solid_children", {}),
    }
    _save_project_state(filename, state)

    return {"success": True}
