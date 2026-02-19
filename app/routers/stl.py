"""
STL generation router - trigger, poll status, list/download generated STL files

STL generation runs in a subprocess (via asyncio.create_subprocess_exec) so that
OCC/OCP mesh generation cannot hold the GIL and block the event loop.  A blocked
event loop prevents the static-file server from responding to Three.js STL fetch
requests, causing the viewer to appear permanently stuck on "loading".
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
import structlog

from app.config import settings
from app.services.task_manager import task_manager, TaskStatus

router = APIRouter()
logger = structlog.get_logger()

_STL_WORKER = Path(__file__).parent.parent / "workers" / "generate_stl.py"
_STL_TIMEOUT = 600  # seconds


def _stl_output_dir(filename: str) -> Path:
    """Build the per-file STL output directory using the 8-char hex run ID."""
    run_id = filename[:8]
    return Path(settings.STL_OUTPUT_DIR) / run_id


async def _run_stl_subprocess(
    mode: str,
    file_path: Path,
    output_dir: Path,
    node_id: str = None,
) -> List[Dict[str, Any]]:
    """
    Run STL generation in a subprocess using asyncio.create_subprocess_exec().

    await proc.communicate() yields the event loop while OCC meshes shapes,
    keeping uvicorn fully responsive so Three.js can fetch generated STL files
    as soon as each one is written.
    """
    cmd = [sys.executable, str(_STL_WORKER), mode, str(file_path), str(output_dir)]
    if node_id:
        cmd.append(node_id)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to launch STL worker: {exc}")

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=float(_STL_TIMEOUT),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"STL generation timed out after {_STL_TIMEOUT // 60} minutes")

    returncode = proc.returncode
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

    if returncode != 0:
        raise RuntimeError(
            f"STL worker failed (exit {returncode}): {stderr}" if stderr
            else f"STL worker crashed with no output (exit {returncode})"
        )

    # Worker prints JSON array; find the first line that starts with '['
    json_line = next(
        (line for line in stdout.splitlines() if line.strip().startswith("[")),
        stdout,
    )
    return json.loads(json_line)


@router.post("/stl/generate/{filename}")
async def generate_stl(filename: str) -> Dict[str, Any]:
    """
    Trigger STL generation for root-level assembly items.

    Idempotent: returns existing task if one is already running or completed.
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    existing = task_manager.get_tasks_for_file(filename, task_type="stl_generation")
    for task in existing:
        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED):
            return {"task_id": task.task_id, "status": task.status.value, "existing": True}

    output_dir = _stl_output_dir(filename)

    async def run(_progress_callback):
        return await _run_stl_subprocess("all", file_path, output_dir)

    task_id = task_manager.submit_async("stl_generation", filename, run)
    return {"task_id": task_id, "status": "pending", "existing": False}


@router.post("/stl/generate-children/{filename}")
async def generate_stl_children(filename: str, parent_id: str) -> Dict[str, Any]:
    """
    Trigger STL generation for children of a specific assembly node.

    Query params:
        parent_id: XCAF label entry string (e.g. "0:1:1:3")

    Idempotent: returns existing task if one matches.
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    task_type = f"stl_explode:{parent_id}"

    existing = task_manager.get_tasks_for_file(filename, task_type=task_type)
    for task in existing:
        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED):
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "parent_id": parent_id,
                "existing": True,
            }

    output_dir = _stl_output_dir(filename)

    async def run(_progress_callback):
        return await _run_stl_subprocess("children", file_path, output_dir, parent_id)

    task_id = task_manager.submit_async(task_type, filename, run)
    return {
        "task_id": task_id,
        "status": "pending",
        "parent_id": parent_id,
        "existing": False,
    }


@router.post("/stl/generate-solids/{filename}")
async def generate_stl_solids(filename: str, node_id: str) -> Dict[str, Any]:
    """
    Trigger STL generation for individual solids in a multi-solid part.

    Query params:
        node_id: XCAF label entry string of the multi-solid part

    Idempotent: returns existing task if one matches.
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    task_type = f"stl_solids:{node_id}"

    existing = task_manager.get_tasks_for_file(filename, task_type=task_type)
    for task in existing:
        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED):
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "node_id": node_id,
                "existing": True,
            }

    output_dir = _stl_output_dir(filename)

    async def run(_progress_callback):
        return await _run_stl_subprocess("solids", file_path, output_dir, node_id)

    task_id = task_manager.submit_async(task_type, filename, run)
    return {
        "task_id": task_id,
        "status": "pending",
        "node_id": node_id,
        "existing": False,
    }


@router.get("/stl/status/{task_id}")
async def get_stl_status(task_id: str) -> Dict[str, Any]:
    """Poll the progress of an STL generation task."""
    info = task_manager.get_task(task_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Task not found", "task_id": task_id},
        )
    return info.to_dict()


@router.get("/stl/files/{filename}")
async def list_stl_files(filename: str) -> Dict[str, Any]:
    """List generated STL files for a given upload."""
    output_dir = _stl_output_dir(filename)
    if not output_dir.exists():
        return {"files": [], "filename": filename}

    files = []
    for stl in sorted(output_dir.glob("*.stl")):
        files.append({
            "name": stl.stem,
            "filename": stl.name,
            "url": f"/outputs/stl/{output_dir.name}/{stl.name}",
            "size_bytes": stl.stat().st_size,
        })

    return {"files": files, "filename": filename}
