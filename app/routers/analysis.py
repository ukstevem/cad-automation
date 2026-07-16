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
from app.services.bom_builder import build_step_bom
from app.services.sidecar import (
    SidecarCorruptError,
    atomic_write_json,
    load_sidecar,
)
from app.services.bom_manifest import pin_bom_items
from app.services.task_manager import task_manager, TaskStatus

# Path to the standalone worker scripts
_WORKER = Path(__file__).parent.parent / "workers" / "analyze_step.py"
_CONSOLIDATE_WORKER = Path(__file__).parent.parent / "workers" / "consolidate_parts.py"

# Hard timeout for a single XCAF parse (seconds)
_ANALYSIS_TIMEOUT = 600  # 10 minutes
_CONSOLIDATE_TIMEOUT = 600  # 10 minutes

router = APIRouter()
logger = structlog.get_logger()

STEP_EXTENSIONS = {".step", ".stp"}
# Extensions shown on the Analysis file list (STEP + IFC).  Kept separate from
# STEP_EXTENSIONS so code that specifically needs STEP-only behaviour still has
# the narrow set available.
ANALYSABLE_EXTENSIONS = STEP_EXTENSIONS | {".ifc"}

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
    try:
        data = load_sidecar(path)
    except SidecarCorruptError as e:
        # Corrupt file has been moved aside (.corrupt) so it is preserved and
        # the next write starts clean — treat as a cache miss so the analysis
        # is regenerated rather than reading garbage or clobbering on write.
        logger.warning("cache_read_failed", filename=filename, error=str(e))
        return None
    return data or None


def _save_analysis(filename: str, analysis_result: dict):
    """Save analysis result to cache, preserving any existing project_state."""
    path = _analysis_json_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Raises SidecarCorruptError (preserving the bad file) rather than falling
    # back to {} — that fallback would write back only this section and destroy
    # every other section (analysis/project_state/cnc_*). Better to fail loudly.
    existing = load_sidecar(path)

    existing["analysis"] = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        **analysis_result,
    }

    # STEP results don't include native_bom (no metadata to build one from at
    # parse time). Synthesise placeholder rows from the tree so every leaf
    # part appears in the BOM immediately; CNC analysis later enriches them.
    # IFC results already carry native_bom from the parser — leave untouched.
    if "native_bom" not in analysis_result:
        tree = analysis_result.get("assembly_tree") or []
        existing["analysis"]["native_bom"] = build_step_bom(
            tree,
            cnc_analysis=existing.get("cnc_analysis"),
            cnc_member_names=existing.get("cnc_member_names"),
            parent_names=existing.get("cnc_parent_names"),
            steel_grade=existing.get("cnc_steel_grade", ""),
            project_number=existing.get("cnc_project_number", ""),
        )

    atomic_write_json(path, existing)
    logger.info("analysis_cached", filename=filename, path=str(path))


def _save_project_state(filename: str, state: dict):
    """Update the project_state section of the cache, preserving analysis.

    Also embeds solid_children into assembly_tree nodes so the tree is
    self-contained for portal consumers.
    """
    path = _analysis_json_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Raises SidecarCorruptError (preserving the bad file) rather than falling
    # back to {} — that fallback would write back only this section and destroy
    # every other section (analysis/project_state/cnc_*). Better to fail loudly.
    existing = load_sidecar(path)

    existing["project_state"] = state
    _embed_solid_children(existing)
    # Pin BOM item numbers as soon as they're issued.  They're allocated from
    # the classifications' insertion order, so without a pin every re-classify
    # renumbers everything after the first change — invalidating an already-
    # issued BOM and the B###.nc1 files named after it.  Best-effort: a BOM
    # numbering failure must never cost the user their project state.
    try:
        pin_bom_items(existing)
    except Exception as e:
        logger.warning("bom_pin_failed", filename=filename, error=str(e))
    atomic_write_json(path, existing)


def _embed_solid_children(cache: dict):
    """Merge solid_children from project_state into assembly_tree nodes.

    For each ``part_multi_solid`` node, looks up its solid children by chiral
    key and writes them as proper child nodes with ``node_type: "solid"``.
    Nodes whose chiral key has no entry in solid_children are left unchanged
    (children stay as-is — typically ``[]``).
    """
    tree = (cache.get("analysis") or {}).get("assembly_tree")
    solid_children = (cache.get("project_state") or {}).get("solid_children", {})
    if not tree or not solid_children:
        return

    def _walk(nodes):
        for node in nodes:
            if node.get("node_type") == "part_multi_solid":
                ref_id = node.get("ref_id", node.get("id", ""))
                suffix = "M" if node.get("is_mirrored") else "N"
                children_info = solid_children.get(f"{ref_id}:{suffix}")
                if children_info:
                    node["children"] = [
                        {
                            "id": child["nodeId"],
                            "name": child["name"],
                            "node_type": "solid",
                            "ref_id": child["nodeId"],
                            "solid_count": 0,
                            "children": [],
                        }
                        for child in children_info
                    ]
            if node.get("children"):
                _walk(node["children"])

    _walk(tree)


def _save_consolidation(filename: str, result: dict):
    """Update the consolidation section of the cache, preserving other sections.

    ``result`` is the dict returned by PartsConsolidator.consolidate():
    ``{"part_groups": [...], "solid_groups": [...], "intra_solid_groups": [...]}``
    """
    path = _analysis_json_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Raises SidecarCorruptError (preserving the bad file) rather than falling
    # back to {} — that fallback would write back only this section and destroy
    # every other section (analysis/project_state/cnc_*). Better to fail loudly.
    existing = load_sidecar(path)

    # Stamp the result so the CNC pipeline can detect when consolidation has
    # been re-run since the last analysis (stale-CNC state).
    result = dict(result)
    result["consolidated_at"] = datetime.now(timezone.utc).isoformat()

    existing["consolidation"] = result
    atomic_write_json(path, existing)
    logger.info(
        "consolidation_cached",
        filename=filename,
        part_groups=len(result.get("part_groups", [])),
        solid_groups=len(result.get("solid_groups", [])),
        intra_solid_groups=len(result.get("intra_solid_groups", [])),
    )


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _build_analysis_response(
    filename: str,
    analysis: dict,
    project_state: Optional[dict],
    classification: Optional[dict] = None,
) -> dict:
    """Assemble the full successful analysis response dict."""
    response: dict = {
        "success": True,
        "filename": filename,
        **analysis,
    }
    if project_state:
        response["project_state"] = project_state
    # The lightweight refined-class cache (ref_id → {refined_class, ...}). The
    # frontend loads this so the BOM's Refined column + auto-classify reflect
    # prior detection on reopen, instead of showing "—" until a re-detect.
    if classification:
        response["classification"] = classification
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
        if entry.is_file() and entry.suffix.lower() in ANALYSABLE_EXTENSIONS:
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
            filename, cache["analysis"], cache.get("project_state"),
            cache.get("classification"),
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
    classification = cache.get("classification") if cache else None

    response = _build_analysis_response(
        info.filename, analysis, project_state, classification)
    # The frontend polls on status.status === 'completed' — make sure the field is present
    response["status"] = "completed"
    return response


def _collect_parts(nodes: List[dict], parent_name: Optional[str], result: dict) -> None:
    """
    Recursively walk an assembly tree and collect all parts (non-assemblies) by ref_id.

    result is modified in place: ref_id -> {ref_id, name, node_type, instances[]}
    where each instance has {node_id, parent_name, is_mirrored}.
    """
    for node in nodes:
        if node.get("node_type") == "assembly":
            _collect_parts(node.get("children", []), node["name"], result)
        else:
            ref_id = node.get("ref_id") or node["id"]
            if ref_id not in result:
                result[ref_id] = {
                    "ref_id": ref_id,
                    "name": node["name"],
                    "node_type": node.get("node_type", "part_single_solid"),
                    "instances": [],
                }
            result[ref_id]["instances"].append({
                "node_id": node["id"],
                "parent_name": parent_name,
                "is_mirrored": node.get("is_mirrored", False),
            })


@router.get("/analysis/parts-list/{filename}")
async def get_parts_list(filename: str) -> Dict[str, Any]:
    """
    Return a consolidated flat list of all unique parts in the assembly.

    Walks the cached analysis tree (no STEP re-parse), deduplicates by ref_id,
    and annotates each part with its classification (from project_state) and
    the list of parent assemblies it appears in.

    Used to build the full BOM even before the user has manually exploded
    every assembly — the server-side tree walk covers all levels.
    """
    cache = _load_cache(filename)
    if not cache or "analysis" not in cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Analysis not cached for this file", "filename": filename},
        )

    tree = cache["analysis"].get("assembly_tree", [])
    classifications: dict = (cache.get("project_state") or {}).get("classifications", {})

    # Walk the full tree
    parts_by_ref: dict = {}
    _collect_parts(tree, None, parts_by_ref)

    result = []
    for info in parts_by_ref.values():
        node_ids = [i["node_id"] for i in info["instances"]]

        # Any instance classified counts for the whole part
        classification = next(
            (classifications[nid] for nid in node_ids if nid in classifications),
            None,
        )

        mirrored_count = sum(1 for i in info["instances"] if i.get("is_mirrored"))

        # Ordered unique list of parent assemblies
        seen: set = set()
        parent_assemblies = []
        for inst in info["instances"]:
            pn = inst.get("parent_name")
            if pn and pn not in seen:
                seen.add(pn)
                parent_assemblies.append(pn)

        result.append({
            "ref_id": info["ref_id"],
            "name": info["name"],
            "node_type": info["node_type"],
            "total_count": len(info["instances"]),
            "mirrored_count": mirrored_count,
            "classification": classification,
            "parent_assemblies": parent_assemblies,
            "node_ids": node_ids,
        })

    # Unclassified first, then by count descending, then name
    result.sort(key=lambda p: (
        p["classification"] is not None,
        -p["total_count"],
        p["name"].lower(),
    ))

    return {
        "parts": result,
        "filename": filename,
        "total_unique_parts": len(result),
    }


@router.get("/analysis/consolidate/{filename}")
async def get_consolidation(filename: str) -> Dict[str, Any]:
    """
    Return cached parts consolidation groups for a STEP file.

    Returns the groups immediately if consolidation has been run before.
    Returns 404 if not yet consolidated (use POST to start).
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    cache = _load_cache(filename)
    # Support new dict format ("consolidation") and old list format ("consolidated_parts")
    if cache and "consolidation" in cache:
        logger.info("consolidation_cache_hit", filename=filename)
        c = cache["consolidation"]
        return {
            "groups": c.get("part_groups", []),
            "solid_groups": c.get("solid_groups", []),
            "intra_solid_groups": c.get("intra_solid_groups", []),
            "filename": filename,
            "status": "cached",
        }
    if cache and "consolidated_parts" in cache:
        logger.info("consolidation_cache_hit_legacy", filename=filename)
        return {
            "groups": cache["consolidated_parts"],
            "solid_groups": [],
            "intra_solid_groups": [],
            "filename": filename,
            "status": "cached",
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "Consolidation not yet run for this file", "filename": filename},
    )


@router.post("/analysis/consolidate/{filename}")
async def start_consolidation(filename: str) -> Dict[str, Any]:
    """
    Start a background parts consolidation task for a STEP file.

    Always submits a fresh run (or returns the existing in-progress task id
    if one is already running).  The GET endpoint serves cached results for
    the BOM panel; the POST endpoint is for the "Consolidate" button which
    should always re-compute to pick up algorithm or geometry changes.

    Returns ``{"consolidation_task_id": "...", "status": "pending"}`` while
    running, or the completed result once polling is done.

    Requires the assembly analysis to be cached first.
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    cache = _load_cache(filename)
    if not cache or "analysis" not in cache:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Assembly analysis not yet available for this file",
                "filename": filename,
            },
        )

    # Check for an existing in-progress task
    existing = task_manager.get_tasks_for_file(filename, task_type="parts_consolidation")
    for task in existing:
        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            logger.info(
                "consolidation_task_already_running",
                filename=filename,
                task_id=task.task_id,
            )
            return {"consolidation_task_id": task.task_id, "status": task.status.value}

    # Submit a new consolidation task
    analysis_json_path = str(_analysis_json_path(filename))

    async def run_consolidation_async(_progress_callback):
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(_CONSOLIDATE_WORKER),
                str(file_path),
                analysis_json_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to launch consolidation worker: {exc}")

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(_CONSOLIDATE_TIMEOUT),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"Consolidation timed out after {_CONSOLIDATE_TIMEOUT // 60} minutes"
            )

        returncode = proc.returncode
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        if returncode != 0:
            stdout_tail = stdout.strip()[-500:] if stdout.strip() else ""
            err = (
                f"Worker failed (exit {returncode}): {stderr}"
                if stderr
                else f"Worker crashed (exit {returncode}), last output: {stdout_tail}"
            )
            logger.error(
                "consolidation_worker_failed",
                filename=filename,
                returncode=returncode,
                stderr=stderr[:1000] if stderr else "",
                stdout_tail=stdout_tail,
            )
            raise RuntimeError(err)

        # Worker outputs a JSON object {"part_groups": [...], "solid_groups": [...], "intra_solid_groups": [...]}
        # (older builds output a bare JSON array — handle both for compatibility)
        json_line = next(
            (
                line
                for line in stdout.splitlines()
                if line.strip().startswith("{") or line.strip().startswith("[")
            ),
            None,
        )
        if json_line is None:
            raise RuntimeError(
                f"No JSON in consolidation worker output. Stdout: {stdout[:300]}"
            )
        raw = json.loads(json_line)
        if isinstance(raw, list):
            # Legacy format — wrap into the new structure
            result = {"part_groups": raw, "solid_groups": [], "intra_solid_groups": []}
        else:
            result = raw
        _save_consolidation(filename, result)
        return result

    task_id = task_manager.submit_async(
        "parts_consolidation", filename, run_consolidation_async
    )
    logger.info("consolidation_task_submitted", filename=filename, task_id=task_id)
    return {"consolidation_task_id": task_id, "status": "pending"}


@router.get("/analysis/consolidate-status/{task_id}")
async def get_consolidation_status(task_id: str) -> Dict[str, Any]:
    """
    Poll the status of a background parts consolidation task.

    While pending/running: returns ``{"status": "pending"|"running"}``.
    On failure:            returns ``{"status": "failed", "error": "..."}``.
    On completion:         returns ``{"status": "completed", "groups": [...], "filename": "..."}``.
    """
    info = task_manager.get_task(task_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Task not found", "task_id": task_id},
        )

    if info.status == TaskStatus.FAILED:
        return {"status": "failed", "error": info.error or "Consolidation failed"}

    if info.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        return {"status": info.status.value, "consolidation_task_id": task_id}

    # Completed — return groups and solid_groups
    result = info.results or {}
    if isinstance(result, list):
        # Legacy format from an older worker run
        return {
            "status": "completed",
            "groups": result,
            "solid_groups": [],
            "intra_solid_groups": [],
            "filename": info.filename,
        }
    return {
        "status": "completed",
        "groups": result.get("part_groups", []),
        "solid_groups": result.get("solid_groups", []),
        "intra_solid_groups": result.get("intra_solid_groups", []),
        "filename": info.filename,
    }


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
        "groups": body.get("groups", {}),
        # User overrides of the auto-detected refined sub-class, keyed by
        # ref_id (single-solid) or "ref_id:sN" (a multi-solid sub-solid).
        # Labelling only — does not regenerate NC1/DXF.
        "refined_overrides": body.get("refined_overrides", {}),
    }
    _save_project_state(filename, state)

    return {"success": True}


@router.get("/analysis/portal-tree/{filename}")
async def get_portal_tree(filename: str) -> Dict[str, Any]:
    """
    Return a portal-ready payload: assembly tree (with solid children already
    embedded), STL map, classifications, and summary metadata.

    Excludes consolidation, cnc_analysis, and other heavy sections that the
    portal viewer does not need.
    """
    cache = _load_cache(filename)
    if not cache or "analysis" not in cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "No analysis found", "filename": filename},
        )

    analysis = cache["analysis"]
    state = cache.get("project_state") or {}

    # runId: 8-char hex prefix of the upload filename
    run_id = Path(filename).stem.split("_", 1)[0] if "_" in filename else ""

    tree = analysis.get("assembly_tree", [])
    _add_world_placements(tree)

    return {
        "runId": run_id,
        "projectName": cache.get("cnc_project_number", ""),
        "assembly_tree": tree,
        "summary": analysis.get("summary", {}),
        "stl_map": state.get("stl_map", {}),
        "classifications": state.get("classifications", {}),
    }


# Column-major 4x4 per-index rounding precision:
#   Rotation  (indices 0-2, 4-6, 8-10) → 3 dp (direction cosines)
#   Translation (indices 12-14)         → 1 dp (mm)
#   Fixed       (indices 3, 7, 11, 15)  → 0 dp (always 0 or 1)
_MAT4_DP = [3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 1, 1, 1, 0]
_IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _mat4_mul(a: list, b: list) -> list:
    """Multiply two 4x4 column-major matrices."""
    r = [0.0] * 16
    for col in range(4):
        for row in range(4):
            r[col * 4 + row] = sum(
                a[k * 4 + row] * b[col * 4 + k] for k in range(4)
            )
    return r


def _round_mat4(m: list) -> list:
    return [round(v, d) for v, d in zip(m, _MAT4_DP)]


def _add_world_placements(nodes: list, parent_world: list = _IDENTITY):
    """Replace local ``placement`` with the world-space transform on every node.

    The field is kept as ``placement`` (what the portal service expects) but
    now contains the accumulated world transform, not the local one.
    """
    for node in nodes:
        local = node.pop("placement", None) or _IDENTITY
        world = _mat4_mul(parent_world, local)
        node["placement"] = _round_mat4(world)
        children = node.get("children")
        if children:
            _add_world_placements(children, world)
