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

from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
import structlog

from app.config import settings
from app.services.bom_builder import build_step_bom
from app.services.sidecar import (
    SidecarCorruptError,
    atomic_write_json,
    load_sidecar,
)
from app.services.bom_manifest import (
    canonical_ref_map,
    finalize_cnc_outputs,
    write_manifest_files,
)
from app.services.task_manager import task_manager, TaskStatus

# Path to the standalone worker script
_CNC_WORKER = Path(__file__).parent.parent / "workers" / "analyse_cnc_parts.py"
# Lightweight classify worker (refined_class only, no NC1/DXF).
_CLASSIFY_WORKER = Path(__file__).parent.parent / "workers" / "classify_parts.py"

# Hard timeout for a single CNC analysis run (seconds).
# The worker saves each part's result progressively, so a timeout only loses
# the part currently being processed.  Re-clicking Analyse resumes from where
# the previous run left off (already-analysed ref_ids are skipped).
_CNC_TIMEOUT = 1800  # 30 minutes

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
    try:
        data = load_sidecar(path)
    except SidecarCorruptError as e:
        logger.warning("cnc_cache_read_failed", filename=filename, error=str(e))
        return None
    return data or None


def _consolidation_state(cache: dict) -> str:
    """Return the gate state for CNC analysis on this cache.

    - ``"missing"``     – consolidation has never been run for this STEP file.
    - ``"stale-tree"``  – the assembly was re-analysed after consolidation, so
      the part_groups may not cover the current tree.  Block.
    - ``"stale-cnc"``   – at least one cnc_analysis result predates the current
      consolidation, so its grouping/canonical assignment may be wrong.  Block
      unless caller passes force=true.
    - ``"fresh"``       – good to go.
    """
    cons = cache.get("consolidation") or {}
    if not cons or "consolidated_at" not in cons:
        return "missing"

    analysis_ts = ((cache.get("analysis") or {}).get("analyzed_at"))
    if analysis_ts and cons["consolidated_at"] < analysis_ts:
        return "stale-tree"

    cons_ts = cons["consolidated_at"]
    for result in (cache.get("cnc_analysis") or {}).values():
        if not isinstance(result, dict):
            continue
        ts = result.get("analysed_at")
        if ts and ts < cons_ts:
            return "stale-cnc"

    return "fresh"


def _clear_cnc_outputs(filename: str, cache: dict) -> None:
    """Wipe the run's outputs/cnc dir and clear cnc_analysis from the cache.

    Called when the user passes force=true on a stale state so the rerun
    starts from a clean slate (matches user intent: 'old NC1 locked out,
    rebuilt only after fresh consolidation + analysis').
    """
    out_dir = _cnc_out_dir(filename)
    for sub in ("nc1", "plates"):
        sub_dir = out_dir / sub
        if not sub_dir.is_dir():
            continue
        for f in sub_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except OSError as e:
                    logger.warning("clear_cnc_file_failed", path=str(f), error=str(e))
    for name in ("manifest.csv", "manifest.json"):
        f = out_dir / name
        if f.exists():
            try:
                f.unlink()
            except OSError as e:
                logger.warning("clear_manifest_failed", path=str(f), error=str(e))
    cache["cnc_analysis"] = {}


def _count_ref_instances(tree: list) -> Dict[str, int]:
    """Walk the assembly tree and return {ref_id: instance_count} for every leaf part.

    Matches PartsConsolidator._collect_parts traversal: assemblies are
    recursed into, leaf nodes contribute one count per occurrence under their
    ref_id (which is shared across all placements of the same prototype).
    """
    counts: Dict[str, int] = {}

    def _walk(nodes: list) -> None:
        for node in nodes or []:
            if node.get("node_type") == "assembly":
                _walk(node.get("children", []))
            else:
                ref_id = node.get("ref_id") or node.get("id")
                if ref_id:
                    counts[ref_id] = counts.get(ref_id, 0) + 1

    _walk(tree)
    return counts


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

    # load_sidecar raises (preserving the bad file) on a corrupt sidecar rather
    # than falling back to {} — the {} fallback is exactly what destroyed the
    # analysis/project_state sections in run c25bf2c8 (2026-06-17).
    existing: dict = load_sidecar(path)

    # Merge into existing results rather than replacing, so that partial results
    # already saved by the worker (progressive saving) are preserved for any
    # ref_ids that aren't in this batch (e.g. resumed after a previous timeout).
    existing.setdefault("cnc_analysis", {}).update(results)
    if member_ids:
        existing["cnc_member_names"] = member_ids
    if parent_names:
        existing["cnc_parent_names"] = parent_names
    if project_number:
        existing["cnc_project_number"] = project_number
    if steel_grade:
        existing["cnc_steel_grade"] = steel_grade

    # Rebuild the unified native BOM from the full cnc_analysis dict (not just
    # this batch) so progressive/resumed runs produce a complete BOM.
    analysis_section = existing.get("analysis") or {}
    tree = analysis_section.get("assembly_tree") or []
    native_bom = build_step_bom(
        tree,
        existing.get("cnc_analysis") or {},
        cnc_member_names=existing.get("cnc_member_names"),
        parent_names=existing.get("cnc_parent_names"),
        steel_grade=existing.get("cnc_steel_grade", ""),
        project_number=existing.get("cnc_project_number", ""),
    )
    if native_bom:
        existing.setdefault("analysis", {})["native_bom"] = native_bom

    atomic_write_json(path, existing)
    logger.info(
        "cnc_analysis_cached",
        filename=filename,
        n_refs=len(results),
        bom_rows=len(native_bom),
    )

    # Reconcile filesystem with the canonical-per-BOM-Item invariant:
    # prune non-canonical refs from cnc_analysis, delete orphan files, then
    # rename surviving NC1/DXF files to <bom_item>[<-sN>].<ext> and rewrite
    # NC1 line 5.  Best-effort — failures here must not break analysis caching.
    out_dir = _cnc_out_dir(filename)
    try:
        finalize_cnc_outputs(existing, out_dir)
        # Paths and hashes changed; re-serialise the sidecar before the
        # manifest builder reads it back.
        atomic_write_json(path, existing)
    except Exception as e:
        logger.warning("cnc_finalize_failed", filename=filename, error=str(e))

    # Emit the BOM↔NC manifest into the run's output directory so operators can
    # tie every NC1/DXF file back to a BOM row.
    try:
        write_manifest_files(existing, out_dir)
    except Exception as e:
        logger.warning("manifest_write_failed", filename=filename, error=str(e))


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------

@router.get("/cnc-analysis/state/{filename}")
async def get_cnc_state(filename: str) -> Dict[str, Any]:
    """Report CNC pipeline freshness so the frontend can gate UI controls.

    Returns ``{"state": "missing"|"stale-tree"|"stale-cnc"|"fresh", ...}``
    plus timestamps and the count of stale refs.  Never raises — a missing
    file still returns a state ("missing") rather than 404 so the UI doesn't
    have to differentiate "file gone" from "no consolidation yet".
    """
    cache = _load_cache(filename) or {}
    state = _consolidation_state(cache)
    cons = cache.get("consolidation") or {}
    analysis_ts = ((cache.get("analysis") or {}).get("analyzed_at"))
    cons_ts = cons.get("consolidated_at")
    cnc = cache.get("cnc_analysis") or {}

    stale_refs: List[str] = []
    if cons_ts:
        for ref, result in cnc.items():
            if isinstance(result, dict):
                ts = result.get("analysed_at")
                if ts and ts < cons_ts:
                    stale_refs.append(ref)

    return {
        "state": state,
        "analyzed_at": analysis_ts,
        "consolidated_at": cons_ts,
        "cnc_ref_count": len(cnc),
        "stale_cnc_refs": stale_refs,
    }


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


@router.post("/cnc-analysis/classify/{filename}")
async def classify_parts(filename: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Lightweight refined-class for a set of refs — features + library match +
    decision tree, NO NC1/DXF.  Assists the top-down triage as the user explores
    the frontier; runs ahead of (and far cheaper than) the heavy CNC analysis.

    Body: ``{"ref_ids": [...], "member_ids": {ref: name}, "steel_grade": "S275"}``
    Returns ``{"results": {ref_id: {refined_class, refined_confidence, ...}}}``.
    Already-classified/analysed refs are returned from cache without recompute.
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )
    ref_ids: List[str] = body.get("ref_ids", [])
    member_ids: dict = body.get("member_ids", {})
    steel_grade: str = body.get("steel_grade", "") or ""
    if not ref_ids:
        return {"filename": filename, "results": {}}

    cache = _load_cache(filename) or {}
    classification = dict(cache.get("classification") or {})
    cnc = cache.get("cnc_analysis") or {}

    def _cached(r):
        if r in classification:
            return classification[r]
        c = cnc.get(r)
        if isinstance(c, dict) and c.get("refined_class") is not None:
            return {"refined_class": c.get("refined_class"),
                    "refined_confidence": c.get("refined_confidence"),
                    "refined_reason": c.get("refined_reason")}
        return None

    pending = [r for r in ref_ids if _cached(r) is None]
    if pending:
        analysis_json_path = str(_analysis_json_path(filename))
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(_CLASSIFY_WORKER),
                str(file_path), analysis_json_path,
                json.dumps(pending), json.dumps(member_ids), steel_grade,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=float(_CNC_TIMEOUT))
        except Exception as exc:
            logger.error("classify_worker_failed", filename=filename, error=str(exc))
            raise HTTPException(status_code=500,
                                detail={"error": f"Classification failed: {exc}"})
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise HTTPException(status_code=500,
                                detail={"error": f"Classify worker exit {proc.returncode}: {stderr[:300]}"})
        # Worker saved progressively to the sidecar; parse stdout for the results.
        json_line = next((l for l in stdout.splitlines() if l.strip().startswith("{")), "{}")
        classification.update(json.loads(json_line).get("results", {}))

    return {"filename": filename,
            "results": {r: _cached(r) or classification.get(r) for r in ref_ids}}


@router.post("/cnc-analysis/analyse/{filename}")
async def start_cnc_analysis(
    filename: str,
    body: Dict[str, Any] = Body(...),
    force_query: bool = Query(False, alias="force"),
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
    force: bool = force_query or bool(body.get("force", False))

    if not ref_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ref_ids must not be empty"},
        )

    # Check for an existing in-progress task for this file
    existing_tasks = task_manager.get_tasks_for_file(filename, task_type="cnc_analysis")
    for task in existing_tasks:
        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            logger.info(
                "cnc_task_already_running",
                filename=filename,
                task_id=task.task_id,
            )
            return {"cnc_task_id": task.task_id, "status": task.status.value}

    cache = _load_cache(filename) or {}

    # Consolidation gate: CNC analysis depends on a fresh consolidation so it
    # can collapse multi-ref groups to a single canonical NC file per BOM Item.
    state = _consolidation_state(cache)
    if state == "missing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "consolidation_required",
                "state": state,
                "detail": "Run /analysis/consolidate before CNC analysis.",
            },
        )
    if state == "stale-tree":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "consolidation_stale",
                "state": state,
                "detail": "Assembly was re-analysed; re-run consolidation before CNC.",
            },
        )
    if state == "stale-cnc" and not force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "cnc_results_stale",
                "state": state,
                "detail": (
                    "Consolidation was re-run after CNC analysis. "
                    "Pass force=true to clear stale NC files and re-analyse."
                ),
            },
        )
    if state == "stale-cnc" and force:
        _clear_cnc_outputs(filename, cache)

    # Canonical filter: for any classified refs that fall in the same
    # consolidation group, only the canonical (first) ref is analysed so the
    # NC1 output ends up 1-to-1 with the BOM rows.
    canonical = canonical_ref_map(cache)
    seen_canons: set = set()
    canonical_pending: List[str] = []
    for r in ref_ids:
        c = canonical.get(r, r)
        if c not in seen_canons:
            seen_canons.add(c)
            canonical_pending.append(c)

    # Skip canonicals that already have a (fresh) cached result.  force=true
    # bypasses the cache; on stale-cnc the cache was just wiped above.
    already_done: set = set() if force else set(cache.get("cnc_analysis", {}).keys())
    pending_ref_ids = [r for r in canonical_pending if r not in already_done]
    skipped = len(canonical_pending) - len(pending_ref_ids)
    if skipped:
        logger.info(
            "cnc_resuming",
            filename=filename,
            total=len(ref_ids),
            canonical=len(canonical_pending),
            already_done=skipped,
            pending=len(pending_ref_ids),
        )

    if not pending_ref_ids:
        # All canonical refs already have results — reconcile and return cache.
        # finalize is idempotent, so running it on this "resumed" path catches
        # any drift (e.g. files left behind from a prior failed rename).
        try:
            finalize_cnc_outputs(cache, _cnc_out_dir(filename))
            atomic_write_json(_analysis_json_path(filename), cache)
            write_manifest_files(cache, _cnc_out_dir(filename))
        except Exception as e:
            logger.warning("cnc_resume_finalize_failed", filename=filename, error=str(e))
        return {"status": "completed", "results": cache.get("cnc_analysis", {}), "resumed": True}

    # Prepare paths and arguments
    analysis_json_path = str(_analysis_json_path(filename))
    out_dir = _cnc_out_dir(filename)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_ids_json = json.dumps(pending_ref_ids)
    member_ids_json = json.dumps(member_ids)
    parent_names_json = json.dumps(parent_names)
    # project_number / steel_grade are plain strings — pass directly as argv

    # Per-ref instance counts from the cached assembly tree.  Each NC1 file is
    # written with the count of its own ref_id (chirality-safe — mirror twins
    # already live in separate consolidation groups via the fingerprint).
    tree = ((cache or {}).get("analysis") or {}).get("assembly_tree") or []
    instance_counts = _count_ref_instances(tree)
    instance_counts_json = json.dumps(instance_counts)

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
                instance_counts_json,
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

    # Refuse downloads when results are stale w.r.t. the current consolidation —
    # the operator should never receive files that don't match the latest BOM.
    state = _consolidation_state(cache)
    if state in ("missing", "stale-tree", "stale-cnc"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "cnc_outputs_stale",
                "state": state,
                "detail": "Re-run consolidation and CNC analysis before downloading.",
            },
        )

    cnc_results: dict = cache["cnc_analysis"]
    member_names: dict = cache.get("cnc_member_names", {})
    parent_names_cache: dict = cache.get("cnc_parent_names", {})
    project_number: str = cache.get("cnc_project_number", "")
    steel_grade: str = cache.get("cnc_steel_grade", "")

    path_key = "dxf_path" if ext == "dxf" else "nc1_path"

    # Build a deduplication map from part-level consolidation groups so that
    # geometrically identical parts (consolidated into one BOM row) also produce
    # only one file in the ZIP rather than one copy per XCAF ref_id.
    #
    # canonical_map: ref_id → canonical_ref_id
    #   Non-canonical ref_ids map to a different canonical; skip them in the loop.
    # group_all_refs: canonical_ref_id → [all ref_ids in that group]
    #   Used to populate consolidated_ref_ids in the manifest.
    canonical_map: dict = {}
    group_all_refs: dict = {}
    part_groups = cache.get("consolidation", {}).get("part_groups", [])
    for group in part_groups:
        grp_refs: List[str] = group.get("ref_ids", [])
        if len(grp_refs) <= 1:
            continue
        # Canonical = first ref_id in this group that has an analysis result
        members_with_results = [r for r in grp_refs if r in cnc_results]
        if not members_with_results:
            continue
        canon = members_with_results[0]
        for rid in grp_refs:
            canonical_map[rid] = canon
        group_all_refs[canon] = grp_refs

    # Collect (arc_name, file_path, meta) for each output file
    entries: List[tuple] = []

    def _collect(ref_id: str, result: dict, solid_idx: Optional[int], consolidated_ref_ids: Optional[List[str]] = None) -> None:
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
            "consolidated_ref_ids": consolidated_ref_ids or [ref_id],
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
        # Skip non-canonical refs — their geometry is identical to the canonical
        # ref_id whose file is already included.
        if canonical_map.get(ref_id, ref_id) != ref_id:
            continue
        consolidated = group_all_refs.get(ref_id, [ref_id])
        if result.get("type") == "multi_solid":
            for idx, solid in enumerate(result.get("solids", [])):
                _collect(ref_id, solid, idx, consolidated)
        else:
            _collect(ref_id, result, None, consolidated)

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


@router.get("/cnc-analysis/bom-xlsx/{filename}")
async def download_bom_xlsx(filename: str) -> Response:
    """
    Download an Excel (.xlsx) BOM workbook with embedded STL thumbnails.

    Sheets: CNC Items, Unknown Items, Bought Out, Excluded, Summary.
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "File not found", "filename": filename},
        )

    # Same staleness gate as the NC1/DXF ZIP download — the BOM is the system
    # of record for what was machined; serving it from stale data would lie.
    cache = _load_cache(filename)
    if cache:
        state = _consolidation_state(cache)
        if state in ("missing", "stale-tree", "stale-cnc"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "bom_stale",
                    "state": state,
                    "detail": "Re-run consolidation and CNC analysis before downloading the BOM.",
                },
            )

    from app.services.bom_excel import generate_bom_xlsx

    xlsx_bytes = generate_bom_xlsx(filename)
    if xlsx_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "No CNC analysis data available for BOM export",
                    "filename": filename},
        )

    cache = _load_cache(filename) or {}
    project_number = cache.get("cnc_project_number", "")
    stem = project_number or Path(filename).stem[:8]

    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{stem}-BOM.xlsx"'},
    )
