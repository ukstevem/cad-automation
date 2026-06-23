"""
Projects router — CRUD for project definitions.

Each project is a JSON file at /app/outputs/projects/{project_number}.json
containing the project number, creation timestamp, and a list of analysis
runs (filename + quantity) that constitute the project's scope of supply.
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/projects", tags=["projects"])

PROJECTS_DIR = os.path.join(settings.ANALYSIS_OUTPUT_DIR, "..", "projects")


def _projects_dir() -> Path:
    p = Path(PROJECTS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _project_path(project_number: str) -> Path:
    safe = re.sub(r'[^\w\-]', '_', project_number)
    return _projects_dir() / f"{safe}.json"


def _read_project(project_number: str) -> dict:
    path = _project_path(project_number)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_number}' not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_project(data: dict) -> None:
    path = _project_path(data["project_number"])
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Models ──────────────────────────────────────────────────────────

class AnalysisEntry(BaseModel):
    filename: str
    quantity: int = Field(ge=1, default=1)
    display_name: Optional[str] = None
    added_at: Optional[str] = None


class CreateProjectRequest(BaseModel):
    project_number: str = Field(min_length=1, max_length=100)


class UpdateAnalysesRequest(BaseModel):
    analyses: List[AnalysisEntry]


# ── Endpoints ───────────────────────────────────────────────────────

@router.get("/")
async def list_projects():
    """Return all projects sorted by creation date (newest first)."""
    projects = []
    for f in sorted(_projects_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            projects.append({
                "project_number": data["project_number"],
                "created_at": data.get("created_at"),
                "analysis_count": len(data.get("analyses", [])),
            })
        except Exception:
            continue
    return {"projects": projects}


@router.post("/")
async def create_project(req: CreateProjectRequest):
    """Create a new project. Fails if the project number already exists."""
    path = _project_path(req.project_number)
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Project '{req.project_number}' already exists")

    data = {
        "project_number": req.project_number,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analyses": [],
    }
    _write_project(data)
    logger.info("project_created", project_number=req.project_number)
    return data


@router.get("/{project_number}")
async def get_project(project_number: str):
    """Get full project data including analysis entries."""
    return _read_project(project_number)


@router.put("/{project_number}/analyses")
async def update_analyses(project_number: str, req: UpdateAnalysesRequest):
    """Replace the project's analysis list (add/remove/update quantities)."""
    data = _read_project(project_number)
    now = datetime.now(timezone.utc).isoformat()

    entries = []
    for a in req.analyses:
        entries.append({
            "filename": a.filename,
            "quantity": a.quantity,
            "display_name": a.display_name,
            "added_at": a.added_at or now,
        })

    data["analyses"] = entries
    _write_project(data)
    logger.info("project_analyses_updated", project_number=project_number, count=len(entries))
    return data


@router.delete("/{project_number}")
async def delete_project(project_number: str):
    """Delete a project."""
    path = _project_path(project_number)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_number}' not found")
    path.unlink()
    logger.info("project_deleted", project_number=project_number)
    return {"deleted": True, "project_number": project_number}


class UpdateNestingTaskRequest(BaseModel):
    nesting_task_id: Optional[str] = None
    nesting_started_at: Optional[str] = None


@router.put("/{project_number}/nesting-task")
async def update_nesting_task(project_number: str, req: UpdateNestingTaskRequest):
    """Save or clear the nesting task ID for reconnection after navigation."""
    data = _read_project(project_number)
    data["nesting_task_id"] = req.nesting_task_id
    data["nesting_started_at"] = req.nesting_started_at
    _write_project(data)
    logger.info("project_nesting_task_updated", project_number=project_number,
                task_id=req.nesting_task_id)
    return data


NESTING_BASE = settings.NESTING_BASE_URL


async def _fetch_cutting_list(task_id: str) -> dict:
    """Fetch the formatted cutting list for a task from the nesting service."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{NESTING_BASE}/api/v1/nesting/cutting-list/{task_id}"
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Nesting service returned {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach nesting service: {exc}",
        )


@router.get("/{project_number}/nesting-pdf")
async def get_nesting_pdf(project_number: str):
    """
    Fetch the cutting list from the nesting service for this project's
    saved task and render it as a PDF report.
    """
    from app.services.nesting_pdf import render_cutting_list_pdf

    data = _read_project(project_number)
    task_id = data.get("nesting_task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="No nesting task saved for this project")

    cutting_list = await _fetch_cutting_list(task_id)
    pdf_bytes = render_cutting_list_pdf(cutting_list, project_number=project_number)

    safe_name = re.sub(r'[^\w\-]', '_', project_number)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="cutting_list_{safe_name}.pdf"',
        },
    )


@router.get("/nesting-pdf/by-task/{task_id}")
async def get_nesting_pdf_by_task(task_id: str):
    """
    Render a cutting-list PDF for an arbitrary nesting task_id.

    Used by the Analysis tab, which runs nesting ad-hoc and holds a live
    task_id rather than a saved project. The PDF is keyed solely by task_id.
    """
    from app.services.nesting_pdf import render_cutting_list_pdf

    cutting_list = await _fetch_cutting_list(task_id)
    pdf_bytes = render_cutting_list_pdf(cutting_list)

    label = cutting_list.get("job_label") or task_id
    safe_name = re.sub(r'[^\w\-]', '_', str(label))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="cutting_list_{safe_name}.pdf"',
        },
    )


@router.get("/{project_number}/nesting-items")
async def get_nesting_items(project_number: str):
    """
    Build expanded nesting items across all analyses in the project.

    For each analysis entry, reads the sidecar JSON to get CNC results
    and consolidation data, then expands by (instance count * project qty).
    Returns the items array ready to POST to the nesting service.
    """
    data = _read_project(project_number)
    analyses = data.get("analyses", [])

    if not analyses:
        raise HTTPException(status_code=400, detail="Project has no analyses")

    items = []
    idx = 0
    errors = []

    for entry in analyses:
        filename = entry["filename"]
        project_qty = entry.get("quantity", 1)

        # Find the sidecar JSON
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        sidecar_path = Path(settings.ANALYSIS_OUTPUT_DIR) / f"{stem}.json"

        if not sidecar_path.exists():
            errors.append(f"No analysis found for {filename}")
            continue

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        cnc = sidecar.get("cnc_analysis", {})
        consolidation = sidecar.get("consolidation", {})
        member_names = sidecar.get("cnc_member_names", {})
        parent_names = sidecar.get("cnc_parent_names", {})

        # Respect classifications — skip excluded and bought-out parts.
        # Classifications may be at part level (0:1:1:81) or solid level
        # (0:1:1:81:s0).  Solid-level exclusions only remove that specific
        # solid from nesting, leaving other solids in the same weldment.
        classifications = sidecar.get("project_state", {}).get("classifications", {})
        excluded_refs = {
            rid for rid, cls_val in classifications.items()
            if cls_val in ("exclude", "bought-out")
        }

        # Build ref_id -> instance count from consolidation part_groups.
        # A consolidation group may contain multiple ref_ids that are
        # geometrically identical.  all_node_ids is the total instance
        # count for the *whole* group, so we must only emit items once
        # per group — mark the canonical ref_id (the first one with CNC
        # results) and skip the rest.
        ref_instance_count = {}
        group_processed = set()   # canonical ref_ids we've already emitted
        ref_to_group_key = {}     # ref_id -> group key (first ref_id in group)

        for group in consolidation.get("part_groups", []):
            ref_ids = group.get("ref_ids", [])
            all_node_ids = group.get("all_node_ids", [])
            count = len(all_node_ids) if all_node_ids else len(ref_ids)
            group_key = ref_ids[0] if ref_ids else None
            for rid in ref_ids:
                ref_instance_count[rid] = count
                ref_to_group_key[rid] = group_key

        # Also count from tree if consolidation is missing
        if not ref_instance_count:
            tree = sidecar.get("analysis", {}).get("assembly_tree", [])
            if not tree:
                # This sidecar may lack the tree (e.g. CNC-only re-run).
                # Search sibling sidecars for the same file.
                tree = _find_tree_for_stem(stem)
            _count_refs(tree, ref_instance_count)

        for ref_id, result in cnc.items():
            if ref_id in excluded_refs:
                continue

            # Collect nestable sections from this CNC entry.
            # Top-level sections have type=section directly.
            # Multi-solid weldments have type=multi_solid with nested
            # solids[] array — each solid may be a section.
            section_results = []

            if result.get("type") == "section":
                section_results.append((ref_id, result))

            elif result.get("type") == "multi_solid":
                for si, solid in enumerate(result.get("solids", [])):
                    if solid.get("type") != "section":
                        continue
                    # Check solid-level exclusion.  The classification key
                    # uses the enumeration index (e.g. "0:1:1:81:s0").
                    solid_idx = solid.get("solid_index")
                    if solid_idx is None:
                        solid_idx = si
                    solid_ref = f"{ref_id}:s{solid_idx}"
                    if solid_ref in excluded_refs:
                        continue
                    section_results.append((ref_id, solid))

            if not section_results:
                continue

            # Skip if another ref_id in the same consolidation group
            # has already been emitted (avoids double-counting).
            gk = ref_to_group_key.get(ref_id)
            if gk is not None:
                if gk in group_processed:
                    continue
                group_processed.add(gk)

            instance_count = ref_instance_count.get(ref_id, 1)
            total_count = instance_count * project_qty
            member = member_names.get(ref_id, "")
            parent = parent_names.get(ref_id, "")

            for _src_ref, sec_result in section_results:
                designation = sec_result.get("designation")
                dims = sec_result.get("dims", {})
                length = dims.get("L")
                if not designation or not length:
                    continue

                for _ in range(total_count):
                    items.append({
                        "item_index": idx,
                        "ref_id": ref_id,
                        "section": designation,
                        "length": round(length),
                        "parent": parent,
                        "member_name": member,
                        "source_file": entry.get("display_name") or filename,
                    })
                    idx += 1

    # Collect unique sections
    sections = sorted(set(it["section"] for it in items))

    return {
        "project_number": project_number,
        "items": items,
        "item_count": len(items),
        "sections": sections,
        "errors": errors,
    }


def _count_refs(nodes: list, counts: dict):
    """Walk the assembly tree and count instances per ref_id."""
    for node in nodes:
        ref_id = node.get("ref_id")
        if ref_id and node.get("node_type", "").startswith("part"):
            counts[ref_id] = counts.get(ref_id, 0) + 1
        children = node.get("children", [])
        if children:
            _count_refs(children, counts)


def _find_tree_for_stem(stem: str) -> list:
    """Search analysis sidecars for one that has a tree for the same file.

    Tries exact prefix match first, then falls back to matching the
    original filename portion (everything after the 8-char hex prefix).
    """
    analysis_dir = Path(settings.ANALYSIS_OUTPUT_DIR)
    prefix = stem[:8]

    # 1) Exact prefix match
    for f in analysis_dir.glob(f"{prefix}*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            tree = data.get("analysis", {}).get("assembly_tree", [])
            if tree:
                return tree
        except Exception:
            continue

    # 2) Match by original filename (after the 9-char "prefix_" part)
    original = stem[9:] if len(stem) > 9 else stem
    if original:
        for f in analysis_dir.glob("*.json"):
            fname = f.stem
            candidate_original = fname[9:] if len(fname) > 9 else fname
            if candidate_original != original:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                tree = data.get("analysis", {}).get("assembly_tree", [])
                if tree:
                    return tree
            except Exception:
                continue

    return []
