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

import structlog
from fastapi import APIRouter, HTTPException
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

        # Build ref_id -> instance count from consolidation part_groups
        ref_instance_count = {}
        for group in consolidation.get("part_groups", []):
            ref_ids = group.get("ref_ids", [])
            all_node_ids = group.get("all_node_ids", [])
            count = len(all_node_ids) if all_node_ids else len(ref_ids)
            for rid in ref_ids:
                ref_instance_count[rid] = count

        # Also count from tree if consolidation is missing
        if not ref_instance_count:
            tree = sidecar.get("analysis", {}).get("assembly_tree", [])
            _count_refs(tree, ref_instance_count)

        for ref_id, result in cnc.items():
            if result.get("type") != "section":
                continue
            designation = result.get("designation")
            dims = result.get("dims", {})
            length = dims.get("L")
            if not designation or not length:
                continue

            instance_count = ref_instance_count.get(ref_id, 1)
            total_count = instance_count * project_qty
            member = member_names.get(ref_id, "")
            parent = parent_names.get(ref_id, "")

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
