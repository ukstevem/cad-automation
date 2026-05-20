"""
BOM↔NC manifest builder.

Generates the per-run manifest that ties every generated NC1/DXF file back to
its BOM row and model node.  Used in three places:

  - app/routers/cnc_analysis.py writes manifest.csv + manifest.json into the
    run's output directory after CNC analysis completes
  - app/services/bom_excel.py renders a "NC Index" sheet and adds the BOM Item
    column to the CNC Items sheet
  - app/routers/cnc_analysis.py /download-all uses the same row schema for the
    ZIP-bundled manifest.json (one entry per file, not per canonical group)

The BOM Item assignment iterates the classifications dict in insertion order,
the same order bom_excel uses to render BOM rows, so item IDs are identical
across the manifest files and the Excel workbook.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


def _walk_count_instances(tree: list) -> Dict[str, int]:
    """Per-ref placement count from the assembly tree (matches the worker)."""
    counts: Dict[str, int] = {}

    def _walk(nodes: list) -> None:
        for n in nodes or []:
            if n.get("node_type") == "assembly":
                _walk(n.get("children", []))
            else:
                rid = n.get("ref_id") or n.get("id")
                if rid:
                    counts[rid] = counts.get(rid, 0) + 1

    _walk(tree)
    return counts


def _build_node_to_ref(tree: list) -> Dict[str, str]:
    """Map every leaf node's instance ``id`` to its prototype ``ref_id``.

    Classifications are stored against instance node IDs (e.g. ``0:1:1:3:1``)
    while CNC analysis, member_names, and consolidation groups are keyed by
    the prototype ``ref_id`` (e.g. ``0:1:1:4``).  This resolver bridges the
    two so BOM rows aggregate by prototype rather than fragment per instance.
    """
    out: Dict[str, str] = {}

    def _walk(nodes: list) -> None:
        for n in nodes or []:
            if n.get("node_type") == "assembly":
                _walk(n.get("children", []))
            else:
                nid = n.get("id")
                rid = n.get("ref_id") or nid
                if nid:
                    out[nid] = rid

    _walk(tree)
    return out


def build_ref_to_stl(stl_map: Dict[str, str], tree: list) -> Dict[str, str]:
    """Invert the instance-keyed stl_map to a ref_id → STL URL lookup.

    ``stl_map`` is populated by the frontend with one entry per placed
    instance.  Multiple instances of the same prototype share an identical
    mesh, so the first URL seen for each ref_id is fine.
    """
    node_to_ref = _build_node_to_ref(tree)
    out: Dict[str, str] = {}
    for nid, url in (stl_map or {}).items():
        rid = node_to_ref.get(nid, nid)
        out.setdefault(rid, url)
    return out


def assign_bom_items(cache: dict) -> Dict[str, Dict[str, Any]]:
    """
    Walk classifications in insertion order and assign a BOM Item ID (B001…)
    keyed by *ref_id* (prototype).  Consolidated group members share an ID.

    Classifications are keyed by *instance* node IDs (e.g. ``0:1:1:3:1``) so
    the function resolves each one to its ref_id via the assembly tree before
    looking it up in the consolidation map.  Several instance classifications
    that resolve to the same ref_id collapse to a single BOM Item.

    Returns {ref_id: {"bom_item": "B042", "bom_total_qty": int, "action": str,
                       "group_name": str|""}}.
    """
    classifications: Dict[str, str] = (cache.get("project_state") or {}).get("classifications", {})
    consolidation: List[dict] = (cache.get("consolidation") or {}).get("part_groups", [])
    tree = ((cache.get("analysis") or {}).get("assembly_tree") or [])

    ref_to_group: Dict[str, dict] = {}
    for g in consolidation:
        for rid in g.get("ref_ids", []):
            ref_to_group[rid] = g

    node_to_ref = _build_node_to_ref(tree)
    instance_counts = _walk_count_instances(tree)

    out: Dict[str, Dict[str, Any]] = {}
    seen_refs: set = set()
    bom_idx = 0

    for node_id, action in classifications.items():
        if action not in ("postprocess", "bought-out", "exclude"):
            continue
        # Resolve instance node_id → prototype ref_id. Fall back to node_id
        # itself for solid sub-IDs or any node not present in the tree map.
        ref_id = node_to_ref.get(node_id, node_id)
        if ref_id in seen_refs:
            continue

        group = ref_to_group.get(ref_id)
        if group:
            # Mark every ref in the group as seen so duplicate instance
            # classifications for any group member don't re-allocate IDs.
            for rid in group["ref_ids"]:
                seen_refs.add(rid)
            bom_idx += 1
            item_id = f"B{bom_idx:03d}"
            total = group.get("total_count") or sum(
                instance_counts.get(r, 0) for r in group["ref_ids"]
            ) or 1
            group_name = group.get("canonical_name", "") if len(group.get("ref_ids", [])) > 1 else ""
            for rid in group["ref_ids"]:
                out[rid] = {
                    "bom_item": item_id,
                    "bom_total_qty": total,
                    "action": action,
                    "group_name": group_name,
                }
        else:
            seen_refs.add(ref_id)
            bom_idx += 1
            out[ref_id] = {
                "bom_item": f"B{bom_idx:03d}",
                "bom_total_qty": instance_counts.get(ref_id, 1),
                "action": action,
                "group_name": "",
            }

    return out


def build_manifest_rows(cache: dict) -> List[Dict[str, Any]]:
    """One row per generated NC1/DXF file (multi-solid parts expand to N rows)."""
    cnc_results: Dict[str, dict] = cache.get("cnc_analysis", {}) or {}
    member_names: Dict[str, str] = cache.get("cnc_member_names", {}) or {}
    parent_names: Dict[str, str] = cache.get("cnc_parent_names", {}) or {}
    stl_map_raw: Dict[str, str] = (cache.get("project_state") or {}).get("stl_map", {}) or {}
    consolidation: List[dict] = (cache.get("consolidation") or {}).get("part_groups", [])
    tree = ((cache.get("analysis") or {}).get("assembly_tree") or [])

    # stl_map is keyed by instance node IDs; invert to ref_id so the manifest
    # rows (and the Excel BOM thumbnails) resolve correctly.
    ref_to_stl = build_ref_to_stl(stl_map_raw, tree)

    # Per-ref canonical name from the consolidation groups — covers singletons
    # where cnc_member_names is empty but consolidation always supplies a name.
    ref_to_canonical_name: Dict[str, str] = {}
    for g in consolidation:
        cname = g.get("canonical_name", "")
        for rid in g.get("ref_ids", []):
            ref_to_canonical_name[rid] = cname

    bom_assignments = assign_bom_items(cache)
    instance_counts = _walk_count_instances(tree)

    rows: List[Dict[str, Any]] = []

    for ref_id, result in cnc_results.items():
        assignment = bom_assignments.get(ref_id)
        if not assignment or assignment["action"] != "postprocess":
            continue

        def _row_for(r: dict, solid_idx: Optional[int]) -> Optional[Dict[str, Any]]:
            fp = r.get("nc1_path") or r.get("dxf_path")
            if not fp:
                return None
            dims = r.get("dims") or {}
            return {
                "bom_item": assignment["bom_item"],
                "filename": Path(fp).name,
                "ext": Path(fp).suffix.lstrip(".").lower(),
                "nc1_hash": r.get("nc1_hash", "") or "",
                "ref_id": ref_id,
                "solid_idx": solid_idx if solid_idx is not None else "",
                "consolidation_group": assignment["group_name"],
                "member_name": (
                    member_names.get(ref_id)
                    or ref_to_canonical_name.get(ref_id, "")
                ),
                "parent_assembly": parent_names.get(ref_id, ""),
                "type": r.get("type", ""),
                "category": r.get("category", ""),
                "designation": r.get("designation", ""),
                "profile_type": r.get("profile_type", ""),
                "L_mm": dims.get("L"),
                "H_mm": dims.get("H"),
                "W_mm": dims.get("W"),
                "T_mm": dims.get("T"),
                "qty": instance_counts.get(ref_id, 1),
                "bom_total_qty": assignment["bom_total_qty"],
                "mass_kg": r.get("mass_kg"),
                "stl_url": ref_to_stl.get(ref_id, ""),
            }

        if result.get("type") == "multi_solid":
            for idx, solid in enumerate(result.get("solids", []) or []):
                row = _row_for(solid, idx)
                if row:
                    rows.append(row)
        else:
            row = _row_for(result, None)
            if row:
                rows.append(row)

    rows.sort(key=lambda r: (
        r.get("bom_item", ""),
        r.get("solid_idx") if isinstance(r.get("solid_idx"), int) else -1,
        r.get("filename", ""),
    ))
    return rows


MANIFEST_HEADERS: List[str] = [
    "bom_item", "filename", "ext", "nc1_hash", "ref_id", "solid_idx",
    "consolidation_group", "member_name", "parent_assembly",
    "type", "category", "designation", "profile_type",
    "L_mm", "H_mm", "W_mm", "T_mm",
    "qty", "bom_total_qty", "mass_kg", "stl_url",
]


def write_manifest_files(cache: dict, out_dir: Path) -> Optional[Path]:
    """Write manifest.csv + manifest.json into out_dir.  Returns CSV path."""
    rows = build_manifest_rows(cache)
    if not rows:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "manifest.csv"
    json_path = out_dir / "manifest.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADERS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    payload = {
        "project_number": cache.get("cnc_project_number", ""),
        "steel_grade": cache.get("cnc_steel_grade", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("manifest_written", csv=str(csv_path), rows=len(rows))
    return csv_path
