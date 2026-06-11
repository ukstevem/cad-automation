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
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger()

# Matches a solid-level classification key: "<prefix>:s<index>" (e.g.
# "0:1:1:2:s0").  Frontend stores per-solid classifications of a multi-solid
# part under "<prototype_ref_id>:s<n>" keys.
_SOLID_KEY_RE = re.compile(r"^(.*):s\d+$")


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


def resolve_classification_ref(node_id: str, node_to_ref: Dict[str, str]) -> str:
    """Resolve a classification key to its prototype ``ref_id``.

    Classifications come in two shapes:

      - *instance* node ids (e.g. ``0:1:1:3:1``) — resolve through the tree's
        instance→ref map.
      - *solid-level* ids (e.g. ``0:1:1:2:s0``) — the frontend stores per-solid
        classifications of a multi-solid part under ``<prototype_ref_id>:s<n>``
        keys.  These never appear in ``node_to_ref`` (the tree leaf is the whole
        multi-solid part, not its sub-solids), so without stripping the suffix
        they fall back to a bogus self-ref and the part fragments into N
        unnamed "Unknown" rows in the BOM export.

    Strip any ``:s<n>`` suffix so solid classifications collapse onto the parent
    multi-solid part (whose ``multi_solid`` CNC result already expands per
    solid), then resolve the remaining base through ``node_to_ref``.
    """
    m = _SOLID_KEY_RE.match(node_id)
    base = m.group(1) if m else node_id
    return node_to_ref.get(base, base)


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
        # Resolve instance node_id → prototype ref_id.  Solid-level keys
        # ("<ref>:s<n>") collapse onto their parent multi-solid part so all of
        # its solids share one BOM Item rather than fragmenting per solid.
        ref_id = resolve_classification_ref(node_id, node_to_ref)
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


def canonical_ref_map(cache: dict) -> Dict[str, str]:
    """{ref_id: canonical_ref_id} — the first ref in each consolidation group.

    Refs not present in any consolidation group map to themselves.  Used by
    the CNC pipeline to ensure only one ref per group is analysed, and by
    cache readers to dereference non-canonical refs to their canonical entry.
    """
    out: Dict[str, str] = {}
    for g in (cache.get("consolidation") or {}).get("part_groups", []):
        refs = g.get("ref_ids") or []
        if not refs:
            continue
        canon = refs[0]
        for r in refs:
            out[r] = canon
    return out


def prune_non_canonical_refs(cache: dict) -> List[str]:
    """Drop non-canonical cnc_analysis entries (consolidation-aware cleanup).

    After the CNC submit filter, only canonical refs are re-analysed.  Stale
    entries from older runs that pre-date the consolidation gate get removed
    here so the sidecar mirrors the new one-NC-per-BOM-Item invariant.
    """
    canonical = canonical_ref_map(cache)
    if not canonical:
        return []
    cnc = cache.get("cnc_analysis") or {}
    dropped: List[str] = []
    for ref in list(cnc.keys()):
        canon = canonical.get(ref, ref)
        if canon != ref:
            cnc.pop(ref, None)
            dropped.append(ref)
    if dropped:
        logger.info("pruned_non_canonical_refs", count=len(dropped))
    return dropped


def prune_non_postprocess_outputs(cache: dict, out_dir: Path) -> List[str]:
    """Drop NC1/DXF files for refs that aren't classified as postprocess.

    Defensive: the frontend only submits postprocess refs to CNC analysis,
    but if any leak through (e.g. force=true with an unfiltered ref list,
    or a re-classification after analysis) the resulting files would not be
    in the BOM manifest and would never get renamed.  Clear them here.
    """
    assignments = assign_bom_items(cache)
    cnc = cache.get("cnc_analysis") or {}
    removed: List[str] = []

    def _clear(target: dict) -> None:
        for key in ("nc1_path", "dxf_path"):
            p = target.get(key)
            if not p:
                continue
            f = Path(p)
            if f.exists():
                try:
                    f.unlink()
                    removed.append(f.name)
                except OSError as e:
                    logger.warning("non_pp_delete_failed", path=str(f), error=str(e))
            target.pop(key, None)
            target.pop("nc1_hash", None)

    for ref, result in cnc.items():
        if not isinstance(result, dict):
            continue
        action = (assignments.get(ref) or {}).get("action")
        if action == "postprocess":
            continue
        if result.get("type") == "multi_solid":
            for solid in result.get("solids") or []:
                if isinstance(solid, dict):
                    _clear(solid)
        else:
            _clear(result)

    if removed:
        logger.info("pruned_non_postprocess_files", count=len(removed))
    return removed


def prune_orphan_files(cache: dict, out_dir: Path) -> List[str]:
    """Delete NC1/DXF files under out_dir that no cnc_analysis entry points at.

    Catches files left over from earlier runs after refs were pruned or the
    consolidation grouping changed.  Walks both single-body results and the
    nested ``solids`` list for multi-solid parts.
    """
    referenced: set = set()

    def _add(p: Optional[str]) -> None:
        if p:
            referenced.add(Path(p).name)

    for ref, result in (cache.get("cnc_analysis") or {}).items():
        if not isinstance(result, dict):
            continue
        if result.get("type") == "multi_solid":
            for solid in result.get("solids") or []:
                _add(solid.get("nc1_path"))
                _add(solid.get("dxf_path"))
        else:
            _add(result.get("nc1_path"))
            _add(result.get("dxf_path"))

    deleted: List[str] = []
    for sub in ("nc1", "plates"):
        sub_dir = out_dir / sub
        if not sub_dir.is_dir():
            continue
        for f in sub_dir.iterdir():
            if f.is_file() and f.name not in referenced:
                try:
                    f.unlink()
                    deleted.append(f.name)
                except OSError as e:
                    logger.warning("orphan_delete_failed", file=str(f), error=str(e))
    if deleted:
        logger.info("pruned_orphan_files", count=len(deleted))
    return deleted


def _rewrite_nc1_member_id_and_qty(path: Path, new_id: str, qty: int) -> None:
    """Rewrite NC1 line 5 (out_filename) and line 7 (qty) in-place.

    After the canonical filter collapses a consolidation group to a single
    surviving NC file, qty must reflect the whole BOM Item count (not just
    the per-ref instance count the worker computed from the canonical ref).
    """
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    if len(lines) >= 5:
        lines[4] = f"  {new_id}"
    if len(lines) >= 7 and qty and qty > 0:
        lines[6] = f"  {int(qty)}"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def rename_files_to_bom_ids(cache: dict, out_dir: Path) -> List[Tuple[str, str]]:
    """Rename surviving NC1/DXF files to ``<bom_item>[<-sN>].<ext>``.

    Single-body parts → ``B042.nc1``.  Multi-solid parts → ``B042-s0.nc1``,
    ``B042-s1.nc1`` etc.  Mutates cache.cnc_analysis paths + nc1_hash in
    place; the caller is expected to re-serialise the sidecar afterwards.
    """
    from app.pipeline.dstv_writer import nc1_group_key  # local import — avoids circular

    rows = build_manifest_rows(cache)
    cnc_results = cache.get("cnc_analysis") or {}
    renames: List[Tuple[str, str]] = []

    for row in rows:
        ref_id = row["ref_id"]
        ext = row["ext"]
        solid_idx = row["solid_idx"]
        bom_item = row["bom_item"]
        if not bom_item:
            continue

        new_stem = f"{bom_item}-s{solid_idx}" if isinstance(solid_idx, int) else bom_item
        new_name = f"{new_stem}.{ext}"

        # Locate the result dict that owns the path we're renaming.
        ref_result = cnc_results.get(ref_id) or {}
        if isinstance(solid_idx, int):
            solids = ref_result.get("solids") or []
            if solid_idx >= len(solids):
                continue
            target = solids[solid_idx]
        else:
            target = ref_result
        if not isinstance(target, dict):
            continue

        key = "nc1_path" if ext == "nc1" else "dxf_path"
        old_path_str = target.get(key)
        if not old_path_str:
            continue
        old_path = Path(old_path_str)
        if not old_path.exists():
            continue
        new_path = old_path.parent / new_name
        if new_path != old_path:
            if new_path.exists():
                # Stale survivor from a previous run — overwrite
                try:
                    new_path.unlink()
                except OSError as e:
                    logger.warning("bom_rename_clear_failed", path=str(new_path), error=str(e))
                    continue
            try:
                shutil.move(str(old_path), str(new_path))
            except OSError as e:
                logger.warning("bom_rename_failed", old=str(old_path), new=str(new_path), error=str(e))
                continue
            target[key] = str(new_path)
            renames.append((str(old_path), str(new_path)))

        # Always re-stamp NC1 header (cheap, idempotent) — keeps line 5
        # and line 7 in sync with the current BOM Item + total qty even on
        # resumed runs where the file is already at its B### name.
        if ext == "nc1":
            _rewrite_nc1_member_id_and_qty(new_path, new_stem, row.get("bom_total_qty") or 1)
            try:
                target["nc1_hash"] = nc1_group_key(new_path)
            except Exception as e:
                logger.warning("nc1_rehash_failed", path=str(new_path), error=str(e))

    if renames:
        logger.info("renamed_to_bom_ids", count=len(renames))
    return renames


def finalize_cnc_outputs(cache: dict, out_dir: Path) -> None:
    """Reconcile sidecar + filesystem with the canonical-per-BOM-Item model.

    Order matters:
      1. Drop non-canonical cnc_analysis entries (multi-ref groups collapse
         to a single canonical result).
      2. Delete files no surviving entry points at (orphans from the prune
         above or from earlier consolidation grouping changes).
      3. Rename survivors to ``<bom_item>[<-sN>].<ext>`` and update sidecar
         paths + nc1_hash.

    Caller writes the sidecar JSON after this returns.
    """
    prune_non_canonical_refs(cache)
    prune_non_postprocess_outputs(cache, out_dir)
    prune_orphan_files(cache, out_dir)
    rename_files_to_bom_ids(cache, out_dir)


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
