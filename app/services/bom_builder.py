"""Unified native-BOM construction for STEP and IFC inputs.

IFC parsers build their own ``native_bom`` at parse time from file metadata
(profile, material, weight, dimensions straight out of Psets). STEP doesn't
carry that data — we derive equivalents from the CNC analysis pass and the
XCAF tree, then emit rows with the *same schema* so downstream consumers
(frontend BOM view, Excel/COBie exporters) treat both sources uniformly.

Row schema (matches ``BaseIFCParser._bom_row_from_node``):

    node_id, guid, type_guid, entity, mark, profile, material, grade,
    length, weight, volume, area,
    assembly_mark, assembly_position, pour, phase, finish, part_class,
    source

Fields not present in STEP (phase, pour, finish, part_class, ...) are left
``None``. Units match IFC: weight kg, length mm, volume m³, area m².
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


_STEP_LEAF_TYPES = ("part_single_solid", "part_multi_solid", "part_no_solid")


def build_step_bom(
    tree: List[Dict[str, Any]],
    cnc_analysis: Optional[Dict[str, Any]] = None,
    *,
    cnc_member_names: Optional[Dict[str, str]] = None,
    parent_names: Optional[Dict[str, str]] = None,
    steel_grade: str = "",
    project_number: str = "",
) -> List[Dict[str, Any]]:
    """Produce one BOM row per physical instance for every STEP tree leaf.

    Tree-first: we walk the assembly tree and emit a row for *every* leaf part,
    regardless of whether CNC analysis has run for that ref_id. Rows with a
    matching cnc_analysis entry are enriched (profile, weight, dims, entity
    category); rows without produce a placeholder with mark + material only so
    the BOM evidences every part, including Bought-Out and Excluded items that
    never see CNC analysis.
    """
    if not tree:
        return []

    cnc_analysis = cnc_analysis or {}
    grade = (steel_grade or "S275").strip()
    material = grade  # STEP has no "STEEL/S275" namespaced form — use bare grade

    rows: List[Dict[str, Any]] = []

    def _walk(nodes: List[Dict[str, Any]], parent_name: Optional[str]) -> None:
        for node in nodes:
            nt = node.get("node_type")
            if nt == "assembly":
                _walk(node.get("children") or [], node.get("name"))
                continue
            if nt not in _STEP_LEAF_TYPES:
                continue

            ref_id = node.get("ref_id") or node.get("id") or ""
            instance = {
                "node_id": node.get("id"),
                "name": node.get("name"),
                "is_mirrored": node.get("is_mirrored", False),
                "parent_name": parent_name,
                "ref_id": ref_id,
                "node_type": nt,
            }
            result = cnc_analysis.get(ref_id)
            rows.extend(_rows_for_instance(
                instance, result, material, grade, parent_names,
            ))

    _walk(tree, None)
    return rows


def _rows_for_instance(
    instance: Dict[str, Any],
    result: Optional[Dict[str, Any]],
    material: str,
    grade: str,
    parent_names: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Emit 0+ BOM rows for a single tree leaf.

    - No CNC result → one placeholder row (minimal data, analysis pending)
    - CNC result, single-body → one enriched row
    - CNC result, multi-solid → one enriched row per sub-solid
    """
    if result is None:
        return [_placeholder_row(instance, material, grade, parent_names)]

    if result.get("type") == "multi_solid":
        out: List[Dict[str, Any]] = []
        for s_idx, solid in enumerate(result.get("solids") or []):
            out.append(_make_row(
                instance, solid,
                entity=_category_to_entity(
                    solid.get("category"), solid.get("type")
                ),
                profile=_profile_for(solid),
                material=material, grade=grade,
                node_id_suffix=f":s{s_idx}",
                mark_suffix=f" - Solid {s_idx + 1}",
                parent_names=parent_names,
            ))
        return out

    return [_make_row(
        instance, result,
        entity=_category_to_entity(
            result.get("category"), result.get("type")
        ),
        profile=_profile_for(result),
        material=material, grade=grade,
        node_id_suffix="",
        mark_suffix="",
        parent_names=parent_names,
    )]


def _placeholder_row(
    instance: Dict[str, Any],
    material: str,
    grade: str,
    parent_names: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Row for a STEP leaf without CNC results — mark, material, source only."""
    node_type = instance.get("node_type")
    if node_type == "part_multi_solid":
        entity = "MultiSolidPart"
    else:
        entity = "Part"

    assembly_mark = None
    if parent_names:
        assembly_mark = parent_names.get(instance.get("ref_id") or "")
    if not assembly_mark:
        assembly_mark = instance.get("parent_name")

    return {
        "node_id": instance.get("node_id"),
        "guid": None,
        "type_guid": None,
        "entity": entity,
        "mark": instance.get("name"),
        "profile": "",
        "material": material,
        "grade": grade,
        "length": None,
        "weight": None,
        "volume": None,
        "area": None,
        "assembly_mark": assembly_mark,
        "assembly_position": None,
        "pour": None,
        "phase": None,
        "finish": None,
        "part_class": None,
        "source": "step",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    instance: Dict[str, Any],
    result: Dict[str, Any],
    *,
    entity: str,
    profile: str,
    material: str,
    grade: str,
    node_id_suffix: str,
    mark_suffix: str,
    parent_names: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    dims = result.get("dims") or {}
    length = _pick_length(dims)
    volume_m3 = _mm3_to_m3(result.get("volume_mm3"))
    mass_kg = result.get("mass_kg")

    assembly_mark = None
    if parent_names:
        assembly_mark = parent_names.get(instance.get("ref_id") or "")
    if not assembly_mark:
        assembly_mark = instance.get("parent_name")

    return {
        "node_id": (instance.get("node_id") or "") + node_id_suffix,
        "guid": None,
        "type_guid": None,
        "entity": entity,
        "mark": (instance.get("name") or "") + mark_suffix,
        "profile": profile,
        "material": material,
        "grade": grade,
        "length": length,
        "weight": mass_kg,
        "volume": volume_m3,
        "area": None,
        "assembly_mark": assembly_mark,
        "assembly_position": None,
        "pour": None,
        "phase": None,
        "finish": None,
        "part_class": None,
        "source": "step",
    }


def _pick_length(dims: Dict[str, Any]) -> Optional[float]:
    """Return the extruded length from whichever key the analyser used."""
    for key in ("length", "Length", "L", "l", "major", "longest"):
        v = dims.get(key)
        if v is not None:
            return v
    return None


def _profile_for(result: Dict[str, Any]) -> str:
    """Derive a profile/designation string from a CNC analyser result.

    Sections: combine category (UC/UB/CHS/...) with designation (203x203x46)
    so the profile reads like an engineer would write it: ``"UC 203x203x46"``.
    Plates don't carry a designation — synthesise a Tekla-style
    ``PL{T}*{W}*{L}`` label from the plate dimensions.
    """
    designation = (result.get("designation") or "").strip()
    category = (result.get("category") or "").strip()
    if designation:
        if category and not designation.upper().startswith(category.upper()):
            return f"{category} {designation}"
        return designation
    if result.get("type") == "plate":
        dims = result.get("dims") or {}
        t, w = dims.get("T"), dims.get("W")
        if t is not None and w is not None:
            # Matches Tekla's PLT{thickness}*{width}; length stays in the
            # row's length field so both sources share one plate label shape.
            return f"PLT{_fmt(t)}*{_fmt(w)}"
    return ""


def _fmt(v: float) -> str:
    """Compact number formatting — integers print without trailing .0."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _mm3_to_m3(volume_mm3: Optional[float]) -> Optional[float]:
    if volume_mm3 is None:
        return None
    return volume_mm3 / 1.0e9


_HOLLOW_CODES = frozenset({"CHS", "SHS", "RHS"})


def _category_to_entity(category: Optional[str], type_: Optional[str]) -> str:
    """Map CNC analyser category/type → synthetic IFC-shaped entity label.

    This is purely a display hint; STEP parts have no real IFC class.
    Type is the primary signal (plate vs section); category disambiguates
    hollow sections (CHS/SHS/RHS) from open sections.
    """
    if type_ == "plate":
        return "Plate"
    if type_ == "section":
        cat = (category or "").strip().upper()
        if cat in _HOLLOW_CODES:
            return "HollowSection"
        return "Section"
    if type_ == "multi_solid":
        return "MultiSolidPart"
    return "Part"
