"""
CNC Shape Analyser service.

Given a STEP file path and a ref_id, opens the XCAF shape and runs the
full detection pipeline to classify the part as:

  - plate   -> export 2D DXF profile for laser cutting
  - section -> identify standard designation, detect holes/end-cuts,
               produce NC1 (DSTV) file for the cutting/drilling machine
  - unknown -> flag for manual review

For multi-solid compounds, each solid is analysed independently.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import structlog

from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.TDF import TDF_Label, TDF_Tool
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds

from app.exceptions import STEPParseError
from app.pipeline.geom_alignment import align_by_longest_straight_edge, align_by_obb
from app.pipeline.geometry_utils import (
    compute_obb_geometry,
    compute_section_area,
    compute_section_dims,
    compute_dstv_pose,
    compute_dstv_origin,
    align_obb_to_dstv_frame,
    swap_width_and_height_if_required,
    count_solids_in_shape,
    get_volume_from_shape,
    _rotY,
    _rotZ,
)
from app.pipeline.classification import classify_profile
from app.pipeline.plate_wrangling import align_plate_to_xy_plane
from app.pipeline.dstv_geometry import (
    classify_and_project_holes_dstv,
    analyze_end_faces_web_and_flange,
    check_duplicate_holes,
)
from app.pipeline.dstv_writer import assemble_dstv_header_data, generate_nc1_file
from app.pipeline.cad_out import export_profile_dxf_with_pca

logger = structlog.get_logger()

# Paths to section classification libraries
_DATA_DIR = Path(__file__).parent.parent / "pipeline" / "data"
_SECTION_LIBRARY = _DATA_DIR / "Shape_classifier_info.json"
_AL_SECTION_LIBRARY = _DATA_DIR / "Aluminium_classifier_info.json"

# Fallback project/material when caller does not supply values
_DEFAULT_PROJECT = "000"
_DEFAULT_MATERIAL = "S275"

# Material density kg/mm³  (steel 7.85 g/cm³, aluminium 2.71 g/cm³)
_DENSITY_STEEL = 7.85e-6
_DENSITY_AL = 2.71e-6

# Substrings in the grade field that indicate aluminium
_AL_GRADE_HINTS = ("6061", "6063", "6082", "6005", "5083", "5052",
                   "aluminium", "aluminum", "alloy 6", "al ")


def _is_aluminium(grade: Optional[str]) -> bool:
    """Return True if the material grade string indicates aluminium."""
    if not grade:
        return False
    g = grade.lower().strip()
    return any(hint in g for hint in _AL_GRADE_HINTS)


def match_confidence(score: Optional[float]) -> str:
    """Map a classification match score to a human-readable confidence label.

    Score composition: dh + dw + 100*ea + 100*el  (mm + %)
      - Excellent: dims within ~0.5mm, area <1%
      - Good:      dims within ~2mm, area <3%
      - Marginal:  dims a few mm off, area ~5%
      - Review:    anything worse — likely wrong profile or bad alignment
    """
    if score is None:
        return "Review"
    if score < 5.0:
        return "Excellent"
    if score < 15.0:
        return "Good"
    if score < 30.0:
        return "Marginal"
    return "Review"


def _library_for_grade(grade: Optional[str]) -> Path:
    """Select the section profile library based on material grade."""
    if _is_aluminium(grade) and _AL_SECTION_LIBRARY.exists():
        return _AL_SECTION_LIBRARY
    return _SECTION_LIBRARY


def _density_for_grade(grade: Optional[str]) -> float:
    """Return material density in kg/mm³ based on material grade."""
    return _DENSITY_AL if _is_aluminium(grade) else _DENSITY_STEEL


def _safe_name(name: Optional[str]) -> str:
    """Sanitise a name for use in a filename: keep alphanumerics and hyphens only."""
    if not name:
        return "unknown"
    # Strip common CAD export suffixes (e.g. "_Default<As Machined>")
    s = re.sub(r'_Default(?:<[^>]*>)?$', '', name)
    s = re.sub(r'[^\w\-]', '_', s).strip('_')
    return s[:64] or "unknown"


def _read_xcaf(file_path: str):
    """Open the STEP file via STEPCAFControl_Reader and return (doc, shape_tool)."""
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(False)
    reader.SetLayerMode(False)

    status = reader.ReadFile(str(file_path))
    if status != IFSelect_RetDone:
        raise STEPParseError(
            "STEPCAFControl_Reader failed",
            details={"file": str(file_path), "status": int(status)},
        )

    if not reader.Transfer(doc):
        raise STEPParseError(
            "STEPCAFControl_Reader transfer failed",
            details={"file": str(file_path)},
        )

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    return doc, shape_tool


def _get_shape(doc, ref_id: str):
    """Look up a prototype shape by its ref_id string."""
    label = TDF_Label()
    TDF_Tool.Label_s(doc.GetData(), TCollection_AsciiString(ref_id), label)
    if label.IsNull():
        raise ValueError(f"ref_id not found in XCAF document: {ref_id!r}")

    shape = XCAFDoc_ShapeTool.GetShape_s(label)
    if shape is None or shape.IsNull():
        raise ValueError(f"Empty shape for ref_id: {ref_id!r}")
    return shape


def _iter_solids(shape):
    """Iterate over all SOLID sub-shapes in a compound."""
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        yield TopoDS.Solid_s(exp.Current())
        exp.Next()


def _detect_hollow_section(shape, obb: Dict,
                           ref_id: str, solid_idx: int,
                           parent_name: Optional[str] = None,
                           steel_grade: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Detect hollow sections (CHS and RHS/SHS) that are not in the standard library.

    Key insight: compute_section_area() can fail for smooth cylinders (OCC section
    algorithm may not close circular wires), falling back to volume/L which equals
    actual_csa — making the hollow ratio = 1.0 and masking the hollow shape.

    Fix: estimate the outer boundary area from OBB dimensions directly (not from
    compute_section_area), and confirm hollowness with a cross-section wire count
    (hollow shapes have ≥2 closed wire loops; solid shapes have 1).

    CHS (round tube):  H ≈ W, est_outer = π/4 · OD², actual_csa << est_outer
    RHS/SHS (rectangular hollow): est_outer = H × W, actual_csa << est_outer
    """
    density = _density_for_grade(steel_grade)
    L, H, W = obb["aligned_extents"]

    if L < 5.0 or H < 5.0 or W < 5.0:
        return None

    try:
        volume = get_volume_from_shape(shape)
    except Exception:
        return None

    if volume <= 0:
        return None

    actual_csa = volume / L
    bounding_rect_area = H * W
    roundness = abs(H - W) / max(H, W, 1.0)

    def _wire_count() -> Optional[int]:
        """
        Try to count closed wire loops in the cross-section at mid-length
        (and fallback positions). Returns count or None if OCC can't slice it.
        A hollow shape produces ≥2 loops; a solid shape produces 1.
        Uses relaxed tolerance (1mm) to handle smooth circular boundaries.
        """
        for frac in (0.50, 0.25, 0.75):
            try:
                x_pos = L * frac
                plane = gp_Pln(gp_Ax3(gp_Pnt(x_pos, 0, 0), gp_Dir(1, 0, 0)))
                sec = BRepAlgoAPI_Section(shape, plane)
                sec.ComputePCurveOn1(True)
                sec.Approximation(True)
                sec.Build()
                if not sec.IsDone():
                    continue
                fb = ShapeAnalysis_FreeBounds(sec.Shape(), 1e-3, False, False)
                seq = fb.GetClosedWires()
                if seq is not None and hasattr(seq, "Length"):
                    n = int(seq.Length())
                    if n > 0:
                        return n
            except Exception:
                continue
        return None

    # ----------------------------------------------------------------
    # CHS check: H ≈ W → circular bounding cross-section
    # est_outer_circle uses OBB-derived OD (reliable even when OCC
    # cross-section slicing fails for smooth cylindrical surfaces)
    # ----------------------------------------------------------------
    if roundness < 0.15:
        OD = (H + W) / 2.0
        est_outer_circle = math.pi / 4.0 * OD ** 2

        if actual_csa < 0.90 * est_outer_circle:
            # Volume-based hollow test: material fills less than 90% of a solid rod
            n_wires = _wire_count()
            if n_wires is not None and n_wires < 2:
                # Cross-section is a solid outline (e.g. equal-leg angle) — not CHS
                logger.info("hollow_chs_rejected_solid_xsec",
                            ref_id=ref_id, n_wires=n_wires)
            else:
                # n_wires >= 2 (confirmed hollow) or None (OCC couldn't slice → trust volume)
                inner_area = est_outer_circle - actual_csa
                ID_sq = 4.0 * inner_area / math.pi
                if ID_sq > 0:
                    ID = math.sqrt(ID_sq)
                    t = (OD - ID) / 2.0
                    if 1.0 <= t <= 0.45 * OD:
                        designation = f"{round(OD, 1)}x{round(t, 1)}"
                        logger.info("hollow_chs_detected", ref_id=ref_id,
                                    OD=round(OD, 1), ID=round(ID, 1),
                                    t=round(t, 1), L=round(L, 1), n_wires=n_wires)
                        return {
                            "type": "section",
                            "category": "CHS",
                            "designation": designation,
                            "profile_type": "RO",
                            "dims": {"L": round(L, 1), "H": round(OD, 1), "W": round(OD, 1)},
                            "volume_mm3": round(volume, 1),
                            "mass_kg": round(volume * density, 3),
                            "holes": 0,
                            "end_cuts": False,
                            "nc1_path": None,
                            "nc1_hash": None,
                            "match_score": None,
                            "analysed_at": datetime.now(timezone.utc).isoformat(),
                            "note": "CHS auto-detected",
                        }

    # ----------------------------------------------------------------
    # RHS/SHS check: material area significantly less than bounding box
    # ----------------------------------------------------------------
    if actual_csa < 0.85 * bounding_rect_area:
        n_wires = _wire_count()
        if n_wires is not None and n_wires < 2:
            logger.info("hollow_rhs_rejected_solid_xsec",
                        ref_id=ref_id, n_wires=n_wires)
        else:
            # Wall thickness estimate (thin-wall): t ≈ (H·W − actual_csa) / (2·(H+W))
            t_approx = (bounding_rect_area - actual_csa) / max(2.0 * (H + W), 1.0)
            if 1.0 <= t_approx <= 0.20 * min(H, W):
                if abs(H - W) < 0.05 * max(H, W):
                    category = "SHS"
                    designation = f"{int(round(H))}x{round(t_approx, 1)}"
                else:
                    category = "RHS"
                    H_r, W_r = int(round(max(H, W))), int(round(min(H, W)))
                    designation = f"{H_r}x{W_r}x{round(t_approx, 1)}"
                logger.info("hollow_rhs_detected", ref_id=ref_id, category=category,
                            H=round(H, 1), W=round(W, 1),
                            t=round(t_approx, 1), L=round(L, 1), n_wires=n_wires)
                return {
                    "type": "section",
                    "category": category,
                    "designation": designation,
                    "profile_type": "RU",
                    "dims": {"L": round(L, 1), "H": round(H, 1), "W": round(W, 1)},
                    "volume_mm3": round(volume, 1),
                    "mass_kg": round(volume * density, 3),
                    "holes": 0,
                    "end_cuts": False,
                    "nc1_path": None,
                    "nc1_hash": None,
                    "match_score": None,
                    "analysed_at": datetime.now(timezone.utc).isoformat(),
                    "note": f"{category} auto-detected",
                }

    return None


def _analyse_single(shape, solid_idx: int, ref_id: str, member_id: str,
                    file_path: str, out_dir: Path,
                    parent_name: Optional[str] = None,
                    project_number: Optional[str] = None,
                    steel_grade: Optional[str] = None,
                    instance_count: int = 1) -> Dict[str, Any]:
    """
    Run the full pipeline on one solid shape and attach measured features.

    Wraps :func:`_analyse_single_impl` and merges a ``features`` block of
    rule-independent measured geometry (OBB extents, cross-section area,
    fill_ratio, inertia, …) into the result.  Persisting these alongside the
    rule verdict turns each analysed sidecar into ML training data without a
    separate re-extraction pass.  Best-effort: a failure here never affects the
    classification result.
    """
    result = _analyse_single_impl(
        shape, solid_idx, ref_id, member_id, file_path, out_dir,
        parent_name, project_number, steel_grade, instance_count,
    )
    if isinstance(result, dict) and "features" not in result:
        try:
            from app.pipeline.feature_extract import extract_solid_features
            result["features"] = extract_solid_features(shape)
        except Exception as e:
            logger.debug("feature_extract_failed", ref_id=ref_id, error=str(e))
    return result


def _analyse_single_impl(shape, solid_idx: int, ref_id: str, member_id: str,
                    file_path: str, out_dir: Path,
                    parent_name: Optional[str] = None,
                    project_number: Optional[str] = None,
                    steel_grade: Optional[str] = None,
                    instance_count: int = 1) -> Dict[str, Any]:
    """
    Run the full pipeline on one solid shape.
    Returns a result dict with type='plate', 'section', or 'unknown'.
    """
    # Select section library and density based on material grade
    section_lib = str(_library_for_grade(steel_grade))
    density = _density_for_grade(steel_grade)

    # Step 1: align by longest straight edge
    try:
        aligned, trsf, world_cs, _, dir_x, dir_y, dir_z, dbg = align_by_longest_straight_edge(shape)
    except Exception as e:
        return {"type": "unknown", "message": f"Alignment failed: {e}"}

    # Step 2: OBB geometry
    try:
        obb = compute_obb_geometry(aligned)
    except Exception as e:
        return {"type": "unknown", "message": f"OBB failed: {e}"}

    L, H, W = obb["aligned_extents"]

    # Step 2b: early AABB inflation check.
    # Mesh-converted shapes have triangulated faces whose diagonal edges are
    # longer than the actual extrusion edges.  align_by_longest_straight_edge
    # picks these diagonals, rotating the shape off-axis and distorting the
    # cross-section.  Detect this early so we don't trust a false positive
    # from the initial classify on the distorted shape.
    _alignment_suspect = False
    try:
        from OCP.BRepBndLib import BRepBndLib as _BRepBndLib
        from OCP.Bnd import Bnd_Box as _Bnd_Box_early
        _raw_bbox_early = _Bnd_Box_early()
        _BRepBndLib.Add_s(shape, _raw_bbox_early)
        _rxn, _ryn, _rzn, _rxx, _ryx, _rzx = _raw_bbox_early.Get()
        _raw_dims_early = sorted([_rxx - _rxn, _ryx - _ryn, _rzx - _rzn], reverse=True)
        _aligned_sorted_early = sorted([L, H, W], reverse=True)
        if _raw_dims_early[0] > 1e-9:
            _dim_ratios_early = [
                a / max(r, 1e-9)
                for a, r in zip(_aligned_sorted_early, _raw_dims_early)
            ]
            _max_infl_early = max(_dim_ratios_early)
            _min_ratio_early = min(_dim_ratios_early)
            _alignment_suspect = _raw_dims_early[0] > 20.0 and _max_infl_early > 1.15
            # For correctly-aligned angled parts the extrusion length grows
            # (exceeds axis-aligned projection) but the cross-section height
            # shrinks dramatically.  Mesh-diagonal inflation grows ALL dims.
            # If any dim ratio dropped below 0.50 the alignment likely found
            # the true extrusion axis of a shape at an angle — not suspect.
            if _alignment_suspect and _min_ratio_early < 0.50:
                _alignment_suspect = False
                logger.debug("angled_alignment_accepted", ref_id=ref_id,
                             max_inflation=round(_max_infl_early, 3),
                             min_ratio=round(_min_ratio_early, 3),
                             raw_dims=[round(d, 1) for d in _raw_dims_early],
                             aligned_dims=[round(L, 1), round(H, 1), round(W, 1)])
            if _alignment_suspect:
                logger.debug("alignment_suspect", ref_id=ref_id,
                             max_inflation=round(_max_infl_early, 3),
                             raw_dims=[round(d, 1) for d in _raw_dims_early],
                             aligned_dims=[round(L, 1), round(H, 1), round(W, 1)])
    except Exception:
        pass

    # Step 3: cross-section dims (area + H/W from core region, avoiding end chamfers)
    try:
        section_dims = compute_section_dims(aligned)
        section_area = section_dims["area"]
        H_cs = section_dims.get("H") or H   # fallback to OBB
        W_cs = section_dims.get("W") or W
    except Exception as e:
        section_area = 0.0
        H_cs, W_cs = H, W
        logger.warning("section_dims_failed", ref_id=ref_id, error=str(e))

    cs = {
        "span_web":    H_cs,
        "span_flange": W_cs,
        "area":        section_area,
        "length":      L,
    }

    # Step 3b: early plate detection.
    # If the thinnest OBB dimension is much smaller than the other two,
    # this is unambiguously a plate — skip section classification entirely
    # to avoid false positives from loose tolerance passes.
    _sorted_dims = sorted([L, H, W])
    _thin = _sorted_dims[0]           # thinnest dimension
    _mid  = _sorted_dims[1]           # middle dimension
    _long = _sorted_dims[2]           # longest dimension
    _obvious_plate = (
        _thin < 25.0
        and _mid > _thin * 4.0
        and _long > _thin * 4.0
    )
    if _obvious_plate:
        logger.info("early_plate_detected", ref_id=ref_id,
                     thin=round(_thin, 1), mid=round(_mid, 1), long=round(_long, 1))

    # Step 4: try to classify as a standard section profile
    # Skip if alignment is suspect — the distorted cross-section can produce
    # false positives (e.g. a channel cross-section cut on a diagonal looks
    # like an equal angle).  The AABB retry (Step 4c) will handle these.
    # Also skip if shape is obviously a plate.
    profile_match = None
    if not _alignment_suspect and not _obvious_plate:
        try:
            profile_match = classify_profile(cs, section_lib)
        except Exception as e:
            profile_match = None
            logger.warning("classify_profile_failed", ref_id=ref_id, error=str(e))

    # Step 4a: sanity-check — reject if OBB height far exceeds the library
    # profile height.  This catches false positives from oblique cross-sections
    # when the alignment picked a non-extrusion axis (e.g. angled parts).
    if profile_match:
        _H_lib_chk = profile_match["JSON"]["height"]
        if H > _H_lib_chk * 1.5 and H > _H_lib_chk + 50:
            logger.warning("classification_rejected_obb_height",
                           ref_id=ref_id,
                           designation=profile_match.get("Designation"),
                           H_obb=round(H, 1), H_lib=round(_H_lib_chk, 1))
            # Fallback: try OBB-based re-alignment on the original shape
            _obb_fallback_ok = False
            try:
                _aligned_obb, _trsf_obb, _, _, _, _, _, _ = align_by_obb(shape)
                _obb_obb = compute_obb_geometry(_aligned_obb)
                _L_o, _H_o, _W_o = _obb_obb["aligned_extents"]
                if _L_o >= 20.0:
                    _sd_o = compute_section_dims(_aligned_obb)
                    _area_o = _sd_o["area"]
                    _H_cs_o = _sd_o.get("H") or _H_o
                    _W_cs_o = _sd_o.get("W") or _W_o
                    _cs_o = {"span_web": _H_cs_o, "span_flange": _W_cs_o,
                             "area": _area_o, "length": _L_o}
                    _match_o = classify_profile(_cs_o, section_lib)
                    if _match_o:
                        _H_lib_o = _match_o["JSON"]["height"]
                        if _H_o <= _H_lib_o * 1.5 or _H_o <= _H_lib_o + 50:
                            logger.info("obb_realign_match", ref_id=ref_id,
                                        designation=_match_o.get("Designation"),
                                        score=_match_o.get("Match_score"),
                                        L=round(_L_o, 1), H=round(_H_o, 1))
                            _match_o.setdefault("Diagnostics", {})["obb_realign"] = True
                            _obb_fallback_ok = True
                            return _process_section(
                                _aligned_obb, _obb_obb, _match_o, ref_id,
                                member_id, solid_idx, file_path, out_dir,
                                parent_name, project_number, steel_grade,
                                instance_count)
            except Exception as e:
                logger.debug("obb_realign_failed", ref_id=ref_id, error=str(e))
            if not _obb_fallback_ok:
                profile_match = None
                _alignment_suspect = True

    if profile_match:
        logger.info("section_match_found", ref_id=ref_id,
                    designation=profile_match.get("Designation"),
                    score=profile_match.get("Match_score"))
        return _process_section(aligned, obb, profile_match, ref_id, member_id,
                                solid_idx, file_path, out_dir, parent_name,
                                project_number, steel_grade, instance_count)

    logger.info("no_section_match", ref_id=ref_id,
                L=round(L, 1), H=round(H, 1), W=round(W, 1),
                section_area=round(section_area, 1))

    # Step 4b: axis-retry — try Y-as-X and Z-as-X before giving up
    # For short items the longest straight edge may be the web height, not the
    # extrusion length.  Rotating so a different OBB axis becomes X lets the
    # existing X-hardcoded pipeline re-attempt classification.
    # Skip when alignment is suspect — axis rotations of a distorted shape
    # are also unreliable; the AABB retry (Step 4c) uses the original shape.
    axis_retry_candidates = []
    for rot_label, rot_fn, rot_args in [] if (_alignment_suspect or _obvious_plate) else [
        ("Y-as-X", _rotZ, 1),   # +90 deg about Z: old Y becomes new X
        ("Z-as-X", _rotY, 3),   # +270 deg about Y: old Z becomes new X
    ]:
        try:
            rotated = rot_fn(aligned, rot_args)
            obb_r = compute_obb_geometry(rotated)
            L_r, H_r, W_r = obb_r["aligned_extents"]
            if L_r < 20.0:
                continue
            sd_r = compute_section_dims(rotated)
            area_r = sd_r["area"]
            H_cs_r = sd_r.get("H") or H_r
            W_cs_r = sd_r.get("W") or W_r
            cs_r = {"span_web": H_cs_r, "span_flange": W_cs_r,
                     "area": area_r, "length": L_r}
            match_r = classify_profile(cs_r, section_lib)
            if match_r:
                logger.info("axis_retry_match", ref_id=ref_id, axis=rot_label,
                            designation=match_r.get("Designation"),
                            score=match_r.get("Match_score"))
                match_r.setdefault("Diagnostics", {})["axis_retry"] = rot_label
                axis_retry_candidates.append((match_r, rotated, obb_r))
        except Exception as e:
            logger.debug("axis_retry_error", ref_id=ref_id, axis=rot_label,
                         error=str(e))

    if axis_retry_candidates:
        axis_retry_candidates.sort(
            key=lambda t: t[0].get("Match_score", float("inf")))
        best_match, best_shape, best_obb = axis_retry_candidates[0]
        logger.info("axis_retry_selected", ref_id=ref_id,
                     designation=best_match.get("Designation"),
                     score=best_match.get("Match_score"))
        return _process_section(best_shape, best_obb, best_match, ref_id,
                                member_id, solid_idx, file_path, out_dir,
                                parent_name, project_number, steel_grade,
                                instance_count)

    # Step 4c: AABB-axis retry for mesh-converted shapes.
    # Mesh triangulation creates diagonal edges longer than the actual extrusion
    # edges, causing align_by_longest_straight_edge to rotate the shape off-axis.
    # Skip for obvious plates — go straight to the plate path.
    if not _obvious_plate:
      try:
        from OCP.BRepBndLib import BRepBndLib
        from OCP.Bnd import Bnd_Box as _Bnd_Box
        raw_bbox = _Bnd_Box()
        BRepBndLib.Add_s(shape, raw_bbox)
        rxn, ryn, rzn, rxx, ryx, rzx = raw_bbox.Get()
        raw_dims = sorted([rxx - rxn, ryx - ryn, rzx - rzn], reverse=True)
        aligned_sorted = sorted([L, H, W], reverse=True)
        # Max inflation of any dimension (sorted largest→smallest)
        _max_infl = max(
            a / max(r, 1e-9) for a, r in zip(aligned_sorted, raw_dims)
        ) if raw_dims[0] > 1e-9 else 1.0
        _aabb_trigger = raw_dims[0] > 20.0 and _max_infl > 1.15
        if not _aabb_trigger:
            # Also trigger on modest L inflation (>2%) as before
            _aabb_trigger = raw_dims[0] > 20.0 and L > raw_dims[0] * 1.02
        if _aabb_trigger:
            # The alignment inflated the length — try each AABB axis as extrusion
            logger.debug("aabb_retry_triggered", ref_id=ref_id,
                         aligned_L=round(L, 1), aabb_max=round(raw_dims[0], 1))
            aabb_candidates = []
            for rot_label, rot_fn, rot_args in [
                ("AABB-Y-as-X", _rotZ, 1),
                ("AABB-Z-as-X", _rotY, 3),
            ]:
                try:
                    rotated = rot_fn(shape, rot_args)  # rotate the ORIGINAL shape
                    obb_r = compute_obb_geometry(rotated)
                    L_r, H_r, W_r = obb_r["aligned_extents"]
                    if L_r < 20.0:
                        continue
                    sd_r = compute_section_dims(rotated)
                    area_r = sd_r["area"]
                    H_cs_r = sd_r.get("H") or H_r
                    W_cs_r = sd_r.get("W") or W_r
                    cs_r = {"span_web": H_cs_r, "span_flange": W_cs_r,
                             "area": area_r, "length": L_r}
                    match_r = classify_profile(cs_r, section_lib)
                    if match_r:
                        logger.info("aabb_retry_match", ref_id=ref_id,
                                    axis=rot_label,
                                    designation=match_r.get("Designation"),
                                    score=match_r.get("Match_score"))
                        match_r.setdefault("Diagnostics", {})["aabb_retry"] = rot_label
                        aabb_candidates.append((match_r, rotated, obb_r))
                except Exception as e:
                    logger.debug("aabb_retry_error", ref_id=ref_id, axis=rot_label,
                                 error=str(e))

            # Also try the original shape unrotated (longest AABB dim already on X?)
            try:
                obb_o = compute_obb_geometry(shape)
                L_o, H_o, W_o = obb_o["aligned_extents"]
                if L_o >= 20.0:
                    sd_o = compute_section_dims(shape)
                    area_o = sd_o["area"]
                    H_cs_o = sd_o.get("H") or H_o
                    W_cs_o = sd_o.get("W") or W_o
                    cs_o = {"span_web": H_cs_o, "span_flange": W_cs_o,
                             "area": area_o, "length": L_o}
                    match_o = classify_profile(cs_o, section_lib)
                    if match_o:
                        match_o.setdefault("Diagnostics", {})["aabb_retry"] = "original"
                        aabb_candidates.append((match_o, shape, obb_o))
            except Exception:
                pass

            if aabb_candidates:
                aabb_candidates.sort(
                    key=lambda t: t[0].get("Match_score", float("inf")))
                best_match, best_shape, best_obb = aabb_candidates[0]
                logger.info("aabb_retry_selected", ref_id=ref_id,
                             designation=best_match.get("Designation"),
                             score=best_match.get("Match_score"))
                return _process_section(best_shape, best_obb, best_match, ref_id,
                                        member_id, solid_idx, file_path, out_dir,
                                        parent_name, project_number, steel_grade,
                                        instance_count)
      except Exception as e:
        logger.debug("aabb_retry_skipped", ref_id=ref_id, error=str(e))

    # Step 4d: area-priority fallback for suspect alignments.
    # When a shape is oriented diagonally in world space (e.g. an angle brace
    # at 45°), the alignment may find the correct extrusion direction (giving
    # correct L and cross-section area) but the cross-section H/W bounding box
    # is distorted because the profile is rotated about its extrusion axis.
    # The area is still reliable — use it with very relaxed dims but strict
    # area tolerance, and guard against plates (thin shapes).
    _min_hw = min(H_cs, W_cs)
    _max_hw = max(H_cs, W_cs)
    _is_plate_like = _min_hw < 15.0 and _max_hw / max(_min_hw, 0.1) > 4.0
    if _alignment_suspect and section_area > 0 and not _is_plate_like and not _obvious_plate:
        try:
            # Match on area with relaxed dims.  Use a direct single-pass
            # approach: classify_profile internally escalates tolerances
            # through multiple passes, so we must check the diagnostics
            # to ensure only PASS 1 (strict area) was used.
            _tol_dim_area = _max_hw  # tolerance as large as the biggest dim
            _tol_area_area = 0.08    # tight area match (8%)
            cs_area = {
                "span_web":    _min_hw,
                "span_flange": _min_hw,
                "area":        section_area,
                "length":      L,
            }
            match_a = classify_profile(cs_area, section_lib,
                                       tol_dim=_tol_dim_area,
                                       tol_area=_tol_area_area)
            if match_a:
                # Reject if classify_profile had to relax beyond PASS 1.
                diag = match_a.get("Diagnostics", {})
                used_area_tol = diag.get("tol_area_used", 1.0)
                match_score = match_a.get("Match_score", float("inf"))
                # Score ceiling: area-priority is a fallback — reject if the
                # composite score is too high (dims wildly off even though
                # area matched).  Prevents matching to the wrong profile
                # when the correct one is missing from the library.
                _AREA_PRIORITY_SCORE_CEILING = 40.0
                if used_area_tol <= _tol_area_area and match_score <= _AREA_PRIORITY_SCORE_CEILING:
                    logger.info("area_priority_match", ref_id=ref_id,
                                designation=match_a.get("Designation"),
                                score=match_a.get("Match_score"),
                                section_area=round(section_area, 1))
                    match_a.setdefault("Diagnostics", {})["area_priority"] = True
                    return _process_section(aligned, obb, match_a, ref_id,
                                            member_id, solid_idx, file_path, out_dir,
                                            parent_name, project_number, steel_grade,
                                            instance_count)
                else:
                    logger.debug("area_priority_rejected",
                                 ref_id=ref_id,
                                 used_area_tol=used_area_tol,
                                 score=match_score,
                                 ceiling=_AREA_PRIORITY_SCORE_CEILING,
                                 designation=match_a.get("Designation"))
        except Exception as e:
            logger.debug("area_priority_failed", ref_id=ref_id, error=str(e))

    # Step 5: try plate path
    try:
        is_plate, aligned_plate, ax3, T, plate_L, plate_W, mass, msg, sig = align_plate_to_xy_plane(aligned)
    except Exception as e:
        return {"type": "unknown", "message": f"Plate alignment error: {e}"}

    if is_plate and ax3 is not None:
        return _process_plate(aligned_plate, ax3, T, plate_L, plate_W, mass,
                              ref_id, member_id, solid_idx, out_dir, parent_name)

    logger.info("plate_path_rejected", ref_id=ref_id, solid_idx=solid_idx,
                msg=msg, sig=sig)

    # Step 6: try hollow section detection (CHS / RHS / SHS)
    try:
        hollow = _detect_hollow_section(aligned, obb,
                                        ref_id, solid_idx, parent_name,
                                        steel_grade)
        if hollow:
            return hollow
    except Exception as e:
        logger.warning("hollow_detection_failed", ref_id=ref_id, error=str(e))

    # Enrich unknown result with measured geometry so users can identify
    # missing profiles and add them to the library.
    unknown_result: Dict[str, Any] = {
        "type": "unknown",
        "message": msg or "Not plate or section",
        "dims": {"L": round(L, 1), "H": round(H, 1), "W": round(W, 1)},
    }
    try:
        vol_mm3 = get_volume_from_shape(aligned)
        mass_kg = round(vol_mm3 * density, 3)
        csa = round(vol_mm3 / L, 1) if L > 1.0 else 0.0
        unknown_result["volume_mm3"] = round(vol_mm3, 1)
        unknown_result["mass_kg"] = mass_kg
        unknown_result["section_area"] = csa
    except Exception:
        pass
    logger.warning("unmatched_section", ref_id=ref_id,
                   L=round(L, 1), H=round(H, 1), W=round(W, 1),
                   section_area=round(section_area, 1),
                   message=unknown_result["message"])
    return unknown_result


def _process_section(aligned, obb, profile_match: Dict, ref_id: str, member_id: str,
                     solid_idx: int, file_path: str, out_dir: Path,
                     parent_name: Optional[str] = None,
                     project_number: Optional[str] = None,
                     steel_grade: Optional[str] = None,
                     instance_count: int = 1) -> Dict[str, Any]:
    """Run the section pipeline: DSTV pose → holes → end cuts → NC1."""
    proj_num = project_number or _DEFAULT_PROJECT
    matl = steel_grade or _DEFAULT_MATERIAL
    try:
        # Optional swap if required_rotation flagged
        aligned, obb = swap_width_and_height_if_required(profile_match, aligned, obb)

        profile_type = profile_match.get("Profile_type", "I")
        profile_dims = profile_match.get("JSON", {})

        # DSTV pose  (channel_mode="web_at_z0" matches original dstv.py and
        # the hole-detection convention: channel web at Z=0, flanges at Z=W)
        refined_shape, dstv_ax3, tr_world_to_dstv, pose_diag = compute_dstv_pose(
            aligned, profile_type, profile_dims, channel_mode="web_at_z0"
        )
        logger.info("dstv_pose", ref_id=ref_id,
                    profile_type=profile_type,
                    turns=pose_diag.get("turns_about_X"),
                    y_norm=round(pose_diag.get("y_norm", 0), 3),
                    z_norm=round(pose_diag.get("z_norm", 0), 3),
                    policy=pose_diag.get("policy"),
                    confidence=pose_diag.get("confidence"))

        # Dimensions — use actual bbox of the *refined* (posed) shape so that
        # hole-detection thresholds reference the true geometry rather than the
        # library nominal values (which may differ by several mm).
        step_vals = profile_match.get("STEP", {})
        L_mm = float(pose_diag.get("L", obb["aligned_extents"][0]))
        H_mm = float(pose_diag.get("H", step_vals.get("height", obb["aligned_extents"][1])))
        W_mm = float(pose_diag.get("W", step_vals.get("width",  obb["aligned_extents"][2])))

        # DSTV origin at (0,0,0) since compute_dstv_pose translates to min-corner
        origin_dstv = gp_Pnt(0.0, 0.0, 0.0)

        logger.info("hole_detection_dims", ref_id=ref_id,
                    profile_type=profile_type,
                    L_mm=round(L_mm, 1), H_mm=round(H_mm, 1), W_mm=round(W_mm, 1))

        # Holes
        try:
            df_holes = classify_and_project_holes_dstv(
                refined_shape, dstv_ax3, origin_dstv,
                W_mm, H_mm, profile_match
            )
            has_dups, _ = check_duplicate_holes(df_holes)
            if has_dups:
                logger.warning("duplicate_holes_detected", ref_id=ref_id)
        except Exception as e:
            df_holes = None
            logger.warning("hole_classification_failed", ref_id=ref_id, error=str(e))

        import pandas as pd
        if df_holes is None:
            df_holes = pd.DataFrame()

        if not df_holes.empty:
            codes = df_holes["Code"].value_counts().to_dict()
            logger.info("holes_detected", ref_id=ref_id, total=len(df_holes), by_face=codes)
        else:
            logger.warning("no_holes_detected", ref_id=ref_id,
                           H_mm=round(H_mm, 1), W_mm=round(W_mm, 1))

        # End-face cuts
        try:
            web_cut = analyze_end_faces_web_and_flange(refined_shape, dstv_ax3)
            if not web_cut:
                web_cut = {"start_web": 0.0, "end_web": 0.0, "start_flange": 0.0, "end_flange": 0.0}
            logger.info("end_face_cuts", ref_id=ref_id, **{k: round(v, 2) for k, v in web_cut.items()})
        except Exception as e:
            web_cut = {"start_web": 0.0, "end_web": 0.0, "start_flange": 0.0, "end_flange": 0.0}
            logger.warning("end_face_analysis_failed", ref_id=ref_id, error=str(e))

        has_end_cuts = any(abs(v) > 0.01 for v in web_cut.values())

        # NC1 filename: {project_number}-{member_id}-{occurrence}.nc1
        nc1_name = f"{_safe_name(proj_num)}-{_safe_name(member_id)}-{solid_idx + 1}"
        nc1_dir = out_dir / "nc1"

        header_data = assemble_dstv_header_data(
            project_number=proj_num,
            step_path=file_path,
            matl_grade=matl,
            member_id=nc1_name,
            profile_match=profile_match,
            quantity=instance_count,
        )

        nc1_path, nc1_hash = generate_nc1_file(df_holes, header_data, nc1_dir, web_cut)

        # Volume / weight (best-effort; aligned shape is the one after swap+pose)
        density = _density_for_grade(steel_grade)
        try:
            vol_mm3 = get_volume_from_shape(aligned)
            mass_kg = round(vol_mm3 * density, 3)
            vol_mm3 = round(vol_mm3, 1)
        except Exception:
            vol_mm3 = None
            mass_kg = None

        _score = profile_match.get("Match_score", 0.0)
        return {
            "type": "section",
            "category": profile_match.get("Category", ""),
            "designation": profile_match.get("Designation", ""),
            "profile_type": profile_type,
            "dims": {"L": round(L_mm, 1), "H": round(H_mm, 1), "W": round(W_mm, 1)},
            "volume_mm3": vol_mm3,
            "mass_kg": mass_kg,
            "holes": len(df_holes) if not df_holes.empty else 0,
            "end_cuts": has_end_cuts,
            "nc1_path": str(nc1_path),
            "nc1_hash": nc1_hash,
            "match_score": _score,
            "confidence": match_confidence(_score),
            "analysed_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error("section_processing_failed", ref_id=ref_id, error=str(e))
        return {"type": "unknown", "message": f"Section processing failed: {e}"}


def _process_plate(aligned_plate, ax3, T: float, plate_L: float, plate_W: float,
                   mass: float, ref_id: str, member_id: str,
                   solid_idx: int, out_dir: Path,
                   parent_name: Optional[str] = None) -> Dict[str, Any]:
    """Run the plate pipeline: export DXF profile."""
    try:
        # Filename: {thickness_mm}-{member_id}-{occurrence}.dxf
        thickness_mm = int(round(T))
        dxf_stem = f"{thickness_mm}-{_safe_name(member_id)}-{solid_idx + 1}"
        dxf_dir = out_dir / "plates"
        dxf_dir.mkdir(parents=True, exist_ok=True)
        dxf_path = dxf_dir / f"{dxf_stem}.dxf"

        _, dxf_out, _ = export_profile_dxf_with_pca(
            aligned_plate,
            dxf_path=dxf_path,
            ax3=ax3,
            canonicalize=True,
        )

        return {
            "type": "plate",
            "dims": {
                "L": round(float(plate_L), 1),
                "W": round(float(plate_W), 1),
                "T": round(float(T), 1),
            },
            "volume_mm3": round(float(plate_L) * float(plate_W) * float(T), 1),
            "mass_kg": round(float(mass), 3),
            "dxf_path": str(dxf_out),
            "confidence": "Excellent",
            "analysed_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error("plate_processing_failed", ref_id=ref_id, error=str(e))
        return {"type": "unknown", "message": f"Plate processing failed: {e}"}


def analyse_ref(file_path: str, ref_id: str, out_dir: Path,
                member_id: Optional[str] = None,
                parent_name: Optional[str] = None,
                project_number: Optional[str] = None,
                steel_grade: Optional[str] = None,
                instance_count: int = 1,
                *,
                _doc=None,
                _shape_tool=None) -> Dict[str, Any]:
    """
    Analyse one XCAF prototype shape identified by ref_id.

    Returns a result dict:
      - type='plate'   -> {type, dims, mass_kg, dxf_path}
      - type='section' -> {type, category, designation, dims, nc1_path, holes, end_cuts}
      - type='multi_solid' -> {type, solids: [result_per_solid, ...]}
      - type='unknown' -> {type, message}

    ``_doc`` and ``_shape_tool`` are optional keyword-only arguments that allow
    the caller to supply a pre-opened XCAF document.  When processing many
    ref_ids from the same file, pass the result of ``_read_xcaf(file_path)``
    once and reuse it for every call — this avoids the OCC memory build-up that
    occurs when the full STEP file is parsed N times in the same process.
    """
    if _doc is not None and _shape_tool is not None:
        doc, shape_tool = _doc, _shape_tool
    else:
        doc, shape_tool = _read_xcaf(file_path)
    shape = _get_shape(doc, ref_id)

    n_solids = count_solids_in_shape(shape)

    safe_member = member_id or ref_id.replace(":", "-")

    if n_solids > 1:
        results = []
        for idx, solid in enumerate(_iter_solids(shape)):
            solid_member = f"{safe_member}-s{idx}"
            r = _analyse_single(solid, idx, ref_id, solid_member, file_path, out_dir,
                                parent_name, project_number, steel_grade,
                                instance_count)
            results.append(r)
        return {
            "type": "multi_solid",
            "n_solids": n_solids,
            "solids": results,
            "analysed_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        return _analyse_single(shape, 0, ref_id, safe_member, file_path, out_dir,
                               parent_name, project_number, steel_grade,
                               instance_count)
