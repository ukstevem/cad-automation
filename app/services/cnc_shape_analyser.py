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
from app.pipeline.geom_alignment import align_by_longest_straight_edge
from app.pipeline.geometry_utils import (
    compute_obb_geometry,
    compute_section_area,
    compute_dstv_pose,
    compute_dstv_origin,
    align_obb_to_dstv_frame,
    swap_width_and_height_if_required,
    count_solids_in_shape,
    get_volume_from_shape,
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

# Path to the section classification library
_SECTION_LIBRARY = Path(__file__).parent.parent / "pipeline" / "data" / "Shape_classifier_info.json"

# Fallback project/material when caller does not supply values
_DEFAULT_PROJECT = "000"


_DEFAULT_MATERIAL = "S275"


def _safe_name(name: Optional[str]) -> str:
    """Sanitise a name for use in a filename: keep alphanumerics and hyphens only."""
    if not name:
        return "unknown"
    s = re.sub(r'[^\w\-]', '_', name).strip('_')
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
                           parent_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
                            "mass_kg": round(volume * 7.85e-6, 3),
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
                    "mass_kg": round(volume * 7.85e-6, 3),
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
                    steel_grade: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the full pipeline on one solid shape.
    Returns a result dict with type='plate', 'section', or 'unknown'.
    """
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

    # Step 3: cross-section area
    try:
        section_area = compute_section_area(aligned)
    except Exception as e:
        section_area = 0.0
        logger.warning("section_area_failed", ref_id=ref_id, error=str(e))

    cs = {
        "span_web":    H,
        "span_flange": W,
        "area":        section_area,
        "length":      L,
    }

    # Step 4: try to classify as a standard section profile
    try:
        profile_match = classify_profile(cs, str(_SECTION_LIBRARY))
    except Exception as e:
        profile_match = None
        logger.warning("classify_profile_failed", ref_id=ref_id, error=str(e))

    if profile_match:
        logger.info("section_match_found", ref_id=ref_id,
                    designation=profile_match.get("Designation"),
                    score=profile_match.get("Match_score"))
        return _process_section(aligned, obb, profile_match, ref_id, member_id,
                                solid_idx, file_path, out_dir, parent_name,
                                project_number, steel_grade)

    logger.info("no_section_match", ref_id=ref_id,
                L=round(L, 1), H=round(H, 1), W=round(W, 1),
                section_area=round(section_area, 1))

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
                                        ref_id, solid_idx, parent_name)
        if hollow:
            return hollow
    except Exception as e:
        logger.warning("hollow_detection_failed", ref_id=ref_id, error=str(e))

    return {"type": "unknown", "message": msg or "Not plate or section"}


def _process_section(aligned, obb, profile_match: Dict, ref_id: str, member_id: str,
                     solid_idx: int, file_path: str, out_dir: Path,
                     parent_name: Optional[str] = None,
                     project_number: Optional[str] = None,
                     steel_grade: Optional[str] = None) -> Dict[str, Any]:
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

        # NC1 filename: {project_number}-{parent_name}-{occurrence}.nc1
        nc1_name = f"{proj_num}-{_safe_name(parent_name)}-{solid_idx + 1}"
        nc1_dir = out_dir / "nc1"

        header_data = assemble_dstv_header_data(
            project_number=proj_num,
            step_path=file_path,
            matl_grade=matl,
            member_id=nc1_name,
            profile_match=profile_match,
        )

        nc1_path, nc1_hash = generate_nc1_file(df_holes, header_data, nc1_dir, web_cut)

        # Volume / weight (best-effort; aligned shape is the one after swap+pose)
        try:
            vol_mm3 = get_volume_from_shape(aligned)
            mass_kg = round(vol_mm3 * 7.85e-6, 3)
            vol_mm3 = round(vol_mm3, 1)
        except Exception:
            vol_mm3 = None
            mass_kg = None

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
            "match_score": profile_match.get("Match_score", 0.0),
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
        # Filename: {thickness_mm}-{parent_name}-{occurrence}.dxf
        thickness_mm = int(round(T))
        dxf_stem = f"{thickness_mm}-{_safe_name(parent_name)}-{solid_idx + 1}"
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
                                parent_name, project_number, steel_grade)
            results.append(r)
        return {
            "type": "multi_solid",
            "n_solids": n_solids,
            "solids": results,
            "analysed_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        return _analyse_single(shape, 0, ref_id, safe_member, file_path, out_dir,
                               parent_name, project_number, steel_grade)
