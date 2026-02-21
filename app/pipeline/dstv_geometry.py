# pipeline/dstv_geometry.py
# Ported from step-gemini: OCC.Core.* -> OCP.*

from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Circle
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.TopoDS import TopoDS
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.gp import gp_Pnt, gp_Vec, gp_Ax3

import numpy as np
import pandas as pd
import math


def get_face_center(face):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    pnt = props.CentreOfMass()
    return np.array([pnt.X(), pnt.Y(), pnt.Z()])


def get_face_normal(face, u=0.5, v=0.5):
    surface   = BRepAdaptor_Surface(face, True)
    gp_pnt    = gp_Pnt()
    gp_vec_u  = gp_Vec()
    gp_vec_v  = gp_Vec()
    surface.D1(u, v, gp_pnt, gp_vec_u, gp_vec_v)
    normal_vec = gp_vec_u.Crossed(gp_vec_v)
    return np.array([normal_vec.X(), normal_vec.Y(), normal_vec.Z()])


def classify_and_project_holes_dstv(
    solid, dstv_frame, origin_dstv,
    width_mm, flange_span_mm,
    profile_match: dict
):
    """
    Classify & project circular holes to DSTV coordinates for U / I / L profiles.

    Assumes the shape is already in FINAL DSTV pose:
      - origin at min corner (0,0,0)
      - X = length (L),  Y = height (H),  Z = width (W)

    Conventions:
      - Channels (U): web at Z=0, flanges at Y≈0 (O) and Y≈H (U).
      - Beams (I):    web at Z≈W/2, flanges at Y≈0 (O) and Y≈H (U).
      - Angles (L):   heel at origin; legs along Y≈0 (-> code 'U') and Z≈0 (-> code 'H').
    """
    ptype = (profile_match or {}).get("Profile_type", "")
    dims  = (profile_match or {}).get("JSON") or (profile_match or {}).get("STEP") or {}

    def _as_float(x, default=None):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            import re
            s = x.replace(",", "").strip()
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    pass
        return default

    tw = _as_float(dims.get("web_thickness"))
    tf = _as_float(dims.get("flange_thickness"))
    t_leg = _as_float(dims.get("thickness"))

    # frame basis
    origin_np = np.array([origin_dstv.X(), origin_dstv.Y(), origin_dstv.Z()])
    L = np.array([dstv_frame.XDirection().X(), dstv_frame.XDirection().Y(), dstv_frame.XDirection().Z()])
    F = np.array([dstv_frame.YDirection().X(), dstv_frame.YDirection().Y(), dstv_frame.YDirection().Z()])
    W = np.array([dstv_frame.Direction().X(),   dstv_frame.Direction().Y(),   dstv_frame.Direction().Z()])

    def _tol_web_at_z0():
        return max(1.0, 0.5*(tw or 0.0) + 1.0) if tw else max(1.0, 0.03*max(width_mm, 1.0))

    def _tol_flange_y():
        return max(1.0, 0.5*(tf or 0.0) + 1.0) if tf else max(1.0, 0.02*max(flange_span_mm, 1.0))

    def _tol_web_at_zmid():
        return max(1.0, 0.5*(tw or 0.0) + 1.0) if tw else max(1.0, 0.03*max(width_mm, 1.0))

    def _tol_angle_y():
        base = (t_leg or tf or 0.0)
        return max(1.0, 0.5*base + 1.0) if base else max(1.0, 0.02*max(flange_span_mm, 1.0))

    def _tol_angle_z():
        base = (t_leg or tw or 0.0)
        return max(1.0, 0.5*base + 1.0) if base else max(1.0, 0.03*max(width_mm, 1.0))

    cos_ok_web = 0.6
    cos_ok_fl  = 0.6

    explorer = TopExp_Explorer(solid, TopAbs_EDGE)
    seen_keys, hole_rows = set(), []

    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        explorer.Next()

        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Circle:
            continue

        circ    = curve.Circle()
        center  = circ.Location()
        axis    = circ.Axis().Direction()
        radius  = circ.Radius()

        center_np = np.array([center.X(), center.Y(), center.Z()], dtype=float)
        axis_np   = np.array([axis.X(),   axis.Y(),   axis.Z()], dtype=float)
        nrm = np.linalg.norm(axis_np)
        if nrm == 0.0:
            continue
        axis_np /= nrm

        v = center_np - origin_np
        y_along = float(np.dot(v, F))
        z_along = float(np.dot(v, W))
        d_web_z0  = abs(z_along)
        d_web_zmid= abs(z_along - 0.5*width_mm)
        dotW = abs(float(np.dot(axis_np, W)))
        dotF = abs(float(np.dot(axis_np, F)))

        code   = None
        y_axis = None

        if ptype == "U":
            tol_web    = _tol_web_at_z0()
            tol_flange = _tol_flange_y()
            if (d_web_z0 <= tol_web) and (dotW >= cos_ok_web):
                code, y_axis = "V", F
            elif abs(y_along) <= tol_flange:
                code, y_axis = "U", W
            elif abs(y_along - flange_span_mm) <= tol_flange:
                code, y_axis = "O", W

        elif ptype == "I":
            tol_web    = _tol_web_at_zmid()
            tol_flange = _tol_flange_y()
            if (d_web_zmid <= tol_web) and (dotW >= cos_ok_web):
                code, y_axis = "V", F
            elif abs(y_along) <= tol_flange:
                code, y_axis = "U", W
            elif abs(y_along - flange_span_mm) <= tol_flange:
                code, y_axis = "O", W

        elif ptype == "L":
            tol_y = _tol_angle_y()
            tol_z = _tol_angle_z()
            if abs(y_along) <= tol_y:
                code, y_axis = "U", W
            elif abs(z_along) <= tol_z:
                code, y_axis = "H", F

        else:
            tol_web    = _tol_web_at_z0()
            tol_flange = _tol_flange_y()
            if (d_web_z0 <= tol_web) and (dotW >= cos_ok_web):
                code, y_axis = "V", F
            elif abs(y_along) <= tol_flange:
                code, y_axis = "U", W
            elif abs(y_along - flange_span_mm) <= tol_flange:
                code, y_axis = "O", W

        if code is None:
            continue

        x = round(float(np.dot(v, L)), 2)
        y = round(float(np.dot(v, y_axis)), 2)

        key = (round(x, 1), round(y, 1), round(radius, 2), code)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        hole_rows.append({
            "Hole #":        len(hole_rows) + 1,
            "Code":          code,
            "Diameter (mm)": round(radius * 2, 2),
            "X (mm)":        x,
            "Y (mm)":        y,
            "Profile": ptype
        })

    if hole_rows:
        codes = [h["Code"] for h in hole_rows]
        print(f"[holes] {ptype}  V={codes.count('V')}  U={codes.count('U')}  O={codes.count('O')}  H={codes.count('H')}")

    return pd.DataFrame(hole_rows)


def check_duplicate_holes(df_holes, tolerance=0.1):
    """
    Checks for duplicate holes (same X, Y, and Code) within a given tolerance (in mm).
    Returns (has_duplicates, duplicate_df).
    """
    if df_holes is None or df_holes.empty:
        return False, df_holes
    df_check = df_holes.copy()
    df_check["X_r"] = (df_check["X (mm)"] / tolerance).round().astype(int)
    df_check["Y_r"] = (df_check["Y (mm)"] / tolerance).round().astype(int)

    duplicates = df_check.duplicated(subset=["Code", "X_r", "Y_r"], keep=False)

    if duplicates.any():
        print(f"Found {duplicates.sum()} potential duplicate holes.")
        dup_df = df_holes[duplicates]
        return True, dup_df
    else:
        return False, pd.DataFrame()


def signed_tilt_to_plane(n_unit: np.ndarray, axis: np.ndarray) -> float:
    """
    Angle between normal and `axis` in [0,180], folded into [-90,90].
    """
    raw = math.degrees(math.acos(np.clip(np.dot(n_unit, axis), -1.0, 1.0)))
    return (raw - 180) if raw > 90 else raw


def _fold_to_plusminus_90(angle: float) -> float:
    if angle > 90:
        return angle - 180
    if angle < -90:
        return angle + 180
    return angle


def analyze_end_faces_web_and_flange(solid, ax3: gp_Ax3, tol: float = 1e-3):
    # frame axes
    axis_x = np.array([ax3.XDirection().X(), ax3.XDirection().Y(), ax3.XDirection().Z()])
    axis_y = np.array([ax3.YDirection().X(), ax3.YDirection().Y(), ax3.YDirection().Z()])
    axis_z = np.array([ax3.Direction().X(),  ax3.Direction().Y(),  ax3.Direction().Z()])

    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_data = []
    while explorer.More():
        f   = explorer.Current()
        c   = get_face_center(f)
        n   = get_face_normal(f)
        x_d = np.dot(c, axis_x)
        face_data.append((n, x_d))
        explorer.Next()
    if not face_data:
        return {}

    face_data.sort(key=lambda item: item[1])
    normals = [face_data[0][0], face_data[-1][0]]
    labels  = ["start", "end"]

    results = []
    for label, normal in zip(labels, normals):
        n_unit = normal / np.linalg.norm(normal)
        nx = float(np.dot(n_unit, axis_x))
        ny = float(np.dot(n_unit, axis_y))
        nz = float(np.dot(n_unit, axis_z))

        raw_web    = math.degrees(math.atan2(ny, nx))
        raw_flange = math.degrees(math.atan2(nz, nx))

        web_angle    = _fold_to_plusminus_90(raw_web)
        flange_angle = _fold_to_plusminus_90(raw_flange)

        results.append({
            "end":               label,
            "web_angle_deg":     round(web_angle,    2),
            "flange_angle_deg":  round(flange_angle, 2),
            "has_web_cut":       abs(web_angle)    > tol,
            "has_flange_cut":    abs(flange_angle) > tol
        })

    web_cuts = {}
    for e in results:
        web_cuts[f"{e['end']}_web"] = (
            e["web_angle_deg"] if e["has_web_cut"] else 0.00
        )
        web_cuts[f"{e['end']}_flange"] = (
            e["flange_angle_deg"] if e["has_flange_cut"] else 0.00
        )

    return web_cuts
