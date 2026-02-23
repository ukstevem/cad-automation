# pipeline/plate_wrangling.py
# Ported from step-gemini: OCC.Core.* -> OCP.*

from typing import Optional
import math
import numpy as np

from OCP.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Trsf
from OCP.Bnd import Bnd_OBB
from OCP.BRepBndLib import BRepBndLib
from OCP.BRep import BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.TopoDS import TopoDS

from app.pipeline.geometry_utils import plate_guard_heuristics

THICKNESS_LIMIT_MM = 101.0
mat_density = 0.000000784  # kg / mm^3 (≈ 7.84 g/cc)


# ---------------- small helpers ----------------
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _proj_inplane_np(v_np, z_np):
    return _unit(v_np - np.dot(v_np, z_np) * z_np)


def _acute_inplane_delta_deg(a_dir, b_dir, z_np):
    """Acute in-plane angle between OCC directions a_dir, b_dir, measured in plane ⟂ z_np."""
    a = _proj_inplane_np(np.array([a_dir.X(), a_dir.Y(), a_dir.Z()], float), z_np)
    b = _proj_inplane_np(np.array([b_dir.X(), b_dir.Y(), b_dir.Z()], float), z_np)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return None
    signed = math.degrees(math.atan2(np.dot(np.cross(a, b), z_np), float(np.dot(a, b))))
    acute  = abs((signed + 180.0) % 360.0 - 180.0)
    return min(acute, 180.0 - acute)


def _yaw_deg_from_xdir(xdir: gp_Dir) -> Optional[float]:
    """Yaw of X relative to world +X in XY-plane."""
    xv = np.array([xdir.X(), xdir.Y(), 0.0])
    n = np.linalg.norm(xv)
    if n < 1e-12:
        return None
    return math.degrees(math.atan2(xv[1], xv[0]))


def compute_section_and_length_and_origin_from_obb(solid):
    obb = Bnd_OBB()
    BRepBndLib.AddOBB_s(solid, obb, True, True)
    center = obb.Center()
    half_extents = [obb.XHSize(), obb.YHSize(), obb.ZHSize()]
    axes = [obb.XDirection(), obb.YDirection(), obb.ZDirection()]  # gp_Dir
    return center, half_extents, axes


def get_face_area(face):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(TopoDS.Face_s(face), props)
    return props.Mass()


def is_planar_by_curvature(face, tol=1e-4):
    face = TopoDS.Face_s(face)
    surface = BRep_Tool.Surface_s(face)
    try:
        umin, umax, vmin, vmax = surface.Bounds()
        u_mid = (umin + umax) / 2.0
        v_mid = (vmin + vmax) / 2.0
        props = GeomLProp_SLProps(surface, u_mid, v_mid, 2, tol)
        if not props.IsCurvatureDefined():
            return False
        return abs(props.MaxCurvature()) < tol and abs(props.MinCurvature()) < tol
    except Exception:
        return False


def get_face_normal(face):
    face = TopoDS.Face_s(face)
    surface = BRep_Tool.Surface_s(face)
    umin, umax, vmin, vmax = surface.Bounds()
    u_mid = (umin + umax) / 2.0
    v_mid = (vmin + vmax) / 2.0
    props = GeomLProp_SLProps(surface, u_mid, v_mid, 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    return props.Normal()  # gp_Dir


def _world_extents_mm(shape):
    obb = Bnd_OBB()
    BRepBndLib.AddOBB_s(shape, obb, True, True)
    return 2.0 * obb.XHSize(), 2.0 * obb.YHSize(), 2.0 * obb.ZHSize()


# ---------------- main ----------------
def align_plate_to_xy_plane(
    solid,
    prefer_dir_x: Optional[gp_Dir] = None,
    lock_x: bool = True,
    square_rel_tol: float = 1e-3,
    warn_obb_delta_deg: Optional[float] = 1.0,
    thickness_limit_mm: Optional[float] = None,
):
    """
    Final plate alignment WITHOUT reintroducing random yaw.

    Returns:
      (is_plate, aligned_solid, dstv_ax3, thickness_mm, length_mm, width_mm, step_mass, message, sig)
    """
    try:
        # guard: bail early if it doesn't look like a plate (or multi-body)
        ok_guard, diag_guard = plate_guard_heuristics(solid)
        if not ok_guard:
            return (
                False, None, None,
                0.0, 0.0, 0.0, 0.0,
                f"Plate guard veto: {diag_guard}",
                {'yaw_deg': None, 'obb_vs_x_deg': None, 'x_choice': 'n/a', 'square_guard': False,
                 'dims_mm': {'L': 0.0, 'W': 0.0, 'T': 0.0}}
            )

        # OBB on INPUT solid (for initial dims & long axis)
        center, half_extents, axes = compute_section_and_length_and_origin_from_obb(solid)
        idx_sorted = sorted(range(3), key=lambda i: -half_extents[i])
        length_idx, width_idx, thickness_idx = idx_sorted
        length_mm    = 2.0 * half_extents[length_idx]
        width_mm     = 2.0 * half_extents[width_idx]
        thickness_mm = 2.0 * half_extents[thickness_idx]

        # mass from volume
        props_v = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, props_v)
        step_mass = props_v.Mass() * mat_density

        # pick largest planar face; its normal => Z
        faces = []
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        while exp.More():
            f = exp.Current()
            if is_planar_by_curvature(f):
                faces.append(f)
            exp.Next()

        if not faces:
            sig = {'yaw_deg': None, 'obb_vs_x_deg': None, 'x_choice': 'n/a', 'square_guard': False,
                   'dims_mm': {'L': length_mm, 'W': width_mm, 'T': thickness_mm}}
            return False, solid, None, thickness_mm, length_mm, width_mm, step_mass, "No planar faces found", sig

        largest = max(faces, key=get_face_area)
        fn = get_face_normal(largest)
        if fn is None:
            sig = {'yaw_deg': None, 'obb_vs_x_deg': None, 'x_choice': 'n/a', 'square_guard': False,
                   'dims_mm': {'L': length_mm, 'W': width_mm, 'T': thickness_mm}}
            return False, solid, None, thickness_mm, length_mm, width_mm, step_mass, "Could not determine face normal", sig

        dir_z = gp_Dir(fn.X(), fn.Y(), fn.Z())
        if dir_z.Z() < 0:
            dir_z = gp_Dir(-dir_z.X(), -dir_z.Y(), -dir_z.Z())
        z_np = np.array([dir_z.X(), dir_z.Y(), dir_z.Z()], float)

        def proj_to_plane(v_np):
            return _unit(v_np - np.dot(v_np, z_np) * z_np)

        x_choice = None
        x_np = None

        if lock_x and prefer_dir_x is not None:
            v = proj_to_plane(np.array([prefer_dir_x.X(), prefer_dir_x.Y(), prefer_dir_x.Z()], float))
            if np.linalg.norm(v) > 1e-12:
                x_np = v
                x_choice = 'prefer'

        is_square = False
        if x_np is None:
            is_square = abs(length_mm - width_mm) <= max(square_rel_tol * max(length_mm, width_mm), 1e-6)
            if is_square:
                v = proj_to_plane(np.array([1.0, 0.0, 0.0], float))
                if np.linalg.norm(v) < 1e-12:
                    v = proj_to_plane(np.array([0.0, 1.0, 0.0], float))
                x_np = v
                x_choice = 'worldX'
            else:
                a = axes[length_idx]
                v = proj_to_plane(np.array([a.X(), a.Y(), a.Z()], float))
                if np.linalg.norm(v) < 1e-12:
                    vdir = gp_Ax3(gp_Pnt(0, 0, 0), dir_z).XDirection()
                    v = proj_to_plane(np.array([vdir.X(), vdir.Y(), vdir.Z()], float))
                    x_choice = 'fallback'
                else:
                    x_choice = 'obb_projected'
                x_np = v

        # stable sign toward +world X when feasible
        if np.dot(x_np, np.array([1.0, 0.0, 0.0])) < 0:
            x_np = -x_np

        # right-handed: Y = Z × X ; then re-orthonormalize X = Y × Z
        y_np = _unit(np.cross(z_np, x_np))
        if np.dot(np.cross(x_np, y_np), z_np) < 0:
            y_np = -y_np
        x_np = _unit(np.cross(y_np, z_np))

        dir_x = gp_Dir(float(x_np[0]), float(x_np[1]), float(x_np[2]))

        yaw_deg = _yaw_deg_from_xdir(dir_x)
        obb_vs_x_deg = None
        try:
            obb_long = axes[length_idx]
            obb_vs_x_deg = _acute_inplane_delta_deg(dir_x, obb_long, z_np)
        except Exception:
            pass

        # build frame and transform
        cx = gp_Pnt(center.X(), center.Y(), center.Z())
        dstv_ax3 = gp_Ax3(cx, dir_z, dir_x)
        world_ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))

        tr = gp_Trsf()
        tr.SetTransformation(dstv_ax3, world_ax3)
        transformed = BRepBuilderAPI_Transform(solid, tr, True).Shape()

        # recompute dims from final orientation; enforce T = min(L, W, T)
        Lx, Ly, Lz = _world_extents_mm(transformed)
        L, W, T = sorted((float(Lx), float(Ly), float(Lz)), reverse=True)
        length_mm, width_mm, thickness_mm = L, W, T

        sig = {
            'yaw_deg': yaw_deg,
            'obb_vs_x_deg': obb_vs_x_deg,
            'x_choice': x_choice,
            'square_guard': bool(is_square),
            'dims_mm': {'L': length_mm, 'W': width_mm, 'T': thickness_mm},
        }

        limit = thickness_limit_mm if thickness_limit_mm is not None else THICKNESS_LIMIT_MM
        if limit is not None and thickness_mm > limit:
            return (False, transformed, dstv_ax3, thickness_mm, length_mm, width_mm, step_mass,
                    "Too thick for plate", sig)

        if (
            warn_obb_delta_deg is not None
            and obb_vs_x_deg is not None
            and obb_vs_x_deg > warn_obb_delta_deg
            and not is_square
        ):
            print(f"OBB long vs chosen X differs by {obb_vs_x_deg:.2f} deg", file=__import__("sys").stderr)

        return (True, transformed, dstv_ax3, thickness_mm, length_mm, width_mm, step_mass,
                "Plate aligned to XY; thickness = min(X,Y,Z)", sig)

    except Exception as e:
        sig = {'yaw_deg': None, 'obb_vs_x_deg': None, 'x_choice': 'error', 'square_guard': False,
               'dims_mm': {'L': None, 'W': None, 'T': None}}
        return False, None, None, 0.0, 0.0, 0.0, 0.0, f"Unhandled error: {e}", sig
