# pipeline/geometry_utils.py
# Ported from step-gemini: OCC.Core.* -> OCP.*

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import math
import numpy as np
import statistics

from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import (
    TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_SOLID, TopAbs_VERTEX
)
from OCP.TopoDS import TopoDS, TopoDS_Shape

from OCP.Bnd import Bnd_Box, Bnd_OBB
from OCP.BRepBndLib import BRepBndLib

from OCP.gp import (
    gp_Pnt, gp_Dir, gp_Vec, gp_Ax1, gp_Ax2, gp_Ax3, gp_Trsf, gp_Pln
)

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
)

from OCP.BRepAdaptor import (
    BRepAdaptor_Curve, BRepAdaptor_Surface
)

from OCP.GeomAbs import GeomAbs_Plane

from OCP.BRepAlgoAPI import BRepAlgoAPI_Section, BRepAlgoAPI_Common

from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.ShapeFix import ShapeFix_Wire

from OCP.BRep import BRep_Tool
from OCP.GeomLProp import GeomLProp_SLProps

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox


# ======================================================================================
# Diagnostics (optional): filled by compute_section_area() in robust mode
# ======================================================================================
CSA_DIAG: Dict[str, Any] = {}


# ======================================================================================
# Simple helpers
# ======================================================================================

def _trsf_world_to_local(ax3: gp_Ax3) -> gp_Trsf:
    """Build world → local(ax3) transform; ax3 uses (origin, Z, X)."""
    world = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    t = gp_Trsf()
    t.SetDisplacement(world, ax3)
    return t


def _bbox_world(shape):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)


def _dstv_min_corner_world(obb_geom):
    """Compute the DSTV min-corner (rear-bottom-left) point in world coords from OBB data."""
    c   = obb_geom["aligned_center"]
    ex  = float(obb_geom["aligned_extents"][0])
    ey  = float(obb_geom["aligned_extents"][1])
    ez  = float(obb_geom["aligned_extents"][2])
    dx  = gp_Vec(obb_geom["aligned_dir_x"]).Scaled(-0.5 * ex)
    dy  = gp_Vec(obb_geom["aligned_dir_y"]).Scaled(-0.5 * ey)
    dz  = gp_Vec(obb_geom["aligned_dir_z"]).Scaled(-0.5 * ez)
    return c.Translated(dx + dy + dz)


def _enforce_long_leg_on_Y_if_angle(shape, profile_type: str, dims: Optional[Dict]):
    if profile_type != "L" or not dims:
        return shape, 0
    legY_nom = float(dims.get("leg_y", dims.get("height", 0.0)))
    legZ_nom = float(dims.get("leg_z", dims.get("width",  0.0)))
    want_Y, want_Z = (max(legY_nom, legZ_nom), min(legY_nom, legZ_nom))
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(shape)
    H, W = (ymax - ymin, zmax - zmin)
    tol = max(2.0, 0.01 * max(want_Y, want_Z, 1.0))

    def _close(a, b):
        return abs(a - b) <= tol

    if _close(H, want_Y) and _close(W, want_Z):
        return shape, 0
    sh90 = _rotX(shape, 1)
    xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = _bbox_local(sh90)
    H2, W2 = (ymax2 - ymin2, zmax2 - zmin2)
    if _close(H2, want_Y) and _close(W2, want_Z):
        return sh90, 1
    return shape, 0


def _to_origin_min_corner(shape):
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(shape)
    tr = gp_Trsf()
    tr.SetTranslation(gp_Pnt(xmin, ymin, zmin), gp_Pnt(0, 0, 0))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def _move_to_dstv_local(shape, obb_geom):
    """
    Translate shape so min-corner → (0,0,0) and rotate axes to world DSTV (Z up, X forward).
    Returns (shape_local, dstv_frame_world, (L,H,W)).
    """
    min_corner = _dstv_min_corner_world(obb_geom)
    t_tr = gp_Trsf()
    t_tr.SetTranslation(gp_Vec(min_corner, gp_Pnt(0, 0, 0)))
    sh_t = BRepBuilderAPI_Transform(shape, t_tr, True).Shape()

    obb_frame = gp_Ax3(gp_Pnt(0, 0, 0),
                       gp_Dir(obb_geom["aligned_dir_z"].XYZ()),
                       gp_Dir(obb_geom["aligned_dir_x"].XYZ()))
    dstv_world = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    t_rot = gp_Trsf()
    t_rot.SetDisplacement(obb_frame, dstv_world)
    sh_local = BRepBuilderAPI_Transform(sh_t, t_rot, True).Shape()

    L, H, W = [float(v) for v in obb_geom["aligned_extents"]]
    return sh_local, dstv_world, (L, H, W)


def _section_yz_points(shape_local, x_pos: float, tol=1e-6):
    """
    Take a section at X = x_pos (DSTV-local), return list of (y,z) vertices from closed wires.
    """
    plane = gp_Pln(gp_Ax3(gp_Pnt(x_pos, 0, 0), gp_Dir(1, 0, 0)))
    sec = BRepAlgoAPI_Section(shape_local, plane)
    sec.ComputePCurveOn1(True)
    sec.Approximation(True)
    sec.Build()
    if not sec.IsDone():
        return []

    fb = ShapeAnalysis_FreeBounds(sec.Shape(), tol, False, False)
    wires = []
    seq = fb.GetClosedWires()
    if seq and hasattr(seq, "Length"):
        for i in range(1, seq.Length() + 1):
            s = seq.Value(i)
            if s.ShapeType() == TopAbs_WIRE:
                wires.append(TopoDS.Wire_s(s))

    pts = []
    for w in wires:
        exp = TopExp_Explorer(w, TopAbs_VERTEX)
        while exp.More():
            v = TopoDS.Vertex_s(exp.Current())
            p = BRep_Tool.Pnt_s(v)
            pts.append((p.Y(), p.Z()))
            exp.Next()
    return pts


def _rotate_180_about_world_x(shape):
    axisX = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
    r = gp_Trsf()
    r.SetRotation(axisX, math.pi)
    return BRepBuilderAPI_Transform(shape, r, True).Shape()


def _canonicalize_channel(shape_local, H, W, tol=1.0):
    """
    Channel canonical: toe at z≈0, y spans [0,H]. If bbox closer to far-planes, flip once.
    """
    _, _, y0, y1, z0, z1 = _bbox_world(shape_local)
    dz_near = min(abs(z0 - 0.0), abs(z1 - 0.0))
    dz_far  = min(abs(z0 - W),   abs(z1 - W))
    dy_near = min(abs(y0 - 0.0), abs(y1 - 0.0))
    dy_far  = min(abs(y0 - H),   abs(y1 - H))
    if (dz_far + 1e-6 < dz_near - 1e-6) or (dy_far + 1e-6 < dy_near - 1e-6):
        shape_local = _rotate_180_about_world_x(shape_local)
    return shape_local


def _canonicalize_angle(shape_local, H, W, tol=1.0):
    """
    Angle canonical: heel at (y≈0, z≈0). If bbox closer to far-planes, flip once.
    """
    _, _, y0, y1, z0, z1 = _bbox_world(shape_local)
    dy_near = min(abs(y0 - 0.0), abs(y1 - 0.0))
    dy_far  = min(abs(y0 - H),   abs(y1 - H))
    dz_near = min(abs(z0 - 0.0), abs(z1 - 0.0))
    dz_far  = min(abs(z0 - W),   abs(z1 - W))
    if (dy_far + 1e-6 < dy_near - 1e-6) or (dz_far + 1e-6 < dz_near - 1e-6):
        shape_local = _rotate_180_about_world_x(shape_local)
    return shape_local


def _rotate_180_about_local_x(shape, ax3: gp_Ax3):
    """Rotate shape 180° about local X (length) axis of ax3."""
    axisX = gp_Ax1(ax3.Location(), ax3.XDirection())
    r = gp_Trsf()
    r.SetRotation(axisX, math.pi)
    return BRepBuilderAPI_Transform(shape, r, True).Shape()


def _wire_area_on_plane(plane: gp_Pln, wire) -> float:
    """Unsigned area of a single wire on a given plane."""
    face = BRepBuilderAPI_MakeFace(plane, wire).Face()
    gp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, gp)
    return abs(gp.Mass())


def count_solids_in_shape(shape: TopoDS_Shape) -> int:
    """Count standalone SOLIDs (detects multi-body STEP)."""
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _dir_to_np(d: gp_Dir) -> np.ndarray:
    return np.array([d.X(), d.Y(), d.Z()], dtype=float)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    u = u / max(np.linalg.norm(u), 1e-12)
    v = v / max(np.linalg.norm(v), 1e-12)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _cluster_dirs_weighted(dirs_areas: List[tuple], ang_tol_deg: float = 10.0) -> List[Dict[str, Any]]:
    """
    Cluster directions (with areas) by angular proximity. Folds ±n into the same cluster.
    Returns: [{'rep': unit_vec, 'area': float, 'count': int}, ...] sorted by area desc.
    """
    clusters: List[Dict[str, Any]] = []
    for nvec, a in dirs_areas:
        v = nvec.copy()
        if v[0] < 0 or (abs(v[0]) < 1e-12 and (v[1] < 0 or (abs(v[1]) < 1e-12 and v[2] < 0))):
            v = -v
        placed = False
        for c in clusters:
            rep = c['rep']
            if _angle_between(v, rep) <= ang_tol_deg or _angle_between(v, -rep) <= ang_tol_deg:
                c['rep'] = (c['rep'] * c['area'] + v * a) / (c['area'] + a)
                c['area'] += a
                c['count'] += 1
                placed = True
                break
        if not placed:
            clusters.append({'rep': v, 'area': float(a), 'count': 1})

    for c in clusters:
        nrm = np.linalg.norm(c['rep'])
        if nrm > 0:
            c['rep'] = c['rep'] / nrm
    clusters.sort(key=lambda c: c['area'], reverse=True)
    return clusters


# ======================================================================================
# Plate guard (alignment-free): multi-body + planar face normal clustering
# ======================================================================================

def plate_guard_faces(shape: TopoDS_Shape,
                      ang_tol_deg: float = 10.0,
                      dominant_min_frac: float = 0.55,
                      second_max_frac: float = 0.30,
                      big_cluster_frac: float = 0.05,
                      min_planar_frac: float = 0.55):
    """
    Alignment-free 'is this plausibly a PLATE?' using planar faces only.
    """
    dirs_areas: List[tuple] = []
    total_area = 0.0
    total_planar_area = 0.0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        gp_props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, gp_props)
        a = float(abs(gp_props.Mass()))
        total_area += a

        srf = BRepAdaptor_Surface(f)
        if srf.GetType() == GeomAbs_Plane:
            n = srf.Plane().Axis().Direction()
            dirs_areas.append((_dir_to_np(n), a))
            total_planar_area += a
        exp.Next()

    if total_area <= 1e-12 or total_planar_area <= 1e-12:
        return False, {"reason": "no_area"}

    planar_frac = total_planar_area / total_area
    clusters = _cluster_dirs_weighted(dirs_areas, ang_tol_deg=ang_tol_deg)
    areas = [c['area'] for c in clusters]
    top = areas[0] if areas else 0.0
    second = areas[1] if len(areas) > 1 else 0.0
    top_frac = top / total_planar_area if total_planar_area > 0 else 0.0
    second_frac = second / total_planar_area if total_planar_area > 0 else 0.0
    big_clusters = sum(1 for a in areas if a / max(total_planar_area, 1e-12) >= big_cluster_frac)

    plausible = (
        planar_frac >= min_planar_frac and
        top_frac >= dominant_min_frac and
        second_frac <= second_max_frac and
        big_clusters <= 6
    )
    diag = {
        "planar_frac": float(planar_frac),
        "top_frac": float(top_frac),
        "second_frac": float(second_frac),
        "big_clusters": int(big_clusters),
        "n_clusters": len(clusters),
    }
    return plausible, diag


def plate_guard_heuristics(shape: TopoDS_Shape,
                           face_ang_tol_deg: float = 10.0,
                           dominant_min_frac: float = 0.40,
                           second_max_frac: float = 0.40,
                           big_cluster_frac: float = 0.05,
                           min_planar_frac: float = 0.55):
    """Decide if we should even ATTEMPT plate handling."""
    n_solids = count_solids_in_shape(shape)
    if n_solids > 1:
        return False, {"reason": "multi_body_shape", "n_solids": n_solids}

    ok, diag_faces = plate_guard_faces(
        shape,
        ang_tol_deg=face_ang_tol_deg,
        dominant_min_frac=dominant_min_frac,
        second_max_frac=second_max_frac,
        big_cluster_frac=big_cluster_frac,
        min_planar_frac=min_planar_frac,
    )
    if not ok:
        diag_faces["reason"] = diag_faces.get("reason", "face_guard_veto")
        return False, diag_faces

    return True, {"reason": "pass_faces"}


# ======================================================================================
# Sectioning helpers
# ======================================================================================

def _assemble_wire_sorted(edges_shape: TopoDS_Shape, tol: float = 1e-6):
    """
    Sort loose edges by endpoint connectivity and build a closed wire.

    TopExp_Explorer returns edges in arbitrary order.  BRepBuilderAPI_MakeWire
    processes them sequentially and drops edges whose endpoints don't touch the
    current wire ends — for I-beam sections this silently loses the web edges,
    producing a self-intersecting wire with inflated area.

    This helper builds a point-adjacency graph, walks the chain in the correct
    order, then feeds the sorted edges to MakeWire.
    """
    # Collect edges with their YZ endpoints (X is the section-plane normal)
    edge_list = []   # [(edge, (y0,z0), (y1,z1)), ...]
    exp = TopExp_Explorer(edges_shape, TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge_s(exp.Current())
        vexp = TopExp_Explorer(e, TopAbs_VERTEX)
        pts = []
        while vexp.More():
            p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(vexp.Current()))
            pts.append((p.Y(), p.Z()))
            vexp.Next()
        if len(pts) >= 2:
            edge_list.append((e, pts[0], pts[1]))
        exp.Next()

    if len(edge_list) < 3:
        return None

    # Build adjacency: for each edge, find neighbours by matching endpoints
    n = len(edge_list)
    match_tol = max(tol, 0.01)  # 0.01mm minimum for floating-point noise

    def _pts_close(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) < match_tol

    # Walk the chain starting from edge 0
    used = [False] * n
    order = [0]       # indices into edge_list
    fwd   = [True]    # whether edge is traversed in its natural direction
    used[0] = True
    current_end = edge_list[0][2]  # start walking from end of first edge

    for _ in range(n - 1):
        found = False
        for j in range(n):
            if used[j]:
                continue
            _, p0, p1 = edge_list[j]
            if _pts_close(p0, current_end):
                order.append(j)
                fwd.append(True)
                used[j] = True
                current_end = p1
                found = True
                break
            elif _pts_close(p1, current_end):
                order.append(j)
                fwd.append(False)
                used[j] = True
                current_end = p0
                found = True
                break
        if not found:
            break

    if len(order) < 3:
        return None

    # Check closure: does the chain end connect back to the start?
    start_pt = edge_list[order[0]][1] if fwd[0] else edge_list[order[0]][2]
    if not _pts_close(current_end, start_pt):
        return None  # not a closed loop

    # Build wire from sorted edges
    mw = BRepBuilderAPI_MakeWire()
    for idx in order:
        mw.Add(edge_list[idx][0])
    if not mw.IsDone():
        return None

    return mw.Wire()


def _extract_outer_and_holes(edges_shape: TopoDS_Shape, plane: gp_Pln, tol: float = 1e-6):
    """
    From section edges return:
      outer_wire (TopoDS_Wire or None),
      inner_wires (list of TopoDS_Wire),
      area_outer (float, outer loop gross area).
    """
    fb = ShapeAnalysis_FreeBounds(edges_shape, tol, False, False)
    seq = fb.GetClosedWires()

    wires = []
    if seq is not None and hasattr(seq, "Length"):
        for i in range(1, seq.Length() + 1):
            s = seq.Value(i)
            if s.ShapeType() == TopAbs_WIRE:
                wires.append(TopoDS.Wire_s(s))

    # Fallback: if FreeBounds found no closed wires, sort edges by connectivity
    # then feed to MakeWire.  Naive MakeWire (random edge order from TopExp_Explorer)
    # drops disconnected edges, producing self-intersecting wires with wrong area.
    if not wires:
        sorted_wire = _assemble_wire_sorted(edges_shape, tol)
        if sorted_wire is not None:
            wires = [sorted_wire]

    if not wires:
        return None, [], 0.0

    areas = [_wire_area_on_plane(plane, w) for w in wires]
    i_outer = int(max(range(len(areas)), key=lambda k: areas[k]))
    outer = wires[i_outer]
    inners = [w for j, w in enumerate(wires) if j != i_outer]
    return outer, inners, float(areas[i_outer])


# ======================================================================================
# Alignment, OBB & transforms
# ======================================================================================

def compute_obb_geometry(aligned_shape: TopoDS_Shape) -> Dict[str, Any]:
    """Axis-aligned bounding box geometry (post alignment)."""
    aabb = Bnd_Box()
    BRepBndLib.Add_s(aligned_shape, aabb)
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = aabb.Get()
    except Exception:
        raise RuntimeError("Failed to extract bounding box from shape.")

    length = xmax - xmin
    height = ymax - ymin
    width  = zmax - zmin

    center = gp_Pnt(
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0
    )

    return {
        "aligned_extents": [length, height, width],
        "aligned_dir_x": gp_Dir(1, 0, 0),
        "aligned_dir_y": gp_Dir(0, 1, 0),
        "aligned_dir_z": gp_Dir(0, 0, 1),
        "aligned_center": center
    }


def get_volume_from_shape(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


# ======================================================================================
# Cross-sectional area (robust)
# ======================================================================================

def compute_section_dims(solid: TopoDS_Shape,
                         axis_dir: gp_Dir = gp_Dir(1, 0, 0),
                         tol: float = 1e-6) -> Dict[str, Any]:
    """
    Robust cross-section dimensions from the core region of a member.

    Samples >=21 sections along the length (avoiding end chamfers/miters),
    clusters by area similarity (3%), and returns the median area, H, and W
    from the dominant cluster.

    Returns {"area": float, "H": float, "W": float}.
    Falls back to volume-based area and 0.0 for H/W if sectioning fails.
    """
    bbox = Bnd_Box()
    BRepBndLib.Add_s(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    L = xmax - xmin
    if L <= 1e-12:
        return {"area": 0.0, "H": 0.0, "W": 0.0}

    def _dims_at_x(x0_abs: float) -> Optional[tuple]:
        """Return (area, H, W) from the section at x0_abs, or None."""
        eps = max(1e-3, 1e-4 * L)
        x0 = max(xmin + eps, min(xmax - eps, x0_abs))
        plane = gp_Pln(gp_Ax3(gp_Pnt(x0, 0, 0), axis_dir))
        sec = BRepAlgoAPI_Section(solid, plane)
        sec.ComputePCurveOn1(True)
        sec.Approximation(True)
        sec.Build()
        if not sec.IsDone():
            return None
        sec_shape = sec.Shape()
        # Get area from outer wire if possible
        outer, _inners, area_outer = _extract_outer_and_holes(sec_shape, plane, tol=tol)
        # Get H/W from the bounding box of ALL section edges — more robust
        # than requiring a closed outer wire (FreeBounds can fail on some shapes)
        sec_bbox = Bnd_Box()
        BRepBndLib.Add_s(sec_shape, sec_bbox)
        if sec_bbox.IsVoid():
            return None
        _, yw0, zw0, _, yw1, zw1 = sec_bbox.Get()
        h_sec = float(yw1 - yw0)
        w_sec = float(zw1 - zw0)
        if h_sec < 1e-6 or w_sec < 1e-6:
            return None
        # Use outer wire area if available, else estimate from volume
        if outer is not None and area_outer > 0:
            return (float(area_outer), h_sec, w_sec)
        return (None, h_sec, w_sec)  # area unknown, but H/W valid

    n_samples = max(5, 21)
    rel_end_margin = 0.06
    abs_min_margin = 15.0
    margin = max(rel_end_margin * L, abs_min_margin)
    x_lo = xmin + margin
    x_hi = xmax - margin
    if x_hi <= x_lo:
        x_lo = xmin + 0.2 * L
        x_hi = xmax - 0.2 * L

    samples: List[tuple] = []  # (area_or_None, H, W)
    xs: List[float] = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        x0_abs = x_lo + t * (x_hi - x_lo)
        d = _dims_at_x(x0_abs)
        if d is not None:
            samples.append(d)
            xs.append(x0_abs)

    if not samples:
        gpv = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, gpv)
        fallback_area = gpv.Mass() / max(L, 1e-9)
        return {"area": fallback_area, "H": 0.0, "W": 0.0}

    # Separate samples with valid area from those with H/W only
    area_samples = [(i, s[0]) for i, s in enumerate(samples) if s[0] is not None]
    all_hs = [s[1] for s in samples]
    all_ws = [s[2] for s in samples]

    # Cluster area samples (3% tolerance) if any
    area_rel_tol = 0.03
    med_area = 0.0
    if area_samples:
        clusters: List[Dict[str, Any]] = []
        for idx, a in sorted(area_samples, key=lambda x: x[1]):
            placed = False
            for c in clusters:
                ref = c['ref']
                if abs(a - ref) <= area_rel_tol * max(ref, 1e-9):
                    c['indices'].append(idx)
                    c['ref'] = statistics.median([samples[j][0] for j in c['indices']])
                    placed = True
                    break
            if not placed:
                clusters.append({'ref': a, 'indices': [idx]})
        clusters.sort(key=lambda c: len(c['indices']), reverse=True)
        core_idx = clusters[0]['indices'] if clusters else [i for i, _ in area_samples]
        core_areas = [samples[j][0] for j in core_idx]
        med_area = float(statistics.median(core_areas))
        # Use H/W from the area-clustered samples if possible
        all_hs = [samples[j][1] for j in core_idx]
        all_ws = [samples[j][2] for j in core_idx]
    else:
        # No valid areas — fallback to volume/length
        gpv = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, gpv)
        med_area = gpv.Mass() / max(L, 1e-9)

    med_h = float(statistics.median(all_hs)) if all_hs else 0.0
    med_w = float(statistics.median(all_ws)) if all_ws else 0.0

    CSA_DIAG.clear()
    CSA_DIAG.update({
        "mode": "robust_cluster",
        "n_total": len(samples),
        "n_with_area": len(area_samples),
        "median_core": med_area,
        "positions": xs,
        "area_rel_tol": area_rel_tol,
        "H_section": med_h,
        "W_section": med_w,
    })

    return {"area": med_area, "H": med_h, "W": med_w}


def compute_section_area(solid: TopoDS_Shape,
                         axis_dir: gp_Dir = gp_Dir(1, 0, 0),
                         sample_pos: Optional[float] = None,
                         tol: float = 1e-6) -> float:
    """
    GROSS cross-sectional area (outer boundary only).
    - If sample_pos is provided (offset from xmin), computes a single slice.
    - Else: delegates to compute_section_dims() and returns just the area.
    """
    if sample_pos is None:
        return compute_section_dims(solid, axis_dir=axis_dir, tol=tol)["area"]

    bbox = Bnd_Box()
    BRepBndLib.Add_s(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    L = xmax - xmin
    if L <= 1e-12:
        return 0.0

    x0_abs = xmin + float(sample_pos)
    eps = max(1e-3, 1e-4 * L)
    x0 = max(xmin + eps, min(xmax - eps, x0_abs))
    plane = gp_Pln(gp_Ax3(gp_Pnt(x0, 0, 0), axis_dir))
    sec = BRepAlgoAPI_Section(solid, plane)
    sec.ComputePCurveOn1(True)
    sec.Approximation(True)
    sec.Build()
    if sec.IsDone():
        outer, _inners, area_outer = _extract_outer_and_holes(sec.Shape(), plane, tol=tol)
        if outer is not None:
            return float(area_outer)
    gpv = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, gpv)
    return gpv.Mass() / max(L, 1e-9)


# ======================================================================================
# Orientation helpers & DSTV alignment
# ======================================================================================

def swap_width_and_height_if_required(profile_match: Dict[str, Any], shape: TopoDS_Shape, obb_geom: Dict[str, Any]):
    """Rotate 90° about X if profile classification flagged a mismatch."""
    requires_swap = profile_match.get("Requires_rotation", False)
    if not requires_swap:
        return shape, obb_geom

    print("Swapping height/width axes to match profile classification", file=__import__("sys").stderr)
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(obb_geom["aligned_center"], obb_geom["aligned_dir_x"]), math.pi / 2.0)
    shape_rotated = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
    obb_geom = compute_obb_geometry(shape_rotated)
    return shape_rotated, obb_geom


def compute_dstv_origin(center: gp_Pnt, extents: List[float], dir_x: gp_Dir, dir_y: gp_Dir, dir_z: gp_Dir) -> gp_Pnt:
    """DSTV origin (rear-bottom-left) for an aligned OBB."""
    dx = -extents[0] / 2.0
    dy = -extents[1] / 2.0
    dz = -extents[2] / 2.0
    offset_vec = gp_Vec(dir_x).Scaled(dx) + gp_Vec(dir_y).Scaled(dy) + gp_Vec(dir_z).Scaled(dz)
    return center.Translated(offset_vec)


def align_obb_to_dstv_frame(shape: TopoDS_Shape, origin_local: gp_Pnt, dir_x: gp_Dir, dir_y: gp_Dir, dir_z: gp_Dir):
    """
    1) Translate origin_local to (0,0,0)
    2) Rotate axes to DSTV global (Z up, X forward)
    """
    translate_vec = gp_Vec(origin_local, gp_Pnt(0, 0, 0))
    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(translate_vec)
    shape_translated = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()

    local_frame = gp_Ax3(gp_Pnt(0, 0, 0), dir_z, dir_x)
    dstv_frame  = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    trsf_rotate = gp_Trsf()
    trsf_rotate.SetDisplacement(local_frame, dstv_frame)
    shape_rotated = BRepBuilderAPI_Transform(shape_translated, trsf_rotate, True).Shape()

    return shape_rotated, trsf_rotate


# ======================================================================================
# CG-based canonicalization and DSTV pose
# ======================================================================================

def _bbox_local(sh) -> Tuple[float, float, float, float, float, float]:
    box = Bnd_Box()
    BRepBndLib.Add_s(sh, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)


def _solid_centroid(shape) -> gp_Pnt:
    g = GProp_GProps()
    try:
        BRepGProp.VolumeProperties_s(shape, g)
        if abs(g.Mass()) > 1e-12:
            return g.CentreOfMass()
    except Exception:
        pass
    try:
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape, g)
        if abs(g.Mass()) > 1e-12:
            return g.CentreOfMass()
    except Exception:
        pass
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(shape)
    return gp_Pnt(0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax))


def _rotX(shape, quarter_turns: int):
    """Rotate about local X axis through origin by k*90 degrees."""
    k = quarter_turns % 4
    if k == 0:
        return shape
    angle = k * (math.pi / 2.0)
    ax = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
    tr = gp_Trsf()
    tr.SetRotation(ax, angle)
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def _rotY(shape, quarter_turns: int):
    """Rotate about local Y axis through origin by k*90 degrees."""
    k = quarter_turns % 4
    if k == 0:
        return shape
    angle = k * (math.pi / 2.0)
    ax = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0))
    tr = gp_Trsf()
    tr.SetRotation(ax, angle)
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def _rotZ(shape, quarter_turns: int):
    """Rotate about local Z axis through origin by k*90 degrees."""
    k = quarter_turns % 4
    if k == 0:
        return shape
    angle = k * (math.pi / 2.0)
    ax = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    tr = gp_Trsf()
    tr.SetRotation(ax, angle)
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def _norm_yz(pt: gp_Pnt, ymin, ymax, zmin, zmax) -> Tuple[float, float]:
    H = max(ymax - ymin, 1e-12)
    W = max(zmax - zmin, 1e-12)
    return (float(pt.Y() - ymin) / H, float(pt.Z() - zmin) / W)


def _rot_180_about_X(shape):
    r = gp_Trsf()
    r.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.pi)
    return BRepBuilderAPI_Transform(shape, r, True).Shape()


def _to_dstv_local(shape):
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(shape)
    t = gp_Trsf()
    t.SetTranslation(gp_Pnt(xmin, ymin, zmin), gp_Pnt(0, 0, 0))
    return BRepBuilderAPI_Transform(shape, t, True).Shape(), (xmax-xmin, ymax-ymin, zmax-zmin)


def _count_wires(compound) -> int:
    n = 0
    exp = TopExp_Explorer(compound, TopAbs_WIRE)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _iter_wires(compound):
    exp = TopExp_Explorer(compound, TopAbs_WIRE)
    while exp.More():
        yield TopoDS.Wire_s(exp.Current())
        exp.Next()


def _outer_wire_at_x(shape_local, x_pos: float, tol=1e-6):
    plane = gp_Pln(gp_Ax3(gp_Pnt(x_pos, 0, 0), gp_Dir(1, 0, 0)))
    sec = BRepAlgoAPI_Section(shape_local, plane)
    sec.ComputePCurveOn1(True)
    sec.Approximation(True)
    sec.Build()
    if not sec.IsDone():
        return None, plane

    fb = ShapeAnalysis_FreeBounds(sec.Shape(), tol, False, False)
    seq = fb.GetClosedWires()
    if not (seq and hasattr(seq, "Length") and seq.Length() > 0):
        return None, plane

    wires = []
    for i in range(1, seq.Length() + 1):
        s = seq.Value(i)
        if s.ShapeType() == TopAbs_WIRE:
            wires.append(TopoDS.Wire_s(s))
    if not wires:
        return None, plane

    def _wire_area(wire):
        face = BRepBuilderAPI_MakeFace(plane, wire).Face()
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, g)
        return abs(g.Mass()), face

    areas = [(_wire_area(w), w) for w in wires]
    (area_max, face_max), wire_max = max(areas, key=lambda t: t[0][0])
    return wire_max, plane


def _outer_cg_yz_slab(shape_local, x_pos: float, slab_thick: float = 2.0,
                      y_pad: float = 20.0, z_pad: float = 20.0) -> Optional[Tuple[float, float]]:
    """Fallback CG extractor using thin-slab boolean section."""
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(shape_local)
    H = float(ymax - ymin)
    W = float(zmax - zmin)

    dx = float(slab_thick)
    dy = float(H + 2*y_pad)
    dz = float(W + 2*z_pad)

    o = gp_Pnt(x_pos - 0.5*dx, ymin - y_pad, zmin - z_pad)
    box = BRepPrimAPI_MakeBox(o, dx, dy, dz).Shape()

    common = BRepAlgoAPI_Common(shape_local, box)
    common.Build()
    if not common.IsDone():
        return None
    sh = common.Shape()

    best_area = -1.0
    best_face = None

    exp = TopExp_Explorer(sh, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        exp.Next()

        ad = BRepAdaptor_Surface(f)
        u0, u1 = ad.FirstUParameter(), ad.LastUParameter()
        v0, v1 = ad.FirstVParameter(), ad.LastVParameter()
        um = 0.5*(u0 + u1)
        vm = 0.5*(v0 + v1)

        pr = GeomLProp_SLProps(ad.Surface().Surface(), um, vm, 1, 1e-6)
        if not pr.IsNormalDefined():
            continue
        n = pr.Normal()
        nx, ny, nz = n.X(), n.Y(), n.Z()
        cos_to_x = abs(nx) / max(1e-12, math.sqrt(nx*nx + ny*ny + nz*nz))
        if cos_to_x < math.cos(math.radians(10.0)):
            continue

        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, g)
        area = abs(g.Mass())
        if area > best_area:
            best_area, best_face = area, f

    if best_face is None or best_area <= 0:
        return None

    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(best_face, g)
    c = g.CentreOfMass()
    return float(c.Y()), float(c.Z())


def _outer_cg_yz(shape_local, x_pos: float) -> Optional[Tuple[float, float]]:
    wire, plane = _outer_wire_at_x(shape_local, x_pos)
    if wire is None:
        cg = _outer_cg_yz_slab(shape_local, x_pos, slab_thick=2.0)
        if cg is None:
            return None
        return cg
    face = BRepBuilderAPI_MakeFace(plane, wire).Face()
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, g)
    c = g.CentreOfMass()
    return float(c.Y()), float(c.Z())


def _outer_cg_yz_robust(shape_local, x_abs: float, *, L: float, H: float, W: float):
    """Return (y_c, z_c) of the OUTER loop at/near absolute X = x_abs (DSTV-local)."""
    fuzzy = max(0.10, 0.002 * max(H, W))
    dx    = max(0.10, 0.001 * L)

    rels   = (0.30, 0.35, 0.40, 0.45, 0.50)
    x_sweep = [(x_abs - 0.25*L) + r*L for r in rels]
    jitters = (0.0, +dx, -dx, +2*dx, -2*dx)

    best_area, best_face = 0.0, None

    for x0 in x_sweep:
        for j in jitters:
            x = x0 + j
            pln = gp_Pln(gp_Ax3(gp_Pnt(x, 0, 0), gp_Dir(1, 0, 0)))
            face_inf = BRepBuilderAPI_MakeFace(pln).Face()

            sec = BRepAlgoAPI_Section(shape_local, face_inf, True)
            sec.Approximation(True)
            sec.ComputePCurveOn1(True)
            sec.ComputePCurveOn2(True)
            sec.SetFuzzyValue(fuzzy)
            sec.Build()
            if not sec.IsDone():
                continue

            edges_comp = sec.Shape()
            fb = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, False, False)
            closed_comp = fb.GetClosedWires()

            if _count_wires(closed_comp) == 0:
                fb_open = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, True, False)
                open_comp = fb_open.GetOpenWires()
                for w in _iter_wires(open_comp):
                    wf = ShapeFix_Wire(w)
                    wf.SetMaxTolerance(fuzzy)
                    wf.Perform()
                    fixed = wf.Wire()
                    face = BRepBuilderAPI_MakeFace(pln, fixed).Face()
                    gp_props = GProp_GProps()
                    BRepGProp.SurfaceProperties_s(face, gp_props)
                    area = abs(gp_props.Mass())
                    if area > best_area:
                        best_area, best_face = area, face

            for w in _iter_wires(closed_comp):
                face = BRepBuilderAPI_MakeFace(pln, w).Face()
                gp_props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, gp_props)
                area = abs(gp_props.Mass())
                if area > best_area:
                    best_area, best_face = area, face

            if best_face and best_area > 1e-6:
                g = GProp_GProps()
                BRepGProp.SurfaceProperties_s(best_face, g)
                c = g.CentreOfMass()
                return float(c.Y()), float(c.Z())

    print(f"[sec] failed near X~{x_abs:.2f} (L={L:.1f}, H={H:.1f}, W={W:.1f})", file=__import__("sys").stderr)
    return None


def _cg_from_best_section(shape_local, *, xmin, xmax, ymin, ymax, zmin, zmax, L, H, W):
    """
    Return (y_c, z_c) at an X where the loop is fully formed (miters avoided).
    """
    guard = max(0.10 * L, 1.5 * W)
    x_lo  = xmin + guard
    x_hi  = xmax - guard
    if not (x_hi > x_lo):
        return None

    xs = [x_lo + i * (x_hi - x_lo) / 6.0 for i in range(7)]

    best_area, best_face = 0.0, None

    fuzzy = max(0.10, 0.002 * max(H, W))
    dx    = max(0.10, 0.001 * L)
    jitters = (0.0, +dx, -dx)

    def _plane_face_x_span(x_abs, margin=5.0):
        pln = gp_Pln(gp_Ax3(gp_Pnt(x_abs, 0.0, 0.0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0)))
        yu, Yv = float(ymin - margin), float(ymax + margin)
        zu, Zv = float(zmin - margin), float(zmax + margin)
        return BRepBuilderAPI_MakeFace(pln, yu, Yv, zu, Zv).Face(), pln

    def _iter_wires_local(comp):
        exp = TopExp_Explorer(comp, TopAbs_WIRE)
        while exp.More():
            yield TopoDS.Wire_s(exp.Current())
            exp.Next()

    for x0 in xs:
        for j in jitters:
            x = x0 + j
            face_rect, pln = _plane_face_x_span(x, margin=max(2.0, 0.02 * max(H, W)))
            sec = BRepAlgoAPI_Section(shape_local, face_rect, True)
            sec.Approximation(True)
            sec.ComputePCurveOn1(True)
            sec.ComputePCurveOn2(True)
            sec.SetFuzzyValue(fuzzy)
            sec.Build()
            if not sec.IsDone():
                continue

            edges_comp = sec.Shape()
            fb = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, False, False)
            closed_comp = fb.GetClosedWires()

            for w in _iter_wires_local(closed_comp):
                face = BRepBuilderAPI_MakeFace(pln, w).Face()
                gp_props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, gp_props)
                area = abs(gp_props.Mass())
                if area > best_area:
                    best_area, best_face = area, face

            if best_area < 1e-6:
                fb_open = ShapeAnalysis_FreeBounds(edges_comp, fuzzy, True, False)
                for w in _iter_wires_local(fb_open.GetOpenWires()):
                    wf = ShapeFix_Wire(w)
                    wf.SetMaxTolerance(fuzzy)
                    wf.Perform()
                    face = BRepBuilderAPI_MakeFace(pln, wf.Wire()).Face()
                    gp_props = GProp_GProps()
                    BRepGProp.SurfaceProperties_s(face, gp_props)
                    area = abs(gp_props.Mass())
                    if area > best_area:
                        best_area, best_face = area, face

            if best_face and best_area > 1e-6:
                g = GProp_GProps()
                BRepGProp.SurfaceProperties_s(best_face, g)
                c = g.CentreOfMass()
                return float(c.Y()), float(c.Z())

    print(f"[sec] no closed loop in core region [{x_lo:.1f},{x_hi:.1f}] (miters?)", file=__import__("sys").stderr)
    return None


def force_canonical_dstv_pose(shape_local, profile_type: str, x_frac: float = 0.25):
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(shape_local)
    L = float(xmax - xmin)
    H = float(ymax - ymin)
    W = float(zmax - zmin)

    cg = _cg_from_best_section(
        shape_local,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
        L=L, H=H, W=W
    )
    y_c, z_c = (None, None)
    if cg is not None:
        y_c, z_c = cg
        if profile_type == "U" and z_c > 0.5 * W:
            print("CG test (U): far side -> rotate 180 deg about X", file=__import__("sys").stderr)
            shape_local = _rot_180_about_X(shape_local)

    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(shape_local)
    L = float(xmax - xmin)
    H = float(ymax - ymin)
    W = float(zmax - zmin)
    return shape_local, (L, H, W, y_c, z_c)


def compute_dstv_pose(primary_aligned_shape, profile_type: str, profile_dims=None,
                      channel_mode: str = "toe_at_z0"):
    """
    Compute the DSTV pose for a section shape.
    Returns (refined_shape, dstv_ax3, tr_world_to_dstv, diag_dict)
    """
    # Stage 0: long leg -> Y (UEA)
    shape0, pre_turns = _enforce_long_leg_on_Y_if_angle(primary_aligned_shape, profile_type, profile_dims or {})
    # Stage 1: arg-min over candidate X-rotations; translate each candidate to min-corner before scoring
    candidates = []
    rots = (0, 2) if profile_type == "U" else (0, 1, 2, 3)
    for k in rots:
        sh = _rotX(shape0, k)
        sh0 = _to_origin_min_corner(sh)
        xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(sh0)
        H, W = (ymax - ymin, zmax - zmin)
        cg = _solid_centroid(sh0)
        y_, z_ = _norm_yz(cg, ymin, ymax, zmin, zmax)
        if profile_type == "U":
            if channel_mode == "toe_at_z0":
                score = abs(z_ - 0.8) + 0.5*abs(y_ - 0.5)
                policy = "U/toe_at_z0"
            else:
                score = abs(z_ - 0.2) + 0.5*abs(y_ - 0.5)
                policy = "U/web_at_z0"
        elif profile_type == "L":
            score = (y_ + z_)
            policy = "L/heel_at_origin"
        else:
            score = 0.0
            policy = "neutral"
        candidates.append((score, pre_turns + k, y_, z_, sh0, (H, W), policy))
    score, turns, y_sel, z_sel, refined_shape, (H, W), policy = min(candidates, key=lambda t: t[0])

    dstv_ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0).Crossed(gp_Dir(0, 1, 0)), gp_Dir(1, 0, 0))
    tr_world_to_dstv = gp_Trsf()  # identity

    conf = "ok"
    if profile_type == "U" and policy.startswith("U/") and z_sel < 0.5:
        conf = "low"
    if profile_type == "L" and (y_sel > 0.35 or z_sel > 0.35):
        conf = "low"
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(refined_shape)
    L = xmax - xmin

    refined_shape = _to_origin_min_corner(refined_shape)
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(refined_shape)
    eps = 1e-6
    if xmin < -eps or ymin < -eps or zmin < -eps:
        refined_shape = _to_origin_min_corner(refined_shape)
        xmin, xmax, ymin, ymax, zmin, zmax = _bbox_local(refined_shape)

    cg_pt = _solid_centroid(refined_shape)
    y_sel = (float(cg_pt.Y()) - ymin) / max(ymax - ymin, 1e-9)
    z_sel = (float(cg_pt.Z()) - zmin) / max(zmax - zmin, 1e-9)

    return refined_shape, dstv_ax3, tr_world_to_dstv, dict(
        turns_about_X=turns % 4, y_norm=y_sel, z_norm=z_sel, L=L, H=H, W=W,
        policy=policy, confidence=conf
    )
