# pipeline/geom_alignment.py
# Minimal aligner: pick the longest straight edge (fallback: longest wire), map to +X.
# Seed +Z from OBB secondary and orthogonalize to X. No extra "smart" rolls.
# Ported from step-gemini: OCC.Core.* -> OCP.*

from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Plane
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Trsf
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.Bnd import Bnd_Box, Bnd_OBB
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

import math
import numpy as np


def _build_xz_frame(dir_x: gp_Dir, dir_z: gp_Dir):
    """
    Build a right-handed (X,Y,Z) frame keeping X exact and re-orthogonalising Z to X.
    Returns: trsf, from_cs, world_cs, dx, dy, dz
    """
    X = np.array([dir_x.X(), dir_x.Y(), dir_x.Z()], dtype=float)
    nX = np.linalg.norm(X)
    X = X / (nX if nX else 1.0)

    Z = np.array([dir_z.X(), dir_z.Y(), dir_z.Z()], dtype=float)
    Z = Z - np.dot(Z, X) * X
    nZ = np.linalg.norm(Z)
    if nZ < 1e-12:
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(up, X))) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        Z = up - np.dot(up, X) * X
        nZ = np.linalg.norm(Z)
    Z = Z / (nZ if nZ else 1.0)

    Y = np.cross(Z, X)
    nY = np.linalg.norm(Y)
    Y = Y / (nY if nY else 1.0)
    if float(np.dot(np.cross(X, Y), Z)) < 0.0:
        Y = -Y

    dx = gp_Dir(float(X[0]), float(X[1]), float(X[2]))
    dy = gp_Dir(float(Y[0]), float(Y[1]), float(Y[2]))
    dz = gp_Dir(float(Z[0]), float(Z[1]), float(Z[2]))

    origin = gp_Pnt(0.0, 0.0, 0.0)
    from_cs = gp_Ax3(origin, dz, dx)                        # local: Z main, X as XDir
    world_cs = gp_Ax3(origin, gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))  # world: Z up, X forward

    trsf = gp_Trsf()
    trsf.SetDisplacement(from_cs, world_cs)
    return trsf, from_cs, world_cs, dx, dy, dz


def _canon_dir(v: np.ndarray) -> np.ndarray:
    """Make direction signless: flip so its largest-magnitude component is positive."""
    i = int(np.argmax(np.abs(v)))
    return v if v[i] >= 0 else -v


def _dominant_perp_face_normals(solid, X_unit: np.ndarray, *, dot_eps=0.2, ang_tol_deg=7.0) -> list:
    """
    Return face-normal clusters (largest-area first) that are ~perpendicular to X.
    dot_eps≈0.2 ≈ 11.5° from perpendicular; ang_tol_deg for clustering.
    """
    X_unit = X_unit / (np.linalg.norm(X_unit) or 1.0)
    clusters = []
    cos_tol = math.cos(math.radians(ang_tol_deg))

    fexp = TopExp_Explorer(solid, TopAbs_FACE)
    while fexp.More():
        f = TopoDS.Face_s(fexp.Current())
        s = BRepAdaptor_Surface(f)
        if s.GetType() != GeomAbs_Plane:
            fexp.Next()
            continue

        pln = s.Plane()
        n = np.array([pln.Axis().Direction().X(), pln.Axis().Direction().Y(), pln.Axis().Direction().Z()], float)
        n /= (np.linalg.norm(n) or 1.0)

        if abs(float(np.dot(n, X_unit))) > dot_eps:
            fexp.Next()
            continue

        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, props)
        A = float(props.Mass()) if math.isfinite(props.Mass()) and props.Mass() > 0 else 1.0

        n = _canon_dir(n)
        assigned = False
        for c in clusters:
            if abs(float(np.dot(c["dir"], n))) > cos_tol:
                c["dir"] = (c["dir"] * c["area"] + n * A)
                c["dir"] /= (np.linalg.norm(c["dir"]) or 1.0)
                c["area"] += A
                assigned = True
                break
        if not assigned:
            clusters.append({"dir": n, "area": A})

        fexp.Next()

    clusters.sort(key=lambda c: c["area"], reverse=True)
    return [c["dir"] for c in clusters]


def _obb_longest_axes(shape):
    """Return [(axis_vec(np3), half_size), ...] sorted by size desc."""
    obb = Bnd_OBB()
    BRepBndLib.AddOBB_s(shape, obb, True, True, True)
    axes = [
        (np.array([obb.XDirection().X(), obb.XDirection().Y(), obb.XDirection().Z()], float), float(obb.XHSize())),
        (np.array([obb.YDirection().X(), obb.YDirection().Y(), obb.YDirection().Z()], float), float(obb.YHSize())),
        (np.array([obb.ZDirection().X(), obb.ZDirection().Y(), obb.ZDirection().Z()], float), float(obb.ZHSize())),
    ]
    axes.sort(key=lambda t: t[1], reverse=True)
    return axes


def _wire_endpoints_vector(wire):
    """Approx wire direction = vector from first to last edge endpoints (model-space)."""
    pts = []
    eexp = TopExp_Explorer(wire, TopAbs_EDGE)
    while eexp.More():
        e = TopoDS.Edge_s(eexp.Current())
        c = BRepAdaptor_Curve(e)
        p1 = c.Value(c.FirstParameter())
        p2 = c.Value(c.LastParameter())
        if not pts:
            pts.append(np.array([p1.X(), p1.Y(), p1.Z()], float))
        pts.append(np.array([p2.X(), p2.Y(), p2.Z()], float))
        eexp.Next()
    if len(pts) < 2:
        return None
    v = pts[-1] - pts[0]
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else None


def _longest_straight_edge_dir(solid):
    """Scan all faces/wires/edges; return (unit_dir, length) for the longest GeomAbs_Line edge."""
    best_len = -1.0
    best_dir = None
    fexp = TopExp_Explorer(solid, TopAbs_FACE)
    while fexp.More():
        f = TopoDS.Face_s(fexp.Current())
        wexp = TopExp_Explorer(f, TopAbs_WIRE)
        while wexp.More():
            w = TopoDS.Wire_s(wexp.Current())
            eexp = TopExp_Explorer(w, TopAbs_EDGE)
            while eexp.More():
                e = TopoDS.Edge_s(eexp.Current())
                c = BRepAdaptor_Curve(e)
                if c.GetType() == GeomAbs_Line:
                    p1 = c.Value(c.FirstParameter())
                    p2 = c.Value(c.LastParameter())
                    v = np.array([p2.X() - p1.X(), p2.Y() - p1.Y(), p2.Z() - p1.Z()], float)
                    L = float(np.linalg.norm(v))
                    if L > best_len and L > 0:
                        best_len = L
                        best_dir = v / L
                eexp.Next()
            wexp.Next()
        fexp.Next()
    return best_dir, best_len


def _longest_wire_dir(solid):
    """If no straight edge found, pick the wire with greatest chord length (end-to-end)."""
    best_len = -1.0
    best_dir = None
    fexp = TopExp_Explorer(solid, TopAbs_FACE)
    while fexp.More():
        f = TopoDS.Face_s(fexp.Current())
        wexp = TopExp_Explorer(f, TopAbs_WIRE)
        while wexp.More():
            w = TopoDS.Wire_s(wexp.Current())
            v = _wire_endpoints_vector(w)
            if v is not None:
                L = 1.0
                if L > best_len:
                    best_len = L
                    best_dir = v
            wexp.Next()
        fexp.Next()
    return best_dir, best_len


def _build_frame_and_trsf(dir_x_np: np.ndarray, dir_z_seed_np: np.ndarray):
    """Orthonormalize Z to X, make RH frame, return (trsf, from_cs, world_cs, dx, dy, dz)."""
    X = dir_x_np / np.linalg.norm(dir_x_np)
    Z = dir_z_seed_np - np.dot(dir_z_seed_np, X) * X
    nz = float(np.linalg.norm(Z))
    if nz < 1e-12:
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(up, X)) > 0.90:
            up = np.array([0.0, 1.0, 0.0])
        Z = up - np.dot(up, X) * X
        Z /= np.linalg.norm(Z)
    else:
        Z /= nz
    Y = np.cross(Z, X)
    Y /= np.linalg.norm(Y)
    if float(np.dot(np.cross(X, Y), Z)) < 0.0:
        Y = -Y

    dx = gp_Dir(float(X[0]), float(X[1]), float(X[2]))
    dy = gp_Dir(float(Y[0]), float(Y[1]), float(Y[2]))
    dz = gp_Dir(float(Z[0]), float(Z[1]), float(Z[2]))

    origin = gp_Pnt(0, 0, 0)
    from_cs = gp_Ax3(origin, dz, dx)
    world_cs = gp_Ax3(origin, gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    trsf = gp_Trsf()
    trsf.SetDisplacement(from_cs, world_cs)
    return trsf, from_cs, world_cs, dx, dy, dz


def align_by_longest_straight_edge(solid, *, debug: bool = False):
    """
    Minimal, semi-working aligner:
      1) Use the LONGEST straight edge direction as +X (fallback: longest wire end-to-end).
      2) Seed +Z from the OBB secondary axis; orthogonalize to X; build RH frame.
      3) Displace (Z->+Z, X->+X).
    Returns: aligned, trsf, world_cs, None, dx, dy, dz, dbg
    """
    dbg = {"used": "longest-edge", "longest_edge_len": 0.0}

    # 1) choose X
    edge_dir, edge_len = _longest_straight_edge_dir(solid)
    axes = None
    if edge_dir is None:
        wire_dir, _ = _longest_wire_dir(solid)
        if wire_dir is None:
            axes = _obb_longest_axes(solid)
            X = axes[0][0] / np.linalg.norm(axes[0][0])
            dbg["used"] = "obb-fallback"
        else:
            X = wire_dir
            dbg["used"] = "longest-wire"
    else:
        X = edge_dir
        dbg["used"] = "longest-edge"
        dbg["longest_edge_len"] = float(edge_len)

    # stabilize sign so X points roughly +globalX
    if X[0] < 0:
        X = -X

    # Try planar-face voting for roll.
    face_dirs = _dominant_perp_face_normals(solid, X, dot_eps=0.2, ang_tol_deg=7.0)
    if face_dirs:
        Yc = face_dirs[0] - np.dot(face_dirs[0], X) * X
        Yc /= (np.linalg.norm(Yc) or 1.0)
        Z = np.cross(X, Yc)
    else:
        if axes is None:
            axes = _obb_longest_axes(solid)
        Zcand = axes[1][0]
        Z = Zcand - np.dot(Zcand, X) * X

    Z /= (np.linalg.norm(Z) or 1.0)

    dx = gp_Dir(float(X[0]), float(X[1]), float(X[2]))
    dz = gp_Dir(float(Z[0]), float(Z[1]), float(Z[2]))
    trsf, from_cs, world_cs, dx, dy, dz = _build_xz_frame(dx, dz)
    aligned = BRepBuilderAPI_Transform(solid, trsf, True).Shape()

    # basic debug bbox
    bb = Bnd_Box()
    BRepBndLib.Add_s(aligned, bb)
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    if debug:
        dbg.update({"extents_after": (xmax - xmin, ymax - ymin, zmax - zmin)})

    return aligned, trsf, world_cs, None, dx, dy, dz, (dbg if debug else None)
