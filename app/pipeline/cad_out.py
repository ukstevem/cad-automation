# pipeline/cad_out.py
# DXF profile export (plate outline).
# Ported from step-gemini: OCC.Core.* -> OCP.*
# shapely-based fingerprint (compute_dxf_fingerprint) is NOT included.

from pathlib import Path
from typing import Union, Tuple, Optional, List
from math import hypot
import numpy as np
import ezdxf

from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.GCPnts import GCPnts_UniformAbscissa
from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Plane
from OCP.gp import gp_Ax3, gp_Pnt, gp_Dir
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.TopoDS import TopoDS


def _wire_to_circle_uv(face, wire, project_uv, tol_r=1e-5, tol_ang=1e-3):
    """
    If 'wire' is a full circle: return (cu, cv, radius), else None.
    """
    wx = BRepTools_WireExplorer(wire, face)

    centers = []
    radii = []
    ang_sum = 0.0
    saw_any = False
    two_pi = 2.0 * np.pi

    while wx.More():
        edge = wx.Current()
        c = BRepAdaptor_Curve(edge)
        if c.GetType() != GeomAbs_Circle:
            return None
        circ = c.Circle()
        centers.append(np.array([circ.Location().X(), circ.Location().Y(), circ.Location().Z()], float))
        radii.append(float(circ.Radius()))

        t0, t1 = float(c.FirstParameter()), float(c.LastParameter())
        ang = abs(t1 - t0)
        while ang > two_pi:
            ang -= two_pi
        ang_sum += ang

        saw_any = True
        wx.Next()

    if not saw_any:
        return None

    if max(radii) - min(radii) > tol_r:
        return None

    if abs(ang_sum - two_pi) > tol_ang:
        return None

    C3 = np.mean(np.vstack(centers), axis=0)
    cu, cv = project_uv(C3)
    r = float(np.mean(radii))
    return (cu, cv, r)


def _ordered_uv_polyline(face, wire, project_uv, sampling_dist=1.0, samples_per_curve=None):
    """
    Return an ordered list of (u,v) along 'wire' so that successive points are contiguous.
    """
    # ---- Phase 1: collect all edges with 3D endpoints ----
    edges_info = []
    wx = BRepTools_WireExplorer(wire, face)
    while wx.More():
        edge = wx.Current()
        c = BRepAdaptor_Curve(edge)
        p0 = c.Value(c.FirstParameter())
        p1 = c.Value(c.LastParameter())
        edges_info.append({
            'curve': c,
            'start_3d': np.array([p0.X(), p0.Y(), p0.Z()], dtype=float),
            'end_3d':   np.array([p1.X(), p1.Y(), p1.Z()], dtype=float),
        })
        wx.Next()

    # ---- Phase 2: build UV polyline ----
    pts_uv = []
    prev = None

    for idx, ei in enumerate(edges_info):
        c = ei['curve']
        ts = None
        try:
            disc = GCPnts_UniformAbscissa(c, float(sampling_dist))
            if disc.IsDone() and disc.NbPoints() >= 2:
                ts = [disc.Parameter(i) for i in range(1, disc.NbPoints() + 1)]
        except Exception:
            pass
        if ts is None:
            t0, t1 = float(c.FirstParameter()), float(c.LastParameter())
            if c.GetType() == GeomAbs_Line:
                ts = [t0, t1]
            else:
                n = samples_per_curve or max(8, int(1 + abs(t1 - t0) / max(1e-6, sampling_dist)))
                ts = np.linspace(t0, t1, n)

        edge_uv = []
        for t in ts:
            p = c.Value(float(t))
            P3 = np.array([p.X(), p.Y(), p.Z()], dtype=float)
            edge_uv.append(project_uv(P3))

        if not edge_uv:
            continue

        if prev is not None:
            d_start = hypot(edge_uv[0][0] - prev[0], edge_uv[0][1] - prev[1])
            d_end   = hypot(edge_uv[-1][0] - prev[0], edge_uv[-1][1] - prev[1])
            if d_end < d_start:
                edge_uv.reverse()

            if hypot(edge_uv[0][0] - prev[0], edge_uv[0][1] - prev[1]) < 1e-8:
                edge_uv = edge_uv[1:]
        elif len(edges_info) > 1:
            # First edge: determine direction by checking which end connects
            # to the next edge (no prev available yet).
            next_ei = edges_info[1]
            ns, ne = next_ei['start_3d'], next_ei['end_3d']
            # Distance from this edge's end to next edge's nearest endpoint
            min_from_end = min(
                float(np.linalg.norm(ei['end_3d'] - ns)),
                float(np.linalg.norm(ei['end_3d'] - ne)),
            )
            # Distance from this edge's start to next edge's nearest endpoint
            min_from_start = min(
                float(np.linalg.norm(ei['start_3d'] - ns)),
                float(np.linalg.norm(ei['start_3d'] - ne)),
            )
            if min_from_start < min_from_end:
                edge_uv.reverse()

        pts_uv.extend(edge_uv)
        prev = pts_uv[-1]

    if pts_uv:
        if hypot(pts_uv[0][0] - pts_uv[-1][0], pts_uv[0][1] - pts_uv[-1][1]) >= 1e-8:
            pts_uv.append(pts_uv[0])

    return pts_uv


def export_profile_dxf_with_pca(
    shape,
    dxf_path: Union[Path, str],
    thumb_path: Optional[Union[Path, str]] = None,
    samples_per_curve: int = 16,
    fingerprint_tol: float = 0.5,
    ax3=None,
    canonicalize: bool = False,
) -> Tuple[str, Path, Optional[Path]]:
    """
    Export a 2D DXF profile of the largest planar face.
    Returns (fingerprint_hash, dxf_path, thumbnail_path).
    fingerprint_hash is always "" (shapely not available).
    """
    dxf_path = Path(dxf_path)
    dxf_path.parent.mkdir(parents=True, exist_ok=True)

    LINE_EPS_MM = 0.1

    def _perp_dist_point_to_segment(p, a, b):
        ax_, ay = a; bx, by = b; px, py = p
        vx, vy = bx - ax_, by - ay
        wx, wy = px - ax_, py - ay
        vv = vx*vx + vy*vy
        if vv < 1e-18:
            return hypot(wx, wy)
        t = max(0.0, min(1.0, (wx*vx + wy*vy)/vv))
        cx, cy = ax_ + t*vx, ay + t*vy
        return hypot(px - cx, py - cy)

    def _dedupe_consecutive(pts, tol=1e-9):
        out = []
        for p in pts:
            if not out or hypot(p[0]-out[-1][0], p[1]-out[-1][1]) > tol:
                out.append(p)
        return out

    def _simplify_closed_polyline(pts_closed, eps):
        if not pts_closed:
            return []
        closed = hypot(pts_closed[0][0]-pts_closed[-1][0], pts_closed[0][1]-pts_closed[-1][1]) < 1e-9
        pts = pts_closed[:-1] if closed else pts_closed[:]
        pts = _dedupe_consecutive(pts)
        if len(pts) <= 2:
            return pts[:2]

        a, b = pts[0], pts[-1]
        max_d = 0.0
        for p in pts[1:-1]:
            d = _perp_dist_point_to_segment(p, a, b)
            if d > max_d:
                max_d = d
        if max_d <= eps:
            return [a, b]

        arr = np.array(pts, float)
        c = arr.mean(axis=0)
        d = np.linalg.norm(arr - c, axis=1)
        i0 = int(np.argmax(d))
        ordered = pts[i0:] + pts[:i0]

        def rdp(poly):
            if len(poly) <= 2:
                return poly
            a_, b_ = poly[0], poly[-1]
            md, mi = -1.0, -1
            for i in range(1, len(poly)-1):
                di = _perp_dist_point_to_segment(poly[i], a_, b_)
                if di > md:
                    md, mi = di, i
            if md > eps:
                L = rdp(poly[:mi+1]); R = rdp(poly[mi:])
                return L[:-1] + R
            else:
                return [a_, b_]

        simp = rdp(ordered)
        if hypot(simp[0][0]-simp[-1][0], simp[0][1]-simp[-1][1]) > 1e-9:
            simp = simp + [simp[0]]
        return simp

    # 1) Largest planar face
    best_face, best_area = None, 0.0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        surf = BRepAdaptor_Surface(f)
        if surf.GetType() == GeomAbs_Plane:
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(f, props)
            a = props.Mass()
            if a > best_area:
                best_area, best_face = a, f
        exp.Next()
    if best_face is None:
        raise RuntimeError("No planar face found on shape")

    # ---- Path A: ax3 projection (no PCA) ----
    if ax3 is not None:
        surf = BRepAdaptor_Surface(best_face)
        pln = surf.Plane()
        Of = np.array([pln.Location().X(), pln.Location().Y(), pln.Location().Z()], dtype=float)
        N  = np.array([pln.Axis().Direction().X(),
                       pln.Axis().Direction().Y(),
                       pln.Axis().Direction().Z()], dtype=float)
        N /= max(np.linalg.norm(N), 1e-12)

        Xa = np.array([ax3.XDirection().X(), ax3.XDirection().Y(), ax3.XDirection().Z()], dtype=float)
        Ya = np.array([ax3.YDirection().X(), ax3.YDirection().Y(), ax3.YDirection().Z()], dtype=float)
        Za = np.array([ax3.Direction().X(),  ax3.Direction().Y(),  ax3.Direction().Z()], dtype=float)
        for v in (Xa, Ya, Za):
            n = np.linalg.norm(v)
            if n > 1e-12:
                v /= n

        def proj_in_plane(v):
            v = np.array(v, dtype=float)
            return v - np.dot(v, N) * N

        U = proj_in_plane(Xa)
        if np.linalg.norm(U) < 1e-9:
            alt = Ya if np.linalg.norm(proj_in_plane(Ya)) >= np.linalg.norm(proj_in_plane(Za)) else Za
            U = proj_in_plane(alt)
        U /= max(np.linalg.norm(U), 1e-12)

        Yp = proj_in_plane(Ya)
        Zy = proj_in_plane(Za)
        if np.linalg.norm(Yp) >= np.linalg.norm(Zy):
            V = Yp
        else:
            V = Zy

        V = V - np.dot(V, U) * U
        if np.linalg.norm(V) < 1e-9:
            V = np.cross(N, U)
        V /= max(np.linalg.norm(V), 1e-12)

        if np.dot(np.cross(U, V), N) < 0:
            V = -V

        O = Of
        project_uv = lambda P3: ((P3 - O).dot(U), (P3 - O).dot(V))

        loops, circles = [], []
        wexp = TopExp_Explorer(best_face, TopAbs_WIRE)
        while wexp.More():
            wire = TopoDS.Wire_s(wexp.Current())
            as_circle = _wire_to_circle_uv(best_face, wire, project_uv)
            if as_circle is not None:
                circles.append(as_circle)
            else:
                uv_pts = _ordered_uv_polyline(best_face, wire, project_uv, sampling_dist=1.0)
                if uv_pts:
                    loops.append(uv_pts)
            wexp.Next()

        line_geoms = []
        poly_loops = []
        for L in loops:
            if L and (abs(L[0][0]-L[-1][0]) > 1e-9 or abs(L[0][1]-L[-1][1]) > 1e-9):
                L = L + [L[0]]
            simp = _simplify_closed_polyline(L, LINE_EPS_MM)
            if len(simp) == 2:
                line_geoms.append((simp[0], simp[1]))
            else:
                poly_loops.append(simp)

        if canonicalize and (poly_loops or circles or line_geoms):
            pts = [p for L in poly_loops for p in L] \
                + [(c[0], c[1]) for c in circles] \
                + [p for seg in line_geoms for p in seg]
            A = np.array(pts, float)
            mn = A.min(axis=0)
            poly_loops = [[(u - mn[0], v - mn[1]) for (u, v) in L] for L in poly_loops]
            circles    = [(cu - mn[0], cv - mn[1], r) for (cu, cv, r) in circles]
            line_geoms = [((p0[0]-mn[0], p0[1]-mn[1]), (p1[0]-mn[0], p1[1]-mn[1])) for (p0, p1) in line_geoms]

        doc = ezdxf.new(dxfversion="R2010")
        try:
            doc.header["$INSUNITS"] = 4  # mm
        except Exception:
            pass
        msp = doc.modelspace()

        for (p0, p1) in line_geoms:
            if hypot(p0[0]-p1[0], p0[1]-p1[1]) > 1e-9:
                msp.add_line((p0[0], p0[1]), (p1[0], p1[1]))

        for pts in poly_loops:
            closed = abs(pts[0][0]-pts[-1][0]) < 1e-9 and abs(pts[0][1]-pts[-1][1]) < 1e-9
            out = pts[:-1] if closed else pts
            if len(out) >= 2:
                msp.add_lwpolyline(out, format="xy", close=closed)

        for (cu, cv, r) in circles:
            msp.add_circle((cu, cv), r)

        doc.saveas(str(dxf_path))
        return "", dxf_path, None

    # ---- Path B: plane-projection + PCA orientation ----
    surf = BRepAdaptor_Surface(best_face)
    pln = surf.Plane()
    origin = np.array([pln.Location().X(), pln.Location().Y(), pln.Location().Z()], float)
    xdir3 = np.array([pln.XAxis().Direction().X(), pln.XAxis().Direction().Y(), pln.XAxis().Direction().Z()], float)
    ydir3 = np.array([pln.YAxis().Direction().X(), pln.YAxis().Direction().Y(), pln.YAxis().Direction().Z()], float)

    proj_plane = lambda P3: ((P3 - origin).dot(xdir3), (P3 - origin).dot(ydir3))

    raw_loops, raw_circles = [], []
    wexp = TopExp_Explorer(best_face, TopAbs_WIRE)
    while wexp.More():
        wire = TopoDS.Wire_s(wexp.Current())
        as_circle = _wire_to_circle_uv(best_face, wire, proj_plane)
        if as_circle is not None:
            raw_circles.append(as_circle)
        else:
            uv_pts = _ordered_uv_polyline(best_face, wire, proj_plane, sampling_dist=1.0)
            if uv_pts:
                raw_loops.append(uv_pts)
        wexp.Next()

    all_pts = [p for L in raw_loops for p in L]
    for (cu, cv, r) in raw_circles:
        for k in range(8):
            a = 2*np.pi * k / 8
            all_pts.append((cu + r*np.cos(a), cv + r*np.sin(a)))

    all_pts_arr = np.array(all_pts, float)
    center2d = all_pts_arr.mean(axis=0)
    cov2 = np.cov((all_pts_arr - center2d).T)
    vals2, vecs2 = np.linalg.eigh(cov2)
    idx = np.argsort(vals2)[::-1]
    R = vecs2[:, idx].T
    if np.linalg.det(R) < 0:
        R[1, :] *= -1

    loops = []
    for L in raw_loops:
        arr = np.array(L, float)
        arr2 = (R @ (arr - center2d).T).T
        loops.append([tuple(p) for p in arr2])

    circles = []
    for (cu, cv, r) in raw_circles:
        c2 = R @ (np.array([cu, cv]) - center2d)
        circles.append((float(c2[0]), float(c2[1]), float(r)))

    line_geoms = []
    poly_loops = []
    for L in loops:
        if L and (abs(L[0][0]-L[-1][0]) > 1e-9 or abs(L[0][1]-L[-1][1]) > 1e-9):
            L = L + [L[0]]
        simp = _simplify_closed_polyline(L, LINE_EPS_MM)
        if len(simp) == 2:
            line_geoms.append((simp[0], simp[1]))
        else:
            poly_loops.append(simp)

    if canonicalize and (poly_loops or circles or line_geoms):
        pts2 = [p for L in poly_loops for p in L] \
             + [(c[0], c[1]) for c in circles] \
             + [p for seg in line_geoms for p in seg]
        A = np.array(pts2, float)
        mn = A.min(axis=0)
        poly_loops = [[(u - mn[0], v - mn[1]) for (u, v) in L] for L in poly_loops]
        circles    = [(cu - mn[0], cv - mn[1], r) for (cu, cv, r) in circles]
        line_geoms = [((p0[0]-mn[0], p0[1]-mn[1]), (p1[0]-mn[0], p1[1]-mn[1])) for (p0, p1) in line_geoms]

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    for (p0, p1) in line_geoms:
        if hypot(p0[0]-p1[0], p0[1]-p1[1]) > 1e-9:
            msp.add_line((p0[0], p0[1]), (p1[0], p1[1]))

    for pts in poly_loops:
        closed = abs(pts[0][0]-pts[-1][0]) < 1e-9 and abs(pts[0][1]-pts[-1][1]) < 1e-9
        out = pts[:-1] if closed else pts
        if len(out) >= 2:
            msp.add_lwpolyline(out, format="xy", close=closed)

    for (cu, cv, r) in circles:
        msp.add_circle((cu, cv), r)

    doc.saveas(str(dxf_path))
    return "", dxf_path, None
