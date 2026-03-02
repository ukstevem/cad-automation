"""
Connection detector — detect weld and bolt connections between solids.

Given an XCAF document and assembly tree, extracts all solids in world
coordinates, filters candidate pairs by AABB overlap, then classifies
each contact as welded (with weld length) or bolted (with bolt count).

This is the core algorithm module; it is called by the subprocess worker
``app/workers/detect_connections.py``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.BRepExtrema import BRepExtrema_IsInFace, BRepExtrema_IsOnEdge
from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane
from OCP.gp import gp_Pnt
from OCP.TCollection import TCollection_AsciiString
from OCP.TDF import TDF_Label, TDF_Tool
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_ShapeTool

logger = structlog.get_logger()

# ── Tolerances ────────────────────────────────────────────────────────────
DISTANCE_TOL = 0.5          # mm — max gap between touching solids
COPLANAR_ANGLE_DEG = 5.0    # degrees — normal anti-parallel tolerance
COPLANAR_DIST = 0.5         # mm — point-on-plane distance tolerance
COLLINEAR_ANGLE_DEG = 5.0   # degrees — hole axis alignment tolerance
BOLT_CONCENTRICITY_TOL = 2.0  # mm — hole axis coincidence tolerance
BOLT_RADIUS_TOL = 1.5      # mm — hole radius match tolerance
HOLE_RADIUS_MIN = 2.0       # mm — min hole radius (M4)
HOLE_RADIUS_MAX = 30.0      # mm — max hole radius (M56)
HOLE_MIN_ANGULAR_EXTENT = 3.0  # radians (~172°) — accept half-cylinders (BRep splits holes into 2 semi-circular faces)
MIN_WELD_PERIMETER = 1.0    # mm — ignore trivially small contact
MAX_SOLIDS = 1000           # safety limit — refuse if more solids than this


# ── Data structures ───────────────────────────────────────────────────────
@dataclass
class SolidInfo:
    node_id: str            # XCAF instance label entry (e.g. "0:1:1:3:1")
    ref_id: str             # prototype ref_id
    name: str               # part name
    solid_index: int        # index within compound (0 for single-solid)
    shape: Any              # TopoDS_Solid in world coordinates
    aabb: Tuple[float, float, float, float, float, float] = field(default=(0,) * 6)
    # (xmin, xmax, ymin, ymax, zmin, zmax)
    parent_node_id: str = ""  # immediate child of the selected scope node


# ── Step 1: Extract all solids ────────────────────────────────────────────

def extract_all_solids(
    doc, shape_tool, tree_nodes: List[dict],
    classifications: Optional[Dict[str, str]] = None,
) -> Tuple[List[SolidInfo], set]:
    """
    Walk assembly tree leaf nodes and extract every solid in world coords.

    *tree_nodes* is the ``children`` list (or top-level list) from the cached
    assembly tree JSON.

    *classifications* is an optional ``{node_id: action}`` dict from
    ``project_state``.  Nodes classified as ``"bought-out"`` are treated as
    opaque leaf nodes — all their solids are extracted under the BO node's
    own ``node_id`` (they are not recursed into).

    Returns ``(solids, bo_node_ids)`` where *bo_node_ids* is the set of
    node IDs that are bought-out.  Callers should filter out intra-BO pairs
    (both solids sharing a BO node_id) using ``build_candidate_pairs()``.
    """
    _cls = classifications or {}
    bo_node_ids: set = set()

    # Build ref_id → effective action lookup.  If every `ref_id:sN` entry
    # in _cls shares the same action (e.g. all "bought-out"), that action
    # applies to any instance of that ref_id.
    _ref_action: Dict[str, str] = {}
    _ref_tmp: Dict[str, set] = {}
    for key, action in _cls.items():
        if ":s" in key:
            ref_part = key.rsplit(":s", 1)[0]
            _ref_tmp.setdefault(ref_part, set()).add(action)
    for ref, actions in _ref_tmp.items():
        if len(actions) == 1:
            _ref_action[ref] = next(iter(actions))

    def _is_bo(node: dict) -> bool:
        """Check if a tree node is classified as bought-out."""
        nid = node.get("id", "")
        if _cls.get(nid) == "bought-out":
            return True
        ref = node.get("ref_id", "")
        if ref and _ref_action.get(ref) == "bought-out":
            return True
        return False

    solids: List[SolidInfo] = []

    def _extract_leaf(node: dict, branch_id: str) -> None:
        node_id = node.get("id", "")
        ref_id = node.get("ref_id", "")
        name = node.get("name", "unknown")

        label = TDF_Label()
        TDF_Tool.Label_s(
            doc.GetData(), TCollection_AsciiString(node_id), label,
        )
        if label.IsNull():
            return

        shape = XCAFDoc_ShapeTool.GetShape_s(label)
        if shape is None or shape.IsNull():
            return

        idx = 0
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp.More():
            solid = TopoDS.Solid_s(exp.Current())
            exp.Next()

            box = Bnd_Box()
            BRepBndLib.Add_s(solid, box)
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
            aabb = (
                float(xmin), float(xmax),
                float(ymin), float(ymax),
                float(zmin), float(zmax),
            )

            solids.append(SolidInfo(
                node_id=node_id,
                ref_id=ref_id,
                name=name,
                solid_index=idx,
                shape=solid,
                aabb=aabb,
                parent_node_id=branch_id,
            ))
            idx += 1

    def _walk(nodes: List[dict], branch_id: str = "") -> None:
        for node in nodes:
            nid = node.get("id", "")

            # Bought-out items: treat as opaque leaf (don't recurse children)
            if _is_bo(node):
                bo_node_ids.add(nid)
                _extract_leaf(node, branch_id or nid)
                continue

            children = node.get("children", [])
            if children:
                if branch_id:
                    # Already inside a branch — keep the same id
                    _walk(children, branch_id=branch_id)
                else:
                    # Top level — each child starts a new branch
                    for child in children:
                        _walk([child], branch_id=child.get("id", ""))
                continue

            _extract_leaf(node, branch_id or nid)

    _walk(tree_nodes)
    logger.info("connection_solids_extracted", count=len(solids),
                bo_nodes=len(bo_node_ids))
    return solids, bo_node_ids


# ── Step 2: AABB pre-filter ──────────────────────────────────────────────

def _aabb_overlap(a: tuple, b: tuple, tol: float = 1.0) -> bool:
    """Check if two AABBs overlap within *tol* mm gap."""
    axmin, axmax, aymin, aymax, azmin, azmax = a
    bxmin, bxmax, bymin, bymax, bzmin, bzmax = b
    return not (
        axmax + tol < bxmin or bxmax + tol < axmin
        or aymax + tol < bymin or bymax + tol < aymin
        or azmax + tol < bzmin or bzmax + tol < azmin
    )


def build_candidate_pairs(
    solids: List[SolidInfo],
    tolerance: float = 1.0,
    scope: str = "all",
    bo_node_ids: Optional[set] = None,
) -> List[Tuple[int, int]]:
    """
    Return (i, j) index pairs whose AABBs overlap.

    Uses sweep-and-prune on the X axis to avoid full O(n^2) comparison.

    *scope* controls which pairs are kept:

    - ``"all"`` — every overlapping pair (default).
    - ``"siblings"`` — only pairs from *different* branches
      (``parent_node_id`` differs).
    - ``"within-part"`` — only pairs from the *same* leaf part
      (``node_id`` matches).

    *bo_node_ids* — set of bought-out node IDs.  Pairs where both solids
    belong to the same BO node are excluded (intra-BO connections are
    not meaningful).
    """
    _bo = bo_node_ids or set()
    n = len(solids)
    # Sort by xmin
    order = sorted(range(n), key=lambda k: solids[k].aabb[0])

    pairs: List[Tuple[int, int]] = []
    for p in range(n):
        i = order[p]
        _, ixmax, _, _, _, _ = solids[i].aabb
        for q in range(p + 1, n):
            j = order[q]
            jxmin = solids[j].aabb[0]
            # Once jxmin exceeds ixmax + tolerance, no further j can overlap
            if jxmin > ixmax + tolerance:
                break
            if _aabb_overlap(solids[i].aabb, solids[j].aabb, tolerance):
                a, b = (i, j) if i < j else (j, i)
                pairs.append((a, b))

    # Apply scope filtering
    raw_count = len(pairs)
    if scope == "siblings":
        pairs = [
            (i, j) for i, j in pairs
            if solids[i].parent_node_id != solids[j].parent_node_id
        ]
    elif scope == "within-part":
        pairs = [
            (i, j) for i, j in pairs
            if solids[i].node_id == solids[j].node_id
        ]

    # Filter out intra-BO pairs (both solids from the same BO node)
    if _bo:
        pairs = [
            (i, j) for i, j in pairs
            if not (
                solids[i].node_id == solids[j].node_id
                and solids[i].node_id in _bo
            )
        ]

    logger.info(
        "connection_candidate_pairs",
        total_solids=n,
        candidate_pairs=len(pairs),
        raw_pairs=raw_count,
        scope=scope,
    )
    return pairs


# ── Helper: planar face extraction ────────────────────────────────────────

def _get_planar_faces(solid) -> List[Tuple[Any, np.ndarray, np.ndarray]]:
    """
    Extract planar faces with outward normal and centroid.

    Returns list of ``(TopoDS_Face, normal_np, centroid_np)``.
    """
    result = []
    exp = TopExp_Explorer(solid, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        exp.Next()

        surf = BRepAdaptor_Surface(face)
        if surf.GetType() != GeomAbs_Plane:
            continue

        pln = surf.Plane()
        n = pln.Axis().Direction()
        normal = np.array([n.X(), n.Y(), n.Z()], dtype=float)

        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = abs(props.Mass())
        if area < 1.0:  # skip tiny faces (< 1 mm²)
            continue

        c = props.CentreOfMass()
        centroid = np.array([c.X(), c.Y(), c.Z()], dtype=float)
        result.append((face, normal, centroid))
    return result


# ── Helper: edge length measurement ──────────────────────────────────────

def _measure_edge_length(shape) -> float:
    """Sum the lengths of all edges in *shape*."""
    total = 0.0
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        exp.Next()
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, props)
        total += abs(props.Mass())
    return total


# ── Helper: weld path polyline extraction ─────────────────────────────────

def _extract_edge_polylines(shape, n_samples: int = 20) -> List[List[List[float]]]:
    """
    Sample points along each edge in *shape*, return list of polylines.

    Each polyline is a list of ``[x, y, z]`` coordinate triples (rounded
    to 2 decimal places).
    """
    polylines: List[List[List[float]]] = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        exp.Next()
        try:
            curve = BRepAdaptor_Curve(edge)
            u0 = curve.FirstParameter()
            u1 = curve.LastParameter()
            if abs(u1 - u0) < 1e-12:
                continue
            pts: List[List[float]] = []
            for k in range(n_samples + 1):
                u = u0 + (u1 - u0) * k / n_samples
                p = curve.Value(u)
                pts.append([round(p.X(), 2), round(p.Y(), 2), round(p.Z(), 2)])
            polylines.append(pts)
        except Exception:
            continue
    return polylines


# ── Helper: hole extraction ──────────────────────────────────────────────

def _extract_holes(solid) -> List[Dict[str, Any]]:
    """
    Extract bolt-hole-like cylindrical faces from *solid*.

    Returns hole descriptors ``{center, axis, radius}`` (all numpy).

    Uses cylindrical **faces** instead of circular edges to avoid
    matching fillet/round arcs.  A real bolt hole is a near-full-
    revolution cylinder (>= ~258°); fillets and chamfers are partial
    cylinders and are rejected.
    """
    holes: List[Dict[str, Any]] = []
    seen: set = set()

    n_cyl = 0
    n_rejected_angle = 0
    n_rejected_radius = 0
    n_dedup = 0

    exp = TopExp_Explorer(solid, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        exp.Next()

        surf = BRepAdaptor_Surface(face)
        if surf.GetType() != GeomAbs_Cylinder:
            continue

        n_cyl += 1

        # Check angular extent — reject partial cylinders (fillets, rounds)
        u_range = abs(surf.LastUParameter() - surf.FirstUParameter())
        if u_range < HOLE_MIN_ANGULAR_EXTENT:
            n_rejected_angle += 1
            continue

        cyl = surf.Cylinder()
        loc = cyl.Location()
        axis_dir = cyl.Axis().Direction()
        radius = cyl.Radius()

        center_np = np.array([loc.X(), loc.Y(), loc.Z()], float)
        axis_np = np.array([axis_dir.X(), axis_dir.Y(), axis_dir.Z()], float)
        nrm = np.linalg.norm(axis_np)
        if nrm < 1e-12:
            continue
        axis_np /= nrm

        # Deduplicate by rounded coordinates
        key = (
            round(center_np[0], 1), round(center_np[1], 1),
            round(center_np[2], 1), round(radius, 1),
        )
        if key in seen:
            n_dedup += 1
            continue
        seen.add(key)

        if radius < HOLE_RADIUS_MIN or radius > HOLE_RADIUS_MAX:
            n_rejected_radius += 1
            continue

        holes.append({"center": center_np, "axis": axis_np, "radius": radius})

    if n_cyl > 0:
        logger.info(
            "hole_extraction",
            cylindrical_faces=n_cyl,
            rejected_angle=n_rejected_angle,
            rejected_radius=n_rejected_radius,
            dedup=n_dedup,
            holes_found=len(holes),
        )
    return holes


# ── Step 3a: bolt detection ───────────────────────────────────────────────

def _detect_bolt_holes(solidA, solidB) -> Optional[Dict[str, Any]]:
    """Match collinear holes between two solids → bolt connection."""
    holes_a = _extract_holes(solidA)
    holes_b = _extract_holes(solidB)
    if not holes_a or not holes_b:
        return None

    cos_limit = math.cos(math.radians(COLLINEAR_ANGLE_DEG))
    matched: List[dict] = []
    used_b: set = set()

    for ha in holes_a:
        for j, hb in enumerate(holes_b):
            if j in used_b:
                continue

            # Axes must be (anti-)parallel
            dot = abs(np.dot(ha["axis"], hb["axis"]))
            if dot < cos_limit:
                continue

            # Radii must be close
            if abs(ha["radius"] - hb["radius"]) > BOLT_RADIUS_TOL:
                continue

            # Axes must be coincident (not just parallel) — concentricity check
            cc = hb["center"] - ha["center"]
            along = np.dot(cc, ha["axis"])
            perp = cc - along * ha["axis"]
            if np.linalg.norm(perp) > BOLT_CONCENTRICITY_TOL:
                continue

            matched.append({
                "diameter_mm": round(ha["radius"] + hb["radius"], 2),
            })
            used_b.add(j)
            break

    if not matched:
        return None

    avg_dia = sum(b["diameter_mm"] for b in matched) / len(matched)
    return {
        "type": "bolted",
        "bolt_count": len(matched),
        "bolt_diameter_mm": round(avg_dia, 1),
        "weld_length_mm": None,
        "contact_faces": None,
    }


# ── Step 3b: weld contact detection ──────────────────────────────────────

def _detect_weld_contact(solidA, solidB) -> Optional[Dict[str, Any]]:
    """Find coplanar face pairs and compute overlap (weld) perimeter."""
    faces_a = _get_planar_faces(solidA)
    faces_b = _get_planar_faces(solidB)
    if not faces_a or not faces_b:
        return None

    cos_limit = -math.cos(math.radians(COPLANAR_ANGLE_DEG))
    total_weld = 0.0
    n_contacts = 0
    all_weld_paths: List[List[List[float]]] = []

    for fa, na, ptA in faces_a:
        for fb, nb, ptB in faces_b:
            # Normals must be anti-parallel
            dot = np.dot(na, nb)
            if dot > cos_limit:
                continue

            # Point from face B must lie on plane of face A
            offset = ptB - ptA
            plane_dist = abs(np.dot(offset, na))
            if plane_dist > COPLANAR_DIST:
                continue

            # Boolean intersection of coplanar faces
            try:
                common = BRepAlgoAPI_Common(fa, fb)
                common.Build()
                if not common.IsDone():
                    continue
                overlap = common.Shape()
                perimeter = _measure_edge_length(overlap)
                if perimeter > MIN_WELD_PERIMETER:
                    total_weld += perimeter
                    n_contacts += 1
                    # Extract weld path polylines
                    paths = _extract_edge_polylines(overlap)
                    all_weld_paths.extend(paths)
            except Exception:
                continue

    if n_contacts == 0:
        return None

    return {
        "type": "welded",
        "weld_length_mm": round(total_weld, 1),
        "contact_faces": n_contacts,
        "weld_paths": all_weld_paths,
        "bolt_count": None,
        "bolt_diameter_mm": None,
    }


# ── Step 3c: T-joint / edge-on-face weld detection ───────────────────────

def _detect_t_joint_weld(solidA, solidB) -> Optional[Dict[str, Any]]:
    """Detect welds where edges of one solid lie on planar faces of the other.

    Covers T-joints, corner joints, and edge-on-face contacts that the
    coplanar face overlap check misses (perpendicular plates, etc.).
    """
    result = _check_edges_on_faces(solidA, solidB)
    if result is None:
        # Try the reverse direction
        result = _check_edges_on_faces(solidB, solidA)
    return result


def _edge_overlap_length(
    edge, face, edge_len: float, n_samples: int = 20,
) -> Tuple[float, List[List[List[float]]]]:
    """Compute the length of *edge* that actually overlaps the finite *face*.

    Returns ``(overlap_length, weld_path_polylines)``.

    Samples the edge at *n_samples* points and checks each against the
    face's bounding box (expanded by ``COPLANAR_DIST``).  Only the
    portion of the edge inside the face extent is counted.
    """
    face_box = Bnd_Box()
    BRepBndLib.Add_s(face, face_box)
    fxmin, fymin, fzmin, fxmax, fymax, fzmax = face_box.Get()
    margin = COPLANAR_DIST + 1.0
    fxmin -= margin; fymin -= margin; fzmin -= margin
    fxmax += margin; fymax += margin; fzmax += margin

    curve = BRepAdaptor_Curve(edge)
    u0 = curve.FirstParameter()
    u1 = curve.LastParameter()
    du = (u1 - u0) / n_samples

    # Track which sample points fall inside the face AABB
    inside: List[bool] = []
    pts: List[List[float]] = []
    for k in range(n_samples + 1):
        u = u0 + du * k
        p = curve.Value(u)
        x, y, z = p.X(), p.Y(), p.Z()
        in_box = (
            fxmin <= x <= fxmax
            and fymin <= y <= fymax
            and fzmin <= z <= fzmax
        )
        inside.append(in_box)
        pts.append([round(x, 2), round(y, 2), round(z, 2)])

    # Count the fraction of segments that are inside
    seg_len = edge_len / n_samples
    overlap = 0.0
    weld_pts: List[List[float]] = []
    for k in range(n_samples):
        if inside[k] or inside[k + 1]:
            overlap += seg_len
            if not weld_pts:
                weld_pts.append(pts[k])
            weld_pts.append(pts[k + 1])

    paths = [weld_pts] if len(weld_pts) >= 2 else []
    return overlap, paths


def _check_edges_on_faces(
    face_solid, edge_solid,
) -> Optional[Dict[str, Any]]:
    """Find edges of *edge_solid* that lie on planar faces of *face_solid*.

    For each planar face, checks whether any edge of the other solid lies
    in the face's plane *and* is actually close to the face (not just in
    the same infinite plane far away).
    """
    faces = _get_planar_faces(face_solid)
    if not faces:
        return None

    # Collect all edges of edge_solid with pre-computed lengths
    edges = []
    exp = TopExp_Explorer(edge_solid, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        exp.Next()
        try:
            props = GProp_GProps()
            BRepGProp.LinearProperties_s(edge, props)
            length = abs(props.Mass())
            if length >= MIN_WELD_PERIMETER:
                edges.append((edge, length))
        except Exception:
            continue
    if not edges:
        return None

    total_weld = 0.0
    n_contacts = 0
    all_weld_paths: List[List[List[float]]] = []
    matched_edges: set = set()  # track by id to avoid double-counting

    for face, normal, face_pt in faces:
        for idx, (edge, edge_len) in enumerate(edges):
            if id(edge) in matched_edges:
                continue
            try:
                curve = BRepAdaptor_Curve(edge)
                u0 = curve.FirstParameter()
                u1 = curve.LastParameter()
                if abs(u1 - u0) < 1e-12:
                    continue

                # Sample start, mid, end — all must lie on the face plane
                on_plane = True
                for u in (u0, (u0 + u1) / 2, u1):
                    p = curve.Value(u)
                    pt = np.array([p.X(), p.Y(), p.Z()])
                    if abs(np.dot(pt - face_pt, normal)) > COPLANAR_DIST:
                        on_plane = False
                        break
                if not on_plane:
                    continue

                # Confirm edge is actually near the face (not far away
                # in the same infinite plane)
                d = BRepExtrema_DistShapeShape(edge, face)
                if not d.IsDone() or d.Value() > COPLANAR_DIST:
                    continue

                # Compute only the portion of the edge that overlaps
                # the finite face (not the full edge length)
                overlap_len, weld_paths = _edge_overlap_length(
                    edge, face, edge_len,
                )
                if overlap_len < MIN_WELD_PERIMETER:
                    continue

                matched_edges.add(id(edge))
                total_weld += overlap_len
                n_contacts += 1
                all_weld_paths.extend(weld_paths)

            except Exception:
                continue

    if n_contacts == 0:
        return None

    return {
        "type": "welded",
        "weld_length_mm": round(total_weld, 1),
        "contact_faces": None,
        "weld_paths": all_weld_paths,
        "bolt_count": None,
        "bolt_diameter_mm": None,
        "auto_classified": True,
        "weld_method": "t-joint",
    }


# ── Helper: classify contact geometry ─────────────────────────────────────

_FACE_TYPE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
}


def _classify_contact(
    dist_calc: BRepExtrema_DistShapeShape,
) -> Dict[str, Any]:
    """Extract face-type info from the extremal contact point(s).

    Returns a dict with:
    - ``contact_subtype`` — e.g. "plane-cylinder", "edge-plane", "vertex"
    - ``cylinder_radius`` — if one support is cylindrical (mm or None)
    """
    subtype_parts = []
    cyl_radius: Optional[float] = None

    for shape_idx in (1, 2):
        try:
            sup_type = (
                dist_calc.SupportTypeShape1(1)
                if shape_idx == 1
                else dist_calc.SupportTypeShape2(1)
            )
            if sup_type == BRepExtrema_IsInFace:
                sup = (
                    dist_calc.SupportOnShape1(1)
                    if shape_idx == 1
                    else dist_calc.SupportOnShape2(1)
                )
                face = TopoDS.Face_s(sup)
                surf = BRepAdaptor_Surface(face)
                geom_type = surf.GetType()
                name = _FACE_TYPE_NAMES.get(geom_type, "other")
                subtype_parts.append(name)

                if geom_type == GeomAbs_Cylinder:
                    cyl_radius = surf.Cylinder().Radius()
            elif sup_type == BRepExtrema_IsOnEdge:
                subtype_parts.append("edge")
            else:
                subtype_parts.append("vertex")
        except Exception:
            subtype_parts.append("unknown")

    return {
        "contact_subtype": "-".join(sorted(subtype_parts)),
        "cylinder_radius": round(cyl_radius, 2) if cyl_radius else None,
    }


# ── Step 3: combined contact detection ────────────────────────────────────

def detect_contact(solidA, solidB) -> Optional[Dict[str, Any]]:
    """
    Detect connection between two solids.

    Returns a dict with ``type`` (welded/bolted/contact) or ``None``.
    """
    # Quick distance gate
    try:
        dist_calc = BRepExtrema_DistShapeShape(solidA, solidB)
        if not dist_calc.IsDone():
            return None
        min_dist = dist_calc.Value()
        if min_dist > DISTANCE_TOL:
            return None
    except Exception:
        return None

    # Check bolts first (more specific signal)
    bolt = _detect_bolt_holes(solidA, solidB)
    if bolt is not None:
        return bolt

    # Check coplanar face overlap (butt/lap welds)
    weld = _detect_weld_contact(solidA, solidB)
    if weld is not None:
        return weld

    # Check T-joint / edge-on-face welds (perpendicular plates, etc.)
    if min_dist < 0.1:
        t_joint = _detect_t_joint_weld(solidA, solidB)
        if t_joint is not None:
            return t_joint

    # Classify the contact geometry for better reporting
    contact_info = _classify_contact(dist_calc)

    # Auto-classify cylinder-on-plane at 0mm as welded (structural convention:
    # tube sitting on plate = fillet weld).  Estimate weld length from the
    # cylinder circumference.
    subtype = contact_info.get("contact_subtype", "")
    cyl_r = contact_info.get("cylinder_radius")
    if min_dist < 0.1 and "cylinder" in subtype and "plane" in subtype and cyl_r:
        estimated_weld = round(2 * math.pi * cyl_r, 1)
        return {
            "type": "welded",
            "weld_length_mm": estimated_weld,
            "contact_faces": 1,
            "weld_paths": [],
            "bolt_count": None,
            "bolt_diameter_mm": None,
            "contact_subtype": subtype,
            "auto_classified": True,
        }

    # Solids are close but no classifiable joint
    return {
        "type": "contact",
        "weld_length_mm": None,
        "contact_faces": None,
        "bolt_count": None,
        "bolt_diameter_mm": None,
        "min_distance_mm": round(min_dist, 3),
        **contact_info,
    }


# ── Main entry point ─────────────────────────────────────────────────────

def detect_connections(
    doc,
    shape_tool,
    tree_nodes: List[dict],
    scope: str = "all",
    classifications: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Detect all connections between solids in *tree_nodes*.

    Parameters
    ----------
    doc : TDocStd_Document
    shape_tool : XCAFDoc_ShapeTool
    tree_nodes : list[dict]
        Assembly tree nodes (top-level list from sidecar JSON).
    scope : str
        ``"all"`` (default), ``"siblings"``, or ``"within-part"``.
    classifications : dict, optional
        ``{node_id: action}`` from project_state.  Nodes classified as
        ``"bought-out"`` are excluded from solid extraction.

    Returns
    -------
    dict with ``connections``, ``summary``, and ``timing`` keys.
    """
    t0 = time.monotonic()

    # Step 1: extract solids
    solids, bo_node_ids = extract_all_solids(
        doc, shape_tool, tree_nodes, classifications=classifications,
    )
    t1 = time.monotonic()

    if len(solids) > MAX_SOLIDS:
        raise ValueError(
            f"Too many solids ({len(solids)}) for connection detection. "
            f"Maximum is {MAX_SOLIDS}. Use node_id to scope to a smaller "
            f"subtree (e.g. a single weldment)."
        )

    # Step 2: AABB pre-filter
    pairs = build_candidate_pairs(
        solids, tolerance=1.0, scope=scope, bo_node_ids=bo_node_ids,
    )
    t2 = time.monotonic()

    # Step 3+4: detect and classify
    connections: List[Dict[str, Any]] = []
    for idx, (i, j) in enumerate(pairs):
        if idx % 50 == 0 and idx > 0:
            logger.info("connection_progress", checked=idx, total=len(pairs))
        result = detect_contact(solids[i].shape, solids[j].shape)
        if result is not None:
            sa, sb = solids[i], solids[j]
            connections.append({
                "solid_a": {
                    "node_id": sa.node_id,
                    "solid_index": sa.solid_index,
                    "name": sa.name,
                },
                "solid_b": {
                    "node_id": sb.node_id,
                    "solid_index": sb.solid_index,
                    "name": sb.name,
                },
                **result,
            })
    t3 = time.monotonic()

    # Summary
    welded = [c for c in connections if c["type"] == "welded"]
    bolted = [c for c in connections if c["type"] == "bolted"]
    contact_only = [c for c in connections if c["type"] == "contact"]

    return {
        "connections": connections,
        "summary": {
            "total_solids": len(solids),
            "candidate_pairs": len(pairs),
            "welded_connections": len(welded),
            "bolted_connections": len(bolted),
            "unclassified_contacts": len(contact_only),
            "total_weld_length_mm": round(
                sum(c.get("weld_length_mm") or 0 for c in welded), 1,
            ),
            "total_bolt_count": sum(c.get("bolt_count") or 0 for c in bolted),
        },
        "timing": {
            "solid_extraction_ms": round((t1 - t0) * 1000, 1),
            "aabb_filter_ms": round((t2 - t1) * 1000, 1),
            "contact_detection_ms": round((t3 - t2) * 1000, 1),
            "total_ms": round((t3 - t0) * 1000, 1),
        },
    }


# ── Cross-assembly detection ─────────────────────────────────────────────

_BOUNDARY_MARGIN = 5.0  # mm — expand overlap region when filtering solids


def detect_cross_assembly(
    doc,
    shape_tool,
    tree_nodes: List[dict],
    classifications: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Detect connections between top-level children of an assembly node.

    Instead of extracting all solids at once (which may exceed MAX_SOLIDS),
    this function:

    1. Computes a bounding box for each child's compound shape (cheap).
    2. Finds pairs of children whose bounding boxes overlap.
    3. For each overlapping pair, extracts only the solids near the shared
       boundary and runs detection on those boundary solids.

    Parameters
    ----------
    doc : TDocStd_Document
    shape_tool : XCAFDoc_ShapeTool
    tree_nodes : list[dict]
        The top-level node (wrapped in a list) whose children are the
        sub-assemblies to check pairwise.
    classifications : dict, optional
        ``{node_id: action}`` from project_state.  Currently unused for
        cross-assembly (BO children participate as opaque units).

    Returns
    -------
    dict with ``connections``, ``summary``, ``timing``, and
    ``pair_details`` keys.
    """
    t0 = time.monotonic()

    # The input should be [root_node] — get its children
    root_node = tree_nodes[0] if tree_nodes else None
    if not root_node:
        raise ValueError("No root node provided for cross-assembly detection")

    children = root_node.get("children", [])
    if not children:
        raise ValueError("Root node has no children for cross-assembly detection")

    # ── Step 1: compute per-child AABBs from compound shapes ──
    # BO children participate — they are treated as opaque units.
    # scope="siblings" in build_candidate_pairs already prevents
    # intra-child (i.e. intra-BO) pairs.
    child_info = []
    for child in children:
        node_id = child.get("id", "")

        label = TDF_Label()
        TDF_Tool.Label_s(
            doc.GetData(), TCollection_AsciiString(node_id), label,
        )
        if label.IsNull():
            continue

        shape = XCAFDoc_ShapeTool.GetShape_s(label)
        if shape is None or shape.IsNull():
            continue

        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        if bbox.IsVoid():
            continue

        child_info.append({
            "node": child,
            "node_id": node_id,
            "name": child.get("name", ""),
            "shape": shape,
            "bnd": bbox,
        })

    t1 = time.monotonic()
    logger.info(
        "cross_assembly_aabbs",
        children=len(child_info),
        aabb_ms=round((t1 - t0) * 1000, 1),
    )

    # ── Step 2: find overlapping child pairs ──
    overlapping_pairs = []
    for i in range(len(child_info)):
        for j in range(i + 1, len(child_info)):
            if not child_info[i]["bnd"].IsOut(child_info[j]["bnd"]):
                overlapping_pairs.append((i, j))

    t2 = time.monotonic()
    logger.info(
        "cross_assembly_pairs",
        possible=len(child_info) * (len(child_info) - 1) // 2,
        overlapping=len(overlapping_pairs),
    )

    # ── Step 3: for each pair, extract boundary solids and detect ──
    all_connections: List[Dict[str, Any]] = []
    total_solids_checked = 0
    total_pairs_checked = 0
    pair_details = []

    for pair_idx, (ci, cj) in enumerate(overlapping_pairs):
        info_a = child_info[ci]
        info_b = child_info[cj]

        # Compute overlap region
        xa1, ya1, za1, xa2, ya2, za2 = info_a["bnd"].Get()
        xb1, yb1, zb1, xb2, yb2, zb2 = info_b["bnd"].Get()

        overlap_box = Bnd_Box()
        overlap_box.Add(gp_Pnt(
            max(xa1, xb1) - _BOUNDARY_MARGIN,
            max(ya1, yb1) - _BOUNDARY_MARGIN,
            max(za1, zb1) - _BOUNDARY_MARGIN,
        ))
        overlap_box.Add(gp_Pnt(
            min(xa2, xb2) + _BOUNDARY_MARGIN,
            min(ya2, yb2) + _BOUNDARY_MARGIN,
            min(za2, zb2) + _BOUNDARY_MARGIN,
        ))

        # Extract boundary solids directly from compound shapes
        # (compound shapes are in world coords; tree-walk shapes are not)
        solids_a = _extract_solids_from_compound(
            info_a["shape"], info_a["node_id"], info_a["name"],
            overlap_box,
        )
        solids_b = _extract_solids_from_compound(
            info_b["shape"], info_b["node_id"], info_b["name"],
            overlap_box,
        )

        if not solids_a or not solids_b:
            continue

        # Combine and run detection (cross-child pairs only)
        combined = solids_a + solids_b
        n_combined = len(combined)
        total_solids_checked += n_combined

        if n_combined > MAX_SOLIDS:
            logger.warning(
                "cross_assembly_pair_too_large",
                pair=f"{info_a['node_id']}<->{info_b['node_id']}",
                solids=n_combined,
                skipped=True,
            )
            continue

        pairs = build_candidate_pairs(combined, tolerance=1.0, scope="siblings")
        total_pairs_checked += len(pairs)

        pair_conns = []
        for pi, pj in pairs:
            result = detect_contact(combined[pi].shape, combined[pj].shape)
            if result is not None:
                sa, sb = combined[pi], combined[pj]
                pair_conns.append({
                    "solid_a": {
                        "node_id": sa.node_id,
                        "solid_index": sa.solid_index,
                        "name": sa.name,
                    },
                    "solid_b": {
                        "node_id": sb.node_id,
                        "solid_index": sb.solid_index,
                        "name": sb.name,
                    },
                    **result,
                })

        all_connections.extend(pair_conns)
        if pair_conns:
            pair_details.append({
                "child_a": info_a["node_id"],
                "child_a_name": info_a["node"].get("name", ""),
                "child_b": info_b["node_id"],
                "child_b_name": info_b["node"].get("name", ""),
                "boundary_solids": n_combined,
                "connections_found": len(pair_conns),
            })

        if (pair_idx + 1) % 10 == 0:
            logger.info(
                "cross_assembly_progress",
                completed=pair_idx + 1,
                total=len(overlapping_pairs),
                connections_so_far=len(all_connections),
            )

    t3 = time.monotonic()

    # Summary
    welded = [c for c in all_connections if c["type"] == "welded"]
    bolted = [c for c in all_connections if c["type"] == "bolted"]
    contact_only = [c for c in all_connections if c["type"] == "contact"]

    return {
        "connections": all_connections,
        "summary": {
            "total_solids": total_solids_checked,
            "candidate_pairs": total_pairs_checked,
            "overlapping_children": len(overlapping_pairs),
            "children_with_connections": len(pair_details),
            "welded_connections": len(welded),
            "bolted_connections": len(bolted),
            "unclassified_contacts": len(contact_only),
            "total_weld_length_mm": round(
                sum(c.get("weld_length_mm") or 0 for c in welded), 1,
            ),
            "total_bolt_count": sum(c.get("bolt_count") or 0 for c in bolted),
        },
        "timing": {
            "aabb_ms": round((t1 - t0) * 1000, 1),
            "pair_finding_ms": round((t2 - t1) * 1000, 1),
            "detection_ms": round((t3 - t2) * 1000, 1),
            "total_ms": round((t3 - t0) * 1000, 1),
        },
        "pair_details": pair_details,
    }


def _extract_solids_from_compound(
    compound_shape,
    node_id: str,
    name: str,
    boundary_box: Optional[Bnd_Box] = None,
) -> List[SolidInfo]:
    """Extract solids from a compound shape (already in world coords).

    If *boundary_box* is given, only solids whose AABBs intersect it are
    returned (for boundary filtering in cross-assembly detection).
    """
    solids: List[SolidInfo] = []
    idx = 0
    exp = TopExp_Explorer(compound_shape, TopAbs_SOLID)
    while exp.More():
        solid = TopoDS.Solid_s(exp.Current())
        exp.Next()

        box = Bnd_Box()
        BRepBndLib.Add_s(solid, box)

        if boundary_box is not None and box.IsOut(boundary_box):
            idx += 1
            continue

        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        aabb = (
            float(xmin), float(xmax),
            float(ymin), float(ymax),
            float(zmin), float(zmax),
        )
        solids.append(SolidInfo(
            node_id=node_id,
            ref_id="",
            name=name,
            solid_index=idx,
            shape=solid,
            aabb=aabb,
            parent_node_id=node_id,
        ))
        idx += 1
    return solids
