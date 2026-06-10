"""
Measured-geometry feature extraction for ML training and live persistence.

Given a single TopoDS solid shape, ``extract_solid_features`` computes a flat
dict of **measured** (not library-derived) geometric features — exactly the
inputs a tabular classifier wants for plate/section/designation prediction.

Two modes:

  - **fast** (default) — derives everything from the cheap geometric
    fingerprint (volume, surface, tight sorted bbox dims, principal inertia,
    chirality) plus algebraic ratios.  No alignment, no OCC cross-section
    slicing, so it scales to tens of thousands of solids.  The longest bbox dim
    is treated as the extrusion length, giving ``csa_from_vol = volume/length``
    and a ``fill_ratio`` proxy ``csa_from_vol / (mid·thin)`` — the near-perfect
    plate-vs-section separator (≈1 for plates, ≪1 for open/hollow sections).

  - **slow** (``fast=False``) — additionally runs longest-straight-edge
    alignment and the robust 21-slice cross-section, overwriting the OBB extents
    and ``cs_H/cs_W/section_area`` with the more accurate measured values.

Nothing here depends on the steel library or the rule-based classifier's verdict,
so these features are an *independent* signal that ML can be trained against.

The function never raises: failed sub-computations leave keys ``None`` and set
the ``features_ok`` / ``align_ok`` flags so callers can filter.
"""
from __future__ import annotations

from typing import Any, Dict

import structlog

from app.services.parts_consolidator import fingerprint_shape

logger = structlog.get_logger()

# Canonical column order the exporter writes and the trainer reads.
# Append new keys at the end to keep old datasets loadable.
FEATURE_KEYS = [
    "obb_length", "obb_height", "obb_width",
    "dim_long", "dim_mid", "dim_thin",
    "section_area", "cs_H", "cs_W",
    "fill_ratio",
    "elongation", "flatness", "compactness",
    "csa_from_vol",
    "t_eff", "t_eff_thin_ratio", "developed_ratio",
    "volume_mm3", "surface_area_mm2",
    "inertia0", "inertia1", "inertia2",
    "inertia_ratio_10", "inertia_ratio_21",
    "chirality",
    # cross-section raster features (only populated when section=True; the
    # raster is expensive so it runs on thin-walled candidates only):
    "n_holes", "thk_max", "thk_cov", "thk_max_over_teff",
    # convex cylindrical bend faces with R/t_eff>=0.9 (excludes rolled toe
    # radii and holes): >=1 => formed plate; >=5 (curved tube) => bent section:
    "n_convex_bends",
]


# Only run the cross-section raster when the part is thin-walled enough to be a
# plate/formed/section candidate; flat plates (fill ~1) and chunky parts skip it.
_SECTION_FILL_GATE = 0.6


def _empty_features() -> Dict[str, Any]:
    d: Dict[str, Any] = {k: None for k in FEATURE_KEYS}
    d["align_ok"] = False
    d["features_ok"] = False
    d["feat_mode"] = "fast"
    return d


def extract_solid_features(shape, fast: bool = True,
                           section: bool = False) -> Dict[str, Any]:
    """Return a flat dict of measured geometric features for one solid shape.

    Keys are ``FEATURE_KEYS`` plus ``align_ok``, ``features_ok``, ``feat_mode``.
    Never raises — failed sub-steps leave ``None`` values.

    ``section=True`` additionally runs the (expensive) cross-section raster to
    populate ``n_holes`` (enclosed voids → closed hollow box section),
    ``thk_max`` (max inscribed wall thickness), ``thk_cov`` (wall-thickness
    variation) and ``thk_max_over_teff`` (flange-vs-web contrast: ~1 uniform,
    >1.5 for I/channel).  To bound cost it only runs on thin-walled candidates
    (``fill_ratio`` below ``_SECTION_FILL_GATE``) — flat plates don't need it.
    """
    feats = _empty_features()

    # --- fingerprint: volume, surface, sorted dims, inertia, chirality -------
    # Reuses the exact routine the consolidator uses, so dims/inertia here are
    # directly comparable to the persisted consolidation fingerprints.
    try:
        fp = fingerprint_shape(shape)
    except Exception as exc:  # pragma: no cover - fingerprint_shape is defensive
        fp = None
        logger.debug("feature_fingerprint_failed", error=str(exc))

    if fp is not None:
        volume, surface, d0, d1, d2, i0, i1, i2, chir = fp  # d0<=d1<=d2
        feats["volume_mm3"] = volume
        feats["surface_area_mm2"] = surface
        feats["dim_thin"], feats["dim_mid"], feats["dim_long"] = d0, d1, d2
        feats["inertia0"], feats["inertia1"], feats["inertia2"] = i0, i1, i2
        feats["chirality"] = chir
        feats["inertia_ratio_10"] = round(i1 / i0, 4) if i0 else None
        feats["inertia_ratio_21"] = round(i2 / i1, 4) if i1 else None
        # Sheet-gauge estimate: for any thin uniform-wall body (flat OR folded),
        # volume = gauge * developed_area and surface ~= 2 * developed_area, so
        # t_eff = 2V/S recovers the metal thickness regardless of folding.
        if surface and surface > 0:
            t_eff = 2.0 * volume / surface
            feats["t_eff"] = round(t_eff, 3)
            # ~1 for a FLAT plate (gauge == smallest bbox dim); <<1 for a FORMED
            # plate or a rolled section (bbox depth is the fold/profile, not metal).
            if d0:
                feats["t_eff_thin_ratio"] = round(t_eff / d0, 4)
            # developed area vs bbox footprint: >1 once the sheet is folded.
            if d1 and d2:
                feats["developed_ratio"] = round((surface / 2.0) / (d1 * d2), 4)
        if d0 and d1 and d2:
            feats["elongation"] = round(d2 / d1, 4) if d1 else None  # long/mid
            feats["flatness"] = round(d1 / d0, 4) if d0 else None    # mid/thin
            if volume:
                feats["compactness"] = round(volume / (d0 * d1 * d2), 4)
            # Treat the longest dim as the extrusion length.
            csa = volume / d2 if d2 else None
            feats["csa_from_vol"] = round(csa, 2) if csa is not None else None
            if csa is not None and d0 and d1:
                feats["fill_ratio"] = round(csa / (d0 * d1), 4)
            # Fast-mode OBB + cross-section proxies (overwritten by slow mode).
            feats["obb_length"], feats["obb_height"], feats["obb_width"] = d2, d1, d0
            feats["cs_H"], feats["cs_W"] = d1, d0
            feats["section_area"] = feats["csa_from_vol"]

    feats["features_ok"] = feats["volume_mm3"] is not None

    # --- cross-section raster (expensive; thin-walled candidates only) -------
    # Populates n_holes / thk_max / thk_cov / thk_max_over_teff.  Gated on
    # fill_ratio so flat plates and chunky parts skip it.
    if section and feats.get("fill_ratio") is not None \
            and feats["fill_ratio"] < _SECTION_FILL_GATE:
        try:
            rast = _cross_section_raster(shape)
            if rast is not None:
                (feats["n_holes"], feats["thk_max"], feats["thk_cov"],
                 cs_h, cs_w, cs_area) = rast
                if feats.get("t_eff"):
                    feats["thk_max_over_teff"] = round(feats["thk_max"] / feats["t_eff"], 3)
                # Overwrite the fast-mode fingerprint proxies with the raster-
                # measured cross-section (used for the library match).
                feats["cs_H"], feats["cs_W"], feats["section_area"] = cs_h, cs_w, cs_area
        except Exception as exc:
            logger.debug("feature_section_raster_failed", error=str(exc))
        try:
            if feats.get("t_eff"):
                feats["n_convex_bends"] = _count_convex_bends(shape, feats["t_eff"])
        except Exception as exc:
            logger.debug("feature_bend_count_failed", error=str(exc))

    if fast:
        return feats

    # --- slow refinement: alignment + robust measured cross-section ----------
    feats["feat_mode"] = "slow"
    try:
        from app.pipeline.geom_alignment import align_by_longest_straight_edge
        from app.pipeline.geometry_utils import (
            compute_obb_geometry, compute_section_dims,
        )
        aligned = align_by_longest_straight_edge(shape)[0]
        feats["align_ok"] = True
    except Exception as exc:
        logger.debug("feature_align_failed", error=str(exc))
        return feats

    try:
        L, H, W = (float(v) for v in compute_obb_geometry(aligned)["aligned_extents"])
        feats["obb_length"], feats["obb_height"], feats["obb_width"] = (
            round(L, 2), round(H, 2), round(W, 2)
        )
        if L > 1.0 and feats["volume_mm3"]:
            feats["csa_from_vol"] = round(feats["volume_mm3"] / L, 2)
    except Exception as exc:
        logger.debug("feature_obb_failed", error=str(exc))

    try:
        sd = compute_section_dims(aligned)
        area = float(sd.get("area") or 0.0)
        cs_h = float(sd.get("H") or 0.0)
        cs_w = float(sd.get("W") or 0.0)
        if area > 0:
            feats["section_area"] = round(area, 2)
        if cs_h > 0:
            feats["cs_H"] = round(cs_h, 2)
        if cs_w > 0:
            feats["cs_W"] = round(cs_w, 2)
        if cs_h > 0 and cs_w > 0:
            feats["fill_ratio"] = round(area / (cs_h * cs_w), 4)
    except Exception as exc:
        logger.debug("feature_section_dims_failed", error=str(exc))

    return feats


# ---------------------------------------------------------------------------
# Cross-section raster (pure-numpy; no scipy).  Rasters the mid-length
# cross-section via point-in-solid classification, then derives:
#   n_holes  – enclosed voids in the section (≥1 → closed hollow box section)
#   thk_max  – max inscribed wall thickness (mm)
#   thk_cov  – wall-thickness variation across the core (≈0 uniform; higher
#              for I/channel with distinct web/flange thicknesses)
# Validated against a true SHS (n_holes=1), channel (thk_max/t_eff~1.7),
# formed plate and flat plate (n_holes=0).
# ---------------------------------------------------------------------------
def _chamfer_dt(mask):
    """8-connected chamfer distance transform: distance of True cells to the
    nearest False cell (two-pass, weights 1 / sqrt2)."""
    import numpy as np
    INF = 1e9
    d = np.where(mask, INF, 0.0)
    H, W = d.shape
    a, b = 1.0, 1.4142135
    for i in range(H):
        for j in range(W):
            if d[i, j] == 0:
                continue
            m = d[i, j]
            if i > 0: m = min(m, d[i-1, j] + a)
            if j > 0: m = min(m, d[i, j-1] + a)
            if i > 0 and j > 0: m = min(m, d[i-1, j-1] + b)
            if i > 0 and j < W-1: m = min(m, d[i-1, j+1] + b)
            d[i, j] = m
    for i in range(H-1, -1, -1):
        for j in range(W-1, -1, -1):
            if d[i, j] == 0:
                continue
            m = d[i, j]
            if i < H-1: m = min(m, d[i+1, j] + a)
            if j < W-1: m = min(m, d[i, j+1] + a)
            if i < H-1 and j < W-1: m = min(m, d[i+1, j+1] + b)
            if i < H-1 and j > 0: m = min(m, d[i+1, j-1] + b)
            d[i, j] = m
    return d


def _count_holes(mask):
    """Count enclosed voids: background cells not reachable from the border."""
    import numpy as np
    from collections import deque
    bg = ~mask
    H, W = bg.shape
    vis = np.zeros_like(bg)
    dq = deque()
    for i in range(H):
        for j in (0, W-1):
            if bg[i, j] and not vis[i, j]:
                vis[i, j] = True; dq.append((i, j))
    for j in range(W):
        for i in (0, H-1):
            if bg[i, j] and not vis[i, j]:
                vis[i, j] = True; dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < H and 0 <= nj < W and bg[ni, nj] and not vis[ni, nj]:
                vis[ni, nj] = True; dq.append((ni, nj))
    enc = bg & ~vis
    lbl = np.zeros_like(bg, int)
    cur = 0
    for i in range(H):
        for j in range(W):
            if enc[i, j] and lbl[i, j] == 0:
                cur += 1; lbl[i, j] = cur; dq = deque([(i, j)])
                while dq:
                    a, c = dq.popleft()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = a+di, c+dj
                        if 0 <= ni < H and 0 <= nj < W and enc[ni, nj] and lbl[ni, nj] == 0:
                            lbl[ni, nj] = cur; dq.append((ni, nj))
    return sum(1 for k in range(1, cur+1) if (lbl == k).sum() > 3)


def _count_convex_bends(solid, t_eff):
    """Count convex cylindrical faces that look like sheet/section bends.

    A formed-plate bend is a CONVEX cylinder (material on the axis side) whose
    radius is at least the metal gauge (R/t_eff >= 0.9) spanning a partial arc
    (30-270 deg).  This excludes drilled holes (concave) and rolled-section toe
    radii (R < gauge).  Empirically: 0 for plates/straight sections, >=1 for
    formed plates, and a large count for curved tubes (bent section).
    """
    import numpy as np
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_IN
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.gp import gp_Pnt
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier

    if not t_eff or t_eff <= 0:
        return 0
    cls = BRepClass3d_SolidClassifier(solid)

    def inside(p):
        cls.Perform(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), 1e-7)
        return cls.State() == TopAbs_IN

    n = 0
    exp = TopExp_Explorer(solid, TopAbs_FACE)
    while exp.More():
        s = BRepAdaptor_Surface(TopoDS.Face_s(exp.Current()))
        if s.GetType() == GeomAbs_Cylinder:
            cyl = s.Cylinder()
            R = cyl.Radius()
            ax = cyl.Axis()
            D = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
            C = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
            uu = (s.FirstUParameter() + s.LastUParameter()) / 2
            vv = (s.FirstVParameter() + s.LastVParameter()) / 2
            arc = np.degrees(abs(s.LastUParameter() - s.FirstUParameter()))
            pv = s.Value(uu, vv)
            P = np.array([pv.X(), pv.Y(), pv.Z()])
            foot = C + np.dot(P - C, D) * D
            rad = P - foot
            nn = np.linalg.norm(rad)
            if nn > 1e-6:
                rad /= nn
                # step from the face toward the axis: inside => material on the
                # axis side => convex outer surface (a bend, not a hole/fillet).
                if inside(P - 0.15 * rad) and 30 <= arc <= 270 and R / t_eff >= 0.9:
                    n += 1
        exp.Next()
    return n


def _cross_section_raster(solid):
    """Return (n_holes, thk_max_mm, thk_cov) from the mid-length cross-section,
    or None if it can't be rastered.  Extrusion axis = longest bbox dim."""
    import numpy as np
    from OCP.gp import gp_Pnt
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN

    bb = Bnd_Box()
    BRepBndLib.Add_s(solid, bb)
    b = bb.Get()
    mn = np.array(b[:3]); mx = np.array(b[3:]); ext = mx - mn
    L = int(np.argmax(ext)); uv = [i for i in range(3) if i != L]
    cls = BRepClass3d_SolidClassifier(solid)

    def inside(pt):
        cls.Perform(gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])), 1e-6)
        return cls.State() == TopAbs_IN

    R = []
    for fr in (0.4, 0.5, 0.6):
        xL = mn[L] + ext[L] * fr
        u0, u1 = mn[uv[0]], mx[uv[0]]
        v0, v1 = mn[uv[1]], mx[uv[1]]
        g = max(max(u1 - u0, v1 - v0) / 70.0, 0.3)
        us = np.arange(u0, u1 + g, g); vs = np.arange(v0, v1 + g, g)
        mask = np.zeros((len(us), len(vs)), bool)
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                pt = [0, 0, 0]; pt[L] = xL; pt[uv[0]] = u; pt[uv[1]] = v
                mask[i, j] = inside(pt)
        if mask.sum() < 5:
            continue
        dt = _chamfer_dt(mask) * g * 2.0
        tmax = float(dt.max())
        core = dt[dt > 0.5 * tmax]
        cov = float(core.std() / core.mean()) if len(core) > 1 and core.mean() > 0 else 0.0
        # Section envelope + material area straight from the mask — reused for
        # the library match in the lightweight classify pass (no re-slicing).
        ys, xs = np.where(mask)
        h = (ys.max() - ys.min() + 1) * g
        w = (xs.max() - xs.min() + 1) * g
        area = float(mask.sum()) * g * g
        R.append((_count_holes(mask), tmax, cov, max(h, w), min(h, w), area))
    if not R:
        return None
    return (int(np.median([r[0] for r in R])),
            round(float(np.median([r[1] for r in R])), 2),
            round(float(np.median([r[2] for r in R])), 3),
            round(float(np.median([r[3] for r in R])), 2),
            round(float(np.median([r[4] for r in R])), 2),
            round(float(np.median([r[5] for r in R])), 1))
