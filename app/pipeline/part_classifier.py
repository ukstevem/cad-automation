"""
Refined part classifier — the validated decision tree (≈91% on 616 labels).

Takes the measured features (``extract_solid_features`` output), the existing
library-match verdict (``rule_type`` = plate/section/unknown) and the part name,
and returns a refined routing class plus a confidence.  This is the SINGLE
source of truth for the rule tree: the live STEP pipeline
(``cnc_shape_analyser``), the offline evaluator (``scripts/eval_rules.py``) and
the labelling-gallery preview all classify through here / mirror this logic.

Classes:
  SECTION       straight rolled/hollow section -> saw/drill (NC1)
  PLATE         flat plate -> laser/plasma cut (DXF)
  FORMED_PLATE  bent sheet -> cut + press-brake bend
  BENT_SECTION  curved tube/section -> section/tube bending
  BOUGHT_OUT    purchased catalogue product (e.g. Unistrut)
  EXCLUDE       tiny / degenerate artifact (filled-in hole, broken geometry)

Thresholds are calibrated against verified.csv; see scripts/eval_rules.py for
the per-class precision/recall.  Confidence < ~0.5 means the part fell to the
uncertain catch-all and should be flagged for human review.
"""
from typing import Any, Dict, Optional

from app.pipeline.catalogue_products import match_catalogue_product

# Tiny / physically degenerate solids — no real fabricated part is this small.
EXCLUDE_VOL_MM3 = 10000.0
# A convex bend with R>=gauge; >=5 means a curved tube (bent section).
BENT_MIN_BENDS = 5
FORMED_MIN_BENDS = 1
# Flat-plate gate on t_eff/dim_thin (~1 for a flat plate, <<1 for a profile).
PLATE_TTHIN = 0.45
# Open profile with distinct flanges (web<<flange) — I/UC/PFC.
SECTION_THK_RATIO = 1.5
# Fastener gate. inertia_ratio_21 (I2/I1) ~ 1 means two equal principal inertias,
# i.e. a solid of revolution about the long axis; elongation bounds it to a
# stubby one (a long threaded rod or tube is not a bolt).
#
# The shape ratios alone are NOT enough — they have no scale, and a *bent*
# member lands right on top of them: its bounding box goes near-square, so
# elongation and inertia_ratio_21 both drift to ~1 by coincidence, while the
# mid-length raster slices across the curve and reports a nonsense "head"
# thickness ratio.  A curved UB (2030x1901, 8.2M mm3) and a 90-degree CHS bend
# (1290x1290, 2.9M mm3) both read as "stubby axisymmetric with a head".  So bound
# the gate by absolute size: a fastener is a SMALL, SHORT part (M20 bolt:
# 68 long, 37 across the head corners, 39k mm3 — 200x smaller than those).
# M36x300 is about the largest common structural bolt; M64's head is ~110 across
# corners.  Anything bigger falls through to the section rules for review.
FASTENER_I21_TOL = 0.03
FASTENER_MAX_ELONGATION = 8.0
FASTENER_MAX_DIM_LONG = 300.0   # mm — overall length
FASTENER_MAX_DIM_MID = 150.0    # mm — head across corners


def classify_part(features: Optional[Dict[str, Any]],
                  rule_type: Optional[str],
                  part_name: Optional[str] = None) -> Dict[str, Any]:
    """Return ``{"class", "confidence", "reason"}`` for one solid.

    ``features``   – an ``extract_solid_features(..., section=True)`` dict.
    ``rule_type``  – the existing classifier's verdict ("section"/"plate"/...).
    ``part_name``  – display name, for catalogue-product (BO) matching.
    """
    f = features or {}

    # 0. No solid geometry (part_no_solid / degenerate) — nothing to fabricate,
    #    so route to EXCLUDE rather than the uncertain catch-all.
    if not f.get("features_ok") or f.get("volume_mm3") is None:
        return {"class": "EXCLUDE", "confidence": 0.90, "reason": "no solid geometry"}

    # 1. Known catalogue products are confidently bought-out (name-based).
    cp = match_catalogue_product(part_name)
    if cp:
        return {"class": cp[0], "confidence": 0.95, "reason": f"catalogue:{cp[1]}"}

    # 2. Flat-walled (uniform thin gauge ~ smallest bbox dim) -> PLATE, checked
    #    FIRST. A flat plate is a plate regardless of curved cut edges (which can
    #    trip the bend detector) or small size (which can trip the artifact gate).
    #    Sections / formed / bent / artifacts all have a low t_eff/dim_thin.
    tthin = f.get("t_eff_thin_ratio")
    if tthin is not None and tthin >= PLATE_TTHIN:
        return {"class": "PLATE", "confidence": 0.90, "reason": "flat plate"}

    # 3. Tiny / degenerate (and not flat) -> artifact.
    vol = f.get("volume_mm3")
    if vol is not None and vol < EXCLUDE_VOL_MM3:
        return {"class": "EXCLUDE", "confidence": 0.90, "reason": "tiny/degenerate"}

    # 4. Closed hollow cross-section -> box/tube section. Only trust it when the
    #    section library actually matched (rule 7's verdict). An unmatched hollow
    #    means the OBB / cross-section doesn't correspond to an available tube
    #    size (handrail posts and other non-standard tube), so keep the SECTION
    #    suggestion but below the 0.5 auto-apply threshold for review — mirrors
    #    the unmatched "flanged" path below.
    holes = f.get("n_holes")
    if holes is not None and holes >= 1:
        if rule_type == "section":
            return {"class": "SECTION", "confidence": 0.90, "reason": "hollow box"}
        return {"class": "SECTION", "confidence": 0.45, "reason": "hollow, no library match"}

    # 4b. Headed solid of revolution -> a fastener, which is always purchased.
    #     A bolt's hex head is thicker than its shank, so rule 8 below reads the
    #     head as a "flange" and calls the bolt a SECTION (M20s landed in the CNC
    #     BOM as B155 qty 245).  Geometry is the only signal available: the
    #     name-based catalogue match can't help on models exported without part
    #     names.  Checked after rule 4 so hollow round tube (handrail standards
    #     are axisymmetric too) is already gone, and before the bend rules so an
    #     incidental head chamfer can't divert a bolt into FORMED_PLATE.  A plain
    #     round bar has no head, so its thk ratio stays ~1 and it falls through.
    thk_ratio = f.get("thk_max_over_teff")
    i21 = f.get("inertia_ratio_21")
    elong = f.get("elongation")
    d_long = f.get("dim_long")
    d_mid = f.get("dim_mid")
    if (d_long is not None and d_long <= FASTENER_MAX_DIM_LONG
            and d_mid is not None and d_mid <= FASTENER_MAX_DIM_MID
            and i21 is not None and abs(i21 - 1.0) <= FASTENER_I21_TOL
            and elong is not None and elong <= FASTENER_MAX_ELONGATION
            and thk_ratio is not None and thk_ratio >= SECTION_THK_RATIO):
        return {"class": "BOUGHT_OUT", "confidence": 0.85,
                "reason": "fastener (headed, axisymmetric)"}

    # 5. Many convex bends -> curved tube (bent section).
    nb = f.get("n_convex_bends")
    if nb is not None and nb >= BENT_MIN_BENDS:
        return {"class": "BENT_SECTION", "confidence": 0.85, "reason": "many bends"}

    # 6. A convex bend (R>=gauge) -> formed plate.
    if nb is not None and nb >= FORMED_MIN_BENDS:
        return {"class": "FORMED_PLATE", "confidence": 0.80, "reason": "convex bend"}

    # 7. The existing pipeline matched a standard section profile.
    if rule_type == "section":
        return {"class": "SECTION", "confidence": 0.85, "reason": "library match"}

    # 8. Distinct flanges (web<<flange) -> looks like an open rolled section, but
    #    the library did NOT match (rule 7 above didn't fire), so the OBB /
    #    cross-section doesn't correspond to an available standard profile. Keep
    #    the SECTION *suggestion* but below the 0.5 auto-apply threshold, so it
    #    surfaces for review instead of auto-committing to CNC. This is where
    #    handrail standards and other non-standard "sections" (frequently
    #    bought-out) land — geometry alone can't tell procured from fabricated.
    if thk_ratio is not None and thk_ratio >= SECTION_THK_RATIO:
        return {"class": "SECTION", "confidence": 0.45, "reason": "flanged, no library match"}

    # 9. Uncertain catch-all: thin uniform open wall — most likely formed, but
    #    could be an unmatched angle. Low confidence -> flag for review.
    return {"class": "FORMED_PLATE", "confidence": 0.40, "reason": "uncertain"}
