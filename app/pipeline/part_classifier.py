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


def classify_part(features: Optional[Dict[str, Any]],
                  rule_type: Optional[str],
                  part_name: Optional[str] = None) -> Dict[str, Any]:
    """Return ``{"class", "confidence", "reason"}`` for one solid.

    ``features``   – an ``extract_solid_features(..., section=True)`` dict.
    ``rule_type``  – the existing classifier's verdict ("section"/"plate"/...).
    ``part_name``  – display name, for catalogue-product (BO) matching.
    """
    f = features or {}

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

    # 4. Closed hollow cross-section -> box section.
    holes = f.get("n_holes")
    if holes is not None and holes >= 1:
        return {"class": "SECTION", "confidence": 0.90, "reason": "hollow box"}

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

    # 8. Distinct flanges (web<<flange) -> open rolled section.
    thk = f.get("thk_max_over_teff")
    if thk is not None and thk >= SECTION_THK_RATIO:
        return {"class": "SECTION", "confidence": 0.70, "reason": "flanged profile"}

    # 9. Uncertain catch-all: thin uniform open wall — most likely formed, but
    #    could be an unmatched angle. Low confidence -> flag for review.
    return {"class": "FORMED_PLATE", "confidence": 0.40, "reason": "uncertain"}
