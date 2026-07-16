"""Unit tests for the refined decision tree (app/pipeline/part_classifier.py).

Feature values are the real measured ones from run 761b26b1 (C24025-Model), so
these pin the two misroutings that run exposed: M20 bolts arriving in the CNC
BOM as SECTION (B155, qty 245), and CHS handrail standards auto-committed to
CNC at 0.90 off the back of a bogus EA 35x35x4 library match.

Pure dict logic — no OCP needed.
"""
from app.pipeline.part_classifier import classify_part


def _feats(**kw):
    """Measured-feature dict with plausible non-triggering defaults."""
    base = {
        "features_ok": True,
        "volume_mm3": 500000.0,
        "t_eff_thin_ratio": 0.2,     # not a flat plate
        "n_holes": 0,
        "n_convex_bends": 0,
        "thk_max_over_teff": 1.0,    # uniform wall
        "inertia_ratio_21": 0.4,     # not axisymmetric
        "elongation": 20.0,
    }
    base.update(kw)
    return base


# --- fasteners (cad-automation-stage2-bsk) ---------------------------------
# 0:1:1:2792 — M20 bolt: 68 long, 37x37 hex head, shank dia 19.56 (CSA 312).


def test_bolt_is_bought_out_not_section():
    r = classify_part(_feats(volume_mm3=39368.3, t_eff_thin_ratio=0.2251,
                             thk_max_over_teff=2.412, inertia_ratio_21=1.0,
                             elongation=1.8378), None, None)
    assert r["class"] == "BOUGHT_OUT"
    assert r["confidence"] >= 0.5      # auto-applies rather than nagging


def test_bolt_rule_survives_incidental_bend_faces():
    # A head chamfer / thread runout must not divert a bolt to FORMED_PLATE.
    r = classify_part(_feats(volume_mm3=39368.3, thk_max_over_teff=2.4,
                             inertia_ratio_21=1.0, elongation=1.84,
                             n_convex_bends=2), None, None)
    assert r["class"] == "BOUGHT_OUT"


def test_round_bar_is_not_a_fastener():
    # Axisymmetric and stubby, but no head (uniform thickness) — must not be
    # swept up as a bolt.
    r = classify_part(_feats(inertia_ratio_21=1.0, elongation=3.0,
                             thk_max_over_teff=1.05), None, None)
    assert r["class"] != "BOUGHT_OUT"


def test_long_headed_shaft_is_not_a_fastener():
    # Elongation guard: a long headed thing is not a bolt.
    r = classify_part(_feats(inertia_ratio_21=1.0, elongation=40.0,
                             thk_max_over_teff=2.4), None, None)
    assert r["class"] != "BOUGHT_OUT"


def test_flanged_open_section_is_not_a_fastener():
    # A UB has a high thickness ratio too, but is nowhere near axisymmetric.
    r = classify_part(_feats(thk_max_over_teff=2.4, inertia_ratio_21=0.3,
                             elongation=15.0), "section", None)
    assert r["class"] == "SECTION"


# --- hollow sections (cad-automation-stage2-5wg) ---------------------------
# 0:1:1:197 — 33.7 CHS handrail standard, 2586.8 long, 3.69 wall.


def test_unmatched_hollow_is_surfaced_for_review_not_auto_cnc():
    # rule_type=None is what the hollowness gate now produces: the library holds
    # no CHS/SHS/RHS, so a tube must never carry a library-match verdict.
    r = classify_part(_feats(volume_mm3=787998.4, t_eff_thin_ratio=0.0948,
                             n_holes=1, thk_max_over_teff=1.155,
                             inertia_ratio_21=1.0, elongation=76.7596),
                      None, None)
    assert r["class"] == "SECTION"
    assert r["confidence"] < 0.5       # below auto-apply -> human review
    assert r["reason"] == "hollow, no library match"


def test_handrail_tube_is_not_caught_by_the_fastener_rule():
    # A round tube is axisymmetric like a bolt; only rule 4 firing first (and
    # the elongation guard) keeps it out of BOUGHT_OUT.
    r = classify_part(_feats(n_holes=1, inertia_ratio_21=1.0,
                             elongation=76.76, thk_max_over_teff=1.155),
                      None, None)
    assert r["class"] == "SECTION"


def test_matched_hollow_still_auto_applies():
    # A genuine library-matched hollow section keeps its confident routing.
    r = classify_part(_feats(n_holes=1), "section", None)
    assert r["class"] == "SECTION"
    assert r["confidence"] >= 0.5
    assert r["reason"] == "hollow box"


# --- guards on the rules the fixes sit between ----------------------------


def test_flat_plate_still_wins_first():
    r = classify_part(_feats(t_eff_thin_ratio=0.9, thk_max_over_teff=2.4,
                             inertia_ratio_21=1.0, elongation=2.0), None, None)
    assert r["class"] == "PLATE"


def test_tiny_solid_still_excluded():
    r = classify_part(_feats(volume_mm3=500.0), None, None)
    assert r["class"] == "EXCLUDE"


def test_no_geometry_excluded():
    assert classify_part({"features_ok": False}, None, None)["class"] == "EXCLUDE"
