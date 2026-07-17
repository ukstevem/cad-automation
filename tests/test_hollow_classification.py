"""Hollow-section library matching (cad-automation-stage2-b5b).

The measured values here are the real ones from run 761b26b1 (C24025-Model),
whose 64 tubes are what motivated the CSA-primary strategy: a raked handrail is
a smooth cylinder with no straight edges for align_by_longest_straight_edge to
grab, so the axis skews and the OBB cross-section inflates (measured ODs smeared
33.7 -> 41.9, none of them standard sizes) while the CSA stayed sound.

No OCP needed — this drives classify_profile against the real library file.
"""
import json
from pathlib import Path

import pytest

from app.pipeline.classification import classify_profile

LIB = str(Path(__file__).parent.parent / "app" / "pipeline" / "data"
          / "Shape_classifier_info.json")


def _cs(H, W, area, length=2000.0):
    return {"span_web": H, "span_flange": W, "area": area, "length": length}


# --- the library itself ----------------------------------------------------


def test_library_has_hollow_and_open_halves():
    lib = json.loads(Path(LIB).read_text(encoding="utf-8"))
    hollow = {c for c, ents in lib.items()
              if any(e.get("hollow") for e in ents.values())}
    open_ = {c for c, ents in lib.items()
             if any(not e.get("hollow") for e in ents.values())}
    assert hollow == {"CHS", "SHS", "RHS"}
    assert {"UB", "UC", "PFC", "EA", "UEA"} <= open_
    assert not (hollow & open_)          # no category is both


def test_hollow_entries_carry_a_hollow_profile_code():
    lib = json.loads(Path(LIB).read_text(encoding="utf-8"))
    for cat in ("CHS", "SHS", "RHS"):
        for name, e in lib[cat].items():
            assert e["hollow"] is True
            assert e["code_profile"] in ("RO", "RU"), f"{cat} {name}"


# --- the two halves must never cross --------------------------------------


def test_open_match_never_returns_a_hollow_profile():
    # A UB cross-section, matched as open — must come back a UB, not an RHS.
    m = classify_profile(_cs(203.2, 133.2, 3200.0), LIB, hollow=False)
    assert m is not None
    assert m["Category"] == "UB"


def test_hollow_match_never_returns_an_open_profile():
    # The exact case that cut the wrong NC1: a 33.7 CHS whose fill ratio sits in
    # the angle band. Asked for hollow, it must not come back EA 35x35x4.
    m = classify_profile(_cs(33.7, 33.7, 306.6), LIB, hollow=True)
    assert m is not None
    assert m["Category"] == "CHS"
    assert m["Designation"] == "33.7x3.2"


# --- CSA-primary matching --------------------------------------------------


@pytest.mark.parametrize("ref,od,csa,want", [
    # ref          OBB OD   CSA     expected — OD is inflated, CSA is not
    ("0:1:1:197",  33.7,   304.6,  "33.7x3.2"),   # box measured true
    ("0:1:1:184",  34.9,   290.3,  "33.7x3"),     # box 1.2mm over
    ("0:1:1:196",  36.7,   302.8,  "33.7x3.2"),   # box 3.0mm over
])
def test_raked_tube_resolves_despite_an_inflated_bounding_box(ref, od, csa, want):
    m = classify_profile(_cs(od, od, csa), LIB, hollow=True)
    assert m is not None, ref
    assert m["Category"] == "CHS"
    assert m["Designation"] == want, f"{ref}: got {m['Designation']}"


def test_csa_pins_the_diameter_but_the_wall_can_be_ambiguous():
    """A known limit, not a bug — the diameter is firm, the wall is not.

    0:1:1:175 measures CSA 297.7, which sits almost exactly between CHS 33.7x3.0
    (289.3) and 33.7x3.2 (306.6) — 2.903% vs 2.904%. Solving CSA = pi*t*(D-t)
    for it gives a wall of 3.1mm, a size that does not exist, because a skewed
    axis over-reads the length and so under-reads CSA = volume/length.

    The diameter is never in doubt, so the part is always the right *tube*; only
    which catalogue wall it is called can flip. Worth knowing before trusting a
    wall thickness straight off a cutting list.
    """
    m = classify_profile(_cs(33.7, 33.7, 297.7), LIB, hollow=True)
    assert m["Category"] == "CHS"
    assert m["Designation"].startswith("33.7x")      # diameter: certain
    assert m["Designation"] in ("33.7x3", "33.7x3.2")  # wall: a coin toss


def test_bounding_box_vetoes_the_csa_collision():
    """42.4x2.6 and 33.7x3.2 are close in CSA; only the box separates them.

    This is the whole reason the measured OBB is still used at all: area alone
    cannot tell a fat thin-walled tube from a slim thick-walled one.
    """
    lib = json.loads(Path(LIB).read_text(encoding="utf-8"))
    a_slim = lib["CHS"]["33.7x3.2"]["csa"]      # 306.6
    a_fat = lib["CHS"]["42.4x2.6"]["csa"]       # ~325
    assert abs(a_slim - a_fat) / a_fat < 0.10   # within the CSA tolerance

    # Measured in a 34.9 box: the 42.4 cannot physically fit, so the slim one wins.
    m = classify_profile(_cs(34.9, 34.9, 315.0), LIB, hollow=True)
    assert m["Designation"] == "33.7x3.2"

    # Measured in a 43 box, same CSA: now the 42.4 fits and is the closer fit.
    m = classify_profile(_cs(43.0, 43.0, 320.0), LIB, hollow=True)
    assert m["Designation"] == "42.4x2.6"


def test_a_section_larger_than_its_own_bounding_box_is_rejected():
    # 114.3x5 CSA (1717) presented in a box far too small to hold it.
    m = classify_profile(_cs(60.0, 60.0, 1717.0), LIB, hollow=True)
    assert m is None or m["JSON"]["height"] <= 60.0 + 1.5


def test_absurdly_oversized_box_finds_nothing():
    # The bent handrail return in run 761b26b1: OBB 1708 wide, CSA 16222.
    # Nothing sane fits that description — it must fall through to review.
    m = classify_profile(_cs(1708.1, 1708.1, 16222.5), LIB, hollow=True)
    assert m is None


def test_poor_hollow_match_is_rejected_by_the_score_ceiling():
    # A UB's cross-section wrongly offered as hollow: the best hollow candidate
    # is RHS 150x100x6.3 but it scores ~95 (8.5% area off, 86mm gap), so the
    # ceiling must throw it out rather than return a plausible-looking wrong one.
    m = classify_profile(_cs(203.2, 133.2, 3200.0), LIB, hollow=True)
    assert m is None


def test_square_hollow_matches_shs():
    lib = json.loads(Path(LIB).read_text(encoding="utf-8"))
    csa = lib["SHS"]["100x100x5"]["csa"]
    m = classify_profile(_cs(100.0, 100.0, csa), LIB, hollow=True)
    assert m["Category"] == "SHS"
    assert m["Designation"] == "100x100x5"


def test_rectangular_hollow_matches_rhs():
    lib = json.loads(Path(LIB).read_text(encoding="utf-8"))
    csa = lib["RHS"]["150x100x6.3"]["csa"]
    m = classify_profile(_cs(150.0, 100.0, csa), LIB, hollow=True)
    assert m["Category"] == "RHS"
    assert m["Designation"] == "150x100x6.3"


def test_rhs_matches_with_height_and_width_swapped():
    # The OBB has no idea which way up the section is.
    lib = json.loads(Path(LIB).read_text(encoding="utf-8"))
    csa = lib["RHS"]["150x100x6.3"]["csa"]
    m = classify_profile(_cs(100.0, 150.0, csa), LIB, hollow=True)
    assert m["Designation"] == "150x100x6.3"
