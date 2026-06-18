"""Tests for user refined sub-class overrides in the BOM export.

Overrides are keyed by ref_id (single-solid) or "ref_id:sN" (a multi-solid
sub-solid) and overwrite refined_class on the request-local CNC results,
clearing the low-confidence flag (cad-automation-stage2-y03).
"""
from app.services.bom_excel import _apply_refined_overrides


def test_single_solid_override_sets_class_and_confidence():
    cnc = {"0:1:1:26": {"type": "plate", "refined_class": "FORMED_PLATE",
                        "refined_confidence": 0.4}}
    _apply_refined_overrides(cnc, {"0:1:1:26": "SECTION"})
    assert cnc["0:1:1:26"]["refined_class"] == "SECTION"
    assert cnc["0:1:1:26"]["refined_confidence"] == 1.0


def test_multi_solid_subsolid_override():
    cnc = {"Z": {"type": "multi_solid",
                 "solids": [{"refined_class": "PLATE"}, {"refined_class": "PLATE"}]}}
    _apply_refined_overrides(cnc, {"Z:s1": "TUBE"})
    assert cnc["Z"]["solids"][1]["refined_class"] == "TUBE"
    assert cnc["Z"]["solids"][1]["refined_confidence"] == 1.0
    assert cnc["Z"]["solids"][0]["refined_class"] == "PLATE"  # untouched


def test_bad_keys_are_ignored():
    cnc = {"Z": {"type": "multi_solid", "solids": [{"refined_class": "PLATE"}]}}
    # unknown ref, out-of-range solid idx, empty code — none should raise
    _apply_refined_overrides(cnc, {"nope": "PLATE", "Z:s9": "PLATE", "Z": ""})
    assert cnc["Z"]["solids"][0]["refined_class"] == "PLATE"


def test_empty_overrides_noop():
    cnc = {"A": {"refined_class": "PLATE"}}
    _apply_refined_overrides(cnc, {})
    _apply_refined_overrides(cnc, None)
    assert cnc["A"]["refined_class"] == "PLATE"
