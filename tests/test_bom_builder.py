"""Unit tests for BOM display-name resolution.

Covers the parent-name fallback for generic STEP leaf names (SOLID/COMPOUND):
a single-part assembly "Column Plate CP1 -> SOLID" should show as "Column Plate
CP1" in the BOM, while an empty no-solid COMPOUND artifact keeps its generic
name so it stays distinct (cad-automation-stage2-22j).
"""
from app.services.bom_builder import resolve_display_name


def test_generic_leaf_uses_parent_name():
    assert resolve_display_name("SOLID", "Column Plate CP1") == "Column Plate CP1"
    assert resolve_display_name("COMPOUND", "Hanging Plate") == "Hanging Plate"
    assert resolve_display_name("solid", "Beam") == "Beam"  # case-insensitive


def test_real_name_is_kept():
    assert resolve_display_name("Splice plate", "TOWER A") == "Splice plate"


def test_no_parent_keeps_leaf_name():
    assert resolve_display_name("SOLID", None) == "SOLID"
    assert resolve_display_name("SOLID", "") == "SOLID"
    assert resolve_display_name("SOLID", "   ") == "SOLID"


def test_blank_leaf_recovers_parent():
    assert resolve_display_name("", "Bridge") == "Bridge"
    assert resolve_display_name(None, "Bridge") == "Bridge"


def test_blank_everything_is_empty_string():
    assert resolve_display_name(None, None) == ""
