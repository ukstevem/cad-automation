"""Unit tests for BOM classification-key resolution.

These cover the bug where solid-level classification keys ("<ref>:s<n>") of
multi-solid parts were not resolved to their parent ref by the server-side BOM
builders, fragmenting one part into N unnamed "Unknown" rows in the export
(cad-automation-stage2-s10).  No OCP/CadQuery needed — pure dict logic.
"""
from app.services.bom_manifest import (
    assign_bom_items,
    pin_bom_items,
    resolve_classification_ref,
)


def test_resolve_instance_id_via_tree_map():
    node_to_ref = {"0:1:1:3:1": "0:1:1:4"}
    assert resolve_classification_ref("0:1:1:3:1", node_to_ref) == "0:1:1:4"


def test_resolve_unknown_instance_falls_back_to_self():
    assert resolve_classification_ref("0:1:1:99", {}) == "0:1:1:99"


def test_resolve_solid_key_collapses_to_parent_ref():
    # "<ref>:s<n>" keys are never in node_to_ref (the tree leaf is the whole
    # multi-solid part); stripping the suffix yields the parent ref.
    assert resolve_classification_ref("0:1:1:2:s0", {}) == "0:1:1:2"
    assert resolve_classification_ref("0:1:1:2:s33", {}) == "0:1:1:2"


def test_resolve_instance_then_solid_strips_only_solid_suffix():
    # An instance:solid id strips just the ":s<n>" tail, leaving the instance id
    # to resolve through the map (here it isn't mapped, so it returns itself).
    assert resolve_classification_ref("0:1:1:4:1:s2", {}) == "0:1:1:4:1"


def _multi_solid_cache():
    """Minimal cache: one multi-solid part classified per-solid as postprocess."""
    return {
        "analysis": {
            "assembly_tree": [
                {
                    "id": "0:1",
                    "node_type": "assembly",
                    "children": [
                        {
                            "id": "0:1:1:2",
                            "ref_id": "0:1:1:2",
                            "node_type": "part_multi_solid",
                        }
                    ],
                }
            ]
        },
        "cnc_analysis": {"0:1:1:2": {"type": "multi_solid", "solids": [{}, {}, {}]}},
        "project_state": {
            "classifications": {
                "0:1:1:2:s0": "postprocess",
                "0:1:1:2:s1": "postprocess",
                "0:1:1:2:s2": "postprocess",
            }
        },
    }


def test_assign_bom_items_collapses_solid_keys_to_one_item():
    out = assign_bom_items(_multi_solid_cache())
    # All three solid classifications collapse onto the single parent ref —
    # one BOM Item, not three fragments.
    assert list(out.keys()) == ["0:1:1:2"]
    assert out["0:1:1:2"]["action"] == "postprocess"
    assert out["0:1:1:2"]["bom_item"] == "B001"


# ---------------------------------------------------------------------------
# BOM item pinning (cad-automation-stage2-jcl)
# ---------------------------------------------------------------------------


def _parts_cache(classifications: dict) -> dict:
    """Three single-solid parts under one assembly, classified as given."""
    return {
        "analysis": {
            "assembly_tree": [
                {
                    "id": "0:1",
                    "node_type": "assembly",
                    "children": [
                        {"id": f"0:1:1:{n}", "ref_id": f"0:1:1:{n}",
                         "node_type": "part_single_solid"}
                        for n in (1, 2, 3)
                    ],
                }
            ]
        },
        "project_state": {"classifications": classifications},
    }


def test_bom_items_are_stable_when_an_earlier_part_drops_out():
    """A part falling below the auto-apply threshold must not renumber the rest.

    This is the whole point of pinning: B-numbers are allocated from the
    classifications' insertion order and also name the generated B###.nc1
    files, so an unpinned re-classify silently repoints an already-issued BOM.
    """
    cache = _parts_cache({"0:1:1:1": "postprocess",
                          "0:1:1:2": "postprocess",
                          "0:1:1:3": "postprocess"})
    pin_bom_items(cache)
    before = {k: v["bom_item"] for k, v in assign_bom_items(cache).items()}
    assert before == {"0:1:1:1": "B001", "0:1:1:2": "B002", "0:1:1:3": "B003"}

    # The first part drops out (reclassified below the confidence threshold).
    del cache["project_state"]["classifications"]["0:1:1:1"]
    after = {k: v["bom_item"] for k, v in assign_bom_items(cache).items()}

    assert after["0:1:1:2"] == "B002"  # unpinned this would slide to B001
    assert after["0:1:1:3"] == "B003"


def test_bom_item_survives_an_action_change():
    cache = _parts_cache({"0:1:1:1": "postprocess", "0:1:1:2": "postprocess"})
    pin_bom_items(cache)
    cache["project_state"]["classifications"]["0:1:1:1"] = "bought-out"
    out = assign_bom_items(cache)
    assert out["0:1:1:1"]["bom_item"] == "B001"
    assert out["0:1:1:1"]["action"] == "bought-out"


def test_new_parts_never_reuse_a_pinned_number():
    cache = _parts_cache({"0:1:1:2": "postprocess", "0:1:1:3": "postprocess"})
    pin_bom_items(cache)  # -> 0:1:1:2=B001, 0:1:1:3=B002

    # A part classified later must draw a free number, not collide with a pin
    # that a not-yet-walked part still owns.
    cache["project_state"]["classifications"] = {
        "0:1:1:1": "postprocess",   # new, walked first
        "0:1:1:2": "postprocess",
        "0:1:1:3": "postprocess",
    }
    out = assign_bom_items(cache)
    assert out["0:1:1:2"]["bom_item"] == "B001"
    assert out["0:1:1:3"]["bom_item"] == "B002"
    assert out["0:1:1:1"]["bom_item"] == "B003"
    assert len({v["bom_item"] for v in out.values()}) == 3


def test_pin_bom_items_records_assignments_in_the_cache():
    cache = _parts_cache({"0:1:1:1": "postprocess"})
    assert "bom_items" not in cache
    pin_bom_items(cache)
    assert cache["bom_items"]["assignments"]["0:1:1:1"] == "B001"
    assert cache["bom_items"]["updated_at"]
