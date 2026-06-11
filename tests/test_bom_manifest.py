"""Unit tests for BOM classification-key resolution.

These cover the bug where solid-level classification keys ("<ref>:s<n>") of
multi-solid parts were not resolved to their parent ref by the server-side BOM
builders, fragmenting one part into N unnamed "Unknown" rows in the export
(cad-automation-stage2-s10).  No OCP/CadQuery needed — pure dict logic.
"""
from app.services.bom_manifest import (
    assign_bom_items,
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
