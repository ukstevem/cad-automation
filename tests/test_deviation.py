"""
Tests for tools/deviation.py - showing an article as built without editing its drawing (bd 0sb).

Synthetic bars made of two halves, like the glued tower. These pin what would make an as-built page
quietly wrong: the untouched half moving, the turn pivoting off the section centre, a weld keeping
the face it had before its half turned, and the drawing's sidecar being modified in place.
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import deviation as DV  # noqa: E402
import weld_faces as WF  # noqa: E402

_TRIS = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def _box(lo, hi):
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                 float)
    return np.array([[v[a], v[b], v[c]] for a, b, c in _TRIS])


def _halves(*extra):
    """A 400 x 100 x 100 bar in two halves either side of x = 0, with a small gap so no triangle sits
    on the cut."""
    return np.concatenate([_box((-200, -50, -50), (-2, 50, 50)), _box((2, -50, -50), (200, 50, 50))]
                          + list(extra))


def _dev(angle, half="-L", split=0.0, extent=(400, 100, 100)):
    return {"schema": DV.SCHEMA, "kind": "half_roll", "turned_half": half, "split_mm": split,
            "angle_deg": angle, "description": "test half turned",
            "article": {"mesh": "bar.stl", "frame_extent_mm": list(extent)}}


def test_the_untouched_half_does_not_move():
    mesh = _halves()
    f = WF.article_frame(mesh)
    out, mask = DV.as_built_mesh(mesh, f, _dev(90))
    assert mask.sum() == 12
    assert np.array_equal(out[~mask], mesh[~mask])
    assert not np.allclose(out[mask], mesh[mask])


def _area(tris):
    return 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum()


def test_faces_running_across_the_cut_are_split_at_it_not_turned_whole():
    """The tower's rails are faces running the full length. Assigned whole by centroid, each half
    carried faces of the other with it; they must be cut at the joint instead."""
    bar = _box((-200, -50, -50), (200, 50, 50))
    f = WF.article_frame(bar)
    out, mask = DV.as_built_mesh(bar, f, _dev(90))
    s = (out - f["centre"]) @ f["axes"][:, 0]
    assert np.all(s[mask] <= 1e-9) and np.all(s[~mask] >= -1e-9)
    assert abs(_area(out) - _area(bar)) < 1e-6
    untouched = out[~mask].reshape(-1, 3)
    assert np.allclose(untouched.min(axis=0), [0, -50, -50]) and np.allclose(untouched.max(axis=0), [200, 50, 50])


def test_the_turn_is_about_the_section_centre_not_the_area_centroid():
    """Material inside the section, off to one corner, drags the area centroid off the section
    centre. A half glued on turned still has its rails in line, so its envelope must not move."""
    inner = [_box((-180, 10, 10), (-20, 40, 40)), _box((20, 10, 10), (180, 40, 40))]
    mesh = _halves(*inner)
    f = WF.article_frame(mesh)
    assert np.linalg.norm(f["centre"][1:]) > 1.0
    out, mask = DV.as_built_mesh(mesh, f, _dev(180))
    after = out[mask].reshape(-1, 3)
    assert np.allclose(after[:, 1:].min(axis=0), [-50, -50])
    assert np.allclose(after[:, 1:].max(axis=0), [50, 50])


def test_a_quarter_turn_moves_long_faces_round_and_leaves_the_ends():
    f = WF.article_frame(_halves())
    # in a right-handed L, W, D frame a quarter turn about +L takes W to D
    assert DV.remap_faces(["+W"], f, _dev(90)) == ["+D"]
    assert DV.remap_faces(["+W"], f, _dev(270)) == ["-D"]
    assert DV.remap_faces(["-L", "+W"], f, _dev(180)) == ["-L", "-W"]


def test_remapped_faces_agree_with_faces_found_on_the_turned_weld():
    f = WF.article_frame(_halves())
    dev = _dev(270)
    corner = f["centre"] + f["axes"] @ np.array([-100.0, f["hi"][1] - 3, f["hi"][2] - 3])
    R, pivot = DV.turn(f, dev)
    turned = (corner - pivot) @ R.T + pivot
    before = WF.faces_of([corner], f, band=10)
    assert len(before) == 2
    assert DV.remap_faces(before, f, dev) == WF.faces_of([turned], f, band=10)


def test_welds_turn_with_their_half_and_the_sidecar_is_left_as_drawn():
    f = WF.article_frame(_halves())
    scale = 0.5

    def weld(name, x):
        p = f["centre"] + f["axes"] @ np.array([x, f["hi"][1] - 3, 0.0])
        q = p + f["axes"][:, 0] * 10
        return {"Name": name, "PSS_WeldGeometry": {"ArticleFaces": ["+W"]},
                "Pset_FastenerWeld": {}, "Representation": {"segments": [[list(p / scale), list(q / scale)]]}}

    welds = [weld("T-W001", -100.0), weld("T-W002", 100.0)]
    original = copy.deepcopy(welds)
    out, turned = DV.as_built_welds(welds, f, _dev(90), scale)
    assert welds == original
    assert turned == ["T-W001"]
    assert out[1] == original[1]
    assert out[0]["PSS_WeldGeometry"]["ArticleFaces"] == ["+D"]
    moved = np.asarray(out[0]["Representation"]["segments"][0]) * scale
    assert WF.faces_of(moved, f, band=10) == ["+D"]


def test_a_deviation_measured_on_another_article_is_refused():
    f = WF.article_frame(_halves())
    DV.check_article(_dev(90), f)
    with pytest.raises(ValueError):
        DV.check_article(_dev(90, extent=(432.6, 116, 116)), f)


def test_records_of_an_unknown_kind_or_a_part_turn_are_refused(tmp_path):
    for bad in (dict(_dev(90), kind="bent"), _dev(45), dict(_dev(90), turned_half="+W")):
        p = tmp_path / "deviation.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            DV.load(str(p))
