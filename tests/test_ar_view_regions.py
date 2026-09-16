"""
Tests for ar_view.deviation_regions - finding the stretch of a part that departs from its model (bd h04).

The glued tower is the case these are drawn from: 2.0 mm at 40-50% of the length against 0.4 mm
either side. A fixed limit would not do, because what counts as normal depends on the article, the
camera distance and the model's fidelity, so a stretch is judged against the part's own typical
deviation - and only where enough samples were measured to mean anything.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ar_view as AV  # noqa: E402


def _bands(medians, n=500):
    return [{"from": 10 * k, "to": 10 * (k + 1), "testable": n, "n": n if m is not None else 0,
             "median_mm": m, "p90_mm": None if m is None else m * 3}
            for k, m in enumerate(medians)]


def test_the_glued_towers_bump_is_found_and_the_rest_is_not():
    """The real tenths measured on tower02, including the 50-60% one where only 39 samples fell and
    which is therefore left out of the baseline. The baseline has to be the quarter-best (0.70 mm),
    not the median of the tenths (1.00 mm) - against the median the threshold would sit above the
    very bump this exists to catch."""
    bands = _bands([1.00, 0.76, 0.45, 0.38, 1.98, 0.06, 1.27, 0.70, 1.18, 1.16])
    bands[5] = dict(bands[5], n=39, testable=39)
    regions, typical, threshold = AV.deviation_regions(bands)
    assert typical == 0.70 and threshold == 1.75
    assert regions == [(40, 50, 1.98)]        # and nothing else, though two tenths sit above 1.1 mm


def test_a_clear_defect_against_a_tight_part_is_flagged():
    bands = _bands([0.40, 0.38, 0.45, 0.38, 1.98, 0.50, 0.42, 0.40, 0.41, 0.44])
    regions, typical, threshold = AV.deviation_regions(bands)
    assert 0.40 <= typical <= 0.45
    assert regions == [(40, 50, 1.98)]


def test_neighbouring_bad_tenths_are_reported_as_one_stretch():
    bands = _bands([0.4, 0.4, 0.4, 0.4, 1.8, 2.4, 0.4, 0.4, 0.4, 0.4])
    regions, _typical, _threshold = AV.deviation_regions(bands)
    assert regions == [(40, 60, 2.4)]


def test_a_tenth_with_almost_no_samples_is_not_flagged():
    """Some tenths of an open frame hold almost nothing a camera can see - 39 samples in one tenth
    of the tower. A wild median from a handful of points is not a finding."""
    bands = _bands([0.4] * 10)
    bands[5] = dict(bands[5], median_mm=9.0, n=12)
    regions, _typical, _threshold = AV.deviation_regions(bands)
    assert regions == []


def test_nothing_is_flagged_below_the_floor_however_it_compares():
    """A part measured to 0.1 mm should not be flagged for a tenth at 0.3 mm."""
    bands = _bands([0.10, 0.10, 0.10, 0.10, 0.30, 0.10, 0.10, 0.10, 0.10, 0.10])
    regions, typical, threshold = AV.deviation_regions(bands)
    assert typical == 0.10 and threshold == 1.0
    assert regions == []


def test_a_part_with_nothing_measured_says_so_rather_than_guessing():
    regions, typical, threshold = AV.deviation_regions(_bands([None] * 10))
    assert regions == [] and typical is None and threshold is None
