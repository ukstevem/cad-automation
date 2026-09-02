"""
Unit tests for image edge extraction (app/services/image_edges.py) — the step that turns a
photo into the ``edge_pixels`` the multi-view fit consumes.

Focus is on the things that fail *silently* in the field: a mask that erases the subject, a
resolution/shape mismatch, and thresholds that collapse on a flat frame.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.services import charuco, image_edges as IE


def _square_image(w=400, h=300, rect=(150, 100, 100, 80)):
    """Black frame with one white rectangle — a known, countable set of edges."""
    img = np.zeros((h, w), np.uint8)
    x, y, rw, rh = rect
    img[y:y + rh, x:x + rw] = 255
    return img


def test_detect_edge_pixels_finds_rectangle_outline():
    pts = IE.detect_edge_pixels(_square_image(), blur_ksize=0, low=50, high=150)
    assert len(pts) > 100
    # Every edge pixel should sit on the rectangle boundary (within a pixel or two of it).
    xs, ys = pts[:, 0], pts[:, 1]
    assert xs.min() >= 145 and xs.max() <= 255
    assert ys.min() >= 95 and ys.max() <= 185


def test_returns_xy_not_rowcol():
    """Nx2 must be (x, y) — projectPoints/KD-tree both expect image-x first."""
    img = np.zeros((300, 400), np.uint8)
    img[:, 200] = 255                      # a single vertical line at x=200
    pts = IE.detect_edge_pixels(img, blur_ksize=0, low=50, high=150)
    assert np.allclose(pts[:, 0].mean(), 200, atol=2.0)
    assert pts[:, 1].max() > 200           # y spans the tall axis, so it exceeds 200


def test_exclude_mask_drops_masked_pixels():
    img = _square_image()
    everything = IE.detect_edge_pixels(img, blur_ksize=0, low=50, high=150)
    mask = np.zeros(img.shape, np.uint8)
    mask[:, :200] = 255                    # blank the left half
    left_gone = IE.detect_edge_pixels(img, blur_ksize=0, low=50, high=150, exclude_mask=mask)
    assert len(left_gone) < len(everything)
    assert left_gone[:, 0].min() >= 200


def test_exclude_mask_shape_mismatch_raises():
    with pytest.raises(IE.EdgeError, match="exclude_mask shape"):
        IE.detect_edge_pixels(_square_image(), exclude_mask=np.zeros((10, 10), np.uint8))


def test_no_edges_raises_rather_than_returning_empty():
    """An empty array would blow up later inside the KD-tree; fail here with a usable message."""
    with pytest.raises(IE.EdgeError, match="no edge pixels"):
        IE.detect_edge_pixels(np.zeros((100, 100), np.uint8), blur_ksize=0, low=50, high=150)


def test_roi_restricts_search():
    img = _square_image()
    pts = IE.detect_edge_pixels(img, blur_ksize=0, low=50, high=150, roi=(0, 0, 200, 300))
    assert pts[:, 0].max() < 200


def test_max_points_caps_and_is_deterministic():
    img = _square_image()
    a = IE.detect_edge_pixels(img, blur_ksize=0, low=50, high=150, max_points=50, seed=7)
    b = IE.detect_edge_pixels(img, blur_ksize=0, low=50, high=150, max_points=50, seed=7)
    assert len(a) == 50 and np.array_equal(a, b)


def test_auto_thresholds_survive_a_flat_frame():
    """A blown or black frame gives median 0/255; the fallback must still be a valid pair."""
    for value in (0, 255):
        low, high = IE.auto_canny_thresholds(np.full((50, 50), value, np.uint8))
        assert 0 <= low < high <= 255


def test_board_grid_mask_covers_the_board_and_not_the_whole_frame():
    board = charuco.build_board_from_config(
        {"squares_x": 5, "squares_y": 7, "square_mm": 30.0, "marker_mm": 22.0,
         "dictionary": "DICT_5X5_100"}
    )
    K = np.array([[1200.0, 0, 640.0], [0, 1200.0, 480.0], [0, 0, 1.0]])
    dist = np.zeros((5, 1))
    # Board face-on, 600 mm away, roughly centred.
    rvec = np.zeros((3, 1))
    tvec = np.array([[-75.0], [-105.0], [600.0]])
    mask = IE.board_grid_mask(board, rvec, tvec, K, dist, (960, 1280), band_px=7)

    frac = np.count_nonzero(mask) / mask.size
    assert 0.01 < frac < 0.5, f"grid mask covers {frac:.1%} — should be bands, not a blanket"
    # The lattice must reach the board's outer edge, not stop at the interior corners:
    # project the board's far corner and check it is inside the masked region.
    w_mm, h_mm = 150.0, 210.0
    proj, _ = cv2.projectPoints(np.array([[[w_mm, h_mm, 0.0]]]), rvec, tvec, K, dist)
    x, y = np.round(proj.reshape(2)).astype(int)
    assert mask[y, x] > 0, "board outer boundary was not masked (lattice not extended)"


def test_marker_hull_mask_is_empty_without_markers():
    assert np.count_nonzero(IE.marker_hull_mask([], (100, 100))) == 0
