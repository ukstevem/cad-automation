"""
Image edge extraction — turns a photo into the ``edge_pixels`` that the multi-view fit consumes.

This is the missing link between a camera and ``app/services/multiview.fit_object_pose``: that
function takes the detected 2D edge pixels of each view as an **input**, and until now nothing
produced them from a real image (the solver was proven on synthetic projections only).

Two things here are easy to get wrong and both fail silently:

1. **Distortion.** ``fit_object_pose`` projects CAD points *through* ``K`` and ``dist``, i.e. into
   the RAW distorted image. So edge pixels must come from the **raw** frame. Do not undistort the
   image first — that matches undistorted observations against distorted predictions.
2. **The board.** A ChArUco board is a dense lattice of maximal-contrast edges, and in this cell
   the part sits *on* it. Fed to the fit unmasked, the KD-tree offers the CAD edges a rich field
   of wrong things to snap to. Because the board pose is known once solved, we can mask the board
   *pattern* precisely — thin bands along its projected grid lines — rather than blanking the whole
   board rectangle, which would erase the object standing on it too.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

try:  # import guard — keep module importable if a wheel is missing
    import cv2

    _CV2_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - import guard
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = str(exc)


class EdgeError(Exception):
    """Raised when edges cannot be extracted from an image."""


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise EdgeError(f"OpenCV is not available: {_CV2_IMPORT_ERROR}")


def to_gray(image: np.ndarray) -> np.ndarray:
    """Accept BGR or already-grey; return single-channel uint8."""
    _require_cv2()
    img = np.asarray(image)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim != 2:
        raise EdgeError(f"expected a 2D or 3D image array, got shape {img.shape}")
    return img


def auto_canny_thresholds(gray: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    """
    Median-based Canny thresholds — adapts to the exposure of the frame rather than
    needing a hand-tuned pair per lighting setup.
    """
    med = float(np.median(gray))
    low = int(max(0, (1.0 - sigma) * med))
    high = int(min(255, (1.0 + sigma) * med))
    if high <= low:                     # flat or blown frame — fall back to a usable spread
        low, high = 50, 150
    return low, high


def board_grid_mask(
    board,
    rvec,
    tvec,
    K,
    dist,
    shape: Tuple[int, int],
    *,
    band_px: int = 7,
    step_mm: float = 2.0,
    include_markers: bool = True,
    marker_pad_px: int = 3,
) -> np.ndarray:
    """
    Build a mask (255 = *exclude*) covering the ChArUco board's own pattern edges.

    Two distinct sources of board edges, and both must go:

    * the **chessboard lattice** — square boundaries, painted as bands of ``band_px`` along every
      projected grid line;
    * the **ArUco markers** — each white square holds a marker whose bit pattern is a dense field
      of edges *inside* the square, nowhere near the lattice. These dominate: masking lattice
      alone leaves well over half the board's edge pixels behind, and the fit happily latches
      onto them. Their 3D corners come from ``board.getObjPoints()``, so every marker is covered
      whether or not the detector decoded it.

    Lines are densely sampled in 3D before projecting so lens distortion is followed correctly —
    a straight 3D line is a curve in a distorted image.

    Masking the pattern rather than the board's bounding quad is deliberate: the part sits on the
    board, so blanking the whole rectangle would erase the object with it.
    """
    _require_cv2()
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), np.uint8)

    corners = np.asarray(board.getChessboardCorners(), np.float64).reshape(-1, 3)
    if len(corners) < 4:
        raise EdgeError("board has too few chessboard corners to build a mask")
    # Infer the square pitch from the interior-corner lattice, then extend one square beyond it
    # in each direction to reach the physical board edge. Derived from the corners themselves so
    # this does not depend on the board-origin convention of the OpenCV build.
    xs = np.unique(np.round(corners[:, 0], 6))
    ys = np.unique(np.round(corners[:, 1], 6))
    if len(xs) < 2 or len(ys) < 2:
        raise EdgeError("degenerate chessboard corner lattice")
    pitch = float(min(np.min(np.diff(xs)), np.min(np.diff(ys))))
    grid_x = np.arange(xs[0] - pitch, xs[-1] + 1.5 * pitch, pitch)
    grid_y = np.arange(ys[0] - pitch, ys[-1] + 1.5 * pitch, pitch)

    K = np.asarray(K, np.float64).reshape(3, 3)
    dist = np.asarray(dist, np.float64).reshape(-1, 1)
    rvec = np.asarray(rvec, np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)

    def _paint(p3d: np.ndarray) -> None:
        proj, _ = cv2.projectPoints(p3d.reshape(-1, 1, 3), rvec, tvec, K, dist)
        pts = np.round(proj.reshape(-1, 2)).astype(np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=int(band_px))

    span_y = np.arange(grid_y[0], grid_y[-1] + step_mm, step_mm)
    span_x = np.arange(grid_x[0], grid_x[-1] + step_mm, step_mm)
    zer_y = np.zeros_like(span_y)
    zer_x = np.zeros_like(span_x)
    for gx in grid_x:                                  # lines of constant X
        _paint(np.stack([np.full_like(span_y, gx), span_y, zer_y], axis=1))
    for gy in grid_y:                                  # lines of constant Y
        _paint(np.stack([span_x, np.full_like(span_x, gy), zer_x], axis=1))

    if include_markers:
        try:
            marker_objs = board.getObjPoints()
        except Exception:                              # pragma: no cover - older API shape
            marker_objs = None
        for quad in (marker_objs or []):
            q = np.asarray(quad, np.float64).reshape(-1, 3)
            if len(q) < 3:
                continue
            proj, _ = cv2.projectPoints(q.reshape(-1, 1, 3), rvec, tvec, K, dist)
            poly = np.round(proj.reshape(-1, 2)).astype(np.int32)
            cv2.fillConvexPoly(mask, cv2.convexHull(poly), 255)
            if marker_pad_px > 0:                      # cover the marker's own outer border
                cv2.polylines(mask, [cv2.convexHull(poly)], isClosed=True, color=255,
                              thickness=int(marker_pad_px) * 2)
    return mask


def working_area_mask(
    board,
    rvec,
    tvec,
    K,
    dist,
    shape: Tuple[int, int],
    *,
    margin_mm: float = 150.0,
    lift_mm: float = 150.0,
) -> np.ndarray:
    """
    Mask (255 = *keep*) of the region where the part can physically be.

    The rig's own 2020 extrusion is long, straight, bright and permanently in frame — and the
    model is a long box, so an edge cost will happily align the part's long axis to a rail. On
    the first real capture both the coarse search and the refinement did exactly that. The rails
    are not noise to be averaged out; they are a better match than the truth.

    Since the board pose is solved, the working plane is known. This projects a box standing on
    that plane — the board extent grown by *margin_mm*, and *lift_mm* tall so the part's upper
    edges are included — and keeps only what falls inside its silhouette. Structure well above
    or outside the working volume falls away.

    Note this is a *keep* mask, the opposite sense to the board and hull masks.

    Keep the margin tight. The first attempt used 400 mm, which around a 360x240 board gives a
    1160x1040 mm region - larger than the ~840-1100 mm the frame actually covers at this
    standoff, so the mask was the whole image and changed nothing at all. Check
    ``working_area_fraction`` in the diagnostics: if it is near 1.0, the mask is doing nothing.
    """
    _require_cv2()
    h, w = int(shape[0]), int(shape[1])
    corners = np.asarray(board.getChessboardCorners(), np.float64).reshape(-1, 3)
    xs, ys = np.unique(np.round(corners[:, 0], 6)), np.unique(np.round(corners[:, 1], 6))
    pitch = float(min(np.min(np.diff(xs)), np.min(np.diff(ys))))
    x0, x1 = xs[0] - pitch - margin_mm, xs[-1] + pitch + margin_mm
    y0, y1 = ys[0] - pitch - margin_mm, ys[-1] + pitch + margin_mm

    # Both faces of the volume: the plane itself and a lifted copy. Which sign is "up" depends
    # on which side the camera is, so include both and take the hull — cheap, and avoids
    # baking in a convention here.
    box = []
    for z in (0.0, -lift_mm, lift_mm):
        box.extend([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
    proj, _ = cv2.projectPoints(np.asarray(box, np.float64).reshape(-1, 1, 3),
                                np.asarray(rvec, np.float64).reshape(3, 1),
                                np.asarray(tvec, np.float64).reshape(3, 1),
                                np.asarray(K, np.float64).reshape(3, 3),
                                np.asarray(dist, np.float64).reshape(-1, 1))
    raw = proj.reshape(-1, 2)
    # Corners behind or level with the camera project to infinity, and casting NaN/inf to int32
    # is undefined — it silently produced a garbage hull (one camera "kept" 39% of frame at a
    # margin that should have kept everything). Drop them, and clamp the rest to a sane range
    # so a near-parallel corner cannot explode the hull.
    finite = np.isfinite(raw).all(axis=1)
    pts = raw[finite]
    if len(pts) < 3:
        return np.full((h, w), 255, np.uint8)          # cannot bound it: keep everything
    limit = 10 * max(w, h)
    pts = np.round(np.clip(pts, -limit, limit)).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    return mask


def marker_hull_mask(
    marker_corners: Sequence,
    shape: Tuple[int, int],
    *,
    dilate_px: int = 9,
) -> np.ndarray:
    """
    Cruder alternative to :func:`board_grid_mask`: fill the convex hull of the detected ArUco
    markers. Appropriate only when the board sits *beside* the object — it erases anything
    standing on the board.
    """
    _require_cv2()
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), np.uint8)
    pts = [np.asarray(c, np.float64).reshape(-1, 2) for c in (marker_corners or [])]
    if not pts:
        return mask
    allpts = np.vstack(pts).astype(np.int32)
    if len(allpts) < 3:
        return mask
    hull = cv2.convexHull(allpts)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate_px), int(dilate_px)))
        mask = cv2.dilate(mask, k)
    return mask


def detect_edge_pixels(
    image: np.ndarray,
    *,
    blur_ksize: int = 5,
    low: Optional[int] = None,
    high: Optional[int] = None,
    sigma: float = 0.33,
    exclude_mask: Optional[np.ndarray] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    max_points: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Canny edge pixels from a **raw (distorted)** frame, as an Nx2 float array of ``(x, y)``.

    *exclude_mask*: uint8 where non-zero means "drop these pixels" (e.g. the board lattice).
    *roi*: ``(x, y, w, h)`` to restrict the search. *max_points*: random-subsample the result
    (deterministic given *seed*) when the frame yields more than this.
    """
    _require_cv2()
    gray = to_gray(image)
    if blur_ksize and blur_ksize >= 3:
        k = int(blur_ksize) | 1                        # OpenCV needs an odd kernel
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if low is not None and high is not None:
        lo, hi = int(low), int(high)
    else:
        # Threshold from the region we actually keep, not the whole frame. Once the bench was
        # covered with a single white sheet the frame median jumped to near-white, which drove
        # the thresholds up and cut camera B from 3122 edge pixels to 529 — the cleaner
        # background silently starved the detector. Sampling inside the mask keeps the statistic
        # tied to the subject.
        sample = gray
        if exclude_mask is not None:
            em = np.asarray(exclude_mask)
            if em.shape[:2] == gray.shape[:2]:
                inside = gray[em == 0]
                if inside.size >= 1000:
                    sample = inside
        lo, hi = auto_canny_thresholds(sample, sigma)
    edges = cv2.Canny(gray, lo, hi, L2gradient=True)

    if exclude_mask is not None:
        em = np.asarray(exclude_mask)
        if em.shape[:2] != edges.shape[:2]:
            raise EdgeError(f"exclude_mask shape {em.shape[:2]} != image shape {edges.shape[:2]}")
        edges[em > 0] = 0

    if roi is not None:
        x, y, w, h = (int(v) for v in roi)
        keep = np.zeros_like(edges)
        keep[max(0, y):y + h, max(0, x):x + w] = 255
        edges = cv2.bitwise_and(edges, keep)

    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        raise EdgeError(
            "no edge pixels found — check focus/exposure, the Canny thresholds, "
            "or whether the exclude mask is covering the object"
        )
    pts = np.stack([xs, ys], axis=1).astype(np.float64)

    if max_points is not None and len(pts) > int(max_points):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=int(max_points), replace=False)
        pts = pts[np.sort(idx)]
    return pts
