"""
Find the geometry that makes a part's orientation unique.

An edge cost treats every CAD point alike, so on a repetitive structure the answer is dominated
by whatever repeats. Measured on the Main Frame: two poses 180 degrees apart — the part flipped
end-for-end — score within ~3 px of each other on a ~23 px baseline. The truss bays match
themselves whichever way round it sits, and the handful of features that actually break the
symmetry are outvoted by the many that do not.

This module locates those features. Apply a candidate symmetry to the model and ask, for each
sampled point, how far it lands from the nearest point of the *original*. Points that map onto
themselves carry no orientation information; points that do not are exactly the evidence that
distinguishes one end from the other.

It is a property of the model alone — no image, no pose — so it can be computed once per part
and reused. That is also why it is trustworthy: it says what the CAD *can* tell you apart by,
independently of any particular photograph.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:  # import guard — keep module importable if a wheel is missing
    import cv2

    _CV2_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - import guard
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = str(exc)


class SymmetryError(Exception):
    """Raised when a symmetry analysis cannot be performed."""


def flip_transform(points: np.ndarray, axis: str = "z", degrees: float = 180.0):
    """
    Rotation about the point cloud's own centroid, as ``(R, t)`` applied as ``R x + t``.

    About the centroid, not the origin, so the transformed part still occupies the same space —
    which is what makes the comparison a test of *shape* symmetry rather than of position.
    """
    if cv2 is None:  # pragma: no cover
        raise SymmetryError(f"OpenCV is not available: {_CV2_IMPORT_ERROR}")
    idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
    if idx is None:
        raise SymmetryError(f"axis must be x, y or z, got {axis!r}")
    vec = np.zeros((3, 1))
    vec[idx] = np.radians(degrees)
    R, _ = cv2.Rodrigues(vec)
    centre = points.reshape(-1, 3).mean(axis=0).reshape(3, 1)
    return R, (centre - R @ centre)


def asymmetry(
    points: np.ndarray,
    axis: str = "z",
    degrees: float = 180.0,
) -> np.ndarray:
    """
    Per-point distance (mm) to the nearest point of the model after applying the symmetry.

    Near zero means the point is indistinguishable under that symmetry and tells you nothing
    about which way round the part is. Large means it is a discriminating feature.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) < 2:
        raise SymmetryError("need at least two points to analyse symmetry")
    R, t = flip_transform(pts, axis=axis, degrees=degrees)
    moved = (R @ pts.T + t).T
    # Distance from each MOVED point back to the original cloud: "if the part were flipped,
    # how far would this feature have to move to look the same?"
    d, _ = cKDTree(pts).query(moved)
    return d


def discriminating_mask(
    points: np.ndarray,
    axis: str = "z",
    degrees: float = 180.0,
    *,
    min_mm: float = 5.0,
    min_fraction: float = 0.02,
) -> Tuple[np.ndarray, dict]:
    """
    Boolean mask of the points worth using to decide orientation, plus a summary.

    *min_mm* is the distance beyond which a point counts as discriminating. If too few points
    clear it, the threshold is relaxed to whatever admits *min_fraction* of the model — a part
    with only a subtle asymmetry still has a best available answer, and reporting "nothing is
    distinctive" would be less useful than reporting the most distinctive there is.

    The summary tells you how ambiguous the part inherently is. A model where almost nothing
    clears the bar cannot be oriented from its edges by any method, and that is worth knowing
    before blaming the solver.
    """
    d = asymmetry(points, axis=axis, degrees=degrees)
    mask = d >= min_mm
    threshold = min_mm
    if mask.mean() < min_fraction:
        threshold = float(np.quantile(d, 1.0 - min_fraction))
        mask = d >= threshold
    return mask, {
        "axis": axis,
        "degrees": degrees,
        "threshold_mm": round(float(threshold), 2),
        "discriminating_points": int(mask.sum()),
        "total_points": int(len(d)),
        "discriminating_fraction": round(float(mask.mean()), 4),
        "max_asymmetry_mm": round(float(d.max()), 2),
        "median_asymmetry_mm": round(float(np.median(d)), 2),
    }
