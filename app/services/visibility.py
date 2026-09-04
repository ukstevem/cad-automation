"""
Hidden-line removal for the multi-view fit — keep only the CAD edges a camera could see.

Without this the fit projects **every** edge of the model at every pose, including all the ones
on the far side of the object. A camera sees roughly half. The rest are phantoms with no
counterpart in the image, and they wreck the cost in three measurable ways:

* they match whatever real edge happens to be nearest, usually unrelated geometry;
* they make a reverse/bidirectional term pointless, because a projection that dense satisfies
  any question asked of it;
* they leave the cost nearly blind — measured on this rig, only ~2.5 px of change across a **2x**
  change in model scale.

The consequence in practice was a pose sitting correctly along one camera's viewing ray and
wrongly along the other's: camera B traced the part closely while camera A was offset and too
large, with the world frames verified consistent.

Method is the standard one (cf. RAPiD, ViSP model-based trackers): rasterise the mesh into a
depth buffer at the candidate pose, then drop any edge sample lying behind the surface at its
own pixel. Visibility depends on pose, so callers use it as an OUTER loop — extract, fit,
re-extract — rather than recomputing inside the optimiser.

Deliberately approximate, because occlusion is a coarse question:

* the buffer is rendered at reduced resolution (a quarter by default);
* triangles are painted far-to-near with a single depth each (painter's algorithm) rather than
  interpolated per-pixel, which is ample for a finely tessellated mesh;
* lens distortion is ignored — but *consistently*, for both the buffer and the lookup, so the
  comparison stays valid.
"""
from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

try:  # import guard — keep module importable if a wheel is missing
    import cv2

    _CV2_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - import guard
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = str(exc)

FAR = 1.0e9


class VisibilityError(Exception):
    """Raised when visibility cannot be computed (bad mesh, missing OpenCV)."""


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise VisibilityError(f"OpenCV is not available: {_CV2_IMPORT_ERROR}")


def load_stl(path: str) -> np.ndarray:
    """Read a binary STL into an ``(n, 3, 3)`` array of triangle vertices."""
    with open(path, "rb") as fh:
        header = fh.read(80)
        if len(header) < 80:
            raise VisibilityError(f"{path}: too short to be a binary STL")
        count = struct.unpack("<I", fh.read(4))[0]
        raw = fh.read(count * 50)
    if len(raw) < count * 50:
        raise VisibilityError(f"{path}: truncated — expected {count} triangles")
    # Each 50-byte record: 12B normal, 36B three vertices, 2B attribute.
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(count, 50)
    verts = buf[:, 12:48].copy().view(np.float32).reshape(count, 3, 3)
    return verts.astype(np.float64)


def _object_to_camera(points: np.ndarray, rvec_obj, tvec_obj, R_cam, t_cam) -> np.ndarray:
    """Object-frame points -> camera frame, as an Nx3 array."""
    R_obj, _ = cv2.Rodrigues(np.asarray(rvec_obj, np.float64).reshape(3, 1))
    t_obj = np.asarray(tvec_obj, np.float64).reshape(3, 1)
    world = (R_obj @ points.reshape(-1, 3).T + t_obj)
    cam = R_cam @ world + t_cam
    return cam.T


def depth_buffer(
    tris: np.ndarray,
    rvec_obj,
    tvec_obj,
    view: dict,
    *,
    downscale: int = 4,
) -> Tuple[np.ndarray, float]:
    """
    Render the mesh's depth at the given pose. Returns ``(depth_image, scale)``.

    Painter's algorithm: triangles are sorted far-to-near and filled with their own mean depth,
    so the nearest surface ends up on top. That is correct for a closed, non-self-intersecting
    mesh and avoids a per-pixel depth comparison entirely.
    """
    _require_cv2()
    R_cam, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    t_cam = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    K = np.asarray(view["K"], np.float64).reshape(3, 3) / float(downscale)
    K[2, 2] = 1.0
    w = max(1, int(view["width"]) // downscale)
    h = max(1, int(view["height"]) // downscale)

    cam = _object_to_camera(tris, rvec_obj, tvec_obj, R_cam, t_cam).reshape(-1, 3, 3)
    z = cam[:, :, 2]
    keep = np.all(z > 1e-6, axis=1)                 # drop anything at or behind the camera
    cam, z = cam[keep], z[keep]
    if not len(cam):
        return np.full((h, w), FAR, np.float32), float(downscale)

    # Pinhole projection, no distortion — see the module docstring: consistency with the lookup
    # matters here, absolute accuracy does not.
    uv = (K @ (cam.reshape(-1, 3) / cam.reshape(-1, 3)[:, 2:3]).T).T[:, :2]
    uv = uv.reshape(-1, 3, 2)
    tri_depth = z.mean(axis=1)

    depth = np.full((h, w), FAR, np.float32)
    for i in np.argsort(-tri_depth):                # far first, so nearer overwrites
        poly = np.round(uv[i]).astype(np.int32)
        if poly[:, 0].max() < 0 or poly[:, 1].max() < 0 or poly[:, 0].min() >= w or poly[:, 1].min() >= h:
            continue
        cv2.fillConvexPoly(depth, poly, float(tri_depth[i]), lineType=cv2.LINE_8)
    return depth, float(downscale)


def visible_points(
    points_obj: np.ndarray,
    rvec_obj,
    tvec_obj,
    view: dict,
    depth: np.ndarray,
    scale: float,
    *,
    tol_mm: float = 3.0,
) -> np.ndarray:
    """
    Boolean mask over *points_obj*: True where the point is not hidden by the mesh.

    A point lying exactly on a surface has the same depth as the buffer, so *tol_mm* of slack is
    needed or the model would occlude its own edges. Points that project outside the frame are
    reported visible — they are off-image, which the fit's own bounds and visibility diagnostic
    handle, and calling them "hidden" here would confuse two different things.
    """
    _require_cv2()
    R_cam, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    t_cam = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    K = np.asarray(view["K"], np.float64).reshape(3, 3) / float(scale)
    K[2, 2] = 1.0
    h, w = depth.shape[:2]

    cam = _object_to_camera(points_obj, rvec_obj, tvec_obj, R_cam, t_cam)
    z = cam[:, 2]
    ok = z > 1e-6
    uv = np.zeros((len(cam), 2))
    if np.any(ok):
        uv[ok] = (K @ (cam[ok] / z[ok, None]).T).T[:, :2]

    xi = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, h - 1)
    inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    surface = depth[yi, xi]
    hidden = inside & (surface < FAR / 2) & (z > surface + tol_mm)
    return ~hidden


def visible_edge_points(
    tris: np.ndarray,
    points_obj: np.ndarray,
    rvec_obj,
    tvec_obj,
    view: dict,
    *,
    downscale: int = 4,
    tol_mm: float = 3.0,
) -> np.ndarray:
    """Convenience: depth-buffer the mesh once, then test the edge samples against it."""
    depth, scale = depth_buffer(tris, rvec_obj, tvec_obj, view, downscale=downscale)
    return visible_points(points_obj, rvec_obj, tvec_obj, view, depth, scale, tol_mm=tol_mm)
