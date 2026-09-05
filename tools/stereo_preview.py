#!/usr/bin/env python3
"""
Preview what a narrow-baseline stereo pair would see, with and without a projector.

The rig's two cameras sit ~50 degrees apart, which is excellent for triangulating a pose and poor
for stereo MATCHING - at that separation a surface is foreshortened differently in each view and
correlation struggles. A depth setup therefore wants a third camera close to an existing one, and
this renders the pair that would produce before anyone mounts anything.

It also renders the pair WITHOUT the projector, which is the argument for buying one: bare steel
carries no texture, so passive stereo has nothing to match and returns holes. The speckle gives
every surface patch a locally unique signature.

The speckle is indexed by the 3D position of the surface point rather than by screen position.
That is what makes it usable: a real projector fixes its pattern in space, so both cameras see the
SAME marks on the SAME points of the object, from different angles. Speckle painted in screen
space would look plausible and match to nonsense.

    docker compose run --rm --no-deps api python tools/stereo_preview.py \
        --mesh outputs/ar_models/mainframe_default_1to5.stl \
        --fit outputs/ar_fits/turn90 --rig outputs/ar_captures/turn90 \
        --baseline 150 --out outputs/ar_fits/stereo_preview.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import charuco, multiview_fit as MVF, visibility as VIS  # noqa: E402

import synth_capture as SC  # noqa: E402


def offset_camera(view: dict, baseline_mm: float) -> dict:
    """
    A virtual second camera, *baseline_mm* to the side of *view*, same orientation.

    Sliding along the camera's own X axis keeps the two image planes parallel, which is the
    canonical stereo arrangement and means rectification is nearly the identity.
    """
    R, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    t = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    centre = -R.T @ t                                  # camera centre in world
    centre2 = centre + R.T @ np.array([[baseline_mm], [0.0], [0.0]])
    out = dict(view)
    out["tvec_cam"] = -R @ centre2
    out["tag"] = view["tag"] + "_virtual"
    return out


_PATTERN_CACHE: dict = {}


def projector_pattern(width: int = 1280, height: int = 800, dot_px: float = 2.5,
                      seed: int = 3) -> np.ndarray:
    """
    The projector's slide: an isotropic random-dot field, generated once and cached.

    Blurring white noise and re-normalising gives blobs with no preferred direction and a
    controllable characteristic size - which is what a diffuser or a speckle slide actually
    produces. It matters because a stereo matcher keys on local texture: a pattern with
    directional structure correlates well ALONG that direction and poorly across it.
    """
    key = (width, height, round(dot_px, 2), seed)
    if key not in _PATTERN_CACHE:
        rng = np.random.default_rng(seed)
        noise = rng.random((height, width)).astype(np.float32)
        k = max(1, int(dot_px) | 1)
        blur = cv2.GaussianBlur(noise, (k * 3 | 1, k * 3 | 1), dot_px / 2.0)
        blur -= blur.min()
        blur /= max(blur.max(), 1e-6)
        _PATTERN_CACHE[key] = blur
    return _PATTERN_CACHE[key]


def projector_pose(views, profile, target=None, board=None, margin_mm: float = 120.0):
    """
    A projector sited between the cameras, AIMED at the working volume and zoomed to cover it.

    Modelled as a camera with its own intrinsics, because optically that is what a projector is -
    light leaves along the rays a camera would gather.

    Both the aim and the focal length are derived from the volume it has to cover rather than
    assumed. The first version copied a camera's orientation and used a fixed throw ratio of 1.4,
    which gives a 39 degree field against the camera's 68: only 20% of the part fell inside the
    cone, speckle touched 1.5% of the frame, and a projector-on/projector-off comparison came out
    identical because the projector was barely illuminating anything. Anyone setting up a real
    cell would aim it and zoom until it covered the work, so the model does the same.
    """
    R, _ = cv2.Rodrigues(np.asarray(views[0]["rvec_cam"], np.float64).reshape(3, 1))
    t = np.asarray(views[0]["tvec_cam"], np.float64).reshape(3, 1)
    centre = -R.T @ t
    if len(views) > 1:
        R2, _ = cv2.Rodrigues(np.asarray(views[1]["rvec_cam"], np.float64).reshape(3, 1))
        t2 = np.asarray(views[1]["tvec_cam"], np.float64).reshape(3, 1)
        centre = 0.5 * (centre + (-R2.T @ t2))         # midway between the camera centres

    # The volume to cover: the board plus a margin, which is where the part is allowed to be.
    if target is None:
        if board is not None:
            bw, bh = MVF.board_extent_mm(board)
        else:
            bw, bh = 356.0, 236.0
        target = np.array([[-margin_mm, -margin_mm, 0.0], [bw + margin_mm, -margin_mm, 0.0],
                           [bw + margin_mm, bh + margin_mm, 0.0], [-margin_mm, bh + margin_mm, 0.0],
                           [bw / 2, bh / 2, -250.0]], np.float64)
    target = np.asarray(target, np.float64).reshape(-1, 3)
    aim = target.mean(axis=0).reshape(3, 1)

    # Look-at, OpenCV convention: z forward, x right, y down.
    z = (aim - centre).ravel()
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(z @ up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rp = np.stack([x, y, z], axis=0)
    tp = (-Rp @ centre).reshape(3, 1)

    # Zoom so the whole target volume fits, with a little slack.
    pw, ph = 1280, 800
    cam = (Rp @ target.T + tp).T
    fwd = cam[:, 2] > 1e-6
    if fwd.any():
        fx = 0.45 * pw / np.max(np.abs(cam[fwd, 0] / cam[fwd, 2]))
        fy = 0.45 * ph / np.max(np.abs(cam[fwd, 1] / cam[fwd, 2]))
        f = float(min(fx, fy))
    else:                                              # pragma: no cover - degenerate placement
        f = 1.2 * pw
    Kp = np.array([[f, 0, pw / 2.0], [0, f, ph / 2.0], [0, 0, 1.0]])
    rvec, _ = cv2.Rodrigues(Rp)
    return {"rvec": rvec, "tvec": tp, "K": Kp, "size": (pw, ph)}


def speckle(img: np.ndarray, tris, rvec, tvec, view, profile,
            grain_mm: float = 2.0, strength: float = 0.55, seed: int = 3,
            proj: dict | None = None, views=None) -> np.ndarray:
    """
    Paint projector speckle by PROJECTING a pattern, the way a projector actually does.

    Each covered pixel is back-projected to the 3D point it sees; that point is then projected
    into the projector's image and the pattern sampled there. Two cameras looking at the same
    physical point sample the SAME pattern pixel, which is the property stereo matching needs -
    and the pattern is correctly foreshortened on tilted surfaces and coarser on distant ones,
    which a world-space lattice hash gets wrong.

    The earlier version quantised world position onto a cubic lattice and hashed it. That is
    cheap and consistent between views, but the lattice imposes axis-aligned structure: the
    speckle appeared as streaks running along the part rather than as isotropic grain, and a
    matcher keying on such a pattern correlates far better along the streaks than across them.
    """
    depth, _sc = VIS.depth_buffer(tris, rvec, tvec, view, downscale=1)
    covered = depth < VIS.FAR / 2
    if not covered.any():
        return img

    K = np.asarray(view.get("K", profile["K"]), np.float64).reshape(3, 3)
    R, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    t = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)

    ys, xs = np.nonzero(covered)
    z = depth[ys, xs]
    xn = (xs - K[0, 2]) / K[0, 0]
    yn = (ys - K[1, 2]) / K[1, 1]
    world = (R.T @ (np.stack([xn * z, yn * z, z], axis=1).T - t)).T

    if proj is None:
        proj = projector_pose(views or [view], profile)
    uv, _ = cv2.projectPoints(world.reshape(-1, 1, 3), proj["rvec"], proj["tvec"],
                              proj["K"], np.zeros((5, 1)))
    uv = uv.reshape(-1, 2)
    pw, ph = proj["size"]
    pat = projector_pattern(pw, ph, dot_px=max(1.5, grain_mm), seed=seed)
    px = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, pw - 1)
    py = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, ph - 1)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < pw) & (uv[:, 1] >= 0) & (uv[:, 1] < ph)

    val = np.zeros(len(uv), np.float32)
    val[inside] = pat[py[inside], px[inside]]
    gain = np.where(inside, 1.0 - strength * val, 1.0)   # outside the cone: unlit, no speckle

    out = img.astype(np.float32)
    out[ys, xs] *= gain[:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def label(img, text, scale=1.1):
    cv2.putText(img, text, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 7)
    cv2.putText(img, text, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--camera", default=None, help="which rig camera to pair with (substring)")
    ap.add_argument("--baseline", type=float, default=150.0)
    ap.add_argument("--grain", type=float, default=2.0, help="speckle grain in mm on the surface")
    ap.add_argument("--crop", type=int, default=1, help="crop to the part (0 = full frame)")
    ap.add_argument("--cross", action="store_true",
                    help="swap left/right for CROSS-EYED free viewing. Without this the layout is "
                         "left-on-left, which is parallel (wall-eyed) order.")
    ap.add_argument("--anaglyph", action="store_true",
                    help="red/cyan composite instead of a side-by-side pair - fusible without "
                         "any free-viewing technique, and it tolerates large disparity.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    views = SC.rig_from_captures(args.rig, profile, board, det)
    for v in views:
        v["K"], v["dist"] = profile["K"], profile["dist"]
    if args.camera:
        views = [v for v in views if args.camera in v["tag"]] or views
    left = views[0]
    right = offset_camera(left, args.baseline)

    tris = VIS.load_stl(args.mesh)
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)

    panels = []
    for tag, use_speckle in (("NO projector - passive stereo", False),
                             ("WITH projector - active stereo", True)):
        row = []
        for name, v in (("LEFT", left), ("RIGHT (+%.0fmm)" % args.baseline, right)):
            img = SC.render(tris, rvec, tvec, v, board, profile, shadow=0.35, noise=1.5)
            if use_speckle:
                img = speckle(img, tris, rvec, tvec, v, profile, grain_mm=args.grain)
            row.append((name, v, img))
        panels.append((tag, row))

    # Crop to the part so the comparison is legible rather than mostly paper.
    box = None
    if args.crop:
        d, _ = VIS.depth_buffer(tris, rvec, tvec, left, downscale=1)
        ys, xs = np.nonzero(d < VIS.FAR / 2)
        if len(xs):
            m = 90
            box = (max(0, xs.min() - m), max(0, ys.min() - m),
                   min(left["width"], xs.max() + m), min(left["height"], ys.max() + m))

    # Report the disparity, because it is what decides whether a pair is free-viewable. The
    # baseline that a human can fuse and the baseline that measures well are different numbers:
    # comfortable fusion wants ~30 px of disparity, measurement wants as much as matching allows.
    d, _ = VIS.depth_buffer(tris, rvec, tvec, left, downscale=1)
    zs = d[d < VIS.FAR / 2]
    if len(zs):
        f = float(np.asarray(profile["K"], np.float64).reshape(3, 3)[0, 0])
        med = float(np.median(zs))
        px = f * args.baseline / med
        print("part at %.0f mm -> disparity %.0f px at %.0f mm baseline" % (med, px, args.baseline))
        print("  (~30 px is comfortable for free viewing => baseline about %.0f mm)"
              % (30.0 * med / f))

    if args.anaglyph:
        _, rowA = panels[-1] if len(panels) > 1 else panels[0]
        imgs = []
        for name, v, img in rowA:
            if box:
                img = img[box[1]:box[3], box[0]:box[2]]
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        h = min(i.shape[0] for i in imgs)
        w2 = min(i.shape[1] for i in imgs)
        L, R = imgs[0][:h, :w2], imgs[1][:h, :w2]
        ana = np.zeros((h, w2, 3), np.uint8)
        ana[:, :, 2] = L                       # red channel  <- left eye
        ana[:, :, 0] = R                       # blue channel <- right eye
        ana[:, :, 1] = R                       # green too, making it red/cyan
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cv2.imwrite(args.out, label(ana, "RED/CYAN anaglyph - %.0f mm baseline" % args.baseline))
        print("wrote %s (anaglyph)" % args.out)
        return 0

    rows = []
    for tag, row in panels:
        imgs = []
        for name, v, img in row:
            if box:
                img = img[box[1]:box[3], box[0]:box[2]]
            img = label(img.copy(), "%s  -  %s" % (name, tag))
            imgs.append(img)
        h = min(i.shape[0] for i in imgs)
        imgs = [i[:h] for i in imgs]
        if args.cross:
            imgs = imgs[::-1]
        rows.append(np.hstack(imgs))
    w = min(r.shape[1] for r in rows)
    grid = np.vstack([r[:, :w] for r in rows])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, grid)
    print("wrote %s  (%dx%d)" % (args.out, grid.shape[1], grid.shape[0]))
    print("baseline %.0f mm; speckle grain %.1f mm on the surface" % (args.baseline, args.grain))
    return 0


if __name__ == "__main__":
    sys.exit(main())
