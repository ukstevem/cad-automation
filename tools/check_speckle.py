#!/usr/bin/env python3
"""
Measure a projected speckle pattern from a photograph, and check what it does to the board.

Two questions, both assumed rather than measured until now:

  1. WHAT SIZE ARE THE BLOBS, really? The pattern was generated for 3 mm on the part, but that
     assumed a projector coverage and a working distance. Throw, focus and the projector's own
     resolution all move it. Blob size is the single parameter that decides whether stereo
     matching works: under ~3 camera pixels it aliases into mush, over ~8 a correlation window
     sees flat area with nothing to lock onto.

  2. DOES THE SPECKLE ACTUALLY BREAK CHARUCO DETECTION? The whole two-shot capture protocol -
     projector off for board pose, on for texture - rests on the claim that it does. If the board
     still decodes under speckle, Monday gets simpler and the protocol is unnecessary ceremony.

Blob size is measured by autocorrelation rather than by counting blobs: correlate the image with
itself at increasing offsets and see how far you can shift before it stops resembling itself. That
distance IS the characteristic feature size, it needs no threshold or blob detector, and it
degrades gracefully on a blurry or unevenly lit photograph.

The same profile also reveals ISOTROPY. Measured along several directions, a pattern with no
preferred direction gives the same width every way; anything axis-aligned - stripes, a grid, the
lattice artefact an earlier version of this project's renderer produced - shows up immediately as
a wide profile one way and narrow the other.

    docker compose run --rm --no-deps api python tools/check_speckle.py \\
        --speckle photo_of_wall.jpg --board photo_of_board_with_projector_on.jpg
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services import charuco, image_edges, multiview_fit as MVF  # noqa: E402


def autocorr_width(gray: np.ndarray, max_lag: int = 40):
    """
    Characteristic feature size, per direction, by normalised autocorrelation.

    Returns (width_px_by_angle, mean_width). The width is the lag at which correlation first
    falls below 0.5 - the half-height of the autocorrelation peak, which for a blob field is
    close to the blob radius, so twice it approximates blob diameter.
    """
    g = gray.astype(np.float32)
    # DETREND FIRST. Autocorrelation measures whatever varies, and a photograph of a projected
    # pattern also carries vignetting, the shadow across the sheet, paper edges and the bright
    # rectangle's own boundary - all far larger in scale than the blobs and far stronger. Left in,
    # they dominate the correlation and the answer depends entirely on where the crop was taken:
    # measuring one photograph three ways gave 4.3, 5.4 and 11.7 mm. Subtracting a broad blur
    # removes everything slower than the blobs and makes the measurement crop-independent.
    g = g - cv2.GaussianBlur(g, (0, 0), 25.0)
    g -= g.mean()
    sd = float(g.std()) or 1.0
    g /= sd
    h, w = g.shape
    # Work on a centred crop: cheaper, and avoids vignetting at the frame edges skewing it.
    ch, cw = min(h, 600), min(w, 600)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    patch = g[y0:y0 + ch, x0:x0 + cw]
    base = float((patch * patch).mean())

    out = {}
    for name, (dy, dx) in (("horizontal", (0, 1)), ("vertical", (1, 0)),
                           ("diag /", (-1, 1)), ("diag \\", (1, 1))):
        # A diagonal step of (1,1) moves sqrt(2) pixels, not one. Comparing raw lag COUNTS across
        # directions therefore makes any isotropic pattern look 1.41x narrower on the diagonals -
        # which is exactly what this tool reported on a pattern known to be isotropic, before the
        # step length was accounted for. Convert lags to distance.
        step = float(np.hypot(dy, dx))
        prev, width = 1.0, None
        for lag in range(1, max_lag):
            sy, sx = dy * lag, dx * lag
            a = patch[max(0, sy):ch + min(0, sy), max(0, sx):cw + min(0, sx)]
            b = patch[max(0, -sy):ch + min(0, -sy), max(0, -sx):cw + min(0, -sx)]
            if a.size == 0:
                break
            c = float((a * b).mean()) / (base or 1.0)
            if c < 0.5:
                lag_at_half = lag - 1 + (prev - 0.5) / max(prev - c, 1e-6)
                width = lag_at_half * step
                break
            prev = c
        out[name] = width
    vals = [v for v in out.values() if v]
    return out, (float(np.mean(vals)) if vals else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speckle", help="photo of the projected pattern on a flat surface")
    ap.add_argument("--board", help="photo of the ChArUco board WITH the projector on")
    ap.add_argument("--board-off", help="the same board with the projector OFF, as the control")
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--distance-mm", type=float, default=800.0)
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    K = np.asarray(profile["K"], np.float64).reshape(3, 3)
    fx = float(K[0][0]) if not isinstance(K[0], np.ndarray) else float(K[0, 0])
    mm_per_px = args.distance_mm / fx

    if args.speckle:
        img = cv2.imread(args.speckle, cv2.IMREAD_COLOR)
        if img is None:
            print("could not read %s" % args.speckle, file=sys.stderr)
            return 2
        g = image_edges.to_gray(img)
        # A photo taken with a different camera has a different scale; report in the photo's own
        # pixels AND in mm using the rig camera's GSD, flagging the assumption.
        widths, mean_w = autocorr_width(g)
        print("=== SPECKLE ===")
        print("  image %dx%d   contrast: p5 %.0f  p95 %.0f  (span %.0f of 255)"
              % (img.shape[1], img.shape[0], np.percentile(g, 5), np.percentile(g, 95),
                 np.percentile(g, 95) - np.percentile(g, 5)))
        for k, v in widths.items():
            print("  autocorr half-width %-11s %s" % (k, "%.1f px" % v if v else "> max lag"))
        if mean_w:
            blob_px = 2.0 * mean_w
            print("  -> blob diameter approx %.1f px in THIS photo" % blob_px)
            print("     at the rig camera's %.2f mm/px that would be %.1f mm on the part"
                  % (mm_per_px, blob_px * mm_per_px))
            vals = [v for v in widths.values() if v]
            aniso = (max(vals) / min(vals)) if len(vals) > 1 and min(vals) > 0 else 1.0
            print("  isotropy ratio %.2f  (%s)"
                  % (aniso, "good - no preferred direction" if aniso < 1.35 else
                     "DIRECTIONAL - the pattern has structure and will match better along one axis"))
            if blob_px < 3:
                print("  VERDICT: too fine - it will alias. Move the projector closer (less area")
                print("           covered) or regenerate with a larger --blob-mm.")
            elif blob_px > 9:
                print("  VERDICT: too coarse - correlation windows will see flat area.")
                print("           Regenerate with a smaller --blob-mm.")
            else:
                print("  VERDICT: in the usable 3-8 px band.")
        span = np.percentile(g, 95) - np.percentile(g, 5)
        if span < 40:
            print("  WARNING: low contrast (%.0f levels). Dim the room or brighten the projector -"
                  % span)
            print("           a matcher needs the pattern to dominate the ambient light.")
        print("")

    if args.board:
        board = charuco.build_board_from_config(profile["board"])
        det = charuco.make_detector(board)
        print("=== CHARUCO UNDER SPECKLE ===")
        for label, path in (("projector ON ", args.board), ("projector OFF", args.board_off)):
            if not path:
                continue
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            if im is None:
                print("  %s: could not read %s" % (label, path))
                continue
            cor, ids, _mc, mi = charuco.detect_board_detailed(det, image_edges.to_gray(im))
            print("  %s  markers %2d   corners %2d"
                  % (label, 0 if mi is None else len(mi), 0 if ids is None else len(ids)))
        print("")
        print("  If ON still decodes the board, the two-shot protocol is unnecessary and Monday")
        print("  gets simpler. If it does not, the protocol is doing real work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
