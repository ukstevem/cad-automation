#!/usr/bin/env python3
"""
Generate a speckle pattern to project onto the work for active stereo.

The projector's only job is to paint texture on bare steel so a stereo matcher has something to
correlate. What makes a pattern good for that is narrower than it looks:

ISOTROPIC. No preferred direction. A pattern with structure - stripes, a grid, anything
axis-aligned - correlates well along that axis and poorly across it, so disparity comes out
accurate in one direction and vague in the other. An early version of this modelled speckle by
hashing a cubic lattice and produced exactly that failure.

BAND-LIMITED, not white noise. Raw random pixels alias badly once projected and re-imaged, and
clump into large empty patches at low frequency. Blurring to the target blob size and then
subtracting a broader blur removes the clumping and leaves an even density of blobs of roughly one
size - which is what a matcher's fixed correlation window wants.

THE BLOB SIZE IS THE ONE NUMBER THAT MATTERS. It should land at roughly 4-6 pixels in the CAMERA
image: smaller and it aliases into mush, larger and a correlation window sees only flat area with
no detail to lock onto. That is a chain of three numbers - projector pixels to millimetres on the
part, then millimetres to camera pixels - so the tool takes the working geometry and does the
arithmetic rather than leaving it to guesswork.

    docker compose run --rm --no-deps api python tools/make_speckle.py \\
        --width 1920 --height 1080 --coverage-mm 650 --out outputs/calibration/speckle.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import cv2
except Exception as exc:                                # pragma: no cover - import guard
    print("OpenCV is required: %s" % exc, file=sys.stderr)
    raise


def speckle(width: int, height: int, blob_px: float, seed: int = 7,
            contrast: float = 1.0) -> np.ndarray:
    """Band-limited isotropic random-dot field, 8-bit."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width)).astype(np.float32)

    # Low-pass to the target blob size, then remove the slower variation. What survives is a band
    # around one spatial frequency: blobs of roughly one size, evenly spread, no clumping.
    fine = cv2.GaussianBlur(noise, (0, 0), blob_px / 2.0)
    coarse = cv2.GaussianBlur(noise, (0, 0), blob_px * 2.0)
    band = fine - coarse

    band -= band.mean()
    sd = float(band.std()) or 1.0
    band = band / sd * (0.30 * contrast)
    img = np.clip(0.5 + band, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1920, help="projector native width in pixels")
    ap.add_argument("--height", type=int, default=1080, help="projector native height")
    ap.add_argument("--coverage-mm", type=float, default=650.0,
                    help="how wide an area the projected image covers on the work")
    ap.add_argument("--blob-mm", type=float, default=3.0,
                    help="blob size ON THE PART. ~3mm suits a camera at ~0.56 mm/px, giving "
                         "about 5 camera pixels per blob.")
    ap.add_argument("--camera-mm-per-px", type=float, default=0.56,
                    help="the stereo camera's ground sample distance, for the sanity check")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--invert", action="store_true",
                    help="dark dots on light, if the projector's black level is poor")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mm_per_projector_px = args.coverage_mm / args.width
    blob_px = args.blob_mm / mm_per_projector_px
    blob_camera_px = args.blob_mm / args.camera_mm_per_px

    print("projector    %dx%d covering %.0f mm  ->  %.3f mm per projector pixel"
          % (args.width, args.height, args.coverage_mm, mm_per_projector_px))
    print("blob         %.1f mm on the part  =  %.1f projector px  =  %.1f CAMERA px"
          % (args.blob_mm, blob_px, blob_camera_px))
    if blob_camera_px < 3:
        print("  WARNING: under ~3 camera px per blob the pattern aliases into mush.")
        print("  Increase --blob-mm, or move the projector closer so it covers less area.")
    elif blob_camera_px > 8:
        print("  WARNING: over ~8 camera px per blob leaves correlation windows looking at flat")
        print("  area. Decrease --blob-mm.")
    else:
        print("  in the 3-8 camera px band that suits an SGBM block size of 7.")

    if blob_px < 2.5:
        print("  NOTE: only %.1f projector pixels per blob - the projector cannot resolve this "
              "cleanly." % blob_px)

    img = speckle(args.width, args.height, max(2.0, blob_px), seed=args.seed,
                  contrast=args.contrast)
    if args.invert:
        img = 255 - img

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, img)
    print("wrote %s" % args.out)
    print("")
    print("Project it FULL SCREEN with no scaling, smoothing or keystone correction - all three")
    print("blur the blobs, and blur is the one thing this pattern cannot survive. Focus the")
    print("projector on the part, not on the bench.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
