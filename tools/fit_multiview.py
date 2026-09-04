#!/usr/bin/env python3
"""
Headless multi-view CAD pose fit: a folder of board-in-shot photos in, pose + overlays out.

This is the CLI shell over ``app/services/multiview_fit.py`` — all the logic lives there so the
future capture/fit endpoint (bead ...-cg0) shares it rather than reimplementing it.

Runs INSIDE the container (that is where OpenCV lives); ``tools/`` is bind-mounted for this:

    docker compose run --rm --no-deps api python tools/fit_multiview.py \\
        outputs/ar_captures/run1 \\
        --profile outputs/calibration/MyCam.json \\
        --model   outputs/ar_models/mainframe_default_1to5.json \\
        --out     outputs/ar_fits/run1

What it does per photo: detect the ChArUco board, solve the board->camera pose (the board IS the
world frame, so every photo self-registers), mask the board's own lattice out of the image, Canny
the rest into edge pixels, then fit the known CAD pose across all views at once.

Always look at the ``*_overlay.png`` files. The amber wireframe is the STARTING guess and the cyan
one is the fit. The edge cost has a narrow basin, so if amber is nowhere near the part the cyan
result is not to be trusted however good its RMS looks — a low residual at a wrong pose is the
exact failure that sank the single-image markerless spikes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services import charuco, multiview_fit as MVF  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_photos(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in IMAGE_EXTS:
                out.extend(sorted(glob.glob(os.path.join(p, "*" + ext))))
                out.extend(sorted(glob.glob(os.path.join(p, "*" + ext.upper()))))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def parse_init(text):
    parts = [float(x) for x in text.replace(",", " ").split()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("--init needs 6 numbers: rx,ry,rz,tx,ty,tz")
    return np.array(parts[:3]).reshape(3, 1), np.array(parts[3:]).reshape(3, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="image files and/or directories of images")
    ap.add_argument("--profile", required=True, help="calibration profile JSON (applies to all photos)")
    ap.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH",
                    help="per-camera override: use PATH for photos whose filename contains SUBSTR")
    ap.add_argument("--model", required=True, help="AR model JSON (outputs/ar_models/*.json)")
    ap.add_argument("--out", default=None, help="output directory (default outputs/ar_fits/<model>)")
    ap.add_argument("--init", type=parse_init, default=None,
                    help="initial object pose 'rx,ry,rz,tx,ty,tz' (rvec rad, tvec mm, board frame)")
    ap.add_argument("--mask", choices=["grid", "hull", "none"], default="grid",
                    help="board masking: grid=mask the lattice lines (default, part may sit on the "
                         "board); hull=blank the whole board area (only if the board is beside the "
                         "part); none=keep board edges (expect the fit to latch onto them)")
    ap.add_argument("--band-px", type=int, default=7, help="grid mask band width in px (default 7)")
    ap.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel before Canny (default 5)")
    ap.add_argument("--canny-low", type=int, default=None)
    ap.add_argument("--canny-high", type=int, default=None)
    ap.add_argument("--max-points", type=int, default=None, help="cap edge pixels per view")
    ap.add_argument("--max-step", type=float, default=2.0, help="CAD sampling step in mm (default 2)")
    ap.add_argument("--huber", type=float, default=10.0, help="Huber delta in px (default 10)")
    ap.add_argument("--max-nfev", type=int, default=400)
    ap.add_argument("--min-corners", type=int, default=6)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--coarse", action="store_true",
                    help="grid-search (x, y, yaw) for the starting pose instead of assuming the "
                         "part is centred on the board. The edge cost is a local optimiser with "
                         "a narrow basin, so a wrong init converges confidently to nonsense.")
    ap.add_argument("--coarse-step", type=float, default=100.0, help="grid pitch in mm")
    ap.add_argument("--coarse-yaw", type=float, default=20.0, help="yaw step in degrees")
    ap.add_argument("--working-margin", type=float, default=400.0,
                    help="keep only edges inside the board extent grown by this margin (mm). "
                         "The rig's own rails are longer and straighter than the part, so an "
                         "unconstrained search aligns the model to a rail instead.")
    ap.add_argument("--no-working-area", action="store_true",
                    help="disable the working-area mask (diagnostic only)")
    ap.add_argument("--no-bounds", action="store_true",
                    help="disable the working-volume constraint on translation. Without it the "
                         "cost has a degenerate minimum at 'far away and tiny' which scores a "
                         "beautiful RMS. Only for diagnosing that behaviour.")
    ap.add_argument("--allow-resolution-mismatch", action="store_true",
                    help="DANGEROUS: use a profile whose calibration resolution differs from the "
                         "photos. Intrinsics do not transfer across resolutions.")
    args = ap.parse_args()

    photos = collect_photos(args.photos)
    if not photos:
        print("No images found.", file=sys.stderr)
        return 2

    overrides = []
    for spec in args.cam_profile:
        if "=" not in spec:
            print(f"--cam-profile needs SUBSTR=PATH, got '{spec}'", file=sys.stderr)
            return 2
        sub, path = spec.split("=", 1)
        overrides.append((sub, MVF.load_profile(path)))

    default_profile = MVF.load_profile(args.profile)
    model = MVF.load_model(args.model)

    out_dir = args.out or os.path.join("outputs", "ar_fits",
                                       os.path.splitext(os.path.basename(args.model))[0])
    os.makedirs(out_dir, exist_ok=True)

    def profile_for(path):
        base = os.path.basename(path)
        for sub, prof in overrides:
            if sub in base:
                return prof
        return default_profile

    # The board comes from the profile, so the fit uses the SAME board the calibration used.
    board = charuco.build_board_from_config(default_profile["board"])
    detector = charuco.make_detector(board)

    print(f"model   : {model.get('name')} ({len(model['edges'])} edge polylines, scale {model.get('scale')})")
    print(f"board   : {default_profile['board']}")
    print(f"photos  : {len(photos)}")

    views, images, failures = [], [], []
    for path in photos:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            failures.append({"photo": path, "error": "could not decode"})
            print(f"  SKIP {os.path.basename(path)}: could not decode")
            continue
        prof = profile_for(path)
        try:
            view = MVF.build_view(
                img, prof, board, detector,
                label=os.path.basename(path),
                mask_mode=args.mask, band_px=args.band_px, blur_ksize=args.blur,
                canny_low=args.canny_low, canny_high=args.canny_high,
                max_points=args.max_points, min_corners=args.min_corners,
                enforce_resolution=not args.allow_resolution_mismatch,
                working_margin_mm=(None if args.no_working_area else args.working_margin),
            )
        except Exception as exc:
            failures.append({"photo": path, "error": str(exc)})
            print(f"  SKIP {os.path.basename(path)}: {exc}")
            continue
        d = view["_diag"]
        print(f"  OK   {d['label']}: {d['board_corners']} corners, {d['edge_pixels']} edge px, "
              f"cam {d['camera_distance_mm']}mm, masked {d['masked_fraction'] * 100:.1f}%")
        views.append(view)
        images.append((path, img))

    if not views:
        print("\nNo usable views — nothing to fit.", file=sys.stderr)
        json.dump({"error": "no usable views", "failures": failures},
                  open(os.path.join(out_dir, "fit.json"), "w"), indent=2)
        return 1
    if len(views) == 1:
        print("\nWARNING: only ONE view. A single view is exactly the degenerate case that failed "
              "on bare sections — treat the result as unverified.")

    if args.init is not None:
        init_rvec, init_tvec = args.init
    elif args.coarse:
        print("coarse  : scanning (x, y, yaw) for a starting pose...")
        init_rvec, init_tvec, score = MVF.coarse_search(
            views, model, board, step_mm=args.coarse_step, yaw_step_deg=args.coarse_yaw)
        print(f"          best grid score {score:.1f} px  "
              f"t={np.asarray(init_tvec).reshape(3).round(1).tolist()}  "
              f"yaw-ish rvec={np.asarray(init_rvec).reshape(3).round(3).tolist()}")
    else:
        init_rvec, init_tvec = MVF.default_init_pose(model, board, views)
        side = MVF.camera_side_of_board(views)
        print(f"init    : default (object centred on board, resting on the camera side, "
              f"board Z {'+' if side > 0 else '-'}ve) "
              f"t={np.asarray(init_tvec).reshape(3).round(1).tolist()}")

    bounds = None if args.no_bounds else MVF.working_volume_bounds(model, board, views)
    if bounds is not None:
        print(f"bounds  : x {bounds[0][0]:.0f}..{bounds[1][0]:.0f}  "
              f"y {bounds[0][1]:.0f}..{bounds[1][1]:.0f}  "
              f"z {bounds[0][2]:.0f}..{bounds[1][2]:.0f} mm (board frame)")
    result = MVF.fit_from_views(
        views, model, init_rvec, init_tvec,
        max_step=args.max_step, huber_delta=args.huber, max_nfev=args.max_nfev,
        tvec_bounds=bounds,
    )
    result["failures"] = failures
    info = result["info"]
    print(f"\nfit     : RMS {info['rms_before_px']} -> {info['rms_after_px']} px "
          f"over {info['n_views']} views / {info['n_points']} CAD points (success={info['success']})")
    print(f"per-view: {info['per_view_rms_px']}")
    print(f"visible : {info.get('visible_fraction', 0) * 100:.1f}% of CAD points project in frame")
    if info.get("degenerate"):
        print(f"  !! DEGENERATE: {info['degenerate']}")
    print(f"rvec    : {[round(x, 5) for x in result['rvec']]}")
    print(f"tvec mm : {[round(x, 2) for x in result['tvec']]}")

    with open(os.path.join(out_dir, "fit.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if not args.no_overlay:
        fit_rvec = np.asarray(result["rvec"], float).reshape(3, 1)
        fit_tvec = np.asarray(result["tvec"], float).reshape(3, 1)
        for view, (path, img) in zip(views, images):
            canvas = MVF.render_overlay(
                img, model, view,
                [("init", init_rvec, init_tvec, MVF.INIT_COLOUR),
                 ("fit", fit_rvec, fit_tvec, MVF.FIT_COLOUR)],
            )
            stem = os.path.splitext(os.path.basename(path))[0]
            dest = os.path.join(out_dir, f"{stem}_overlay.png")
            cv2.imwrite(dest, canvas)
            print(f"  overlay -> {dest}")

    print(f"\nWrote {out_dir}. Check the overlays: amber = init, cyan = fit. "
          f"A good RMS at a visibly wrong pose is a WRONG pose, not a good fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
