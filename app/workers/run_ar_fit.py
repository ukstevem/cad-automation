#!/usr/bin/env python3
"""
Subprocess worker for the multi-view AR fit.

Runs as a standalone child process, per the project's rule that long CPU-bound work never
executes in the FastAPI process. The fit is scipy/OpenCV rather than OCC, so it does not hold
the GIL the way XCAF does — but a coarse yaw scan takes minutes, which would stall the event
loop just as effectively.

Prints a single JSON object on stdout; diagnostics go to stderr so the parent can scan for the
first line starting with ``{``, matching the existing workers.

    python -m app.workers.run_ar_fit --captures DIR --profile P [--cam-profile TAG=P] \
        --model M --out DIR [--coarse] [--canny-low N --canny-high N]
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import threading
import traceback

faulthandler.enable()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_RESULT: dict = {}


def _work(args) -> None:
    import cv2
    import numpy as np

    from app.services import charuco, multiview_fit as MVF

    default_profile = MVF.load_profile(args.profile)
    overrides = []
    for spec in args.cam_profile or []:
        sub, path = spec.split("=", 1)
        overrides.append((sub, MVF.load_profile(path)))
    model = MVF.load_model(args.model)
    board = charuco.build_board_from_config(default_profile["board"])
    detector = charuco.make_detector(board)

    exts = (".png", ".jpg", ".jpeg")
    photos = sorted(
        os.path.join(args.captures, n) for n in os.listdir(args.captures)
        if n.lower().endswith(exts)
    )
    if not photos:
        raise RuntimeError(f"no images in {args.captures}")

    def profile_for(path):
        base = os.path.basename(path)
        for sub, prof in overrides:
            if sub in base:
                return prof
        return default_profile

    views, images, failures = [], [], []
    for path in photos:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            failures.append({"photo": os.path.basename(path), "error": "could not decode"})
            continue
        try:
            view = MVF.build_view(
                img, profile_for(path), board, detector,
                label=os.path.basename(path),
                mask_mode=args.mask, blur_ksize=args.blur,
                canny_low=args.canny_low, canny_high=args.canny_high,
                working_margin_mm=args.working_margin,
            )
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            failures.append({"photo": os.path.basename(path), "error": str(exc)})
            continue
        views.append(view)
        images.append((path, img))

    if not views:
        raise RuntimeError("no usable views: " + "; ".join(f["error"] for f in failures))

    if args.coarse:
        init_rvec, init_tvec, grid_score = MVF.coarse_search(
            views, model, board, step_mm=args.coarse_step, yaw_step_deg=args.coarse_yaw)
        init_source = f"coarse scan (grid score {grid_score:.1f} px)"
    else:
        seeded = MVF.estimate_position_from_edges(views, model, board)
        if seeded is not None:
            init_rvec, init_tvec = np.zeros((3, 1)), seeded
            init_source = "edge centroid back-projected to the plane"
        else:
            init_rvec, init_tvec = MVF.default_init_pose(model, board, views)
            init_source = "default (centred on board)"

    bounds = MVF.working_volume_bounds(model, board, views)
    result = MVF.fit_from_views(
        views, model, init_rvec, init_tvec,
        max_step=args.max_step, huber_delta=args.huber, max_nfev=args.max_nfev,
        tvec_bounds=bounds, planar=not args.full_6dof,
    )
    result["init_source"] = init_source
    result["init"] = {
        "rvec": np.asarray(init_rvec).reshape(3).tolist(),
        "tvec": np.asarray(init_tvec).reshape(3).tolist(),
    }
    result["failures"] = failures

    os.makedirs(args.out, exist_ok=True)
    overlays = []
    fit_rvec = np.asarray(result["rvec"], float).reshape(3, 1)
    fit_tvec = np.asarray(result["tvec"], float).reshape(3, 1)
    for view, (path, img) in zip(views, images):
        canvas = MVF.render_overlay(
            img, model, view,
            [("init", init_rvec, init_tvec, MVF.INIT_COLOUR),
             ("fit", fit_rvec, fit_tvec, MVF.FIT_COLOUR)],
        )
        stem = os.path.splitext(os.path.basename(path))[0]
        dest = os.path.join(args.out, f"{stem}_overlay.png")
        cv2.imwrite(dest, canvas)
        overlays.append(os.path.relpath(dest, "outputs").replace("\\", "/"))
    result["overlays"] = overlays

    with open(os.path.join(args.out, "fit.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    _RESULT.update(result)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--cam-profile", action="append", default=[])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", default="grid")
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--canny-low", type=int, default=None)
    ap.add_argument("--canny-high", type=int, default=None)
    ap.add_argument("--working-margin", type=float, default=150.0)
    ap.add_argument("--max-step", type=float, default=2.0)
    ap.add_argument("--huber", type=float, default=10.0)
    ap.add_argument("--max-nfev", type=int, default=400)
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--coarse-step", type=float, default=60.0)
    ap.add_argument("--coarse-yaw", type=float, default=10.0)
    ap.add_argument("--full-6dof", action="store_true")
    args = ap.parse_args()

    err: list = []

    def runner():
        try:
            _work(args)
        except Exception as exc:                       # noqa: BLE001
            err.append(exc)
            traceback.print_exc(file=sys.stderr)

    # Generous stack, matching the other workers - OpenCV and scipy both recurse.
    threading.stack_size(64 * 1024 * 1024)
    t = threading.Thread(target=runner)
    t.start()
    t.join()

    if err:
        print(f"ar-fit failed: {err[0]}", file=sys.stderr)
        return 1
    print(json.dumps(_RESULT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
