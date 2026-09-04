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

    from app.services import (charuco, model_symmetry as SYM, multiview_fit as MVF,
                              visibility as VIS)

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

    # A mesh beside the model JSON enables hidden-line removal. Without it the fit projects
    # every far-side edge too - on this model 81% of the samples - matching geometry no camera
    # can see.
    mesh = None
    mesh_path = os.path.splitext(args.model)[0] + ".stl"
    if os.path.exists(mesh_path):
        try:
            mesh = VIS.load_stl(mesh_path)
        except Exception as exc:                       # noqa: BLE001 - degrade, do not fail
            print(f"mesh unusable ({exc}); continuing without hidden-line removal", file=sys.stderr)

    seed_for_scan = MVF.estimate_position_from_edges(views, model, board)
    if args.coarse:
        init_rvec, init_tvec, grid_score = MVF.coarse_search(
            views, model, board, step_mm=args.coarse_step, yaw_step_deg=args.coarse_yaw,
            mesh=mesh,
            seed_pose=(np.zeros((3, 1)), seed_for_scan) if seed_for_scan is not None else None)
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
    if mesh is not None:
        result = MVF.fit_with_visibility(
            views, model, mesh, init_rvec, init_tvec, rounds=args.visibility_rounds,
            max_step=args.max_step, huber_delta=args.huber, max_nfev=args.max_nfev,
            tvec_bounds=bounds, planar=not args.full_6dof,
        )
    else:
        result = MVF.fit_from_views(
            views, model, init_rvec, init_tvec,
            max_step=args.max_step, huber_delta=args.huber, max_nfev=args.max_nfev,
            tvec_bounds=bounds, planar=not args.full_6dof,
        )
    # The 180-degree end-for-end flip. On a repetitive part the edge cost separates these two
    # poses by a couple of pixels on a ~23px baseline - it cannot decide, and pretending it can
    # is how a confidently wrong pose gets shipped. So solve both and present both; a person
    # (or, in the cell, the fixture) settles it in a second by looking at an end feature.
    def _fit_from(r0, t0):
        if mesh is not None:
            return MVF.fit_with_visibility(
                views, model, mesh, r0, t0, rounds=args.visibility_rounds,
                max_step=args.max_step, huber_delta=args.huber, max_nfev=args.max_nfev,
                tvec_bounds=bounds, planar=not args.full_6dof)
        return MVF.fit_from_views(
            views, model, r0, t0, max_step=args.max_step, huber_delta=args.huber,
            max_nfev=args.max_nfev, tvec_bounds=bounds, planar=not args.full_6dof)

    alternatives = []
    if not args.no_flip:
        R0, _ = cv2.Rodrigues(np.asarray(result["rvec"], float).reshape(3, 1))
        Rz, _ = cv2.Rodrigues(np.array([[0.0], [0.0], [np.pi]]))
        rflip, _ = cv2.Rodrigues(Rz @ R0)
        # Rotating about the object's own centre keeps it on the part rather than swinging it
        # a full length away, which is what makes this a fair second hypothesis.
        lo, hi = MVF.model_bbox(model)
        centre = ((lo + hi) / 2.0).reshape(3, 1)
        t0 = np.asarray(result["tvec"], float).reshape(3, 1)
        tflip = (R0 @ centre + t0) - (Rz @ R0) @ centre
        tflip[2] = t0[2]
        flipped = _fit_from(rflip, tflip)
        alternatives.append(flipped)

    result["init_source"] = init_source
    result["hidden_line_removal"] = mesh is not None
    result["mesh"] = os.path.basename(mesh_path) if mesh is not None else None
    result["init"] = {
        "rvec": np.asarray(init_rvec).reshape(3).tolist(),
        "tvec": np.asarray(init_tvec).reshape(3).tolist(),
    }
    result["failures"] = failures

    os.makedirs(args.out, exist_ok=True)
    # Clear stale overlays: a rerun with a different capture set was leaving the previous run's
    # images beside the new ones, which is a good way to draw a conclusion from the wrong photo.
    for old_name in os.listdir(args.out):
        if old_name.endswith("_overlay.png"):
            try:
                os.remove(os.path.join(args.out, old_name))
            except OSError:
                pass

    def _render(res, suffix):
        out = []
        r = np.asarray(res["rvec"], float).reshape(3, 1)
        t = np.asarray(res["tvec"], float).reshape(3, 1)
        for view, (path, img) in zip(views, images):
            canvas = MVF.render_overlay(img, model, view,
                                        [("fit", r, t, MVF.FIT_COLOUR)], show_edge_pixels=False)
            stem = os.path.splitext(os.path.basename(path))[0]
            dest = os.path.join(args.out, f"{stem}{suffix}_overlay.png")
            cv2.imwrite(dest, canvas)
            out.append(os.path.relpath(dest, "outputs").replace("\\", "/"))
        return out

    result["overlays"] = _render(result, "")
    result["alternatives"] = []
    for i, alt in enumerate(alternatives):
        result["alternatives"].append({
            "label": "flipped 180 deg end-for-end",
            "rvec": alt["rvec"], "tvec": alt["tvec"],
            "rms_after_px": alt["info"]["rms_after_px"],
            "per_view_rms_px": alt["info"]["per_view_rms_px"],
            "yaw_deg": alt["info"].get("yaw_deg"),
            "overlays": _render(alt, f"_alt{i + 1}"),
        })
    if result["alternatives"]:
        best = min([result["info"]["rms_after_px"]] +
                   [a["rms_after_px"] for a in result["alternatives"]])
        worst = max([result["info"]["rms_after_px"]] +
                    [a["rms_after_px"] for a in result["alternatives"]])
        result["ambiguity_px"] = round(worst - best, 2)

        # Break the tie on the geometry that actually differs between the two orientations.
        # Scoring every edge equally lets the repetitive truss outvote the features that carry
        # direction: measured on this part the two poses came out 0.06 px apart - a dead heat.
        # Restricted to the points that move under the flip, the same comparison separates them
        # by ~8 px. The discriminating set is a property of the CAD, computed once.
        try:
            from app.services.multiview import sample_polylines, _View
            pts = sample_polylines(model["edges"], max_step=args.max_step)
            disc, sym_info = SYM.discriminating_mask(pts, "z", 180.0, min_mm=10.0)
            result["symmetry"] = sym_info

            def _disc_score(rv, tv):
                rv = np.asarray(rv, float).reshape(3, 1)
                tv = np.asarray(tv, float).reshape(3, 1)
                per = []
                for v in views:
                    m = disc.copy()
                    if mesh is not None:
                        m &= VIS.visible_edge_points(mesh, pts, rv, tv, v)
                    if m.sum() < 50:
                        return None
                    vw = _View(v["K"], v["dist"], v["rvec_cam"], v["tvec_cam"],
                               v["edge_pixels"], v.get("width"), v.get("height"), point_mask=m)
                    per.append(float(np.sqrt(np.mean(vw.residuals(rv, tv, pts) ** 2))))
                return float(np.sqrt(np.mean(np.square(per))))

            cands = [("primary", result["rvec"], result["tvec"], result)]
            for a in result["alternatives"]:
                cands.append((a["label"], a["rvec"], a["tvec"], a))
            scored = [(n, _disc_score(r, t), obj) for n, r, t, obj in cands]
            scored = [(n, sc, obj) for n, sc, obj in scored if sc is not None]
            if len(scored) > 1:
                scored.sort(key=lambda x: x[1])
                for n, sc, obj in scored:
                    obj["discriminating_rms_px"] = round(sc, 2)
                result["orientation_choice"] = {
                    "chosen": scored[0][0],
                    "margin_px": round(scored[1][1] - scored[0][1], 2),
                    "scores": {n: round(sc, 2) for n, sc, _ in scored},
                    "points_used": int(disc.sum()),
                }
                # Promote the winner if the tie-break disagrees with the raw edge cost.
                if scored[0][0] != "primary":
                    win = scored[0][2]
                    swapped = {k: result[k] for k in ("rvec", "tvec")}
                    swapped_info = result["info"]
                    result["rvec"], result["tvec"] = win["rvec"], win["tvec"]
                    result["info"] = dict(result["info"])
                    result["info"]["rms_after_px"] = win["rms_after_px"]
                    result["info"]["per_view_rms_px"] = win["per_view_rms_px"]
                    result["info"]["yaw_deg"] = win.get("yaw_deg")
                    win["label"] = "superseded: raw edge cost preferred this"
                    win.update(swapped)
                    win["rms_after_px"] = swapped_info["rms_after_px"]
                    win["per_view_rms_px"] = swapped_info["per_view_rms_px"]
                    win["yaw_deg"] = swapped_info.get("yaw_deg")
        except Exception as exc:                       # noqa: BLE001 - diagnostic, never fatal
            print(f"orientation tie-break unavailable: {exc}", file=sys.stderr)

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
    ap.add_argument("--visibility-rounds", type=int, default=3)
    ap.add_argument("--no-flip", action="store_true",
                    help="skip the 180-degree alternative hypothesis")
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
