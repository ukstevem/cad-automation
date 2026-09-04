#!/usr/bin/env python3
"""
Sweep a part through orientations in simulation and score every fit against known truth.

This is the experiment the rig cannot run cheaply. Three questions are open and all three need
many trials with the answer known in advance:

  1. WHERE IS THE END-ON LIMIT? Orientation was recovered with the part 23-26 degrees off the
     camera axis (3 of 3) and failed at 3 degrees (1 of 1). Everything between is unmeasured, and
     the gate that would make the pipeline refuse unsafe geometry needs that number.
  2. DOES IT GENERALISE? One printed part cannot say. Run the same sweep on many meshes and the
     answer becomes a distribution rather than an anecdote.
  3. IS THE MARGIN EVER TRUSTWORTHY? On real captures the tie-break margin was ANTI-correlated
     with correctness. With hundreds of labelled trials that can be settled properly.

Scoring is automatic and needs no images: the true pose is known, so the fitted long axis either
points the same way as the truth (correct) or the opposite way (flipped end-for-end). No
plate-end marking, no eyeballing, no judgement calls.

READ THE CEILING WARNING in tools/synth_capture.py before believing any number here. Synthetic
images lack shadow, texture and print error, so these results bound what is achievable rather
than predict field performance. Anchor first: run the sweep at the four poses actually shot on
the rig and confirm it reproduces those outcomes, INCLUDING the near end-on failure. A harness
that passes everything the rig failed is measuring its own optimism, not the pipeline.

    docker compose run --rm --no-deps api python tools/synth_sweep.py \
        --mesh outputs/ar_models/mainframe_default_1to5.stl \
        --rig outputs/ar_captures/turn90 --yaw 0:180:15 --csv outputs/ar_fits/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services import charuco, multiview_fit as MVF  # noqa: E402
from app.services.visibility import load_stl  # noqa: E402

import synth_capture as SC  # noqa: E402  (same directory)


def long_axis_of(tris: np.ndarray) -> np.ndarray:
    pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    d = pts - pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    return evecs[:, int(np.argmax(evals))]


def end_on_angles(rvec, tvec, axis_obj, views) -> list:
    """Degrees between the part's long axis and each camera ray. 90 = broadside, 0 = end-on."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    a = R @ axis_obj
    a = a / np.linalg.norm(a)
    out = []
    for v in views:
        Rc, _ = cv2.Rodrigues(np.asarray(v["rvec_cam"], np.float64).reshape(3, 1))
        ray = Rc.T @ np.array([0.0, 0.0, 1.0])
        out.append(90.0 - float(np.degrees(np.arccos(np.clip(abs(float(a @ ray)), 0, 1)))))
    return out


def parse_range(spec: str) -> list:
    if ":" in spec:
        lo, hi, step = (float(v) for v in spec.split(":"))
        n = int(round((hi - lo) / step)) + 1
        return [lo + i * step for i in range(n)]
    return [float(v) for v in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--model", default=None, help="AR model JSON (default: mesh name with .json)")
    ap.add_argument("--rig", default="outputs/ar_captures/turn90")
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--yaw", default="0:165:15", help="lo:hi:step or comma list, degrees")
    ap.add_argument("--xy", default="180,-70", help="where to seat the part, board mm")
    ap.add_argument("--base-fit", default=None,
                    help="STRONGLY PREFERRED. Take a pose verified on the real rig and rotate it "
                         "about the board normal by each swept angle, keeping the part where it "
                         "actually sat. Inventing an (x, y) instead varies placement and viewing "
                         "angle together: the first attempt at this sweep seated the part ON the "
                         "board, obscuring the pattern and running off-frame, and produced 9 of 12 "
                         "'failures' that were measuring the placement, not the pipeline.")
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--work", default="outputs/ar_synth", help="scratch dir for renders and fits")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--keep", action="store_true", help="keep renders (default: delete each)")
    args = ap.parse_args()

    model_path = args.model or os.path.splitext(args.mesh)[0] + ".json"
    if not os.path.exists(model_path):
        print("no AR model JSON at %s" % model_path, file=sys.stderr)
        return 2

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    views = SC.rig_from_captures(args.rig, profile, board, det)
    tris = load_stl(args.mesh)
    axis_obj = long_axis_of(tris)
    x, y = (float(v) for v in args.xy.split(","))
    base, base_centroid = None, None
    if args.base_fit:
        src = args.base_fit
        if os.path.isdir(src):
            src = os.path.join(src, "fit.json")
        with open(src, "r", encoding="utf-8") as fh:
            base = json.load(fh)
        pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
        base_centroid = pts.mean(axis=0)
        print("sweeping yaw about the verified pose in %s" % src)

    os.makedirs(args.work, exist_ok=True)
    rows = []
    for yaw in parse_range(args.yaw):
        tag = "y%03d" % int(round(yaw))
        capdir = os.path.join(args.work, "cap_" + tag)
        fitdir = os.path.join(args.work, "fit_" + tag)
        os.makedirs(capdir, exist_ok=True)
        if base is not None:
            # Rotate the verified pose about the board normal, through the part's OWN centre so it
            # turns on the spot rather than swinging away across the bench.
            Rz, _ = cv2.Rodrigues(np.array([[0.0], [0.0], [np.radians(yaw)]]))
            R0, _ = cv2.Rodrigues(np.asarray(base["rvec"], np.float64).reshape(3, 1))
            t0 = np.asarray(base["tvec"], np.float64).reshape(3, 1)
            centre = (R0 @ base_centroid.reshape(3, 1)) + t0
            rvec_t, _ = cv2.Rodrigues(Rz @ R0)
            tvec_t = Rz @ (t0 - centre) + centre
        else:
            rvec_t, tvec_t = SC.seat_on_board(tris, x, y, yaw, args.roll)
        for v in views:
            img = SC.render(tris, rvec_t, tvec_t, v, board, profile)
            cv2.imwrite(os.path.join(capdir, "cap_%s_%s_synth.png" % (tag, v["tag"])), img)

        cmd = [sys.executable, "app/workers/run_ar_fit.py",
               "--captures", capdir, "--profile", args.profile,
               "--model", model_path, "--out", fitdir, "--coarse"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        fit_path = os.path.join(fitdir, "fit.json")
        if proc.returncode != 0 or not os.path.exists(fit_path):
            print("  %s  FIT FAILED (%s)" % (tag, (proc.stderr or "").strip()[-120:]))
            rows.append({"yaw_deg": yaw, "status": "fit_failed"})
            continue

        with open(fit_path, "r", encoding="utf-8") as fh:
            fit = json.load(fh)
        Rt, _ = cv2.Rodrigues(np.asarray(rvec_t, np.float64).reshape(3, 1))
        Rf, _ = cv2.Rodrigues(np.asarray(fit["rvec"], np.float64).reshape(3, 1))
        at, af = Rt @ axis_obj, Rf @ axis_obj
        # Same way round, or flipped? The sign of the dot product is the whole test.
        flipped = float(at @ af) < 0.0
        ang = float(np.degrees(np.arccos(np.clip(abs(float(at @ af)), 0, 1))))
        seat = float(np.linalg.norm(np.asarray(fit["tvec"])[:2] - np.asarray(tvec_t).ravel()[:2]))
        eo = end_on_angles(rvec_t, tvec_t, axis_obj, views)
        oc = fit.get("orientation_choice") or {}
        rows.append({
            "yaw_deg": round(yaw, 1),
            "min_end_on_deg": round(min(eo), 1),
            "orientation": "FLIPPED" if flipped else "ok",
            "axis_err_deg": round(ang, 2),
            "seating_err_mm": round(seat, 1),
            "margin_px": oc.get("margin_px"),
            "rms_px": fit.get("info", {}).get("rms_after_px"),
            "status": "ok",
        })
        print("  yaw %6.1f   end-on %5.1f   %-7s  axis %5.2f deg  seat %6.1f mm  margin %s"
              % (yaw, min(eo), rows[-1]["orientation"], ang, seat, oc.get("margin_px")))
        if not args.keep:
            shutil.rmtree(capdir, ignore_errors=True)

    good = [r for r in rows if r.get("status") == "ok"]
    if good:
        flips = [r for r in good if r["orientation"] == "FLIPPED"]
        print("\n=== %d trials: %d correct, %d flipped ==="
              % (len(good), len(good) - len(flips), len(flips)))
        if flips:
            print("  flipped at min-end-on angles: %s"
                  % ", ".join("%.0f" % r["min_end_on_deg"] for r in flips))
            print("  correct  at min-end-on angles: %s"
                  % ", ".join("%.0f" % r["min_end_on_deg"] for r in good
                              if r["orientation"] == "ok"))
        # Is the tie-break margin worth anything as confidence? On real captures it was
        # anti-correlated with being right, which is worse than useless.
        m_ok = [r["margin_px"] for r in good if r["orientation"] == "ok" and r["margin_px"]]
        m_no = [r["margin_px"] for r in flips if r["margin_px"]]
        if m_ok and m_no:
            print("  margin when CORRECT  mean %.2f px (n=%d)" % (np.mean(m_ok), len(m_ok)))
            print("  margin when FLIPPED  mean %.2f px (n=%d)" % (np.mean(m_no), len(m_no)))
            print("  -> margin %s separate the two"
                  % ("DOES" if np.mean(m_ok) > 2 * np.mean(m_no) else "does NOT"))

    if args.csv and rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("\nwrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
