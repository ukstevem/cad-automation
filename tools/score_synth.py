#!/usr/bin/env python3
"""
Score a fit of a synthetic capture against the pose it was rendered from.

Needs no images and makes no judgement calls: the true pose is known exactly, so the fitted long
axis either points with the truth or against it. That single sign test is the orientation verdict
that took plate-end marking and eyeballing to reach on real photographs.

Used first for the ANCHOR - render at poses verified on the real rig and check the harness
reproduces those outcomes, the near end-on FAILURE included. If it gets everything right, the
renderer is too kind to be evidence and no sweep run on it means anything.

    docker compose run --rm --no-deps api python tools/score_synth.py \
        outputs/ar_captures/synth_turn90 outputs/ar_fits/synth_turn90
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services.visibility import load_stl  # noqa: E402


def long_axis(tris: np.ndarray) -> np.ndarray:
    pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    d = pts - pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    return evecs[:, int(np.argmax(evals))]


def score(capdir: str, fitdir: str, mesh_dir: str = "outputs/ar_models") -> dict:
    with open(os.path.join(capdir, "truth.json"), "r", encoding="utf-8") as fh:
        truth = json.load(fh)
    with open(os.path.join(fitdir, "fit.json"), "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    tris = load_stl(os.path.join(mesh_dir, truth["mesh"]))
    e = long_axis(tris)

    Rt, _ = cv2.Rodrigues(np.asarray(truth["rvec"], np.float64).reshape(3, 1))
    Rf, _ = cv2.Rodrigues(np.asarray(fit["rvec"], np.float64).reshape(3, 1))
    at, af = Rt @ e, Rf @ e
    dot = float(at @ af)
    # Seating must be judged on where the PART is, not on tvec. tvec translates the object
    # ORIGIN, and flipping a part end-for-end in place swings that origin by up to a full part
    # length - reporting ~430mm of "seating error" for a part that has not moved.
    pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    cen = pts.mean(axis=0).reshape(3, 1)
    ct = (Rt @ cen).ravel() + np.asarray(truth["tvec"], np.float64).ravel()
    cf = (Rf @ cen).ravel() + np.asarray(fit["tvec"], np.float64).ravel()
    oc = fit.get("orientation_choice") or {}
    return {
        "case": os.path.basename(os.path.normpath(capdir)),
        "orientation": "FLIPPED" if dot < 0 else "ok",
        "axis_err_deg": round(float(np.degrees(np.arccos(np.clip(abs(dot), 0, 1)))), 2),
        "seating_err_mm": round(float(np.linalg.norm(cf[:2] - ct[:2])), 1),
        "margin_px": oc.get("margin_px"),
        "rms_px": fit.get("info", {}).get("rms_after_px"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="alternating <capture_dir> <fit_dir>")
    args = ap.parse_args()
    if len(args.pairs) % 2:
        print("give capture/fit directories in pairs", file=sys.stderr)
        return 2

    print("%-16s %-9s %10s %12s %10s %9s"
          % ("case", "orient", "axis err", "seating err", "margin", "rms"))
    rows = []
    for i in range(0, len(args.pairs), 2):
        try:
            r = score(args.pairs[i], args.pairs[i + 1])
        except (OSError, KeyError) as exc:
            print("  %-16s could not score (%s)" % (os.path.basename(args.pairs[i]), exc))
            continue
        rows.append(r)
        print("%-16s %-9s %8.2f d %9.1f mm %8s %9s"
              % (r["case"], r["orientation"], r["axis_err_deg"], r["seating_err_mm"],
                 r["margin_px"], r["rms_px"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
