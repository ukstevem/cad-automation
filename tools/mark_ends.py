#!/usr/bin/env python3
"""
Mark, on the photograph, which end of the part the solver thinks is which.

An overlay shows whether the model sits in the right PLACE; it is a poor way to judge whether it
sits the right way ROUND. The wireframe is dense at both ends and reads much the same either way -
during this project a fit was called wrong from its overlay and the projection test showed the
opposite. This draws the one thing that settles it: the model's distinctive end, projected at the
fitted pose, ringed on the image. If the ring lands on the real end plate, the orientation is
right; if it lands on the open end, it is not.

The plate end is identified from the model itself - the end decile holding more sampled points is
the solid one - so nothing is hand-labelled and this works on any part with an asymmetric end.

    docker compose run --rm --no-deps api python tools/mark_ends.py outputs/ar_fits/turn90
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

from app.services import charuco, image_edges, multiview_fit as MVF  # noqa: E402
from app.services.board_pose import charuco_board_pose  # noqa: E402


def end_points(model: dict):
    """The model's two extreme points along its own length, plus which is the solid end."""
    pts = np.vstack([np.asarray(e, np.float64).reshape(-1, 3) for e in model["edges"]])
    c = pts.mean(axis=0)
    d = pts - c
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    axis = evecs[:, int(np.argmax(evals))]
    t = d @ axis
    lo, hi = float(t.min()), float(t.max())
    band = 0.10 * (hi - lo)
    n_lo = int((t < lo + band).sum())
    n_hi = int((t > hi - band).sum())
    plate_at_hi = n_hi > n_lo
    plate = c + axis * (hi if plate_at_hi else lo)
    open_ = c + axis * (lo if plate_at_hi else hi)
    return plate, open_, n_lo, n_hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit_dir", help="an outputs/ar_fits/<name> directory")
    ap.add_argument("--captures", default=None, help="default: outputs/ar_captures/<name>")
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--model", default=None, help="default: taken from the fit's mesh name")
    args = ap.parse_args()

    name = os.path.basename(os.path.normpath(args.fit_dir))
    caps = args.captures or os.path.join("outputs/ar_captures", name)
    with open(os.path.join(args.fit_dir, "fit.json"), "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    model_path = args.model or os.path.join(
        "outputs/ar_models", os.path.basename(fit.get("mesh", "")).replace(".stl", ".json"))
    model = MVF.load_model(model_path)
    plate, open_, n_lo, n_hi = end_points(model)
    print("model end-decile counts: lo=%d hi=%d -> plate is the %s end"
          % (n_lo, n_hi, "hi" if n_hi > n_lo else "lo"))

    prof = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(prof["board"])
    det = charuco.make_detector(board)
    R, _ = cv2.Rodrigues(np.asarray(fit["rvec"], np.float64).reshape(3, 1))
    T = np.asarray(fit["tvec"], np.float64).reshape(3, 1)

    made = 0
    for path in sorted(glob.glob(os.path.join(caps, "*"))):
        if "overlay" in os.path.basename(path) or "endcheck" in os.path.basename(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cor, ids, _m, _i = charuco.detect_board_detailed(det, image_edges.to_gray(img))
        if ids is None or len(ids) < 6:
            print("  skip %s: board not found" % os.path.basename(path))
            continue
        rc, tc, _n = charuco_board_pose(cor, ids, board, prof["K"], prof["dist"])
        for label, P, col in (("CAD says PLATE END", plate, (60, 220, 60)),
                              ("CAD says OPEN END", open_, (60, 160, 255))):
            w = R @ P.reshape(3, 1) + T
            uv, _ = cv2.projectPoints(w.reshape(1, 1, 3), rc, tc, prof["K"], prof["dist"])
            x, y = uv.reshape(2)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 46, col, 5)
            cv2.circle(img, (x, y), 4, col, -1)
            ty = y - 62 if y > 200 else y + 88
            tx = max(8, min(x - 190, img.shape[1] - 470))
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 0), 7)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.05, col, 2)
        cv2.putText(img, name, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8)
        cv2.putText(img, name, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        out = os.path.join(args.fit_dir,
                           os.path.splitext(os.path.basename(path))[0] + "_endcheck.png")
        cv2.imwrite(out, img)
        print("  wrote %s" % out)
        made += 1
    if not made:
        print("no images marked", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
