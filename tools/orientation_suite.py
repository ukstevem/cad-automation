#!/usr/bin/env python3
"""
The orientation test suite: a part turned through 360 degrees, at four rolls, projector on and off.

Every orientation claim on this project so far has come from a handful of trials on one part, and
each one has been overturned by a wider test - 12/12 on the Main Frame became 43% across 68 parts,
and an end-on hypothesis filed as P1 died to two sweeps. So the standard is now a matrix rather
than a demonstration.

The matrix, per part:

    yaw    0..330 in steps        - the part turned on the bench, which is what an operator varies
    roll   0, 90, 180, 270        - which face is up; the axis NO image method has ever recovered
    light  projector off / on     - the sensor question, held against everything else fixed

Each cell is an independent capture-and-decide: render the rig's stereo pair at that true pose,
match, build a cloud, and score the four roll hypotheses against the CAD. It scores CORRECT only
if the true roll wins. Yaw is not scored - it is the nuisance variable being swept over, there to
stop a result resting on one lucky viewing angle.

The projector-off half is the control. If the projector matters, off should collapse toward chance
(25% with four hypotheses) while on stays high; if both are high, the speckle is not doing the
work and the hardware is not needed.

    docker compose run --rm --no-deps api python tools/orientation_suite.py \
        --mesh outputs/ar_models/mainframe_default_1to5.stl \
        --fit outputs/ar_fits/turn90 --rig outputs/ar_captures/turn90 \
        --yaw-step 45 --csv outputs/ar_fits/suite.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import charuco, image_edges, multiview_fit as MVF, visibility as VIS  # noqa: E402

import synth_capture as SC  # noqa: E402
from stereo_preview import offset_camera, projector_pose, speckle  # noqa: E402
from stereo_roll_test import disparity_range, stereo_cloud, to_world, score_against  # noqa: E402
from depth_discriminability import rotate_about_own_axis  # noqa: E402

ROLLS = [0.0, 90.0, 180.0, 270.0]


def pose_at(tris, r0, t0, centroid, yaw_deg, roll_deg):
    """The true pose: verified rig pose, turned by yaw on the bench, then rolled on its own axis."""
    Rz, _ = cv2.Rodrigues(np.array([[0.0], [0.0], [np.radians(yaw_deg)]]))
    R0, _ = cv2.Rodrigues(r0)
    centre = R0 @ centroid + t0
    rvec, _ = cv2.Rodrigues(Rz @ R0)
    tvec = Rz @ (t0 - centre) + centre
    if roll_deg:
        rvec, tvec = rotate_about_own_axis(tris, rvec, tvec, "long", roll_deg)
    return rvec, tvec


def decide(tris, rvec, tvec, left, right, board, profile, baseline, grain, lit, proj):
    """One cell: render, match, and score the four roll hypotheses. Returns (winner, scores, n)."""
    imgs = []
    for v in (left, right):
        im = SC.render(tris, rvec, tvec, v, board, profile, shadow=0.35, noise=1.5)
        if lit:
            im = speckle(im, tris, rvec, tvec, v, profile, grain_mm=grain, proj=proj)
        imgs.append(im)

    lo, span = disparity_range(tris, rvec, tvec, left, profile, baseline)
    cloud_cam, valid, _disp = stereo_cloud(imgs[0], imgs[1], left, profile, baseline,
                                           min_disp=lo, num_disp=span)
    if len(cloud_cam) < 500:
        return None, {}, 0
    subj = image_edges.segment_subject(imgs[0], grow_px=0)
    ys, xs = np.nonzero(valid)
    on = np.ones(len(ys), bool) if subj is None else (subj[ys, xs] > 0)
    if on.sum() < 500:
        return None, {}, 0
    cloud = to_world(cloud_cam[on], left)

    scores = {}
    for d in ROLLS:
        if d == 0.0:
            rv, tv = rvec, tvec
        else:
            rv, tv = rotate_about_own_axis(tris, rvec, tvec, "long", d)
        _m, _mn, _n, near = score_against(tris, rv, tv, cloud)
        scores[d] = near
    return max(scores, key=scores.get), scores, len(cloud)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--baseline", type=float, default=60.0)
    ap.add_argument("--grain", type=float, default=3.0)
    ap.add_argument("--yaw-step", type=float, default=45.0)
    ap.add_argument("--rolls", default="0,90,180,270")
    ap.add_argument("--light", default="on,off", help="which lighting conditions to run")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    views = SC.rig_from_captures(args.rig, profile, board, det)
    for v in views:
        v["K"], v["dist"] = profile["K"], profile["dist"]
    left, right = views[0], offset_camera(views[0], args.baseline)
    proj = projector_pose(views, profile, board=board)

    tris = VIS.load_stl(args.mesh)
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    r0 = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    t0 = np.asarray(fit["tvec"], np.float64).reshape(3, 1)
    centroid = np.vstack([tris.reshape(-1, 3),
                          tris.mean(axis=1)]).mean(axis=0).reshape(3, 1)

    yaws = [y for y in np.arange(0.0, 360.0, args.yaw_step)]
    rolls = [float(r) for r in args.rolls.split(",")]
    lights = [s.strip() == "on" for s in args.light.split(",")]
    print("suite: %d yaw x %d roll x %d lighting = %d captures"
          % (len(yaws), len(rolls), len(lights), len(yaws) * len(rolls) * len(lights)))

    rows = []
    for lit in lights:
        tag = "projector ON " if lit else "projector OFF"
        ok = tot = 0
        for roll in rolls:
            for yaw in yaws:
                rvec, tvec = pose_at(tris, r0, t0, centroid, yaw, roll)
                win, scores, n = decide(tris, rvec, tvec, left, right, board, profile,
                                        args.baseline, args.grain, lit, proj)
                if win is None:
                    rows.append({"light": "on" if lit else "off", "roll": roll, "yaw": yaw,
                                 "points": 0, "winner": "", "correct": "", "margin": ""})
                    continue
                tot += 1
                good = (win == 0.0)          # hypothesis 0 = the pose as rendered = correct
                ok += int(good)
                others = [v for k, v in scores.items() if k != 0.0]
                margin = scores[0.0] - max(others) if others else 0.0
                rows.append({"light": "on" if lit else "off", "roll": roll, "yaw": yaw,
                             "points": n, "winner": win, "correct": int(good),
                             "margin": round(margin, 4)})
        rate = 100.0 * ok / tot if tot else float("nan")
        print("  %s   %3d/%-3d correct  (%.0f%%)" % (tag, ok, tot, rate))

    print("")
    print("chance is 25%% with four hypotheses.")
    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
