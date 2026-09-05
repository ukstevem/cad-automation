#!/usr/bin/env python3
"""
Does a stereo cloud pick the right orientation RELIABLY, or did it get lucky once?

A single trial showed the true pose beating a 180-degree roll by better than 2:1 on the fraction
of cloud explained. That is one trial. The edge method was held to 68 parts and 271 trials before
anyone believed a number from it, and this deserves the same standard - especially since the edge
method also looked convincing on its first part (12/12 on the Main Frame) and then turned out to
be at chance on everything else.

So: sweep the part through orientations, and at each one build a stereo cloud from scratch and ask
which of the four seatings it scores best. The answer is a win rate, comparable directly with the
44% (individual parts) and 64% (weldments) the edge method achieved.

Scoring is the fraction of cloud points within 5 mm of the hypothesised CAD surface. That beat the
median in testing, because a wrong hypothesis makes FEWER, LARGER errors rather than uniformly
worse ones - so counting explained surface separates better than averaging error.

    docker compose run --rm --no-deps api python tools/stereo_orientation_sweep.py \
        --mesh outputs/ar_models/mainframe_default_1to5.stl \
        --fit outputs/ar_fits/turn90 --rig outputs/ar_captures/turn90 --yaw 0:150:30
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
from stereo_preview import offset_camera, speckle  # noqa: E402
from stereo_roll_test import (disparity_range, stereo_cloud, to_world,  # noqa: E402
                              score_against)
from depth_discriminability import rotate_about_own_axis  # noqa: E402


HYPOTHESES = [
    ("true", None, 0.0),
    ("roll 180", "long", 180.0),
    ("roll  90", "long", 90.0),
    ("roll 270", "long", 270.0),
    ("end-for-end", "short", 180.0),
]


def trial(tris, rvec, tvec, left, right, board, profile, baseline, grain, speckled=True):
    """One capture-and-decide cycle. Returns the scores per hypothesis, or None if it failed."""
    imgs = []
    for v in (left, right):
        im = SC.render(tris, rvec, tvec, v, board, profile, shadow=0.35, noise=1.5)
        if speckled:
            im = speckle(im, tris, rvec, tvec, v, profile, grain_mm=grain)
        imgs.append(im)

    lo, span = disparity_range(tris, rvec, tvec, left, profile, baseline)
    cloud_cam, valid, disp = stereo_cloud(imgs[0], imgs[1], left, profile, baseline,
                                          min_disp=lo, num_disp=span)
    if len(cloud_cam) < 500:
        return None
    # Subject mask from the IMAGE, never from the CAD - using CAD coverage would favour whichever
    # pose was used to select the points, which is the hypothesis under test.
    subj = image_edges.segment_subject(imgs[0], grow_px=0)
    ys, xs = np.nonzero(valid)
    on_part = np.ones(len(ys), bool) if subj is None else (subj[ys, xs] > 0)
    if on_part.sum() < 500:
        return None
    cloud = to_world(cloud_cam[on_part], left)

    out = {}
    for name, axis, deg in HYPOTHESES:
        if axis is None:
            rv, tv = rvec, tvec
        else:
            rv, tv = rotate_about_own_axis(tris, rvec, tvec, axis, deg)
        _med, _mean, _n, near = score_against(tris, rv, tv, cloud)
        out[name] = near
    return out, len(cloud)


def parse_range(spec):
    if ":" in spec:
        lo, hi, step = (float(v) for v in spec.split(":"))
        return [lo + i * step for i in range(int(round((hi - lo) / step)) + 1)]
    return [float(v) for v in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--baseline", type=float, default=60.0)
    ap.add_argument("--grain", type=float, default=2.0)
    ap.add_argument("--yaw", default="0:150:30")
    ap.add_argument("--no-speckle", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    views = SC.rig_from_captures(args.rig, profile, board, det)
    for v in views:
        v["K"], v["dist"] = profile["K"], profile["dist"]
    left = views[0]
    right = offset_camera(left, args.baseline)

    tris = VIS.load_stl(args.mesh)
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    r0 = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    t0 = np.asarray(fit["tvec"], np.float64).reshape(3, 1)
    pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    centroid = pts.mean(axis=0).reshape(3, 1)

    print("%-7s %8s  %s" % ("yaw", "points", "  ".join("%-11s" % h[0] for h in HYPOTHESES)))
    rows, wins, n = [], 0, 0
    for yaw in parse_range(args.yaw):
        # Turn the verified pose about the board normal, through the part's own centre.
        Rz, _ = cv2.Rodrigues(np.array([[0.0], [0.0], [np.radians(yaw)]]))
        R0, _ = cv2.Rodrigues(r0)
        centre = R0 @ centroid + t0
        rvec, _ = cv2.Rodrigues(Rz @ R0)
        tvec = Rz @ (t0 - centre) + centre

        res = trial(tris, rvec, tvec, left, right, board, profile,
                    args.baseline, args.grain, speckled=not args.no_speckle)
        if res is None:
            print("%-7.0f %8s  matching failed" % (yaw, "-"))
            continue
        scores, npts = res
        best = max(scores, key=scores.get)
        ok = best == "true"
        wins += int(ok)
        n += 1
        print("%-7.0f %8d  %s  -> %s" % (
            yaw, npts, "  ".join("%10.1f%%" % (100 * scores[h[0]]) for h in HYPOTHESES),
            "OK" if ok else "WRONG (%s)" % best))
        row = {"yaw": yaw, "points": npts, "winner": best, "correct": int(ok)}
        row.update({h[0]: round(scores[h[0]], 4) for h in HYPOTHESES})
        rows.append(row)

    if n:
        print("")
        print("=== orientation chosen correctly in %d of %d trials (%.0f%%) ===" % (wins, n, 100 * wins / n))
        print("    edge method, for comparison: 44%% on individual parts, 64%% on weldments,")
        print("    and roll specifically was NEVER recovered - 0 for 4 physical, 0 across sweeps.")
    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
