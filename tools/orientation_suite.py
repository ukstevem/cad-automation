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
from real_stereo import rectified_cloud  # noqa: E402
from depth_discriminability import rotate_about_own_axis  # noqa: E402

ROLLS = [0.0, 90.0, 180.0, 270.0]


def self_symmetries(tris, rolls, tol_frac: float = 0.01):
    """
    Which of the candidate rolls map the part ONTO ITSELF, and are therefore meaningless to ask.

    A square-section bar rolled 90 degrees is the same object in the same place. No sensor can
    distinguish that, and neither counting it correct nor counting it wrong says anything about
    the method - the question simply has no answer. Discovered the hard way: a plain bar scored
    100% because every hypothesis tied and max() happened to return the true one first.

    Compared as a point cloud: roll the sampled points about the long axis and measure how far
    each lands from the original shape, in units of the part's own length.
    """
    from scipy.spatial import cKDTree

    pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    if len(pts) > 4000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
    c = pts.mean(axis=0)
    d = pts - c
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    axis = evecs[:, int(np.argmax(evals))]
    tol = tol_frac * float(np.ptp(d @ axis))
    tree = cKDTree(pts)

    out = {}
    for deg in rolls:
        if deg == 0.0:
            out[deg] = True
            continue
        rv = axis.reshape(3, 1) * np.radians(deg)
        R, _ = cv2.Rodrigues(rv)
        moved = (R @ d.T).T + c
        dist, _ = tree.query(moved)
        out[deg] = bool(np.median(dist) < tol)      # True == indistinguishable from the original
    return out


def pose_at(tris, r0, t0, centroid, yaw_deg, roll_deg, seat=None):
    """The true pose: verified rig pose, turned by yaw on the bench, then rolled on its own axis."""
    if seat is not None:
        rvec, tvec = SC.seat_on_board(tris, seat[0], seat[1], yaw_deg, 0.0)
        if roll_deg:
            rvec, tvec = rotate_about_own_axis(tris, rvec, tvec, "long", roll_deg)
        return rvec, tvec
    Rz, _ = cv2.Rodrigues(np.array([[0.0], [0.0], [np.radians(yaw_deg)]]))
    R0, _ = cv2.Rodrigues(r0)
    centre = R0 @ centroid + t0
    rvec, _ = cv2.Rodrigues(Rz @ R0)
    tvec = Rz @ (t0 - centre) + centre
    if roll_deg:
        rvec, tvec = rotate_about_own_axis(tris, rvec, tvec, "long", roll_deg)
    return rvec, tvec


def decide(tris, rvec, tvec, left, right, board, profile, baseline, grain, lit, proj,
           distinct=None, spec=0.0, rig_pair=False):
    """One cell: render, match, and score the four roll hypotheses. Returns (winner, scores, n)."""
    imgs = []
    for v in (left, right):
        im = SC.render(tris, rvec, tvec, v, board, profile, shadow=0.35, noise=1.5,
                       specular=spec)
        if lit:
            im = speckle(im, tris, rvec, tvec, v, profile, grain_mm=grain, proj=proj)
        imgs.append(im)

    if rig_pair:
        # Two real cameras aimed inward are NOT a rectified pair; go through stereoRectify, the
        # same path the real-capture tool uses. The fast formula would silently return nonsense.
        Ra, _ = cv2.Rodrigues(np.asarray(left["rvec_cam"], np.float64).reshape(3, 1))
        d, _ = VIS.depth_buffer(tris, rvec, tvec, left, downscale=1)
        zh = d[d < VIS.FAR / 2]
        K = np.asarray(profile["K"], np.float64).reshape(3, 3)
        dc = np.asarray(profile["dist"], np.float64).reshape(-1, 1)
        cc, _rec, n = rectified_cloud(imgs[0], imgs[1],
                                      left["rvec_cam"], left["tvec_cam"],
                                      right["rvec_cam"], right["tvec_cam"],
                                      K, dc, K, dc, z_hint=zh if len(zh) else None)
        if cc is None or len(cc) < 500:
            return None, {}, 0
        cloud = (Ra.T @ (cc.T - np.asarray(left["tvec_cam"], np.float64).reshape(3, 1))).T
        scores = {}
        for dd in (distinct or ROLLS):
            if dd == 0.0:
                rv, tv = rvec, tvec
            else:
                rv, tv = rotate_about_own_axis(tris, rvec, tvec, "long", dd)
            _m, _mn, _n, near = score_against(tris, rv, tv, cloud)
            scores[dd] = near
        return max(scores, key=scores.get), scores, len(cloud)

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
    for d in (distinct or ROLLS):
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
    ap.add_argument("--seat", default=None, metavar="X,Y",
                    help="seat the part's CENTROID here (board mm) instead of reusing the fit's "
                         "pose. REQUIRED for any mesh other than the one the fit was solved for: "
                         "a fit positions the object ORIGIN, and every mesh has its geometry "
                         "somewhere different relative to its origin, so borrowing the pose puts "
                         "a foreign part off-frame - where the segmenter grabs the ChArUco board "
                         "instead and the cloud is of the board, not the part.")
    ap.add_argument("--specular", type=float, default=0.0,
                    help="0 matches the matte 1:5 print; ~0.5 approximates bare steel")
    ap.add_argument("--rig-pair", action="store_true",
                    help="use the rig's two real cameras as the stereo pair instead of a virtual "
                         "narrow-baseline one. Wide separation triangulates well and matches "
                         "badly; this measures which effect wins.")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    views = SC.rig_from_captures(args.rig, profile, board, det)
    for v in views:
        v["K"], v["dist"] = profile["K"], profile["dist"]
    if args.rig_pair:
        # Use the rig's OWN two cameras as the stereo pair, rather than a virtual narrow offset.
        # This is the honest test of a wide-baseline pair: good for pose, questionable for
        # correlation matching, and the difference is exactly what needs measuring.
        left, right = views[0], views[1]
        Ra, _ = cv2.Rodrigues(np.asarray(left["rvec_cam"], np.float64).reshape(3, 1))
        Rb, _ = cv2.Rodrigues(np.asarray(right["rvec_cam"], np.float64).reshape(3, 1))
        ca = -Ra.T @ np.asarray(left["tvec_cam"], np.float64).reshape(3, 1)
        cb = -Rb.T @ np.asarray(right["tvec_cam"], np.float64).reshape(3, 1)
        args.baseline = float(np.linalg.norm(ca - cb))
        za = (Ra.T @ np.array([[0.], [0.], [1.]])).ravel()
        zb = (Rb.T @ np.array([[0.], [0.], [1.]])).ravel()
        sep = np.degrees(np.arccos(np.clip(abs(float(za @ zb)), 0, 1)))
        print("using the RIG pair: baseline %.0f mm, optical axes %.1f deg apart"
              % (args.baseline, sep))
    else:
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

    seat = None
    if args.seat:
        seat = [float(v) for v in args.seat.split(",")]
        print("seating the part's centroid at (%.0f, %.0f) on the board" % (seat[0], seat[1]))
    yaws = [y for y in np.arange(0.0, 360.0, args.yaw_step)]
    rolls = [float(r) for r in args.rolls.split(",")]
    # Drop hypotheses that are self-symmetries: they are the same object in the same place, so
    # scoring them says nothing about the method either way.
    sym = self_symmetries(tris, rolls)
    distinct = [0.0] + [d for d in rolls if d != 0.0 and not sym[d]]
    dropped = [d for d in rolls if d != 0.0 and sym[d]]
    if dropped:
        print("self-symmetric rolls, excluded as unanswerable: %s"
              % ", ".join("%.0f deg" % d for d in dropped))
    if len(distinct) < 2:
        print("this part is rotationally symmetric about its long axis - roll is UNDEFINED,")
        print("not merely hard. Nothing to test.")
        return 0
    print("hypotheses actually distinguishable: %s"
          % ", ".join("%.0f" % d for d in distinct))
    lights = [s.strip() == "on" for s in args.light.split(",")]
    print("suite: %d yaw x %d roll x %d lighting = %d captures"
          % (len(yaws), len(rolls), len(lights), len(yaws) * len(rolls) * len(lights)))

    rows = []
    for lit in lights:
        tag = "projector ON " if lit else "projector OFF"
        ok = tot = amb = 0
        for roll in rolls:
            for yaw in yaws:
                rvec, tvec = pose_at(tris, r0, t0, centroid, yaw, roll, seat)
                win, scores, n = decide(tris, rvec, tvec, left, right, board, profile,
                                        args.baseline, args.grain, lit, proj, distinct,
                                        args.specular, args.rig_pair)
                if win is None:
                    rows.append({"light": "on" if lit else "off", "roll": roll, "yaw": yaw,
                                 "points": 0, "winner": "", "correct": "", "margin": ""})
                    continue
                tot += 1
                others = [v for k, v in scores.items() if k != 0.0]
                margin = scores[0.0] - max(others) if others else 0.0
                # A tie is not a win. Requiring a real margin stops the first-key-wins behaviour
                # of max() from silently crediting hypotheses the data cannot separate.
                good = (win == 0.0) and margin > 1e-3
                ok += int(good)
                if abs(margin) <= 1e-3:
                    amb += 1
                rows.append({"light": "on" if lit else "off", "roll": roll, "yaw": yaw,
                             "points": n, "winner": win, "correct": int(good),
                             "margin": round(margin, 4)})
        rate = 100.0 * ok / tot if tot else float("nan")
        print("  %s   %3d/%-3d correct (%.0f%%)   %d too close to call"
              % (tag, ok, tot, rate, amb))

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
