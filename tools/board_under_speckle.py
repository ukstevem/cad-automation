#!/usr/bin/env python3
"""
Does projector speckle degrade the BOARD POSE, even where detection still succeeds?

The bench answered the easier half: with the board fully inside the projected pattern, all 27
markers and all 40 corners were still found. But detection surviving is not the same as the pose
being as good. Speckle lands on the black/white boundaries the corner refinement measures, and a
corner that shifts by a fraction of a pixel moves everything downstream - the board IS the world
frame, so its pose error propagates into every part pose derived from it.

The rig cannot answer this, because a photograph has no ground truth: comparing a speckled pose
against an unspeckled one only shows they differ, not which is wrong. In simulation the camera
pose is known exactly, so both can be measured against the truth.

Rendered scene-wide - the board gets the pattern too, as a real projector would.

    docker compose run --rm --no-deps api python tools/board_under_speckle.py \\
        --rig outputs/ar_captures/turn90 --mesh outputs/ar_models/mainframe_default_1to5.stl \\
        --fit outputs/ar_fits/turn90
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import charuco, image_edges, multiview_fit as MVF, visibility as VIS  # noqa: E402
from app.services.board_pose import charuco_board_pose  # noqa: E402

import synth_capture as SC  # noqa: E402
from stereo_preview import projector_pose, speckle_scene  # noqa: E402


def pose_error(rv_true, tv_true, rv, tv):
    """Rotation error in degrees and translation error in mm."""
    Rt, _ = cv2.Rodrigues(np.asarray(rv_true, np.float64).reshape(3, 1))
    Rm, _ = cv2.Rodrigues(np.asarray(rv, np.float64).reshape(3, 1))
    dr, _ = cv2.Rodrigues(Rm @ Rt.T)
    ang = float(np.degrees(np.linalg.norm(dr)))
    dt = float(np.linalg.norm(np.asarray(tv).ravel() - np.asarray(tv_true).ravel()))
    return ang, dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--grain", type=float, default=4.0,
                    help="blob size in mm; the bench measured 3.95 on the real projector")
    ap.add_argument("--strengths", default="0,0.3,0.55,0.8",
                    help="speckle contrast levels to sweep; 0 is the unspeckled control")
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    views = SC.rig_from_captures(args.rig, profile, board, det)
    for v in views:
        v["K"], v["dist"] = profile["K"], profile["dist"]
    proj = projector_pose(views, profile, board=board)

    tris = VIS.load_stl(args.mesh)
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)

    print("The rendered camera pose IS the ground truth - the board is drawn from it, so any")
    print("difference in the solved pose is error introduced by the speckle.")
    print("")
    print("%-8s %-10s %8s %9s %12s %12s"
          % ("camera", "speckle", "markers", "corners", "rot err deg", "trans err mm"))

    for v in views:
        for st in [float(x) for x in args.strengths.split(",")]:
            img = SC.render(tris, rvec, tvec, v, board, profile, shadow=0.35, noise=1.5)
            if st > 0:
                img = speckle_scene(img, tris, rvec, tvec, v, profile, grain_mm=args.grain,
                                    strength=st, proj=proj)
            g = image_edges.to_gray(img)
            cor, ids, _mc, mi = charuco.detect_board_detailed(det, g)
            nm = 0 if mi is None else len(mi)
            nc = 0 if ids is None else len(ids)
            if nc < 6:
                print("%-8s %-10s %8d %9d %12s %12s"
                      % (v["tag"][:8], ("%.2f" % st) if st else "none", nm, nc, "-", "-"))
                continue
            rc, tc, _n = charuco_board_pose(cor, ids, board, profile["K"], profile["dist"])
            ang, dt = pose_error(v["rvec_cam"], v["tvec_cam"], rc, tc)
            print("%-8s %-10s %8d %9d %12.4f %12.3f"
                  % (v["tag"][:8], ("%.2f" % st) if st else "none", nm, nc, ang, dt))
        print("")

    print("A pose error that rises with speckle strength means the two-shot protocol earns its")
    print("keep. Flat across the sweep means the board can simply be shot with the projector on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
