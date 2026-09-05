#!/usr/bin/env python3
"""
Ask whether DEPTH separates two orientations that the silhouette cannot.

The pose fit sees outlines and edges. Measured on the Main Frame, rolling it 180 degrees about
its long axis changes only ~3% of that outline, which is why roll has never once been recovered -
the information is not in the sensor rather than not in the algorithm.

Depth should carry it, because rolling the part swaps which faces point at the camera even when
the outline barely moves. This measures that directly, and it does so BEFORE any stereo rig is
built: render the depth the cameras would see at each orientation and compare them, in
millimetres, over the pixels where both orientations put material.

The comparison that matters is the ratio. If depth differs by tens of millimetres where the
silhouette differs by a few percent, then a depth sensor resolves what two days of edge work
could not, and the hardware is worth mounting. If depth barely differs either, the part really is
ambiguous and no sensor will help - which would be worth knowing before buying anything.

    docker compose run --rm --no-deps api python tools/depth_discriminability.py \
        --mesh outputs/ar_models/mainframe_default_1to5.stl \
        --fit outputs/ar_fits/turn90 --rig outputs/ar_captures/turn90
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

from app.services import charuco, multiview_fit as MVF, visibility as VIS  # noqa: E402

import synth_capture as SC  # noqa: E402


def rotate_about_own_axis(tris, rvec, tvec, axis_name: str, degrees: float):
    """Rotate the posed object about one of its OWN principal axes, through its own centroid."""
    pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    c = pts.mean(axis=0)
    d = pts - c
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    evecs = evecs[:, np.argsort(evals)[::-1]]
    axis_obj = evecs[:, {"long": 0, "mid": 1, "short": 2}[axis_name]]

    R0, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t0 = np.asarray(tvec, np.float64).reshape(3, 1)
    axis_world = R0 @ axis_obj
    Rd, _ = cv2.Rodrigues(axis_world.reshape(3, 1) / np.linalg.norm(axis_world)
                          * np.radians(degrees))
    centre = R0 @ c.reshape(3, 1) + t0
    rvec2, _ = cv2.Rodrigues(Rd @ R0)
    tvec2 = Rd @ (t0 - centre) + centre
    return rvec2, tvec2


def compare(tris, poseA, poseB, view, downscale=2):
    """Depth and silhouette difference between two poses, from one camera."""
    dA, sc = VIS.depth_buffer(tris, poseA[0], poseA[1], view, downscale=downscale)
    dB, _ = VIS.depth_buffer(tris, poseB[0], poseB[1], view, downscale=downscale)
    okA, okB = dA < VIS.FAR / 2, dB < VIS.FAR / 2
    both = okA & okB
    either = okA | okB
    if either.sum() == 0:
        return None
    # Silhouette difference: pixels covered by one orientation and not the other, as a fraction
    # of the area either covers. This is what an edge/outline cost can see.
    sil_diff = float((either & ~both).sum()) / float(either.sum())
    # Depth difference: only where BOTH put material, so it isolates "the surface moved toward or
    # away from the camera" from "the outline changed".
    dd = np.abs(dA[both] - dB[both]) if both.sum() else np.array([0.0])
    return {
        "silhouette_diff_frac": round(sil_diff, 4),
        "depth_median_mm": round(float(np.median(dd)), 2),
        "depth_p90_mm": round(float(np.percentile(dd, 90)), 2),
        "depth_max_mm": round(float(dd.max()), 2),
        "depth_over_2mm_frac": round(float((dd > 2.0).mean()), 4),
        "overlap_px": int(both.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--fit", required=True, help="a verified fit dir, giving a realistic pose")
    ap.add_argument("--rig", required=True, help="capture dir, for real camera poses")
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--downscale", type=int, default=2)
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    views = SC.rig_from_captures(args.rig, profile, board, det)
    # rig_from_captures returns poses only; the depth buffer needs the intrinsics too.
    for v in views:
        v["K"], v["dist"] = profile["K"], profile["dist"]
    tris = VIS.load_stl(args.mesh)

    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    pose0 = (np.asarray(fit["rvec"], np.float64).reshape(3, 1),
             np.asarray(fit["tvec"], np.float64).reshape(3, 1))

    print("Comparing orientations of %s at the verified pose in %s\n"
          % (os.path.basename(args.mesh), args.fit))
    print("%-26s %-10s  %10s %10s %10s %10s"
          % ("transformation", "camera", "silhouette", "depth med", "depth p90", ">2mm"))

    cases = [
        ("roll 180 about long axis", "long", 180.0),
        ("flip 180 end-for-end", "short", 180.0),
        ("roll 90 about long axis", "long", 90.0),
    ]
    for label, axis, deg in cases:
        for v in views:
            poseB = rotate_about_own_axis(tris, pose0[0], pose0[1], axis, deg)
            r = compare(tris, pose0, poseB, v, downscale=args.downscale)
            if r is None:
                continue
            print("%-26s %-10s  %9.1f%% %9.1f mm %9.1f mm %9.1f%%"
                  % (label, v["tag"][:10], 100 * r["silhouette_diff_frac"],
                     r["depth_median_mm"], r["depth_p90_mm"],
                     100 * r["depth_over_2mm_frac"]))
        print()

    print("READ IT AS A RATIO. A large depth difference where the silhouette barely changes means")
    print("the orientation IS distinguishable, just not by an outline - which is the case for a")
    print("depth sensor and against more edge work. Both small would mean the part is genuinely")
    print("ambiguous and no sensor helps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
