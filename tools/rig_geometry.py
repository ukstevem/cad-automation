#!/usr/bin/env python3
"""
Measure the actual rig geometry from simultaneous board captures.

The build spec asks for cameras ~600-630 mm from the object, ~40 degrees apart. Those are
targets set on paper; this reports what the rig is really doing, in millimetres, from the
images themselves.

Every ``webcam_capture.py shot`` writes both cameras at once, so a calibration set is already a
set of simultaneous stereo pairs. For each pair we solve the board pose in each camera, which
gives the camera-to-camera transform for that pair; averaging over all of them yields the rig
extrinsics plus a spread figure.

    docker compose run --rm --no-deps api python tools/rig_geometry.py \\
        outputs/ar_captures/cal_20260904 \\
        --profile-a outputs/calibration/RigCam_52FD1B1F.json \\
        --profile-b outputs/calibration/RigCam_B68DE55F.json

**The spread is the important number, not the average.** The cameras did not move between poses,
so their relative pose must come out the same every time. If it does not, either the rig is
flexing, a camera is loose, or the board pose is poorly conditioned — and every later
measurement inherits that. A rigid, well-conditioned rig should land inside a degree or two.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services import charuco  # noqa: E402
from app.services.board_pose import average_relative_pose, charuco_board_pose  # noqa: E402
from app.services.multiview_fit import load_profile  # noqa: E402

NAME_RE = re.compile(r"^(?P<label>.+)_(?P<tag>[A-Za-z0-9]+)_(?P<stamp>\d{8}_\d{6})\.[^.]+$")


def index_pairs(directory: str):
    """Group images by capture timestamp — that is what makes them simultaneous."""
    by_stamp = defaultdict(dict)
    for fn in sorted(os.listdir(directory)):
        m = NAME_RE.match(fn)
        if not m:
            continue
        by_stamp[m.group("stamp")][m.group("tag")] = os.path.join(directory, fn)
    return by_stamp


def solve(path, profile, board, detector, min_corners):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if (w, h) != tuple(profile["image_size"]):
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _mc, _mi = detector.detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    try:
        rvec, tvec, n = charuco_board_pose(corners, ids, board, profile["K"], profile["dist"],
                                           min_corners=min_corners)
    except Exception:
        return None
    return rvec, tvec, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--profile-a", required=True)
    ap.add_argument("--profile-b", required=True)
    ap.add_argument("--min-corners", type=int, default=8)
    ap.add_argument("--part-mm", type=float, default=433.0,
                    help="longest dimension of the part, for the frame-fill figure "
                         "(default 433 = Main Frame_Default at 1:5)")
    args = ap.parse_args()

    pa, pb = load_profile(args.profile_a), load_profile(args.profile_b)
    board = charuco.build_board_from_config(pa["board"])
    detector = charuco.make_detector(board)
    tag_a = os.path.splitext(os.path.basename(args.profile_a))[0].split("_")[-1]
    tag_b = os.path.splitext(os.path.basename(args.profile_b))[0].split("_")[-1]

    print(f"board   {pa['board']}")
    print(f"A = {tag_a}   B = {tag_b}\n")

    pairs, dist_a, dist_b = [], [], []
    for stamp, files in sorted(index_pairs(args.directory).items()):
        if tag_a not in files or tag_b not in files:
            continue
        ra = solve(files[tag_a], pa, board, detector, args.min_corners)
        rb = solve(files[tag_b], pb, board, detector, args.min_corners)
        if ra is None or rb is None:
            print(f"  {stamp}  skipped ({'A' if ra is None else 'B'} did not solve)")
            continue
        pairs.append(((ra[0], ra[1]), (rb[0], rb[1])))
        dist_a.append(float(np.linalg.norm(ra[1])))
        dist_b.append(float(np.linalg.norm(rb[1])))
        print(f"  {stamp}  A {ra[2]:2d} corners @ {dist_a[-1]:6.1f}mm   "
              f"B {rb[2]:2d} corners @ {dist_b[-1]:6.1f}mm")

    if not pairs:
        sys.exit("\nno usable simultaneous pairs")

    # Ground sample distance at the measured standoff: mm of subject per pixel, = Z / f.
    # This is the number that decides what the rig can actually resolve, and it is only
    # meaningful once the standoff is real rather than assumed.
    print(f"\n=== RESOLUTION AT THE MEASURED STANDOFF ===")
    for label, prof, dists in (("A", pa, dist_a), ("B", pb, dist_b)):
        z = float(np.mean(dists))
        fx = float(prof["K"][0, 0])
        gsd = z / fx
        print(f"  {label}: standoff {z:6.1f} mm  ->  GSD {gsd:.3f} mm/px"
              f"   ({args.part_mm:.0f}mm part spans {args.part_mm / gsd:.0f} px, "
              f"{100 * (args.part_mm / gsd) / prof['image_size'][0]:.0f}% of frame width)")

    if len(pairs) < 2:
        print("\n(Only one pair: reporting standoff only. Relative-pose averaging and its "
              "consistency figure need several pairs — use a calibration set for that.)")
        return 0

    R_ab, t_ab, info = average_relative_pose(pairs)

    # Angle between the two optical axes. Each camera looks down its own +Z; expressing B's
    # axis in A's frame and comparing with A's own gives the convergence angle — the number the
    # build spec calls "camera-to-camera separation ~40 degrees".
    axis_a = np.array([0.0, 0.0, 1.0])
    axis_b_in_a = R_ab.T @ axis_a
    sep_deg = float(np.degrees(np.arccos(np.clip(float(np.dot(axis_a, axis_b_in_a)), -1, 1))))

    print(f"\n=== RIG GEOMETRY (from {len(pairs)} simultaneous pairs) ===")
    print(f"  baseline (camera centres apart)   {info['baseline_mm']:.1f} mm")
    print(f"  convergence angle between axes    {sep_deg:.1f} deg")
    print(f"  standoff A -> board               {np.mean(dist_a):.0f} mm  "
          f"(range {min(dist_a):.0f}-{max(dist_a):.0f})")
    print(f"  standoff B -> board               {np.mean(dist_b):.0f} mm  "
          f"(range {min(dist_b):.0f}-{max(dist_b):.0f})")
    print(f"\n  CONSISTENCY (this is the number that matters)")
    print(f"  rotation spread across pairs      {info['rot_spread_deg']:.2f} deg")
    print(f"  translation spread across pairs   {info['trans_spread_mm']:.1f} mm")

    print("\n=== vs build spec ===")
    verdict = "OK" if 30 <= sep_deg <= 60 else ("LOW" if sep_deg < 30 else "HIGH")
    print(f"  camera separation          {sep_deg:7.1f} deg target 30-60 -> {verdict}")
    print(f"  baseline                   {info['baseline_mm']:7.1f} mm  spec builds for ~420mm")
    print("\n  NOTE: the standoff figures above are NOT the working distance. A calibration set")
    print("  deliberately moves the board all over the volume, so their mean says nothing about")
    print("  where the part actually sits. To measure the real standoff, put the board flat in")
    print("  the part's position and take ONE pair, then re-run this on that directory.")
    if info["rot_spread_deg"] > 2.0:
        print("\n  WARNING: the relative pose is not consistent across pairs. The cameras did "
              "not move, so it should be. Check the mounts are locked and the board is flat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
