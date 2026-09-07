#!/usr/bin/env python3
"""
Pick the rig's exposure from evidence: how bright can it go before the board stops detecting?

The setting is a genuine trade and both ends of it matter.

TOO LOW and the dark end dies. Exposure on this rig was pinned low on purpose - auto-metering
took the whole frame, let a dark background drag the average down and blew the subject out, so 8
was chosen over auto's 77. But the line check's misses are concentrated on creases inside
shadowed webs, where both faces catch the light alike and there is no contrast to find. Those are
the edges a higher floor would rescue.

TOO HIGH and the world frame goes. The ChArUco board's white squares clip before anything else,
and the board IS the world frame - every camera pose, and so every part pose, is derived from it.
An image that looks pleasantly bright and detects thirty corners instead of forty has quietly
made every measurement downstream worse.

So the ceiling is not a histogram threshold, it is the detector's own answer, which is why this
runs here rather than on the rig: `tools/webcam_capture.py expose` sweeps and saves a frame per
setting, and this reads that folder and asks the detector.

    docker compose run --rm --no-deps api python tools/exposure_sweep.py \\
        --frames outputs/ar_captures/exposure \\
        --profile outputs/calibration/RigCam_52FD1B1F.json
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import charuco, image_edges, multiview_fit as MVF  # noqa: E402
from app.services.board_pose import charuco_board_pose  # noqa: E402


def tone(gray):
    """Dark end, bright end and clipped fraction - the three numbers that describe an exposure."""
    h = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    c = np.cumsum(h) / max(h.sum(), 1)
    p10 = int(np.searchsorted(c, 0.10))
    p90 = int(np.searchsorted(c, 0.90))
    clip = 100.0 * float(h[254:].sum()) / max(h.sum(), 1)
    return p10, p90, clip


def dark_detail(gray, frac=0.25):
    """
    How much texture survives in the DARKEST quarter of the part.

    A plain percentile says how dark the shadows are but not whether anything can be seen in
    them, and that is the question: a web at level 40 with gradient is measurable, a web at level
    40 that is uniformly level 40 is not. Local standard deviation in the darkest region answers
    it directly, and it is the number that should rise as exposure comes up.
    """
    thr = np.quantile(gray, frac)
    dark = gray <= thr
    if dark.sum() < 500:
        return 0.0
    g = gray.astype(np.float32)
    mu = cv2.blur(g, (9, 9))
    var = cv2.blur(g * g, (9, 9)) - mu * mu
    return float(np.sqrt(np.maximum(var, 0))[dark].mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder of expo<value>_<tag>.png frames")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--max-clip", type=float, default=0.5,
                    help="clipped-pixel percentage treated as the hard ceiling")
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.frames, "*.png"))
                       + glob.glob(os.path.join(args.frames, "*.jpg"))):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        m = re.search(r"expo(\d+)", os.path.basename(path))
        val = int(m.group(1)) if m else -1
        gray = image_edges.to_gray(img)
        p10, p90, clip = tone(gray)
        detail = dark_detail(gray)
        cor, ids, _mc, mi = charuco.detect_board_detailed(det, gray)
        nm = 0 if mi is None else len(mi)
        nc = 0 if ids is None else len(ids)
        rms = None
        if nc >= 6:
            try:
                rc, tc, _n = charuco_board_pose(cor, ids, board, profile["K"], profile["dist"])
                obj = np.asarray(board.getChessboardCorners(), np.float64).reshape(-1, 3)[
                    np.asarray(ids).ravel()]
                proj, _ = cv2.projectPoints(obj, rc, tc, np.asarray(profile["K"], np.float64),
                                            np.asarray(profile["dist"], np.float64))
                rms = float(np.sqrt(np.mean(np.sum(
                    (proj.reshape(-1, 2) - np.asarray(cor).reshape(-1, 2)) ** 2, axis=1))))
            except Exception:
                rms = None
        rows.append(dict(val=val, name=os.path.basename(path), p10=p10, p90=p90, clip=clip,
                         detail=detail, markers=nm, corners=nc, rms=rms))

    if not rows:
        print("no frames found in %s" % args.frames, file=sys.stderr)
        return 2

    best_corners = max(r["corners"] for r in rows)
    print("%9s %6s %7s %8s %8s %8s %8s %8s"
          % ("exposure", "dark", "bright", "clipped", "shadow", "markers", "corners", "rms px"))
    for r in sorted(rows, key=lambda r: r["val"]):
        print("%9s %6d %7d %7.2f%% %8.2f %8d %8d %8s"
              % (r["val"] if r["val"] >= 0 else r["name"][:9], r["p10"], r["p90"], r["clip"],
                 r["detail"], r["markers"], r["corners"],
                 "-" if r["rms"] is None else "%.2f" % r["rms"]))

    # The recommendation: the brightest setting that still detects the board as well as the best
    # frame did, and does not clip. Shadow detail is the tie-breaker, since raising the floor is
    # the entire reason for touching exposure.
    ok = [r for r in rows if r["clip"] <= args.max_clip and r["corners"] >= best_corners]
    print("")
    if not ok:
        print("NOTHING QUALIFIES. Either every frame clips beyond %.1f%%, or none matches the best"
              % args.max_clip)
        print("corner count (%d). Widen the sweep, or re-light before choosing an exposure."
              % best_corners)
        return 1
    pick = max(ok, key=lambda r: (r["detail"], r["val"]))
    base = min(rows, key=lambda r: r["val"])
    if pick["val"] < 0:
        # Frames that did not come from the sweep carry no exposure in their name, so there is
        # nothing to recommend - but the measurements above still describe the setting in use.
        print("These frames carry no exposure value in their filenames, so there is no setting to")
        print("recommend - the rows above simply describe whatever exposure took them. Headroom")
        print("shows as a bright end below ~205 with clipping at zero: that is room to raise.")
        print("Run `webcam_capture.py expose` on the rig to sweep properly.")
        return 0
    print("RECOMMEND exposure %s" % pick["val"])
    print("  keeps all %d board corners, clips %.2f%% of pixels, and lifts shadow detail to %.2f"
          % (pick["corners"], pick["clip"], pick["detail"]))
    if base["val"] != pick["val"]:
        gain = (pick["detail"] / base["detail"] - 1) * 100 if base["detail"] > 0 else float("nan")
        print("  against %.2f at the lowest setting tried (%s) - %+.0f%% more texture in the"
              % (base["detail"], base["val"], gain))
        print("  darkest quarter of the frame, which is where the creases are.")
    print("")
    print("Apply it on the rig, and re-lock so every later shot uses the same value:")
    print("  python3 webcam_capture.py lock --exposure %s" % pick["val"])
    print("")
    print("Exposure does not disturb the intrinsics, so the existing calibration still stands -")
    print("but ONLY if focus did not move. Check the lock output says focus is unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
