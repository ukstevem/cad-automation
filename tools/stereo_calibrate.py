#!/usr/bin/env python3
"""
Calibrate the two rig cameras as a pair (bd ghn): camera B's fixed position relative to camera A.

    python tools/stereo_calibrate.py --captures outputs/ar_captures/recal_20260917_all \\
        --profile-a outputs/calibration/RigCam_52FD1B1F.json \\
        --profile-b outputs/calibration/RigCam_B68DE55F.json

Uses photo pairs from webcam_capture.py (``<label>_<SERIAL>_<stamp>.png``, one stamp per pair) in which
both cameras found the board. Each camera's lens calibration is held fixed; see app/services/stereo_rig.py
for why the pair has to be solved together at all. Writes ``outputs/calibration/RigStereo_<A>_<B>.json``,
which ``--stereo`` on pose_refine, ar_view and click_pose then uses.

The rig shoots its two cameras one after the other, a second or two apart. A board that moved in that
gap makes a pair no fixed transform fits, and those pairs are dropped by their own error.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services import multiview_fit as MVF, stereo_rig as SR  # noqa: E402

NAME_RE = re.compile(r"^(?P<label>.+)_(?P<tag>[A-Za-z0-9]+)_(?P<stamp>\d{8}_\d{6})\.(png|jpe?g)$", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True, action="append", help="folder of photo pairs; repeatable")
    ap.add_argument("--profile-a", required=True)
    ap.add_argument("--profile-b", required=True)
    ap.add_argument("--min-common", type=int, default=12, help="corners both cameras must share in a pair")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pa, pb = MVF.load_profile(args.profile_a), MVF.load_profile(args.profile_b)
    sa, sb = SR.serial_of_profile(pa["name"]), SR.serial_of_profile(pb["name"])
    if pa["board"] != pb["board"]:
        print("the two profiles were calibrated with different boards", file=sys.stderr)
        return 2

    groups = {}
    for folder in args.captures:
        for path in sorted(glob.glob(os.path.join(folder, "*"))):
            m = NAME_RE.match(os.path.basename(path))
            if m:
                groups.setdefault((m.group("label"), m.group("stamp")), {})[m.group("tag")] = path
    pairs, skipped = [], []
    for (label, _stamp), files in sorted(groups.items()):
        if sa not in files or sb not in files:
            continue
        oa = SR.detect_corners(cv2.imread(files[sa]), pa["board"])
        ob = SR.detect_corners(cv2.imread(files[sb]), pb["board"])
        if oa is None or ob is None:
            skipped.append((label, "board not found by %s" % ("both" if oa is None and ob is None else (sa if oa is None else sb))))
            continue
        obj, ia, ib = SR.common_corners(oa, ob)
        if len(obj) < args.min_common:
            skipped.append((label, "%d common corners" % len(obj)))
            continue
        pairs.append((label, obj, ia, ib))
    print("%d photo pairs found, %d usable (%s)" % (len(groups), len(pairs),
          "; ".join("%s: %s" % s for s in skipped) or "none skipped"))

    res = SR.calibrate(pairs, pa["K"], pa["dist"], pb["K"], pb["dist"], pa["image_size"],
                       min_common=args.min_common)
    for i, r in enumerate(res["rounds"]):
        print("  round %d: %d pairs, rms %.3f px, median pair %.2f px, worst %.2f px%s"
              % (i, r["pairs"], r["rms_px"], r["median_pair_px"], r["worst_pair_px"],
                 ("  dropping " + ", ".join(r["dropped"])) if r["dropped"] else ""))
    R, T = res["R"], res["T"]
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    baseline = float(np.linalg.norm(T))

    def intr(p, path):
        K = p["K"]
        return {"profile": path, "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "dist": np.asarray(p["dist"]).ravel().tolist()}

    out = args.out or os.path.join(os.path.dirname(args.profile_a), "RigStereo_%s_%s.json" % (sa, sb))
    doc = {
        "schema": SR.SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cameras": {"A": sa, "B": sb},
        "convention": "x_B = R @ x_A + T, millimetres, in the ChArUco board's units",
        "R": R.tolist(), "T": T.tolist(),
        "baseline_mm": round(baseline, 2), "rotation_deg": round(angle, 3),
        "rms_px": round(res["rms_px"], 4),
        "pairs_used": res["pairs_used"], "pairs_total": len(pairs),
        "per_pair_px": res["per_pair_px"], "rounds": res["rounds"],
        "captures": args.captures, "board": pa["board"],
        "intrinsics": {"A": intr(pa, args.profile_a), "B": intr(pb, args.profile_b)},
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("camera B relative to A: baseline %.1f mm, rotation %.2f deg, rms %.3f px over %d of %d pairs -> %s"
          % (baseline, angle, res["rms_px"], len(res["pairs_used"]), len(pairs), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
