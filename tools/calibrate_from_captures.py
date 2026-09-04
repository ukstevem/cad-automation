#!/usr/bin/env python3
"""
Calibrate the rig cameras from a folder of captures, without touching the browser.

The Calibrate tab's live capture uses getUserMedia, which runs in the browser on the laptop and
can therefore only ever see cameras plugged into the laptop. The rig cameras are on the capture
host. This posts the captured image sets straight at the same ``/api/v1/calibration/compute``
endpoint the tab uses, so there is one calibration code path, not two.

    python tools/calibrate_from_captures.py <dir> --squares-x 9 --squares-y 6 \\
        --square-mm 40 --marker-mm 30 --dictionary DICT_5X5_100

Images are grouped by the camera serial embedded in the filename by ``webcam_capture.py``
(``<label>_<SERIAL>_<timestamp>.png``), so one folder of pairs calibrates both cameras in one
pass and nothing can get crossed over.

Runs on the laptop against the running container (default http://localhost:8000).

THE BOARD VALUES MUST MATCH THE PRINTED SHEET.
Every one of them. A wrong dictionary detects nothing; wrong square counts decode the markers
but yield zero corners; a wrong square_mm calibrates happily and silently scales every distance
you derive from it afterwards. The sheet has its own configuration printed on it — read it off
the paper, not from memory.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

try:
    import requests
except ImportError:
    sys.exit("needs requests:  pip install requests")

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
# webcam_capture.py writes <label>_<TAG>_<YYYYmmdd>_<HHMMSS>.<ext>
NAME_RE = re.compile(r"^(?P<label>.+)_(?P<tag>[A-Za-z0-9]+)_(?P<stamp>\d{8}_\d{6})\.[^.]+$")


def group_by_camera(directory: str):
    groups = defaultdict(list)
    for fn in sorted(os.listdir(directory)):
        if not fn.lower().endswith(IMAGE_EXTS):
            continue
        m = NAME_RE.match(fn)
        if not m:
            print(f"  skipping unrecognised filename: {fn}")
            continue
        groups[m.group("tag")].append(os.path.join(directory, fn))
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--squares-x", type=int, required=True)
    ap.add_argument("--squares-y", type=int, required=True)
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--marker-mm", type=float, required=True)
    ap.add_argument("--dictionary", required=True)
    ap.add_argument("--prefix", default="RigCam",
                    help="profile name prefix; the serial is appended (default RigCam)")
    ap.add_argument("--dry-run", action="store_true", help="group and report, do not calibrate")
    args = ap.parse_args()

    if not os.path.isdir(args.directory):
        sys.exit(f"not a directory: {args.directory}")
    groups = group_by_camera(args.directory)
    if not groups:
        sys.exit("no recognisable capture files found")

    print(f"board: {args.squares_x}x{args.squares_y} @ {args.square_mm}mm "
          f"(marker {args.marker_mm}mm) {args.dictionary}")
    for tag, files in groups.items():
        print(f"  {tag}: {len(files)} frames")
    if args.dry_run:
        return 0

    url = f"{args.base_url}/api/v1/calibration/compute"
    failed = False
    for tag, files in sorted(groups.items()):
        name = f"{args.prefix}_{tag}"
        print(f"\n=== {name} ({len(files)} frames) ===")
        if len(files) < 4:
            print("  fewer than 4 frames — calibration needs more views. Skipping.")
            failed = True
            continue
        handles = []
        try:
            payload = []
            for path in files:
                fh = open(path, "rb")
                handles.append(fh)
                payload.append(("frames", (os.path.basename(path), fh, "image/png")))
            data = {
                "name": name,
                "squares_x": str(args.squares_x),
                "squares_y": str(args.squares_y),
                "square_mm": str(args.square_mm),
                "marker_mm": str(args.marker_mm),
                "dictionary": args.dictionary,
                "camera_label": tag,
            }
            resp = requests.post(url, files=payload, data=data, timeout=300)
        finally:
            for fh in handles:
                fh.close()

        if resp.status_code != 200:
            print(f"  FAILED {resp.status_code}: {resp.text[:400]}")
            failed = True
            continue

        r = resp.json()
        used = r.get("views_used")
        total = r.get("views_total")
        rms = r.get("rms_reproj_error_px")
        size = r.get("image_size")
        intr = r.get("intrinsics") or {}
        print(f"  views used     {used}/{total}")
        print(f"  RMS reproj     {rms} px at {size}")
        print(f"  fx, fy         {intr.get('fx')}, {intr.get('fy')}")
        print(f"  cx, cy         {intr.get('cx')}, {intr.get('cy')}")
        if size and intr.get("cx") is not None:
            # A healthy pinhole fit: fx ~ fy, and the principal point near the centre. A wild
            # cx/cy usually means too few views or all of them clustered in one part of frame.
            dx = abs(intr["cx"] - size[0] / 2) / size[0]
            dy = abs(intr["cy"] - size[1] / 2) / size[1]
            skew = abs(intr["fx"] - intr["fy"]) / max(intr["fx"], intr["fy"])
            flags = []
            if skew > 0.05:
                flags.append(f"fx/fy differ by {skew * 100:.1f}%")
            if dx > 0.1 or dy > 0.1:
                flags.append(f"principal point {dx * 100:.0f}%/{dy * 100:.0f}% off centre")
            print("  sanity         " + ("OK" if not flags else "; ".join(flags)))
        for v in r.get("per_view", []):
            if not v.get("used"):
                print(f"    dropped {v.get('name')}: {v.get('reason', '?')} "
                      f"({v.get('corners', 0)} corners)")

    print("\nProfiles land in outputs/calibration/. Use the one whose serial matches each "
          "photo — fit_multiview.py --cam-profile matches on the filename tag.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
