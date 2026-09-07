#!/usr/bin/env python3
"""
Find the gain that pairs with an exposure stop, by measuring rather than by eye.

The C920 gives exposure only in stops - 3, 4, 9, 19, 38 and up, each a doubling - so the useful
setting is nearly always "one stop up, then trim". Gain is the trim. But gain is amplification,
not light: it lifts the noise floor along with the signal, and the noise floor is exactly what a
low-contrast crease has to clear. So the goal is the LOWEST gain that still reaches the target
brightness at the higher exposure, never the other way about.

Runs on the capture host. Frames come from the preview server's own MJPEG stream rather than by
opening the camera again, which would collide with it over USB bandwidth - so leave the preview
running while this works.

    python3 gain_sweep.py --tag 52FD1B1F --exposure 19
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import webcam_capture as WC  # noqa: E402

BY_ID = "/dev/v4l/by-id"


def device_for(tag: str):
    for n in sorted(os.listdir(BY_ID)):
        if tag in n and n.endswith("-video-index0"):
            return os.path.join(BY_ID, n)
    return None


def grab(url: str, timeout: float = 6.0):
    """One complete JPEG from an MJPEG stream, as bytes."""
    buf = b""
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = fh.read(16384)
            if not chunk:
                break
            buf += chunk
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
            if s >= 0 and e > 0:
                return buf[s:e + 2]
    return None


def stats(jpeg: bytes):
    """(dark p10, bright p90, clipped %) via ffmpeg - no image library on the capture host."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                        "-vf", "scale=480:270,format=gray", "-frames:v", "1",
                        "-f", "rawvideo", "-"],
                       input=jpeg, capture_output=True, check=False)
    d = p.stdout
    if not d:
        return None
    hist = [0] * 256
    for b in d:
        hist[b] += 1
    n = len(d)

    def pct(f):
        want, run = n * f, 0
        for v in range(256):
            run += hist[v]
            if run >= want:
                return v
        return 255
    return pct(0.10), pct(0.90), 100.0 * (hist[254] + hist[255]) / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="camera serial, e.g. 52FD1B1F")
    ap.add_argument("--exposure", type=int, default=19, help="exposure stop to hold")
    ap.add_argument("--gains", default="255,220,190,160,130,100,70,40",
                    help="gain values to try, brightest first")
    ap.add_argument("--url", default=None, help="stream URL (default the local preview server)")
    ap.add_argument("--target-bright", type=int, default=235)
    ap.add_argument("--max-clip", type=float, default=0.5)
    args = ap.parse_args()

    dev = device_for(args.tag)
    if dev is None:
        print("no device matching %s in %s" % (args.tag, BY_ID), file=sys.stderr)
        return 2
    url = args.url or ("http://127.0.0.1:8088/stream/%s" % args.tag)

    WC.set_ctrl(dev, "exposure_time_absolute", args.exposure)
    time.sleep(1.5)                       # the snap to its own ladder takes about a second
    got = WC.get_ctrl(dev, "exposure_time_absolute")
    print("%s at exposure %s (asked %s)" % (args.tag, got, args.exposure))
    print("%6s %7s %8s %9s" % ("gain", "dark", "bright", "clipped"))

    rows = []
    for g in [int(x) for x in args.gains.split(",")]:
        WC.set_ctrl(dev, "gain", g)
        time.sleep(1.2)
        j = grab(url)
        st = stats(j) if j else None
        if st is None:
            print("%6d %7s %8s %9s" % (g, "-", "-", "no frame"))
            continue
        dark, bright, clip = st
        print("%6d %7d %8d %8.2f%%" % (g, dark, bright, clip))
        rows.append((g, dark, bright, clip))

    ok = [r for r in rows if r[3] <= args.max_clip]
    print("")
    if not ok:
        print("Everything clips at this exposure. Drop a stop and sweep again.")
        return 1
    # The lowest gain that still reaches the target, else the brightest that does not clip.
    reach = [r for r in ok if r[2] >= args.target_bright]
    pick = min(reach, key=lambda r: r[0]) if reach else max(ok, key=lambda r: r[2])
    print("RECOMMEND exposure %s, gain %s  ->  bright %d, dark %d, clipped %.2f%%"
          % (got, pick[0], pick[2], pick[1], pick[3]))
    if not reach:
        print("  (nothing reached bright %d without clipping, so this is the brightest clean one)"
              % args.target_bright)
    base = [r for r in rows if r[0] == max(r0[0] for r0 in rows)]
    if base and base[0][0] != pick[0]:
        print("  against gain %d, which is %d points of amplification you no longer pay for in"
              % (base[0][0], base[0][0] - pick[0]))
        print("  noise - and noise is the floor a low-contrast crease has to clear.")
    print("")
    print("Persist it:  python3 webcam_capture.py lock --exposure %s" % got)
    print("(check the focus line is unchanged - exposure and gain are safe for the calibration,")
    print("focus is not)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
