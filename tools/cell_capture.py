#!/usr/bin/env python3
"""
Test-cell capture for two Raspberry Pi Camera Module 3 units on ONE Pi (dual CSI, e.g.
Pi 5) — for the AR multi-view alignment spike (ADR 0001).

Camera Module 3 has AUTOFOCUS, and auto-exposure / auto-white-balance. Any of those
drifting silently changes the camera intrinsics and breaks calibration. So this script:

  1. `lock`  — runs AF/AE/AWB ONCE, lets them settle, reads the resulting lens position /
               exposure / gains / colour gains, and writes them to settings.json.
  2. `shot`  — loads settings.json, applies those values as FIXED manual controls on both
               cameras, and captures a still from each.

Run `lock` once after the rig is built and lit, then use the SAME settings.json for both
the ChArUco calibration shots and every measurement shot. Identical locked controls across
calibration + capture is the whole point — don't re-`lock` between them.

Static part → rolling shutter and sequential capture are both fine (nothing moves).

Usage (on the Pi):
    python3 cell_capture.py lock                 # once, after building + lighting the cell
    python3 cell_capture.py shot board           # both cams photograph the ChArUco board
    python3 cell_capture.py shot part_view1      # both cams photograph the part on the board
    python3 cell_capture.py shot part_view2      # ... move nothing but trigger again if wanted

Files land in ./captures/<label>_cam<N>_<timestamp>.jpg . Copy them to the box running
cad-automation and feed them to the Calibrate tab (intrinsics) / multi-view fit.

Notes
-----
- Use the STANDARD Module 3 (66° FOV), not the wide (120°) — the wide's heavy distortion
  is calibratable but hurts edge precision. Whichever you use, keep it fixed.
- Resolution is pinned here (full sensor 4608x2592). Calibrate and capture at this same
  resolution — the app's resolution gate rejects mismatches.
- Assumes Picamera2 camera indices 0 and 1. With an Arducam multiplexer instead of native
  dual-CSI, only one camera is live at a time; see SEQUENTIAL_MUX below.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from picamera2 import Picamera2
    from libcamera import controls
except ImportError:  # pragma: no cover - only runs on the Pi
    print("This script runs on the Raspberry Pi (needs picamera2 + libcamera).")
    raise

HERE = Path(__file__).resolve().parent
SETTINGS = HERE / "settings.json"
CAPTURES = HERE / "captures"
RESOLUTION = (4608, 2592)          # Module 3 full sensor; MUST match calibration resolution
CAM_INDICES = [0, 1]               # native dual-CSI; both cameras live at once
SETTLE_S = 2.0                     # let AE/AWB settle before reading them back


def _still_config(cam):
    return cam.create_still_configuration(main={"size": RESOLUTION})


def _open(idx):
    cam = Picamera2(idx)
    cam.configure(_still_config(cam))
    return cam


def lock():
    """Auto-focus/expose/white-balance once per camera, then persist the settled values."""
    settings = {}
    for idx in CAM_INDICES:
        cam = _open(idx)
        cam.start()
        # Let auto-everything settle, then run one AF cycle and read the result back.
        cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
        time.sleep(SETTLE_S)
        try:
            cam.autofocus_cycle()            # blocks until focus converges
        except Exception as exc:             # noqa: BLE001 - some stacks lack the helper
            print(f"  cam{idx}: autofocus_cycle unavailable ({exc}); using settled metadata")
        time.sleep(0.5)
        md = cam.capture_metadata()
        settings[str(idx)] = {
            "LensPosition": float(md.get("LensPosition", 0.0)),
            "ExposureTime": int(md.get("ExposureTime", 10000)),
            "AnalogueGain": float(md.get("AnalogueGain", 1.0)),
            "ColourGains": [float(x) for x in md.get("ColourGains", (1.0, 1.0))],
        }
        print(f"  cam{idx} locked: {settings[str(idx)]}")
        cam.stop()
        cam.close()
    settings["_resolution"] = list(RESOLUTION)
    settings["_locked_at"] = datetime.now().isoformat(timespec="seconds")
    SETTINGS.write_text(json.dumps(settings, indent=2))
    print(f"Saved {SETTINGS}. Use this same file for calibration AND measurement shots.")


def _apply_locked(cam, s):
    cam.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": s["LensPosition"],
        "AeEnable": False,
        "ExposureTime": s["ExposureTime"],
        "AnalogueGain": s["AnalogueGain"],
        "AwbEnable": False,
        "ColourGains": tuple(s["ColourGains"]),
    })


def shot(label):
    if not SETTINGS.exists():
        sys.exit("No settings.json — run `python3 cell_capture.py lock` first.")
    settings = json.loads(SETTINGS.read_text())
    CAPTURES.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for idx in CAM_INDICES:
        s = settings[str(idx)]
        cam = _open(idx)
        cam.start()
        _apply_locked(cam, s)
        time.sleep(0.4)                      # let the fixed controls take effect
        out = CAPTURES / f"{label}_cam{idx}_{ts}.jpg"
        cam.capture_file(str(out))
        print(f"  cam{idx} -> {out}")
        cam.stop()
        cam.close()
    print("Done. Move the part/board only BETWEEN shots, never the cameras.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("lock", "shot"):
        sys.exit(__doc__)
    if sys.argv[1] == "lock":
        lock()
    else:
        if len(sys.argv) < 3:
            sys.exit("Usage: python3 cell_capture.py shot <label>")
        shot(sys.argv[2])


if __name__ == "__main__":
    main()
