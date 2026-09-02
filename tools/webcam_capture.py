#!/usr/bin/env python3
"""
Test-cell capture for USB webcams (Logitech C920 and friends) on any Linux box.

The USB-webcam counterpart to ``cell_capture.py`` (which is Raspberry Pi / Picamera2 only).
Same discipline, different plumbing: settle autofocus/exposure/white-balance ONCE, read the
settled values back, pin them as manual controls, and reuse the identical values for the
ChArUco calibration shots AND every measurement shot. Drifting controls silently change the
intrinsics and quietly invalidate the calibration.

    python3 webcam_capture.py list                 # find your cameras and their controls
    python3 webcam_capture.py lock                 # once, after the rig is built and lit
    python3 webcam_capture.py shot board           # both cams photograph the ChArUco board
    python3 webcam_capture.py shot part_view1      # both cams photograph the part

Files land in ./captures/<label>_cam<N>_<timestamp>.png. Copy them to the cad-automation box
and feed them to the Calibrate tab (intrinsics), then tools/fit_multiview.py.

Why this runs one camera at a time
----------------------------------
The part is static, so nothing needs simultaneous exposure — and opening cameras sequentially
sidesteps the classic USB bandwidth wall where two high-resolution streams on one controller
simply fail to start.

The trap this tool exists to catch
----------------------------------
UVC controls can reset when a device is closed and reopened. Since each shot reopens the
camera, a silent reset between the calibration shot and the measurement shot would leave you
with correct-looking photos taken at the wrong focus — undetectable afterwards. So every shot
re-applies the locked values and **reads them back to verify**, shouting if they did not stick.

Requirements: ``v4l2-ctl`` (apt install v4l-utils) and ``opencv-python``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

try:
    import cv2
except ImportError:  # pragma: no cover - runs on the capture host, not in CI
    sys.exit("Needs OpenCV: pip install opencv-python")

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(HERE, "webcam_settings.json")
CAPTURES = os.path.join(HERE, "captures")

DEFAULT_DEVICES = ["/dev/video0", "/dev/video2"]   # UVC cams often claim two nodes each
RESOLUTION = (1920, 1080)
SETTLE_S = 3.0
WARMUP_FRAMES = 12          # webcams need frames in flight before the sensor settles

# The controls that must not drift. Names are the modern V4L2 spellings; older kernels use
# the aliases on the right and we fall back to them automatically.
CONTROLS = [
    ("focus_automatic_continuous", "focus_auto"),
    ("focus_absolute", "focus_absolute"),
    ("auto_exposure", "exposure_auto"),
    ("exposure_time_absolute", "exposure_absolute"),
    ("white_balance_automatic", "white_balance_temperature_auto"),
    ("white_balance_temperature", "white_balance_temperature"),
    ("sharpness", "sharpness"),
    ("backlight_compensation", "backlight_compensation"),
]
AUTO_OFF = {                       # value meaning "manual" for each auto control
    "focus_automatic_continuous": 0, "focus_auto": 0,
    "auto_exposure": 1, "exposure_auto": 1,             # 1 = manual, 3 = aperture priority
    "white_balance_automatic": 0, "white_balance_temperature_auto": 0,
}


def _require_v4l2() -> None:
    if shutil.which("v4l2-ctl") is None:
        sys.exit("v4l2-ctl not found. Install it:  sudo apt install v4l-utils")


def _v4l2(device: str, args):
    return subprocess.run(["v4l2-ctl", "-d", device] + args,
                          capture_output=True, text=True, check=False)


def supported_controls(device: str):
    """Map of control name -> current value, as reported by the driver."""
    out = _v4l2(device, ["--list-ctrls"])
    found = {}
    for line in out.stdout.splitlines():
        line = line.strip()
        if ":" not in line or " " not in line:
            continue
        name = line.split(" ", 1)[0]
        for part in line.split():
            if part.startswith("value="):
                try:
                    found[name] = int(part.split("=", 1)[1])
                except ValueError:
                    pass
    return found


def get_ctrl(device: str, name: str):
    out = _v4l2(device, ["--get-ctrl", name])
    if out.returncode != 0 or ":" not in out.stdout:
        return None
    try:
        return int(out.stdout.split(":", 1)[1].strip())
    except ValueError:
        return None


def set_ctrl(device: str, name: str, value: int) -> bool:
    return _v4l2(device, ["--set-ctrl", f"{name}={value}"]).returncode == 0


def resolve_names(device: str):
    """Pick whichever spelling of each control this kernel actually exposes."""
    available = supported_controls(device)
    resolved = []
    for modern, legacy in CONTROLS:
        if modern in available:
            resolved.append(modern)
        elif legacy in available:
            resolved.append(legacy)
    return resolved


def open_camera(device: str, fourcc: str = "YUYV"):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"Could not open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    return cap


def warmup(cap, n: int = WARMUP_FRAMES) -> None:
    for _ in range(n):
        cap.read()


def cmd_list(args) -> int:
    _require_v4l2()
    for device in args.devices:
        if not os.path.exists(device):
            print(f"{device}: not present")
            continue
        name = _v4l2(device, ["--info"]).stdout
        card = next((l.split(":", 1)[1].strip() for l in name.splitlines() if "Card type" in l), "?")
        print(f"\n{device}  ({card})")
        ctrls = supported_controls(device)
        for modern, legacy in CONTROLS:
            key = modern if modern in ctrls else (legacy if legacy in ctrls else None)
            mark = "ok " if key else "MISSING"
            print(f"  {mark} {modern:32s} {'= ' + str(ctrls[key]) if key else '(not exposed)'}")
        if not any(m in ctrls or l in ctrls for m, l in CONTROLS[:3]):
            print("  WARNING: this camera does not expose focus/exposure locks. "
                  "Intrinsics will drift and calibration cannot be trusted.")
    return 0


def cmd_lock(args) -> int:
    _require_v4l2()
    settings = {}
    for device in args.devices:
        if not os.path.exists(device):
            print(f"{device}: not present, skipping")
            continue
        print(f"\n{device}: settling auto controls for {SETTLE_S}s...")
        names = resolve_names(device)
        # Turn the autos ON so the camera can converge on the real scene, then read the result.
        for n in names:
            if n in AUTO_OFF:
                set_ctrl(device, n, 3 if "exposure" in n else 1)
        cap = open_camera(device, args.fourcc)
        t0 = time.time()
        while time.time() - t0 < SETTLE_S:
            cap.read()
        cap.release()

        values = {}
        for n in names:
            v = get_ctrl(device, n)
            if v is not None:
                values[n] = v
        for n in names:                       # now pin the autos to manual
            if n in AUTO_OFF:
                values[n] = AUTO_OFF[n]
        if args.sharpness is not None:
            for n in names:
                if n == "sharpness":
                    values[n] = args.sharpness
        settings[device] = values
        print(f"  locked: {values}")

    if not settings:
        sys.exit("No cameras locked.")
    settings["_resolution"] = list(RESOLUTION)
    settings["_fourcc"] = args.fourcc
    settings["_locked_at"] = datetime.now().isoformat(timespec="seconds")
    with open(SETTINGS, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)
    print(f"\nSaved {SETTINGS}. Use this SAME file for calibration AND measurement shots — "
          f"do not re-lock in between.")
    return 0


def apply_and_verify(device: str, values: dict) -> list:
    """Apply the locked controls, then read them back. Returns a list of controls that drifted."""
    for name, value in values.items():
        set_ctrl(device, name, value)
    time.sleep(0.3)
    drifted = []
    for name, value in values.items():
        got = get_ctrl(device, name)
        if got is not None and got != value:
            drifted.append((name, value, got))
    return drifted


def cmd_shot(args) -> int:
    _require_v4l2()
    if not os.path.exists(SETTINGS):
        sys.exit(f"No {SETTINGS} — run `python3 webcam_capture.py lock` first.")
    with open(SETTINGS, encoding="utf-8") as fh:
        settings = json.load(fh)
    fourcc = settings.get("_fourcc", args.fourcc)
    os.makedirs(CAPTURES, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    any_drift = False
    for idx, device in enumerate(args.devices):
        values = settings.get(device)
        if values is None:
            print(f"{device}: not in settings.json, skipping")
            continue
        cap = open_camera(device, fourcc)
        drifted = apply_and_verify(device, values)
        if drifted:
            any_drift = True
            print(f"  !! {device} CONTROLS DID NOT STICK — the camera reset them on reopen:")
            for name, want, got in drifted:
                print(f"       {name}: wanted {want}, got {got}")
        warmup(cap)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  {device}: capture FAILED")
            continue
        h, w = frame.shape[:2]
        if (w, h) != tuple(settings.get("_resolution", RESOLUTION)):
            print(f"  !! {device} returned {w}x{h}, not {settings.get('_resolution')} — "
                  f"the calibration resolution gate will reject these.")
        # PNG, not JPEG: no compression artefacts on the very edges we are about to detect.
        dest = os.path.join(CAPTURES, f"{args.label}_cam{idx}_{stamp}.png")
        cv2.imwrite(dest, frame)
        print(f"  {device} -> {dest}  ({w}x{h})")

    if any_drift:
        print("\nWARNING: at least one camera did not hold its locked controls. Those photos "
              "were taken with different intrinsics than the calibration — do not trust a pose "
              "from them. Re-run `lock`, or keep the device open for the whole session.")
        return 1
    print("\nDone. Move the part/board only BETWEEN shots, never the cameras.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES,
                    help=f"video devices (default {' '.join(DEFAULT_DEVICES)})")
    ap.add_argument("--fourcc", default="YUYV", choices=["YUYV", "MJPG"],
                    help="YUYV (default) is uncompressed — slower fps, no artefacts on edges. "
                         "Frame rate is irrelevant for a static part.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show cameras and whether they expose the needed locks")
    lock = sub.add_parser("lock", help="settle then pin focus/exposure/white balance")
    lock.add_argument("--sharpness", type=int, default=0,
                      help="force sharpness (default 0 — in-camera sharpening fabricates edges)")
    shot = sub.add_parser("shot", help="capture one labelled frame per camera")
    shot.add_argument("label")
    args = ap.parse_args()

    return {"list": cmd_list, "lock": cmd_lock, "shot": cmd_shot}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
