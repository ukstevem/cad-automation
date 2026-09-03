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

Requirements: ``v4l2-ctl`` and ``ffmpeg`` — nothing else.

Frames are grabbed with ffmpeg rather than OpenCV deliberately. A capture host should carry as
little as possible, and on Arch ``python-opencv`` drags in vtk, qt6-base, openmpi and hdf5 to do
a job ffmpeg already does. ffmpeg's CLI is also far more stable across major versions than
OpenCV's Python API, and it gives explicit control of the pixel format, which is the one thing
that actually matters here (uncompressed YUYV, not MJPEG).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(HERE, "webcam_settings.json")
CAPTURES = os.path.join(HERE, "captures")

# Device paths are resolved via /dev/v4l/by-id/, NOT /dev/videoN. The numbered nodes are
# assigned in enumeration order and can swap on reboot or replug; the by-id paths are keyed on
# the USB serial and are stable per physical camera. That matters more than it sounds: if two
# cameras swap numbers between the calibration shots and the measurement shots, each photo gets
# the other camera's intrinsics. The pose is then wrong, with nothing in the data to show it.
BY_ID_DIR = "/dev/v4l/by-id"
RESOLUTION = (1920, 1080)
SETTLE_S = 3.0
WARMUP_FRAMES = 12          # webcams need frames in flight before the sensor settles

# The controls that must not drift. Names are the modern V4L2 spellings; older kernels use
# the aliases on the right and we fall back to them automatically.
CONTROLS = [
    ("focus_automatic_continuous", "focus_auto"),
    ("focus_absolute", "focus_absolute"),
    ("zoom_absolute", "zoom_absolute"),
    ("auto_exposure", "exposure_auto"),
    ("exposure_time_absolute", "exposure_absolute"),
    ("white_balance_automatic", "white_balance_temperature_auto"),
    ("white_balance_temperature", "white_balance_temperature"),
    ("sharpness", "sharpness"),
    ("backlight_compensation", "backlight_compensation"),
]

# Which controls actually invalidate a calibration if they move.
#
# Only the ones that change the camera's GEOMETRY. Focus shifts the lens elements and with them
# the focal length; zoom scales it outright. Move either between the calibration shots and the
# measurement shots and the intrinsics are silently wrong.
#
# Exposure, white balance, sharpness and backlight change how the image LOOKS, not where things
# project to. Observed on a real C920: exposure_time_absolute shifts by ~7% when the stream opens
# (verified as a genuine stream-open adjustment, not value quantisation — step is 1 and a set
# without streaming reads straight back). Failing the run over that would cry wolf, and a check
# that cries wolf gets ignored — which is exactly how a real focus drift would later slip past.
GEOMETRY_CRITICAL = {
    "focus_absolute", "focus_automatic_continuous", "focus_auto", "zoom_absolute",
}
AUTO_OFF = {                       # value meaning "manual" for each auto control
    "focus_automatic_continuous": 0, "focus_auto": 0,
    "auto_exposure": 1, "exposure_auto": 1,             # 1 = manual, 3 = aperture priority
    "white_balance_automatic": 0, "white_balance_temperature_auto": 0,
}


def _require_v4l2() -> None:
    if shutil.which("v4l2-ctl") is None:
        sys.exit("v4l2-ctl not found. Install it:  sudo apt install v4l-utils")


def discover_devices():
    """
    Stable capture-device paths, newest-sorted, from /dev/v4l/by-id.

    A UVC webcam claims more than one node — typically ``-video-index0`` for capture and a
    further index for metadata — so we take index0 only. Falls back to /dev/video[0-9] if the
    by-id directory is absent, with a warning, because that path is not reboot-stable.
    """
    if os.path.isdir(BY_ID_DIR):
        found = sorted(
            os.path.join(BY_ID_DIR, n) for n in os.listdir(BY_ID_DIR)
            if n.endswith("-video-index0")
        )
        if found:
            return found
    numbered = sorted(p for p in ("/dev/video%d" % i for i in range(10)) if os.path.exists(p))
    if numbered:
        print("WARNING: no /dev/v4l/by-id entries; falling back to numbered nodes, which are "
              "NOT stable across reboots. Verify identities before trusting a calibration.")
    return numbered


def device_identity(device: str) -> str:
    """A human-readable identity for a device: the by-id basename, or the driver's card name."""
    if BY_ID_DIR in device:
        return os.path.basename(device)
    info = _v4l2(device, ["--info"]).stdout
    for line in info.splitlines():
        if "Card type" in line:
            return line.split(":", 1)[1].strip()
    return device


def short_tag(device: str, fallback_index: int) -> str:
    """
    A short, stable tag for one physical camera, used in capture filenames.

    It ends up in the photo name, which is what ``fit_multiview.py --cam-profile SUBSTR=PATH``
    matches on — so each camera's photos automatically pick up that camera's calibration
    profile. Derived from the USB serial in the by-id name where possible.
    """
    ident = device_identity(device)
    if ident.endswith("-video-index0"):
        ident = ident[: -len("-video-index0")]
    token = ident.split("_")[-1].split("-")[-1]
    token = "".join(ch for ch in token if ch.isalnum())
    return token if len(token) >= 4 else "cam%d" % fallback_index


def unique_tags(devices):
    """
    ``device -> tag``, guaranteed distinct.

    Two cameras of the same model that report no USB serial derive the *same* tag. Since a shot
    writes ``<label>_<tag>_<timestamp>.png`` and both cameras share the timestamp, a collision
    would have one camera silently overwrite the other's photo. So collisions get an index
    suffix rather than being left to chance.
    """
    tags = {d: short_tag(d, i) for i, d in enumerate(devices)}
    seen = {}
    for d in devices:
        seen.setdefault(tags[d], []).append(d)
    for tag, owners in seen.items():
        if len(owners) > 1:
            for i, d in enumerate(owners):
                tags[d] = f"{tag}{i}"
    return tags


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


PIXFMT = {"YUYV": "yuyv422", "MJPG": "mjpeg"}


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found. Install it:  sudo pacman -S ffmpeg   (or apt install ffmpeg)")


def stream_briefly(device: str, seconds: float, fourcc: str = "YUYV") -> None:
    """Hold the camera streaming for *seconds* and discard it — lets the auto controls converge."""
    _require_ffmpeg()
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "v4l2", "-input_format", PIXFMT.get(fourcc, "yuyv422"),
         "-video_size", f"{RESOLUTION[0]}x{RESOLUTION[1]}",
         "-i", device, "-t", str(seconds), "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )


def capture_frame(device: str, dest: str, fourcc: str = "YUYV",
                  warmup: int = WARMUP_FRAMES) -> str:
    """
    Grab one frame to *dest* as PNG. Returns "" on success, else the ffmpeg error.

    The ``select`` filter discards the first *warmup* frames rather than saving them: a webcam's
    opening frames are dark or half-converged even with the controls pinned, and that would be
    baked into a calibration. PNG, not JPEG — no compression artefacts on the very edges the
    fit is about to detect.
    """
    _require_ffmpeg()
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "v4l2", "-input_format", PIXFMT.get(fourcc, "yuyv422"),
         "-video_size", f"{RESOLUTION[0]}x{RESOLUTION[1]}",
         "-i", device,
         "-vf", f"select=gte(n\\,{int(warmup)})", "-frames:v", "1",
         "-y", dest],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0 or not os.path.exists(dest):
        return (proc.stderr or "ffmpeg failed").strip().splitlines()[-1:][0] if proc.stderr else "ffmpeg failed"
    return ""


def png_size(path: str):
    """(width, height) from a PNG header — avoids pulling in an image library just to check."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def cmd_list(args) -> int:
    _require_v4l2()
    tags = unique_tags(args.devices)
    for device in args.devices:
        if not os.path.exists(device):
            print(f"{device}: not present")
            continue
        name = _v4l2(device, ["--info"]).stdout
        card = next((l.split(":", 1)[1].strip() for l in name.splitlines() if "Card type" in l), "?")
        print(f"\n{device}")
        print(f"    card {card}   tag '{tags[device]}'  (this tag lands in the filenames)")
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
        stream_briefly(device, SETTLE_S, args.fourcc)

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
        # Apply, then stream once and re-read. A C920 nudges exposure when the stream opens
        # (observed ~7%, and it is a genuine stream-open adjustment, not quantisation). Recording
        # the pre-stream value would mean every later shot reported a drift it could do nothing
        # about. So settle it here: record what the camera actually produces once streaming.
        apply_controls(device, values)
        stream_briefly(device, 1.0, args.fourcc)
        settled = verify_controls(device, values)
        for name, wanted, got in settled:
            if name in GEOMETRY_CRITICAL:
                print(f"  !! {name} will not hold ({wanted} -> {got} on stream open). This camera "
                      f"cannot keep stable intrinsics — calibration from it is not trustworthy.")
            else:
                values[name] = got          # accept the settled appearance value
                print(f"     {name} settles to {got} once streaming (was {wanted}) — recorded")

        settings[device] = {"_identity": device_identity(device), "controls": values}
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


def apply_controls(device: str, values: dict) -> None:
    for name, value in values.items():
        set_ctrl(device, name, value)
    time.sleep(0.3)


def verify_controls(device: str, values: dict) -> list:
    """Read the controls back. Returns ``[(name, wanted, got), ...]`` for any that differ."""
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
    tags = unique_tags(args.devices)
    for idx, device in enumerate(args.devices):
        entry = settings.get(device)
        if entry is None:
            print(f"{device}: not in settings.json, skipping")
            continue
        values = entry["controls"] if isinstance(entry, dict) and "controls" in entry else entry
        want_id = entry.get("_identity") if isinstance(entry, dict) else None
        got_id = device_identity(device)
        if want_id and want_id != got_id:
            any_drift = True
            print(f"  !! {device} is now '{got_id}' but was locked as '{want_id}' — the cameras "
                  f"have swapped device nodes. Re-run `lock`.")
        dest = os.path.join(CAPTURES, f"{args.label}_{tags[device]}_{stamp}.png")
        apply_controls(device, values)
        err = capture_frame(device, dest, fourcc)
        if err:
            print(f"  {device}: capture FAILED — {err}")
            continue

        # Verify AFTER the capture, not before. If the driver resets controls when the device is
        # opened for streaming — the exact trap this tool exists to catch — a pre-flight check
        # would read back the values we just set and report all-clear on a photo that was
        # actually taken with different ones.
        drifted = verify_controls(device, values)
        critical = [d for d in drifted if d[0] in GEOMETRY_CRITICAL]
        advisory = [d for d in drifted if d[0] not in GEOMETRY_CRITICAL]
        if critical:
            any_drift = True
            print(f"  !! {device} GEOMETRY CONTROLS DID NOT STICK — this invalidates the "
                  f"calibration:")
            for name, want, got in critical:
                print(f"       {name}: wanted {want}, got {got}")
        for name, want, got in advisory:
            print(f"     note: {name} settled to {got}, not {want} (appearance only, "
                  f"not intrinsics)")

        size = png_size(dest)
        if size and list(size) != list(settings.get("_resolution", RESOLUTION)):
            any_drift = True
            print(f"  !! {device} returned {size[0]}x{size[1]}, not "
                  f"{settings.get('_resolution')} — the calibration resolution gate will "
                  f"reject these.")
        print(f"  {device} -> {dest}  ({size[0]}x{size[1]})" if size else f"  {device} -> {dest}")

    if any_drift:
        print("\nWARNING: a camera did not hold its FOCUS or ZOOM, or its identity changed. "
              "Those photos have different intrinsics from the calibration — do not trust a "
              "pose from them. Re-run `lock`, or keep the device open for the whole session.")
        return 1
    print("\nDone. Move the part/board only BETWEEN shots, never the cameras.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=None,
                    help="video devices (default: auto-discover stable /dev/v4l/by-id paths)")
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
    if not args.devices:
        _require_v4l2()
        args.devices = discover_devices()
        if not args.devices:
            sys.exit("No video devices found. Camera plugged in? And are you in the 'video' "
                     "group?   sudo usermod -aG video $USER   (then log out and back in)")

    return {"list": cmd_list, "lock": cmd_lock, "shot": cmd_shot}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
