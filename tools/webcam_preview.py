#!/usr/bin/env python3
"""
Live side-by-side preview of both test-cell cameras, in a browser. For AIMING only.

Build-order step 3 is a framing check — whole part plus enough ChArUco board visible in both
views, roughly 40 degrees apart — and that is impossible to dial in by taking stills one at a
time. This serves both cameras as MJPEG streams plus a page showing them together with centre
crosshairs and thirds guides.

    # on the capture host
    python3 webcam_preview.py

    # from the laptop, no firewall change needed:
    ssh -f -N -L 8088:127.0.0.1:8088 administrator@<host>
    # then open http://localhost:8088/

Requirements: ffmpeg only. Binds 127.0.0.1 by default, so nothing is exposed to the network
unless you ask for it with --bind.

PREVIEW IS NOT MEASUREMENT
--------------------------
Two 1920x1080 YUYV streams are ~20 MB/s each against a ~35 MB/s USB 2.0 bus, so simultaneous
uncompressed preview is physically impossible. This therefore streams **MJPG at 1280x720**:
compressed, and a different capture format from the one used for measurement.

That is fine for judging framing and nothing else. Do not calibrate from these frames, and
**re-run `webcam_capture.py lock` after you finish aiming** — you will have changed the scene the
cameras are looking at, and opening a stream at a different format can perturb exposure.
Focus and zoom are not touched, so the geometry the calibration depends on is unaffected.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the capture tool's v4l2 helpers rather than copying them. In particular
# apply_controls() knows the auto-toggles must precede the values they gate, which is not
# optional: set exposure while auto_exposure is still on and it is silently discarded.
import webcam_capture as WC  # noqa: E402

BY_ID_DIR = "/dev/v4l/by-id"
SOI, EOI = b"\xff\xd8", b"\xff\xd9"     # JPEG start/end of image markers
EXPOSURE_CTRLS = ("exposure_time_absolute", "exposure_absolute")
_LETTERS = "ABCDEFGH"


def discover_devices():
    if os.path.isdir(BY_ID_DIR):
        found = sorted(
            os.path.join(BY_ID_DIR, n) for n in os.listdir(BY_ID_DIR)
            if n.endswith("-video-index0")
        )
        if found:
            return found
    return sorted(p for p in ("/dev/video%d" % i for i in range(10)) if os.path.exists(p))


def short_tag(device: str, index: int) -> str:
    base = os.path.basename(device)
    if base.endswith("-video-index0"):
        base = base[: -len("-video-index0")]
    token = "".join(ch for ch in base.split("_")[-1].split("-")[-1] if ch.isalnum())
    return token if len(token) >= 4 else "cam%d" % index


def load_locked_controls(device: str) -> dict:
    """
    The controls `webcam_capture.py lock` recorded for this device, if any.

    Without these the preview streams at whatever the camera defaults to when ffmpeg opens it —
    which is auto-metered, and against a dark background that means a blown-out subject. You
    would then be aiming using a picture that looks nothing like what you are about to capture.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webcam_settings.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            settings = json.load(fh)
    except Exception:
        return {}
    entry = settings.get(device)
    if isinstance(entry, dict) and "controls" in entry:
        return dict(entry["controls"])
    return dict(entry) if isinstance(entry, dict) else {}


class Camera:
    """One ffmpeg process producing MJPEG on stdout, with the latest whole frame kept."""

    def __init__(self, device: str, tag: str, size: str, fps: int,
                 controls: dict | None = None, label: str = ""):
        self.device, self.tag = device, tag
        # Humans say "camera A"; filenames and profile matching use the serial, which is the
        # thing that stays true when the cameras are unplugged and swapped around.
        self.label = label or tag
        self.size, self.fps = size, fps
        self.controls = dict(controls or {})
        self.frame: bytes | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def exposure(self):
        for name in EXPOSURE_CTRLS:
            v = WC.get_ctrl(self.device, name)
            if v is not None:
                return name, v
        return None, None

    def set_exposure(self, value: int):
        """Exposure can be changed while streaming, so this takes effect live."""
        name, _ = self.exposure()
        if name is None:
            return None
        # Make sure we are in manual mode first, or the write is discarded.
        for auto in ("auto_exposure", "exposure_auto"):
            if WC.get_ctrl(self.device, auto) is not None:
                WC.set_ctrl(self.device, auto, WC.AUTO_OFF[auto])
                break
        WC.set_ctrl(self.device, name, int(value))
        self.controls[name] = int(value)
        return WC.get_ctrl(self.device, name)

    def _run(self) -> None:
        if self.controls:
            WC.apply_controls(self.device, self.controls)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "v4l2", "-input_format", "mjpeg",
               "-video_size", self.size, "-framerate", str(self.fps),
               "-i", self.device,
               # The C920 emits MJPEG natively, so copy the stream rather than re-encoding.
               "-c:v", "copy", "-f", "mjpeg", "pipe:1"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          bufsize=0)
        except Exception as exc:                       # pragma: no cover
            self.error = str(exc)
            return
        if self.controls:
            # Re-apply once the stream is actually up: opening it perturbs exposure on a C920,
            # so a pre-flight apply alone leaves the preview brighter than the real capture.
            threading.Timer(1.5, WC.apply_controls, (self.device, self.controls)).start()
        buf = b""
        while True:
            chunk = self._proc.stdout.read(65536)
            if not chunk:
                err = (self._proc.stderr.read() or b"").decode(errors="replace").strip()
                self.error = err.splitlines()[-1] if err else "ffmpeg exited"
                return
            buf += chunk
            # Emit every complete JPEG in the buffer, keeping only the most recent.
            while True:
                start = buf.find(SOI)
                if start < 0:
                    buf = b""
                    break
                end = buf.find(EOI, start + 2)
                if end < 0:
                    buf = buf[start:]
                    break
                with self._lock:
                    self.frame = buf[start:end + 2]
                buf = buf[end + 2:]

    def latest(self):
        with self._lock:
            return self.frame

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


CAMERAS: dict = {}

PAGE = """<!doctype html><meta charset=utf-8><title>Test cell preview</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif}
 header{padding:8px 12px;background:#1b1b1b;border-bottom:1px solid #333}
 .wrap{display:flex;flex-wrap:wrap;gap:10px;padding:10px}
 .cam{position:relative;flex:1 1 640px;min-width:420px}
 .cam img{width:100%;display:block;background:#000}
 .cam .tag{position:absolute;top:6px;left:8px;background:rgba(0,0,0,.6);padding:2px 6px;
   border-radius:3px;font-weight:600}
 /* Centre cross + thirds: aiming aids, drawn over the stream not into it. */
 .g{position:absolute;inset:0;pointer-events:none}
 .g::before,.g::after{content:"";position:absolute;background:rgba(255,80,80,.75)}
 .g::before{left:50%;top:0;bottom:0;width:1px}
 .g::after{top:50%;left:0;right:0;height:1px}
 .t{position:absolute;inset:0;pointer-events:none;
   background:
     linear-gradient(to right,transparent 33.33%,rgba(255,255,255,.18) 33.33% calc(33.33% + 1px),
       transparent calc(33.33% + 1px) 66.66%,rgba(255,255,255,.18) 66.66% calc(66.66% + 1px),transparent 0),
     linear-gradient(to bottom,transparent 33.33%,rgba(255,255,255,.18) 33.33% calc(33.33% + 1px),
       transparent calc(33.33% + 1px) 66.66%,rgba(255,255,255,.18) 66.66% calc(66.66% + 1px),transparent 0)}
 code{background:#000;padding:1px 4px;border-radius:3px}
 .ctl{position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,.7);padding:5px 8px;
   border-radius:4px;display:flex;gap:6px;align-items:center}
 .ctl button{background:#333;color:#eee;border:1px solid #555;border-radius:3px;
   width:30px;height:26px;font-size:15px;cursor:pointer}
 .ctl button:hover{background:#444}
 .ctl .v{min-width:52px;text-align:center;font-variant-numeric:tabular-nums}
</style>
<header><b>Test cell preview</b> &mdash; MJPG __SIZE__, aiming only.
Whole part <i>plus</i> board in both views, cameras ~40&deg; apart.
Exposure changes below are live and persist on the camera; still re-run
<code>webcam_capture.py lock</code> when you have finished aiming.</header>
<div class=wrap>__CAMS__</div>
<script>
async function ex(tag, delta, absolute) {
  const el = document.getElementById('v_' + tag);
  const cur = parseInt(el.textContent, 10);
  const want = absolute !== undefined ? absolute : Math.max(1, cur + delta);
  const r = await fetch('/ctrl?tag=' + encodeURIComponent(tag) + '&exposure=' + want);
  const j = await r.json();
  if (j.exposure !== null && j.exposure !== undefined) el.textContent = j.exposure;
}
async function poll() {
  for (const tag of window.__TAGS__) {
    try {
      const r = await fetch('/ctrl?tag=' + encodeURIComponent(tag));
      const j = await r.json();
      const el = document.getElementById('v_' + tag);
      if (el && j.exposure !== null) el.textContent = j.exposure;
    } catch (e) {}
  }
}
window.__TAGS__ = __TAGLIST__;
poll();
</script>
"""

CAM_BLOCK = """<div class=cam><span class=tag><b>__LABEL__</b> &nbsp;<small>__TAG__</small></span>
<img src="/stream/__TAG__" alt="__LABEL__"><div class=t></div><div class=g></div>
<div class=ctl><span>exposure</span>
 <button onclick="ex('__TAG__',-20)" title="much darker">&laquo;</button>
 <button onclick="ex('__TAG__',-4)" title="darker">&minus;</button>
 <span class=v id="v___TAG__">?</span>
 <button onclick="ex('__TAG__',4)" title="brighter">+</button>
 <button onclick="ex('__TAG__',20)" title="much brighter">&raquo;</button>
</div></div>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):                 # keep the console quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            blocks = "".join(
                CAM_BLOCK.replace("__TAG__", t).replace("__LABEL__", c.label)
                for t, c in CAMERAS.items()
            )
            body = PAGE.replace("__CAMS__", blocks)
            first = next(iter(CAMERAS.values()), None)
            body = body.replace("__SIZE__", first.size if first else "?")
            body = body.replace("__TAGLIST__", json.dumps(list(CAMERAS)))
            raw = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if self.path.startswith("/ctrl"):
            q = parse_qs(urlparse(self.path).query)
            tag = (q.get("tag") or [""])[0]
            cam = CAMERAS.get(tag)
            if cam is None:
                self.send_error(404, "no such camera")
                return
            if "exposure" in q:
                try:
                    cam.set_exposure(int(q["exposure"][0]))
                except (ValueError, TypeError):
                    pass
            _name, value = cam.exposure()
            raw = json.dumps({"tag": tag, "exposure": value}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if self.path.startswith("/stream/"):
            tag = self.path[len("/stream/"):]
            cam = CAMERAS.get(tag)
            if cam is None:
                self.send_error(404, "no such camera")
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=--frameboundary")
            self.end_headers()
            last = None
            try:
                while True:
                    frame = cam.latest()
                    if frame is None or frame is last:
                        if cam.error:
                            return
                        threading.Event().wait(0.02)
                        continue
                    last = frame
                    self.wfile.write(b"--frameboundary\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(frame))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        self.send_error(404)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=None)
    ap.add_argument("--size", default="1280x720", help="preview size (default 1280x720)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="display names in discovery order (default A B C D). The serial stays "
                         "the identity used in URLs and filenames; this is only what you read.")
    ap.add_argument("--exposure", type=int, default=None,
                    help="override exposure_time_absolute for the preview (units of 100us). "
                         "Also adjustable live from the page.")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="default 127.0.0.1 — reach it over an SSH tunnel rather than opening "
                         "a firewall port. Use 0.0.0.0 to serve the LAN.")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found.")
    devices = args.devices or discover_devices()
    if not devices:
        sys.exit("No cameras found. Plugged in, and are you in the 'video' group?")

    for i, dev in enumerate(devices):
        tag = short_tag(dev, i)
        while tag in CAMERAS:
            tag += "_"
        label = args.labels[i] if args.labels and i < len(args.labels) else _LETTERS[i]
        controls = load_locked_controls(dev)
        if args.exposure is not None:
            for name in EXPOSURE_CTRLS:
                if name in controls or not controls:
                    controls[name] = args.exposure
                    break
            else:
                controls["exposure_time_absolute"] = args.exposure
        CAMERAS[tag] = Camera(dev, tag, args.size, args.fps, controls, label)
        src = "locked settings" if controls else "CAMERA DEFAULTS (run `lock` first)"
        print(f"  {label}  {tag}  {dev}  [{src}]")

    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"\nServing on http://{args.bind}:{args.port}/   (Ctrl-C to stop)")
    print("From the laptop:  ssh -f -N -L "
          f"{args.port}:127.0.0.1:{args.port} <user>@<host>   then http://localhost:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        for cam in CAMERAS.values():
            cam.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
