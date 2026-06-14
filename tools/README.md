# Test-cell tools (run on the Raspberry Pi)

Capture utilities for the AR multi-view alignment spike. These run **on the Pi**
(need `picamera2` + `libcamera`), not in the cad-automation container.

## `cell_capture.py` — locked dual-camera capture (2× Camera Module 3, one Pi)

Camera Module 3 autofocuses and auto-exposes; if those drift, the camera intrinsics
drift and calibration breaks. So we lock focus/exposure/white-balance once and reuse
the same locked values for **both** calibration and measurement shots.

```bash
python3 cell_capture.py lock              # ONCE, after the rig is built + lit
python3 cell_capture.py shot board        # both cams shoot the ChArUco board
python3 cell_capture.py shot part_view1   # both cams shoot the part on the board
```

Output: `captures/<label>_cam<N>_<timestamp>.jpg`. Copy to the box running
cad-automation → Calibrate tab (per-camera intrinsics) → multi-view fit.

### Golden rules
- **Run `lock` once.** Do NOT re-lock between calibration and measurement — identical
  intrinsics across both is the whole point.
- **Never move the cameras** after calibration. Move only the part/board, between shots.
- **Calibrate and capture at the same resolution** (pinned to 4608×2592 here) — the app's
  resolution gate rejects mismatches.
- Use the **standard** Module 3 (66° FOV), not the wide (120°).
- Mount cameras **rigidly**, 30–60° apart, convergent on one working volume.

Assumes native dual-CSI (camera indices 0 and 1, e.g. Pi 5). With an Arducam multiplexer,
only one camera is live at a time — adapt the capture loop to switch the mux GPIO.
