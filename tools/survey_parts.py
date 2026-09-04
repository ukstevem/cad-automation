#!/usr/bin/env python3
"""
Run the synthetic orientation test across many REAL parts, and label every trial.

This is the answer to "what is our confidence on a different item", which could not be given
honestly before: two CAD-only screens were tried and both were rejected by the single physical
part available, and one part can reject a screen but cannot calibrate one.

It now can be calibrated, because three things are in place. The synthetic harness reproduces the
rig's behaviour AND its end-on failure (tools/synth_capture.py, anchored). Mesh-derived edge
models fit real captures as well as CAD-derived ones - 21.09 px against 21.26 px, same pose to
0.2 degrees - so any of the ~4000 part meshes under outputs/stl/ can be tested without its STEP
file. And scoring needs no images, only the sign of a dot product against known truth.

Each part is SCALED so its longest dimension matches the test article. The rig has one working
volume and one camera distance; feeding it a 5.5 m bridge member would test whether the part fits
the frame, not whether its shape carries its orientation. Scaling isolates SHAPE, which is the
variable of interest.

The output is a labelled dataset: for every part, how often its orientation was recovered, beside
CAD-only measures of its symmetry. Fit a screen to THAT and the screen has ground truth behind it.

    docker compose run --rm --no-deps api python tools/survey_parts.py \
        --picks outputs/ar_models/_survey_picks.json --yaw 0:150:30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mesh_to_model as M2M  # noqa: E402


def write_stl(tris: np.ndarray, path: str) -> None:
    """Write a binary STL. Needed because the render and the model must be the SAME scale."""
    n = len(tris)
    out = bytearray(b"\0" * 80 + struct.pack("<I", n))
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 1e-12, nrm / np.maximum(ln, 1e-12), 0.0)
    for i in range(n):
        out += np.asarray(nrm[i], np.float32).tobytes()
        out += np.asarray(tris[i], np.float32).tobytes()
        out += b"\0\0"
    with open(path, "wb") as fh:
        fh.write(bytes(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picks", required=True, help="JSON list of STL paths")
    ap.add_argument("--target-mm", type=float, default=432.0,
                    help="scale each part to this longest dimension (the test article's size)")
    ap.add_argument("--yaw", default="0:150:30")
    ap.add_argument("--rig", default="outputs/ar_captures/turn90")
    ap.add_argument("--base-fit", default="outputs/ar_fits/turn90")
    ap.add_argument("--work", default="outputs/ar_survey")
    ap.add_argument("--csv", default="outputs/ar_fits/survey.csv")
    ap.add_argument("--angle", type=float, default=25.0)
    args = ap.parse_args()

    with open(args.picks, "r", encoding="utf-8") as fh:
        picks = json.load(fh)
    os.makedirs(args.work, exist_ok=True)
    os.makedirs("outputs/ar_models/_survey", exist_ok=True)

    summary = []
    for k, src in enumerate(picks):
        base = "p%02d" % k
        try:
            tris = M2M.load_stl(src)
        except Exception as exc:
            print("  %s skip (%s)" % (base, exc))
            continue
        pts = tris.reshape(-1, 3)
        longest = float((pts.max(axis=0) - pts.min(axis=0)).max())
        if longest < 1e-6:
            continue
        scale = args.target_mm / longest
        stl_path = os.path.join("outputs/ar_models/_survey", base + ".stl")
        json_path = os.path.join("outputs/ar_models/_survey", base + ".json")
        write_stl(tris * scale, stl_path)
        try:
            model = M2M.build(stl_path, angle_deg=args.angle, name=base)
        except Exception as exc:
            print("  %s skip (%s)" % (base, exc))
            continue
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(model, fh)

        out_csv = os.path.join(args.work, base + ".csv")
        print("\n=== %s  %s  (scaled x%.4f, %d edges) ==="
              % (base, os.path.basename(src)[:48], scale, model["summary"]["edges"]))
        cmd = [sys.executable, "tools/synth_sweep.py", "--mesh", stl_path,
               "--model", json_path, "--rig", args.rig, "--base-fit", args.base_fit,
               "--yaw", args.yaw, "--work", os.path.join(args.work, base), "--csv", out_csv]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_csv):
            print("  sweep failed: %s" % (proc.stderr or "").strip()[-200:])
            continue
        rows = list(csv.DictReader(open(out_csv)))
        ok = [r for r in rows if r.get("orientation") == "ok"]
        n = len([r for r in rows if r.get("status") == "ok"])
        rate = len(ok) / n if n else float("nan")
        print("  recovered %d/%d" % (len(ok), n))
        summary.append({
            "part": base,
            "source": os.path.basename(src),
            "scale": round(scale, 5),
            "len_mm": round(longest, 1),
            "edges": model["summary"]["edges"],
            "trials": n,
            "recovered": len(ok),
            "rate": round(rate, 3) if n else "",
        })

    if summary:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print("\n=== SURVEY ===")
        for s in summary:
            print("  %-5s %-46s %2d/%-2d  %s"
                  % (s["part"], s["source"][:46], s["recovered"], s["trials"], s["rate"]))
        print("\nwrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
