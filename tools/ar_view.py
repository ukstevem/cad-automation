#!/usr/bin/env python3
"""
The tower in AR: its outline and its weld positions on both rig photographs, from a solved pose and
the IFC weld sidecar.

Everything on the page is computed for THIS placement from two inputs - the pose, and the welds in
`weld_locate.py extract`'s IFC-shaped sidecar. Nothing is drawn by hand, so a new placement is a new
page rather than an edit.

    docker compose run --rm --no-deps api python tools/ar_view.py \\
        --captures outputs/ar_captures/rot01 --fit outputs/ar_fits/rot01_seated \\
        --welds outputs/welds/mainframe_ifc.json --scale 0.2 \\
        --cam-profile B68DE55F=outputs/calibration/RigCam_B68DE55F.json \\
        --out outputs/ar_fits/rot01_seated

    # the app serves outputs/ar_fits, so open
    #   http://localhost:8000/outputs/ar_fits/rot01_seated/ar_view.html

THE OUTLINE IS COLOURED BY EVIDENCE, NOT DRAWN IN ONE COLOUR. Each sample along a visible model edge
asks the photograph whether a real line runs there, within --tol millimetres:
  green  confirmed   the line is in the photo where the model puts it
  red    missed      the photo has contrast nearby, but no line where the model puts one
  grey   untestable  no contrast at all, so this view cannot say either way
A single colour would hide the one thing worth seeing. On an article that does not match its model,
the red gathers where the two part company.

WELDS ON THE FACES EACH CAMERA LOOKS AT (bd bn5). Every weld carries the article faces it is on -
top, the long sides, the ends - from `weld_faces`, and a camera shows the welds on the faces it is
presented. That replaced a convex-hull test which made each camera show only its own end of the
tower. See tools/weld_faces.py for the reasoning and the measurements behind the defaults.

ALONG THE LENGTH, IN MILLIMETRES (bd h04). Deviation is pooled across the cameras by tenths of the
part's length and reported as a median, because that is the measurement an inspector can act on and
defend. A pass fraction cannot do that job: on the tower it reads 54% while the pose is good to about
half a millimetre, since it counts every sample with no contrast as a failure. Deviation is measured
in a WIDER window than the pass tolerance (--dev-tol), or the tolerance would truncate the very tail
that marks a defect. Bands whose median stands out from the rest of the part are flagged: on the
glued tower that is 2.0 mm at 40-50% of the length against 0.4 mm either side, which is where its two
halves meet.

TRUST BEFORE POSITION. Silhouette confirmation is checked first. Below --min-silhouette the weld
positions are withheld: a weld from a wrong pose is not approximately right, it is somewhere
plausible and wrong. The outline is still shown, so the page says why.

AS BUILT (bd 0sb). An article that does not match its drawing fails the trust gate, which is right
but says nothing about why. --deviation applies a measured departure from the drawing (see
tools/deviation.py) to the outline model and to the welds on the part that departs, so the page shows
the article as it is, draws those welds in blue where they are on it, and says plainly that it is not
as drawn. The IFC sidecar is never edited.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import visibility as VIS  # noqa: E402
import deviation as DV  # noqa: E402
import line_check as LC  # noqa: E402
import pose_refine as PR  # noqa: E402
import weld_faces as WF  # noqa: E402
import weld_locate as WL  # noqa: E402

BANDS = 10
CONFIRMED, MISSED, UNTESTABLE = 2, 1, 0


def length_axis(mesh, rvec, tvec):
    """The part's long axis in the board frame, its centre, and the model's extent along it.

    Taken from the part's own geometry - the principal axis of its vertices - not from the board,
    so 'along the length' means the same stretch of steel however the part is lying."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    pts = mesh.reshape(-1, 3).astype(np.float64)
    c = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))
    ax = evecs[:, int(np.argmax(evals))]
    s = (pts - c) @ ax
    return R @ ax, R @ c + np.asarray(tvec, np.float64).ravel(), float(s.min()), float(s.max())


def low_stretches(bands, below):
    """Consecutive tenths under a threshold, merged into ``[(from_pct, to_pct, worst_pct)]``."""
    out, cur = [], None
    for b in bands:
        low = b["pct"] is not None and b["pct"] < below
        if low and cur is None:
            cur = [b["from"], b["to"], b["pct"]]
        elif low:
            cur[1], cur[2] = b["to"], min(cur[2], b["pct"])
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def deviation_regions(bands, factor=2.5, least_mm=1.0, min_samples=100):
    """Consecutive tenths whose median deviation stands out from the rest of the part.

    Compared against the part's own typical deviation rather than a fixed limit, because what counts
    as normal depends on the article, the camera distance and the model's fidelity. A stretch needs
    enough measured samples to be worth flagging - some tenths of an open frame hold almost nothing
    the cameras can see.

    Returns ``([(from_pct, to_pct, worst_mm)], typical_mm, threshold_mm)``."""
    usable = sorted(b["median_mm"] for b in bands
                    if b["median_mm"] is not None and b["n"] >= min_samples)
    if not usable:
        return [], None, None
    # The part's own standard is what it achieves WHERE IT FITS - the lower quartile of the tenths -
    # not their median. A defect drags the median up and then hides behind it: on the glued tower the
    # median of the tenths is 1.00 mm, so a median-based threshold would sit above the 1.98 mm bump
    # it exists to catch, while the quarter-best is 0.70 mm and the bump stands well clear.
    typical = usable[max(0, (len(usable) - 1) // 4)]
    threshold = max(least_mm, factor * typical)
    out, cur = [], None
    for b in bands:
        bad = (b["median_mm"] is not None and b["n"] >= min_samples
               and b["median_mm"] >= threshold)
        if bad and cur is None:
            cur = [b["from"], b["to"], b["median_mm"]]
        elif bad:
            cur[1], cur[2] = b["to"], max(cur[2], b["median_mm"])
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out, typical, threshold


def jpeg_data_uri(img, width):
    """The photograph, shrunk to the page width and embedded, so the page is one portable file."""
    s = min(1.0, width / float(img.shape[1]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    _ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    return uri, s, small.shape[1], small.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--fit", required=True, help="fit.json, or a directory holding one")
    ap.add_argument("--welds", required=True,
                    help="the IFC-shaped weld sidecar written by weld_locate.py extract")
    ap.add_argument("--mesh", default=None, help="defaults to the mesh named in the fit")
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="weld paths into the fitted model's units - the 1:5 tower needs 0.2")
    ap.add_argument("--tol", type=float, default=3.0,
                    help="mm either side of a model edge a photographed line may sit and still "
                         "confirm it")
    ap.add_argument("--min-silhouette", type=float, default=60.0,
                    help="below this, weld positions are withheld")
    ap.add_argument("--low-band", type=float, default=70.0,
                    help="flag stretches of the length whose outline confirmation falls below this")
    ap.add_argument("--dev-tol", type=float, default=5.0,
                    help="mm either side of a model edge to LOOK for the photographed line when "
                         "measuring how far off it is. Wider than --tol on purpose: measuring "
                         "deviation inside the pass tolerance truncates the tail that marks a defect")
    ap.add_argument("--dev-factor", type=float, default=2.5,
                    help="a tenth of the length is flagged when its median deviation reaches this "
                         "multiple of what the part achieves where it fits well - the lower quartile "
                         "of the tenths, not their median, which a defect would drag up with it "
                         "(and at least --dev-least)")
    ap.add_argument("--dev-least", type=float, default=1.0,
                    help="mm below which a stretch is never flagged, however it compares")
    ap.add_argument("--face-band", type=float, default=None,
                    help="mm in from a face plane a weld may sit and still be on that face; "
                         "default 20%% of the smaller cross-section extent. Used only where the "
                         "sidecar carries no ArticleFaces.")
    ap.add_argument("--min-face-cos", type=float, default=0.2,
                    help="how squarely a camera must face an article face to be shown its welds")
    ap.add_argument("--occlusion-mm", type=float, default=8.0,
                    help="how far behind the surface in front of it a weld point may sit and still "
                         "count as visible")
    ap.add_argument("--min-weld", type=float, default=0.0,
                    help="leave out joints shorter than this, in the fitted model's units")
    ap.add_argument("--deviation", default=None,
                    help="a measured departure from the drawing (tools/deviation.py), to show the "
                         "article as built; the mesh then defaults to the one it was measured on")
    ap.add_argument("--width", type=int, default=1400, help="photo width on the page, px")
    ap.add_argument("--out", required=True, help="directory for ar_view.html and ar_view.json")
    args = ap.parse_args()

    with open(args.welds, "r", encoding="utf-8") as fh:
        sidecar = json.load(fh)
    if not sidecar.get("schema"):
        print("%s is not an IFC-shaped sidecar - re-run weld_locate.py extract" % args.welds,
              file=sys.stderr)
        return 2
    if sidecar["welds"] and "PSS_WeldGeometry" not in sidecar["welds"][0]:
        print("%s was written before its property set dropped the reserved Pset_ prefix (bd nlk) - "
              "re-run weld_locate.py extract" % args.welds, file=sys.stderr)
        return 2
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)
    deviation = DV.load(args.deviation) if args.deviation else None
    drawn = deviation["article"]["mesh"] if deviation else os.path.basename(fit.get("mesh") or "")
    mesh = VIS.load_stl(args.mesh or os.path.join("outputs/ar_models", drawn))
    # The article frame always comes from the DRAWN mesh: face codes and deviations are defined in it.
    frame = WF.article_frame(mesh)
    turned = []
    if deviation:
        DV.check_article(deviation, frame)
        mesh, _mask = DV.as_built_mesh(mesh, frame, deviation)

    views = WL.load_views(args.captures, args.profile, args.cam_profile)
    if not views:
        print("no usable captures in %s" % args.captures, file=sys.stderr)
        return 2

    confirmed, silhouette = PR.score(mesh, rvec.ravel(), tvec.ravel(), views)
    withheld = silhouette < args.min_silhouette
    print("pose trust: %.0f%% confirmed, %.0f%% silhouette%s"
          % (confirmed, silhouette, "   -> WELD POSITIONS WITHHELD" if withheld else ""))

    ax_w, c_w, s_lo, s_hi = length_axis(mesh, rvec, tvec)
    span = max(s_hi - s_lo, 1e-9)
    # TWO passes, because the two questions want different windows. Pass or fail is asked at --tol,
    # and asking it there also keeps "untestable" meaning "no contrast within the pass tolerance",
    # which is what the outline colours and every confirmation figure have always meant. Deviation is
    # asked in the wider --dev-tol window, because a line found 4 mm away is a measurement, and the
    # pass tolerance would record it as "nothing there".
    per_view = LC.check(mesh, rvec, tvec, views, tol_mm=args.tol, reach=1.0, with_world=True)
    window = max(args.tol, args.dev_tol)
    per_dev = (per_view if window <= args.tol else
               LC.check(mesh, rvec, tvec, views, tol_mm=window, reach=1.0, with_world=True))

    welds = sidecar["welds"]
    if args.min_weld > 0:
        welds = [w for w in welds
                 if w["PSS_WeldGeometry"]["MeasuredLengthMm"] * args.scale >= args.min_weld]
    if deviation:
        welds, turned = DV.as_built_welds(welds, frame, deviation, args.scale)
        print("as built: %s - %d welds turned with it" % (deviation["description"], len(turned)))
    placed = WL.place_welds(welds, rvec, tvec, args.scale)

    band = args.face_band if args.face_band else WF.default_band(frame)
    faces_by_weld, computed = WF.faces_for_welds(welds, frame, args.scale, band)
    name_of = {code: WF.face_name(frame, rvec, code) for code in WF.FACES}
    no_face = sum(1 for f in faces_by_weld.values() if not f)
    print("faces: %s%s" % ("from the sidecar" if not computed else
                           "computed here for %d welds (band %.1f mm)" % (computed, band),
                           ", %d welds on no face" % no_face if no_face else ""))

    band_hit, band_test = np.zeros(BANDS), np.zeros(BANDS)
    band_dev = [[] for _ in range(BANDS)]
    seen_by = {name: [] for name, _w, _r in placed}
    page_views = []
    for i, (v, pv, pd) in enumerate(zip(views, per_view, per_dev)):
        uri, s, w, h = jpeg_data_uri(v["image"], args.width)
        letter = "ABCDEFGH"[i % 8]
        entry = {"tag": v["tag"], "letter": letter, "img": uri, "w": w, "h": h, "outline": [],
                 "counts": {"confirmed": 0, "missed": 0, "untestable": 0}, "welds": {},
                 "faces_presented": [], "weld_status": {}}
        pts = pv.get("pts")
        if pts is not None and len(pts) and pv.get("world") is not None:
            code = np.where(pv["found"], CONFIRMED, np.where(pv["blind"], UNTESTABLE, MISSED))
            pos = np.clip(((pv["world"] - c_w) @ ax_w - s_lo) / span, 0.0, 0.999999)
            band_idx = (pos * BANDS).astype(int)
            testable = code != UNTESTABLE
            np.add.at(band_test, band_idx[testable], 1)
            np.add.at(band_hit, band_idx[code == CONFIRMED], 1)
            # Deviation comes from the wider pass: every edge whose line was found, however far off.
            if pd.get("world") is not None and len(pd["dev"]):
                dpos = np.clip(((pd["world"] - c_w) @ ax_w - s_lo) / span, 0.0, 0.999999)
                didx = (dpos * BANDS).astype(int)
                for k in range(BANDS):
                    sel = pd["found"] & (didx == k)
                    if sel.any():
                        band_dev[k].append(pd["dev"][sel])
            # tenths of a page pixel as integers: exact enough to draw, a third of the size as JSON
            xy = np.round(pts * s * 10).astype(int)
            entry["outline"] = np.column_stack([xy, code]).ravel().tolist()
            entry["counts"] = {"confirmed": int((code == CONFIRMED).sum()),
                               "missed": int((code == MISSED).sum()),
                               "untestable": int((code == UNTESTABLE).sum())}
        lines, status, presented, cosines = WF.visible_weld_points(
            placed, faces_by_weld, mesh, rvec, tvec, v, frame, args.min_face_cos, args.occlusion_mm)
        entry["faces_presented"] = [name_of[c] for c in WF.FACES if c in presented]
        entry["face_cos"] = {name_of[c]: round(cosines[c], 2) for c in WF.FACES}
        entry["weld_status"] = dict(Counter(status.values()))
        if not withheld:
            for name, ls in lines.items():
                entry["welds"][name] = [np.round(np.asarray(line) * s * 10).astype(int)
                                        .ravel().tolist() for line in ls]
                seen_by[name].append(letter)
        page_views.append(entry)
        c = entry["counts"]
        print("%-32s outline %6d confirmed %6d missed %6d untestable"
              % (v["tag"][:32], c["confirmed"], c["missed"], c["untestable"]))
        print("%-32s faces presented: %s | welds %s"
              % ("", ", ".join(entry["faces_presented"]) or "none",
                 ", ".join("%s %d" % kv for kv in sorted(entry["weld_status"].items()))))

    bands = []
    for k in range(BANDS):
        pct = 100.0 * band_hit[k] / band_test[k] if band_test[k] else None
        d = np.concatenate(band_dev[k]) if band_dev[k] else np.zeros(0)
        bands.append({"from": 10 * k, "to": 10 * (k + 1), "testable": int(band_test[k]),
                      "pct": None if pct is None else round(float(pct), 1),
                      "n": int(len(d)),
                      "median_mm": round(float(np.median(d)), 2) if len(d) else None,
                      "p90_mm": round(float(np.percentile(d, 90)), 2) if len(d) else None})
    lows = low_stretches(bands, args.low_band)
    regions, typical_mm, dev_threshold = deviation_regions(bands, args.dev_factor, args.dev_least)

    rows = []
    if not withheld:
        for name, _w, r in placed:
            rows.append({"name": name, "centre": [round(float(x), 1) for x in r["centre"]],
                         "length": round(float(r["length"]), 1), "type": r["type"],
                         "runs": r["runs"], "seen": seen_by[name],
                         "faces": [name_of[c] for c in faces_by_weld.get(name, [])],
                         "as_built": name in turned})
        rows.sort(key=lambda r: -r["length"])

    data = {
        "generated": datetime.datetime.now().replace(microsecond=0).isoformat(sep=" "),
        "captures": os.path.basename(os.path.normpath(args.captures)),
        "pose": {"source": fit.get("init") or "multiview fit", "fit": os.path.normpath(src),
                 "resting_index": fit.get("resting_index")},
        "trust": {"confirmed": round(float(confirmed), 1), "silhouette": round(float(silhouette), 1),
                  "min_silhouette": args.min_silhouette, "withheld": bool(withheld),
                  "default_min_silhouette": ap.get_default("min_silhouette")},
        "sidecar": {"file": os.path.basename(args.welds), "schema": sidecar.get("schema"),
                    "piece_mark": (sidecar.get("piece_mark") or {}).get("value"),
                    "project": sidecar.get("project"), "welds": len(sidecar["welds"])},
        "faces": {"band_mm": round(float(band), 1), "min_cos": args.min_face_cos,
                  "source": "sidecar" if not computed else "computed", "no_face": no_face},
        "deviation": DV.summary(deviation, turned, args.deviation) if deviation else None,
        "tol_mm": args.tol, "scale": args.scale, "low_band": args.low_band,
        "dev": {"window_mm": window, "typical_mm": typical_mm, "threshold_mm": dev_threshold,
                "factor": args.dev_factor, "least_mm": args.dev_least},
        "dev_regions": regions,
        "views": page_views, "bands": bands, "low_runs": lows, "welds": rows,
    }

    os.makedirs(args.out, exist_ok=True)
    slim = dict(data, views=[{k: val for k, val in pv.items() if k not in ("img", "outline", "welds")}
                             for pv in page_views])
    with open(os.path.join(args.out, "ar_view.json"), "w", encoding="utf-8") as fh:
        json.dump(slim, fh, indent=1)
    html = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    dest = os.path.join(args.out, "ar_view.html")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(html)

    print("")
    print("deviation along the length, median mm (measured in a +/-%.0f mm window):" % window)
    print("  " + "  ".join("%d-%d%% %s" % (b["from"], b["to"],
                           "-" if b["median_mm"] is None else "%.2f" % b["median_mm"])
                           for b in bands))
    if typical_mm is not None:
        print("  typical for this part %.2f mm; a stretch is flagged at %.2f mm"
              % (typical_mm, dev_threshold))
    for lo, hi, worst in regions:
        print("  DEPARTS FROM THE MODEL at %d-%d%% of the length: %.2f mm against %.2f mm typical"
              % (lo, hi, worst, typical_mm))
    print("confirmed within %.0f mm: " % args.tol
          + "  ".join("%d-%d%% %s" % (b["from"], b["to"],
                      "-" if b["pct"] is None else "%.0f" % b["pct"]) for b in bands))
    for lo, hi, worst in lows:
        print("  outline confirmation below %.0f%% from %d%% to %d%% of the length (lowest %.0f%%)"
              % (args.low_band, lo, hi, worst))
    if len(page_views) >= 2 and not withheld:
        sets = [set(pv["welds"]) for pv in page_views]
        print("welds shown: %s | on both of the first two cameras: %d"
              % (", ".join("%s %d" % (pv["letter"], len(st)) for pv, st in zip(page_views, sets)),
                 len(sets[0] & sets[1])))
    print("%s weld positions" % ("withheld -" if withheld else "%d" % len(rows)))
    print("wrote %s (%.1f MB)" % (dest, os.path.getsize(dest) / 1e6))
    norm = os.path.normpath(args.out).replace("\\", "/")
    if "outputs/ar_fits/" in norm + "/":
        print("open http://localhost:8000/" + norm[norm.index("outputs/ar_fits"):] + "/ar_view.html")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tower AR</title>
<style>
:root{--bg:#121417;--panel:#1b1e23;--line:#2a2e35;--text:#e6e8ec;--dim:#9aa1ab;
      --weld:#ffb000;--asbuilt:#5ec8ff;--ok:#35c97a;--miss:#ff5a4f;--blind:#7d838c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
     font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:20px 24px 40px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 18px;margin-bottom:10px}
h1{font-size:22px;margin:0}
.meta{color:var(--dim);font-size:13px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 14px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 10px;
      font-variant-numeric:tabular-nums}
.chip b{font-weight:600;margin-left:4px}
.banner{border-radius:6px;padding:10px 14px;margin:0 0 12px;border:1px solid;max-width:110ch}
.banner.bad{background:#3a1715;border-color:#7a2a24}
.banner.note{background:#2c2410;border-color:#6b5316}
.banner.dev{background:#0f2536;border-color:#1f5578}
.tag{display:inline-block;margin-left:6px;padding:0 6px;border-radius:4px;font-size:11px;
     color:var(--asbuilt);background:#0f2536;border:1px solid #1f5578}
.tools{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;margin:0 0 12px;color:var(--dim)}
.tools label{display:inline-flex;gap:6px;align-items:center;cursor:pointer;color:var(--text)}
.tools input:focus-visible{outline:2px solid var(--weld);outline-offset:2px}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px}
.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:14px}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
figcaption{display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 10px;padding:8px 12px;
           color:var(--dim);font-size:13px;font-variant-numeric:tabular-nums}
figcaption .faces{flex-basis:100%;color:var(--text)}
canvas{display:block;width:100%;height:auto}
section{margin-top:22px}
h2{font-size:15px;margin:0 0 8px}
.strip{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:4px}
.bar{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:6px 2px;
     text-align:center;font-variant-numeric:tabular-nums;font-size:12px;color:var(--dim)}
.bar b{display:block;color:var(--text);font-size:14px}
.bar .sub{display:block;font-size:11px;color:var(--dim)}
.bar.flag{border-color:#7a2a24;background:#2a1613}
.bar.flag b{color:#ff8b80}
.bar i{display:block;height:6px;border-radius:3px;margin:4px 3px 0}
.muted{color:var(--dim)}
p.muted{margin:6px 0 10px;max-width:110ch}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th.l,td.l{text-align:left}
th{color:var(--dim);font-weight:500;background:var(--panel)}
tr[data-n]{cursor:pointer}
tr[data-n]:hover{background:#20242a}
tr.sel{background:#3a2e10}
</style>
</head>
<body>
<header><h1>Tower AR</h1><span class="meta" id="meta"></span></header>
<div class="chips" id="chips"></div>
<div id="banners"></div>
<div class="tools">
  <span>Outline</span>
  <label><input type="checkbox" id="t2"><span class="sw" style="background:var(--ok)"></span>confirmed</label>
  <label><input type="checkbox" id="t1"><span class="sw" style="background:var(--miss)"></span>missed</label>
  <label><input type="checkbox" id="t0"><span class="sw" style="background:var(--blind)"></span>untestable</label>
  <span></span>
  <label><input type="checkbox" id="tw"><span class="sw" style="background:var(--weld)"></span>welds</label>
  <label><input type="checkbox" id="tl">weld numbers</label>
  <span id="asbuiltkey" hidden><span class="sw" style="background:var(--asbuilt)"></span> welds as built</span>
</div>
<div class="views" id="views"></div>
<section>
  <h2>How far the part is from its model, along the tower's length</h2>
  <div class="strip" id="strip"></div>
  <p class="muted" id="stripnote"></p>
</section>
<section>
  <h2>Weld positions</h2>
  <p class="muted" id="weldnote"></p>
  <div class="tablewrap"><table>
    <thead><tr><th class="l">Weld</th><th>x mm</th><th>y mm</th><th>z mm</th><th>length mm</th>
      <th class="l">detected as</th><th class="l">faces</th><th class="l">in view</th></tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
</section>
<script>
const D = __DATA__;
const el = id => document.getElementById(id);
const S = {2: true, 1: true, 0: false, welds: true, labels: true, sel: null};
const COL = {2: '#35c97a', 1: '#ff5a4f', 0: '#7d838c'};
const TURNED = new Set(D.deviation ? D.deviation.turned_welds : []);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

el('meta').textContent = `${D.captures} · pose from ${D.pose.source} · ${D.sidecar.file} ` +
  `(${D.sidecar.schema}, ${D.sidecar.piece_mark}, ${D.sidecar.welds} welds) · ${D.generated}`;
const chip = (k, v) => `<span class="chip">${k}<b>${v}</b></span>`;
el('chips').innerHTML =
  chip('silhouette confirmed', D.trust.silhouette.toFixed(0) + '%') +
  chip('outline confirmed', D.trust.confirmed.toFixed(0) + '%') +
  chip('tolerance', '±' + D.tol_mm + ' mm') +
  chip('face band', D.faces.band_mm + ' mm') +
  chip('weld positions', D.trust.withheld ? 'withheld' : D.welds.length) +
  (D.deviation ? chip('as built', `${D.deviation.turned_half} half turned ${D.deviation.angle_deg}°`) : '');

let banners = '';
if (D.deviation) {
  el('asbuiltkey').hidden = false;
  banners += `<div class="banner dev"><b>As built, not as drawn.</b> ${esc(D.deviation.description)}` +
    (D.deviation.joint ? ` (${esc(D.deviation.joint)})` : '') + `, found by fitting the photographs.
    The outline is checked against the article as built, and the ${TURNED.size} welds on the turned
    half are drawn in blue where they are on this article. The drawing and its IFC weld sidecar are
    unchanged.</div>`;
}
if (!D.trust.withheld && D.trust.min_silhouette < D.trust.default_min_silhouette)
  banners += `<div class="banner note"><b>Trust gate lowered for this page.</b> Silhouette confirmation
  is ${D.trust.silhouette.toFixed(0)}%. Weld positions are shown from ${D.trust.min_silhouette}% instead
  of the usual ${D.trust.default_min_silhouette}%, so treat them as indicative and check them against
  the photographs.</div>`;
if (D.trust.withheld) banners += `<div class="banner bad"><b>Weld positions withheld.</b>
  Silhouette confirmation is ${D.trust.silhouette.toFixed(0)}%, below the ${D.trust.min_silhouette}%
  needed to trust the pose. A weld placed from a wrong pose is somewhere plausible and wrong, so
  none are shown. The outline still is, so you can see where the fit fails.</div>`;
if (D.low_runs.length) banners += `<div class="banner note"><b>The part and the model disagree along
  its length.</b> Outline confirmation falls below ${D.low_band}% at ` +
  D.low_runs.map(r => `${r[0]}–${r[1]}% of the length (lowest ${r[2].toFixed(0)}%)`).join(', ') +
  `. Weld positions in that stretch are where the model says they should be, which may not be
  where they are on this part.</div>`;
if (D.dev_regions.length) banners += `<div class="banner bad"><b>The part departs from its model.</b>
  ` + D.dev_regions.map(r => `${r[0]}–${r[1]}% of the length is out by ${r[2].toFixed(2)} mm`).join(', ') +
  `, against ${D.dev.typical_mm.toFixed(2)} mm typical for the rest of it. Weld positions there are
  where the model puts them, which is not where they are on this part.</div>`;
if (D.faces.no_face) banners += `<div class="banner note">${D.faces.no_face} welds are on no face of
  the article at a ${D.faces.band_mm} mm band, so no camera shows them.</div>`;
el('banners').innerHTML = banners;

el('views').innerHTML = D.views.map((v, i) => `<figure>
  <canvas id="cv${i}" width="${v.w}" height="${v.h}"></canvas>
  <figcaption><span>Camera ${v.letter} · ${esc(v.tag)}</span>
  <span>${v.counts.confirmed} confirmed · ${v.counts.missed} missed · ${v.counts.untestable} untestable
  · ${Object.keys(v.welds).length} welds</span>
  <span class="faces">Faces this camera looks at: ${v.faces_presented.length ? v.faces_presented.join(', ') : 'none'}</span>
  </figcaption></figure>`).join('');

const imgs = [];
function draw() {
  D.views.forEach((v, i) => {
    const img = imgs[i];
    if (!img || !img.complete) return;
    const x = el('cv' + i).getContext('2d');
    x.drawImage(img, 0, 0);
    const o = v.outline;
    for (const code of [0, 2, 1]) {                     // missed last, so it is never painted over
      if (!S[code]) continue;
      x.fillStyle = COL[code];
      for (let k = 0; k < o.length; k += 3)
        if (o[k + 2] === code) x.fillRect(o[k] / 10 - 1.1, o[k + 1] / 10 - 1.1, 2.2, 2.2);
    }
    if (!S.welds) return;
    x.lineCap = 'round'; x.lineJoin = 'round';
    for (const [name, lines] of Object.entries(v.welds)) {
      const hot = S.sel === name, tone = TURNED.has(name) ? '#5ec8ff' : '#ffb000';
      for (const l of lines) {
        x.beginPath();
        for (let k = 0; k < l.length; k += 2) {
          const px = l[k] / 10, py = l[k + 1] / 10;
          k ? x.lineTo(px, py) : x.moveTo(px, py);
        }
        if (l.length === 2) x.lineTo(l[0] / 10 + 0.1, l[1] / 10);
        x.lineWidth = hot ? 8 : 5; x.strokeStyle = 'rgba(0,0,0,.65)'; x.stroke();
        x.lineWidth = hot ? 5 : 2.5; x.strokeStyle = hot ? '#ffffff' : tone; x.stroke();
      }
      if (S.labels || hot) {
        const l = lines.reduce((a, b) => b.length > a.length ? b : a);
        const m = Math.floor(l.length / 4) * 2;
        const lx = l[m] / 10 + 6, ly = l[m + 1] / 10 - 6, t = name.split('-').pop();
        x.font = '600 13px system-ui,sans-serif';
        x.lineWidth = 3; x.strokeStyle = '#000'; x.strokeText(t, lx, ly);
        x.fillStyle = hot ? '#ffffff' : tone; x.fillText(t, lx, ly);
      }
    }
  });
}
D.views.forEach((v, i) => { const im = new Image(); im.onload = draw; im.src = v.img; imgs[i] = im; });
[['t2', 2], ['t1', 1], ['t0', 0], ['tw', 'welds'], ['tl', 'labels']].forEach(([id, k]) => {
  el(id).checked = S[k];
  el(id).onchange = e => { S[k] = e.target.checked; draw(); };
});

const flagged = pct => D.dev_regions.some(r => pct >= r[0] && pct < r[1]);
el('strip').innerHTML = D.bands.map(b => {
  const p = b.pct, m = b.median_mm;
  const c = p === null ? '#3a3f47' : p >= 80 ? '#35c97a' : p >= D.low_band ? '#e3a42a' : '#ff5a4f';
  return `<div class="bar${flagged(b.from) ? ' flag' : ''}"
    title="${b.n} measured deviations, ${b.testable} testable samples, 90th percentile ${b.p90_mm === null ? '–' : b.p90_mm + ' mm'}">${b.from}–${b.to}%
    <b>${m === null ? '–' : m.toFixed(2) + ' mm'}</b>
    <span class="sub">${p === null ? '–' : p.toFixed(0) + '% within ' + D.tol_mm + ' mm'}</span>
    <i style="background:${c}"></i></div>`;
}).join('');
el('stripnote').textContent =
  `Median distance from each model edge to the line found in the photographs, all cameras pooled, ` +
  `searched up to ±${D.dev.window_mm} mm. Typical for this part is ` +
  `${D.dev.typical_mm === null ? '–' : D.dev.typical_mm.toFixed(2) + ' mm'}; a stretch is flagged at ` +
  `${D.dev.threshold_mm === null ? '–' : D.dev.threshold_mm.toFixed(2) + ' mm'} ` +
  `(${D.dev.factor}× typical, never below ${D.dev.least_mm} mm). The percentage underneath is the ` +
  `share of testable samples confirmed within ${D.tol_mm} mm, and the bar's colour follows it. ` +
  `The two ends of the strip are the two ends of the part, taken from its own geometry.`;

const tb = el('rows');
if (D.trust.withheld) {
  el('weldnote').textContent = 'Withheld: the pose is not trusted.';
} else {
  el('weldnote').textContent = `Each weld's centre in the rig (board) frame, from the IFC sidecar ` +
    `through the solved pose, in millimetres on this article. A camera shows the welds on the ` +
    `article faces it looks at. Click a row to find the weld on the photos.` +
    (D.deviation ? ` Welds tagged as built are on the turned half: they are drawn where they are on ` +
      `this article, not where the drawing puts them.` : '');
  tb.innerHTML = D.welds.map(r => `<tr data-n="${esc(r.name)}"><td class="l">${esc(r.name)}${r.as_built ? ' <span class="tag">as built</span>' : ''}</td>
    <td>${r.centre[0].toFixed(1)}</td><td>${r.centre[1].toFixed(1)}</td><td>${r.centre[2].toFixed(1)}</td>
    <td>${r.length.toFixed(0)}</td><td class="l">${r.type ? esc(r.type) : '<span class="muted">–</span>'}</td>
    <td class="l">${r.faces.length ? esc(r.faces.join(', ')) : '<span class="muted">none</span>'}</td>
    <td class="l">${r.seen.length ? r.seen.join(' ') : '<span class="muted">not shown</span>'}</td></tr>`).join('');
  tb.onclick = e => {
    const tr = e.target.closest('tr[data-n]');
    if (!tr) return;
    S.sel = S.sel === tr.dataset.n ? null : tr.dataset.n;
    tb.querySelectorAll('tr').forEach(t => t.classList.toggle('sel', t.dataset.n === S.sel));
    draw();
  };
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
