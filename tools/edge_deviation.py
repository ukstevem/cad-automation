#!/usr/bin/env python3
"""
Per-edge deviation in millimetres, and the go/no-go that follows from it.

The requirement is stated on edges: every observable edge of the article within 10 mm of its
modelled position. This measures exactly that, in the images, using what the rig already has - no
depth sensor, no projector, no point cloud.

HOW A PIXEL BECOMES A MILLIMETRE. Each sampled CAD edge point is projected into every camera
through the real intrinsics, and the distance to the nearest detected image edge is measured in
pixels. That distance is then scaled by the point's own depth: one pixel spans Z/f millimetres, so
10 mm is 17.5 px at 800 mm and 11.8 px at 1200 mm. Depth comes from the fitted pose, which is
known here because this is verification against a nominal placement, not a search.

WHY THE WORST CAMERA WINS. A 2D distance is the projection of a 3D one, so it can only
understate the true error - an edge displaced along one camera's viewing ray looks perfectly
placed to that camera. Taking the largest deviation across cameras is therefore the safe reading,
and it is why two views at 50 degrees measure something a single view cannot: what hides along
one ray is plainly visible along the other.

WHAT IT CANNOT SEE. An edge hidden from every camera is not tested and is reported as such, never
as a pass. Roughly four fifths of a model's edges are self-occluded from any one viewpoint, so the
coverage figure belongs beside the verdict - "all observable edges pass" is a different claim from
"all edges pass", and only the first is ever earned.

    docker compose run --rm --no-deps api python tools/edge_deviation.py \\
        --captures outputs/ar_captures/turn90 --fit outputs/ar_fits/turn90 \\
        --out outputs/ar_fits/deviation.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import charuco, image_edges, multiview_fit as MVF, visibility as VIS  # noqa: E402
from app.services.board_pose import charuco_board_pose  # noqa: E402
from critical_edges import geometric_edges, sensitivity  # noqa: E402

GREEN = (90, 190, 90)
AMBER = (60, 180, 235)
RED = (70, 70, 225)
GREY = (150, 150, 150)


def informative_deviation(mesh, rvec, tvec, views, min_sens=0.5, step=4):
    """
    Deviation measured only on edges that can actually SEE an error.

    Testing every edge equally is what makes the reading under-report: on an elongated part about
    half the edges run along its length and are blind to length-wise motion, so averaging them in
    drags the answer toward zero regardless of how far the part has moved. Keeping only points
    whose sensitivity exceeds *min_sens* - measured per point, per axis, by exact correspondence -
    leaves the ones whose position carries information.

    Sensitivity is taken as the best of the three axes, because the error's direction is not known
    in advance; a point that can see any direction well is worth measuring.
    """
    from scipy.spatial import cKDTree

    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    pts_o = np.vstack([mesh.reshape(-1, 3), mesh.mean(axis=1)])
    d = pts_o - pts_o.mean(axis=0)
    ev, evec = np.linalg.eigh(np.cov(d.T))
    o = np.argsort(ev)[::-1]
    axes = {"length": R @ evec[:, o[0]], "width": R @ evec[:, o[1]], "depth": R @ evec[:, o[2]]}

    per_view, all_mm, all_keep = [], [], []
    for v in views:
        pts, tan, z, sens = sensitivity(mesh, rvec, tvec, v, axes, step=step)
        if not len(pts):
            per_view.append({"view": v, "uv": pts, "usable": np.zeros(0, bool), "mm": np.zeros(0)})
            all_mm.append(np.zeros(0))
            continue
        K = np.asarray(v["K"], np.float64).reshape(3, 3)
        fx = float(K[0, 0])
        best = np.max(np.stack([sens[k] * z / fx for k in axes], axis=0), axis=0)
        keep = best >= min_sens
        mm = np.full(len(pts), np.nan)
        if keep.any() and len(v["edge_pixels"]):
            dd, _ = cKDTree(v["edge_pixels"]).query(pts[keep])
            mm[keep] = dd * z[keep] / fx
        per_view.append({"view": v, "uv": pts, "usable": keep, "mm": mm})
        all_mm.append(mm)
        all_keep.append(keep)
    return per_view, all_mm


def silhouette_points(mesh, rvec, tvec, view, step=3):
    """
    The model's OCCLUDING CONTOUR in this view, as image points with their depths.

    Two reasons this beats testing every un-occluded edge. First, an edge can be geometrically
    visible and still produce no image edge - a join between near-coplanar faces has no brightness
    step to detect - so scoring its distance to the nearest edge measures nothing but the spacing
    of the structure. Second, and decisively, the outline is NOT repetitive: displace the model and
    the silhouette moves somewhere there is no other silhouette, so nearest-neighbour cannot latch
    onto a wrong counterpart the way it does among interior lattice members.
    """
    depth, _sc = VIS.depth_buffer(mesh, rvec, tvec, view, downscale=1)
    mask = (depth < VIS.FAR / 2).astype(np.uint8)
    if not mask.any():
        return np.zeros((0, 2)), np.zeros(0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c.reshape(-1, 2) for c in cnts if len(c) > 8])[::step]
    # depth just inside the contour, for the pixel -> mm scale
    h, w = mask.shape
    xi = np.clip(pts[:, 0].astype(int), 0, w - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
    z = depth[yi, xi]
    ok = z < VIS.FAR / 2
    return pts[ok].astype(np.float64), z[ok]


def silhouette_deviation(mesh, rvec, tvec, views, tol_mm=10.0, step=3):
    """Deviation of the model's outline from the observed one, worst camera wins."""
    from scipy.spatial import cKDTree

    per_view, all_mm = [], []
    for v in views:
        K = np.asarray(v["K"], np.float64).reshape(3, 3)
        fx = float(K[0, 0])
        pts, z = silhouette_points(mesh, rvec, tvec, v, step=step)
        mm = np.full(len(pts), np.nan)
        if len(pts) and len(v["edge_pixels"]):
            d, _ = cKDTree(v["edge_pixels"]).query(pts)
            mm = d * z / fx
        per_view.append({"view": v, "uv": pts, "usable": np.ones(len(pts), bool), "mm": mm})
        all_mm.append(mm)
    return per_view, all_mm


def deviation(edge_pts, rvec, tvec, views, tol_mm=10.0):
    """
    Per-edge-point deviation in mm, worst camera wins.

    Returns (dev_mm, tested, per_view) where *dev_mm* is NaN for points no camera can see.
    """
    from scipy.spatial import cKDTree

    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t = np.asarray(tvec, np.float64).reshape(3, 1)
    world = (R @ edge_pts.T + t).T
    n = len(edge_pts)
    dev = np.full(n, np.nan)
    seen = np.zeros(n, bool)
    per_view = []

    for v in views:
        K = np.asarray(v["K"], np.float64).reshape(3, 3)
        fx = float(K[0, 0])
        Rc, _ = cv2.Rodrigues(np.asarray(v["rvec_cam"], np.float64).reshape(3, 1))
        tc = np.asarray(v["tvec_cam"], np.float64).reshape(3, 1)

        vis = VIS.visible_edge_points(v["mesh"], edge_pts, rvec, tvec, v, tol_mm=3.0)
        uv, _ = cv2.projectPoints(world.reshape(-1, 1, 3), v["rvec_cam"], v["tvec_cam"],
                                  K, np.asarray(v["dist"], np.float64).reshape(-1, 1))
        uv = uv.reshape(-1, 2)
        h, w = v["height"], v["width"]
        inframe = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        usable = vis & inframe

        d_px = np.full(n, np.nan)
        if usable.any() and len(v["edge_pixels"]):
            d, _ = cKDTree(v["edge_pixels"]).query(uv[usable])
            d_px[usable] = d
        # depth of each point in THIS camera, for the pixel -> mm scale
        Z = (Rc @ world.T + tc)[2]
        mm = d_px * Z / fx

        better = usable & (~np.isnan(mm))
        dev[better] = np.where(np.isnan(dev[better]), mm[better],
                               np.maximum(dev[better], mm[better]))
        seen |= better
        per_view.append({"view": v, "uv": uv, "usable": usable, "mm": mm})

    return dev, seen, per_view


def draw(img, pv, dev, seen, tol_mm, label, radius=2):
    """Overlay the edge points coloured by deviation."""
    out = img.copy()
    uv, usable = pv["uv"], pv["usable"]
    order = np.argsort(np.nan_to_num(dev, nan=-1))       # worst drawn last, on top
    for i in order:
        if not usable[i]:
            continue
        x, y = int(round(uv[i, 0])), int(round(uv[i, 1]))
        d = dev[i]
        if np.isnan(d):
            col = GREY
        elif d <= tol_mm * 0.5:
            col = GREEN
        elif d <= tol_mm:
            col = AMBER
        else:
            col = RED
        cv2.circle(out, (x, y), radius, col, -1, lineType=cv2.LINE_AA)
    # points no camera could see, drawn faintly so the untested region is visible not invisible
    for i in np.nonzero(~seen)[0][::7]:
        x, y = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), 1, GREY, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 64), (28, 28, 28), -1)
    cv2.putText(out, label, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 240), 2)
    return out


def legend(width, tol_mm, stats):
    """A caption strip carrying the verdict and the numbers behind it."""
    h = 150
    bar = np.full((h, width, 3), 24, np.uint8)
    go = stats["fail"] == 0 and stats["tested"] > 0
    verdict = "GO" if go else "NO-GO"
    vcol = GREEN if go else RED
    cv2.putText(bar, verdict, (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.8, vcol, 3)
    txt = ("%d of %d edge points observable (%.0f%%)   |   within %.0f mm: %d    over: %d"
           % (stats["tested"], stats["total"], 100.0 * stats["tested"] / max(stats["total"], 1),
              tol_mm, stats["pass"], stats["fail"]))
    cv2.putText(bar, txt, (24, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (215, 215, 215), 1)
    txt2 = ("worst %.1f mm   95th pct %.1f mm   median %.1f mm   |   unobservable edges are NOT "
            "tested and NOT passed" % (stats["worst"], stats["p95"], stats["median"]))
    cv2.putText(bar, txt2, (24, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (165, 165, 165), 1)
    x = width - 470
    for col, lab in ((GREEN, "<= %.0f mm" % (tol_mm / 2)), (AMBER, "<= %.0f mm" % tol_mm),
                     (RED, "over"), (GREY, "not visible")):
        cv2.circle(bar, (x, 56), 8, col, -1, lineType=cv2.LINE_AA)
        cv2.putText(bar, lab, (x + 18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        x += 120
    return bar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--fit", required=True, help="fit.json giving the pose to verify")
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--tol", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=4.0, help="CAD edge sampling in mm")
    ap.add_argument("--informative", action="store_true",
                    help="measure only on edges that can see an error - filters by per-point "
                         "sensitivity instead of weighting blind edges into the average")
    ap.add_argument("--min-sens", type=float, default=0.5)
    ap.add_argument("--silhouette", action="store_true",
                    help="test the model's OUTLINE rather than every un-occluded edge")
    ap.add_argument("--offset", default=None, metavar="DX,DY,DZ",
                    help="displace the model by this much before testing - used to check the "
                         "tool reports a deviation it was given")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)

    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    model_path = args.model or os.path.join(
        "outputs/ar_models", os.path.basename(fit.get("mesh", "")).replace(".stl", ".json"))
    mesh_path = args.mesh or os.path.join("outputs/ar_models", os.path.basename(fit.get("mesh", "")))
    model = MVF.load_model(model_path)
    mesh = VIS.load_stl(mesh_path)
    edge_pts = MVF.sample_polylines(model["edges"], max_step=args.step)

    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)
    if args.offset:
        d = np.asarray([float(x) for x in args.offset.split(",")], np.float64).reshape(3, 1)
        tvec = tvec + d
        print("model displaced by %s mm before testing" % args.offset)

    views = []
    for path in sorted(glob.glob(os.path.join(args.captures, "*"))):
        base = os.path.basename(path)
        if any(k in base for k in ("overlay", "endcheck", "containment", "deviation")):
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        v = MVF.build_view(img, profile, board, det, label=base)
        v.update({"K": profile["K"], "dist": profile["dist"], "mesh": mesh,
                  "image": img, "tag": base})
        views.append(v)
    if not views:
        print("no usable captures in %s" % args.captures, file=sys.stderr)
        return 2

    if args.informative:
        per_view, all_mm = informative_deviation(mesh, rvec, tvec, views, min_sens=args.min_sens)
        tested = np.concatenate([m[~np.isnan(m)] for m in all_mm]) if all_mm else np.zeros(0)
        dev = np.concatenate(all_mm) if all_mm else np.zeros(0)
        seen = ~np.isnan(dev)
        edge_pts = dev
    elif args.silhouette:
        per_view, all_mm = silhouette_deviation(mesh, rvec, tvec, views, tol_mm=args.tol)
        tested = np.concatenate([m[~np.isnan(m)] for m in all_mm]) if all_mm else np.zeros(0)
        dev = np.concatenate(all_mm) if all_mm else np.zeros(0)
        seen = ~np.isnan(dev)
        edge_pts = dev                                   # only used for the total below
    else:
        dev, seen, per_view = deviation(edge_pts, rvec, tvec, views, tol_mm=args.tol)
        tested = dev[seen]
    stats = {
        "total": len(edge_pts), "tested": int(seen.sum()),
        "pass": int((tested <= args.tol).sum()), "fail": int((tested > args.tol).sum()),
        "worst": float(np.nanmax(tested)) if len(tested) else float("nan"),
        "p95": float(np.nanpercentile(tested, 95)) if len(tested) else float("nan"),
        "median": float(np.nanmedian(tested)) if len(tested) else float("nan"),
    }

    print("")
    print("edge points        %d sampled at %.0f mm" % (stats["total"], args.step))
    print("observable         %d (%.0f%%) - the rest are self-occluded and NOT tested"
          % (stats["tested"], 100.0 * stats["tested"] / max(stats["total"], 1)))
    print("within %.0f mm       %d" % (args.tol, stats["pass"]))
    print("over               %d" % stats["fail"])
    print("median / p95 / worst   %.1f / %.1f / %.1f mm"
          % (stats["median"], stats["p95"], stats["worst"]))
    print("")
    print("VERDICT: %s" % ("GO" if stats["fail"] == 0 and stats["tested"] else "NO-GO"))

    if args.silhouette or args.informative:
        off = 0
        panels = []
        for pv in per_view:
            n = len(pv["uv"])
            panels.append(draw(pv["view"]["image"], pv, dev[off:off + n], seen[off:off + n],
                               args.tol, "%s  -  outline, %d pts"
                               % (pv["view"]["tag"][:32], n)))
            off += n
    else:
        panels = [draw(pv["view"]["image"], pv, dev, seen, args.tol,
                   "%s  -  %d px edges" % (pv["view"]["tag"][:34], len(pv["view"]["edge_pixels"])))
                  for pv in per_view]
    hgt = min(p.shape[0] for p in panels)
    row = np.hstack([p[:hgt] for p in panels])
    grid = np.vstack([row, legend(row.shape[1], args.tol, stats)])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, grid)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
