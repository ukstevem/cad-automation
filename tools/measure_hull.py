#!/usr/bin/env python3
"""
Measure the part from the photographs alone - no CAD model, no pose fit.

The fit's own RMS cannot tell you whether the fit is *right*. It is the quantity the solver
minimised, so it reports how well the solver did at its own task; a confidently wrong pose scores
well (this project has produced several). What is needed is a yardstick built from different
information, and the silhouettes are exactly that: they never touch the CAD.

Method is the classic visual hull (shape-from-silhouette). Sweep a voxel grid through the working
volume, project each voxel into every view, and keep the ones that land inside the subject
silhouette in *all* of them. What survives is a bound on the space the real object occupies.

Two views bound it loosely - the intersection of two cones keeps phantom volume that a third view
would carve away - so treat the *extents* as upper bounds. The centroid and the principal axis are
far more robust than the volume, because the phantom volume is roughly symmetric about the true
object, and those are the two numbers this tool exists to provide.

With ``--fit`` it also measures the CAD at the fitted pose and prints the difference, in mm. That
is the honest accuracy statement: how far the fit sits from where the photographs say the part is.

    docker compose run --rm --no-deps api python tools/measure_hull.py \
        outputs/ar_captures/run1 --profile outputs/calibration/Cam.json \
        --fit outputs/ar_fits/fit03/fit.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services import charuco, image_edges, multiview_fit as MVF  # noqa: E402
from app.services.board_pose import charuco_board_pose  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_photos(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in IMAGE_EXTS:
                out.extend(sorted(glob.glob(os.path.join(p, "*" + ext))))
                out.extend(sorted(glob.glob(os.path.join(p, "*" + ext.upper()))))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return [p for p in uniq if "overlay" not in os.path.basename(p).lower()]


def footprint_stats(pts_xy: np.ndarray) -> dict:
    """Centroid, principal axis and extents of a 2-D point set, via PCA."""
    c = pts_xy.mean(axis=0)
    d = pts_xy - c
    # Principal axis: the direction of greatest spread. For a long part that is its length axis,
    # which is what makes this comparable with the CAD's own long axis.
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    proj = d @ evecs
    length = float(proj[:, 0].max() - proj[:, 0].min())
    width = float(proj[:, 1].max() - proj[:, 1].min())
    angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
    # Axis direction is arbitrary in sign; fold to [-90, 90) so two measurements compare.
    angle = (angle + 90.0) % 180.0 - 90.0
    return {
        "centre_mm": [round(float(c[0]), 1), round(float(c[1]), 1)],
        "axis_deg": round(angle, 2),
        "length_mm": round(length, 1),
        "width_mm": round(width, 1),
        "aspect": round(length / width, 2) if width > 1e-6 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH")
    ap.add_argument("--fit", default=None, help="fit.json to compare the CAD pose against")
    ap.add_argument("--model", default=None, help="model JSON (default: taken from --fit)")
    ap.add_argument("--voxel", type=float, default=4.0, help="voxel pitch in mm (default 4)")
    ap.add_argument("--z-range", type=float, default=260.0, help="how far to sweep off the board (mm)")
    ap.add_argument("--margin", type=float, default=150.0, help="xy margin beyond the board (mm)")
    ap.add_argument("--min-corners", type=int, default=6)
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--no-self-test", dest="self_test", action="store_false",
                    help="measure THIS TOOL's own bias. Renders the CAD's silhouettes at the "
                         "fitted pose, builds a hull from those, and reports how far that hull's "
                         "centroid falls from the CAD's known centre. Two views bound a shape "
                         "loosely and the phantom volume need not be symmetric, so the hull "
                         "centroid carries a bias of its own - and a seating error smaller than "
                         "that bias is not a measurement, it is noise. On by default.")
    ap.set_defaults(self_test=True)
    args = ap.parse_args()

    overrides = [(s.split("=", 1)[0], MVF.load_profile(s.split("=", 1)[1])) for s in args.cam_profile]
    default_profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(default_profile["board"])
    detector = charuco.make_detector(board)

    photos = collect_photos(args.photos)
    if not photos:
        print("no photos found", file=sys.stderr)
        return 2

    views, silhouettes = [], []
    for path in photos:
        name = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print("  skip %s: unreadable" % name, file=sys.stderr)
            continue
        prof = default_profile
        for sub, p in overrides:
            if sub in name:
                prof = p
                break
        gray = image_edges.to_gray(img)
        corners, ids, _mc, _mi = charuco.detect_board_detailed(detector, gray)
        if ids is None or len(ids) < args.min_corners:
            print("  skip %s: board not found" % name, file=sys.stderr)
            continue
        rvec_cam, tvec_cam, _n = charuco_board_pose(
            corners, ids, board, prof["K"], prof["dist"], min_corners=args.min_corners
        )
        h, w = img.shape[:2]
        exclude = image_edges.board_grid_mask(
            board, rvec_cam, tvec_cam, prof["K"], prof["dist"], (h, w), band_px=7
        )
        keep = image_edges.working_area_mask(
            board, rvec_cam, tvec_cam, prof["K"], prof["dist"], (h, w), margin_mm=args.margin
        )
        # grow_px=0: the fit dilates the mask so Canny can see the gradient either side of the
        # boundary, but here the boundary IS the measurement and must not be inflated.
        sil = image_edges.segment_subject(
            img, exclude_mask=MVF.keep_to_exclude(keep, exclude), grow_px=0
        )
        if sil is None:
            print("  skip %s: no subject segmented" % name, file=sys.stderr)
            continue
        views.append({"K": prof["K"], "dist": prof["dist"], "rvec_cam": rvec_cam,
                      "tvec_cam": tvec_cam, "label": name, "shape": (h, w), "image": img,
                      "width": w, "height": h,
                      "unknown": exclude})
        silhouettes.append(sil > 0)
        print("  %s: silhouette %d px" % (name, int((sil > 0).sum())))

    if len(views) < 2:
        print("need >=2 usable views, got %d" % len(views), file=sys.stderr)
        return 2

    # The part rests on the board, and the board's Z points AWAY from the camera, so the object
    # occupies NEGATIVE Z. Measured from the solved poses rather than assumed - see CLAUDE.md.
    side = MVF.camera_side_of_board(views)
    bw, bh = MVF.board_extent_mm(board)
    m = args.margin
    xs = np.arange(-m, bw + m, args.voxel)
    ys = np.arange(-m, bh + m, args.voxel)
    zs = np.arange(0.0, args.z_range, args.voxel) * side
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    vox = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    print("\nsweeping %d voxels at %g mm over x[%.0f,%.0f] y[%.0f,%.0f] z[%.0f,%.0f]"
          % (len(vox), args.voxel, xs[0], xs[-1], ys[0], ys[-1], zs[0], zs[-1]))

    occupied = np.ones(len(vox), bool)
    for view, sil in zip(views, silhouettes):
        h, w = view["shape"]
        uv, _ = cv2.projectPoints(vox[occupied].reshape(-1, 1, 3), view["rvec_cam"],
                                  view["tvec_cam"], view["K"], view["dist"])
        uv = uv.reshape(-1, 2)
        xi = np.round(uv[:, 0]).astype(np.int64)
        yi = np.round(uv[:, 1]).astype(np.int64)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        hit = np.zeros(len(uv), bool)
        hit[inside] = sil[yi[inside], xi[inside]]
        idx = np.flatnonzero(occupied)
        occupied[idx[~hit]] = False
        print("  after %s: %d voxels" % (view["label"], int(occupied.sum())))

    if not occupied.any():
        print("visual hull is empty - silhouettes do not intersect", file=sys.stderr)
        return 1

    pts = vox[occupied]
    hull = footprint_stats(pts[:, :2])
    hull["height_mm"] = round(float(abs(pts[:, 2]).max()), 1)
    hull["voxels"] = int(len(pts))
    result = {"hull": hull, "views": [v["label"] for v in views], "voxel_mm": args.voxel}

    print("\n=== MEASURED FROM PHOTOGRAPHS (no CAD) ===")
    for k, v in hull.items():
        print("  %-14s %s" % (k, v))

    if args.fit:
        with open(args.fit, "r", encoding="utf-8") as fh:
            fit = json.load(fh)
        model_path = args.model or os.path.join(
            "outputs/ar_models", os.path.basename(fit.get("mesh", "")).replace(".stl", ".json")
        )
        model = MVF.load_model(model_path)
        mp = np.vstack([np.asarray(e, np.float64).reshape(-1, 3) for e in model["edges"]])
        R, _ = cv2.Rodrigues(np.asarray(fit["rvec"], np.float64).reshape(3, 1))
        world = (R @ mp.T + np.asarray(fit["tvec"], np.float64).reshape(3, 1)).T
        cad = footprint_stats(world[:, :2])
        cad["height_mm"] = round(float(abs(world[:, 2]).max()), 1)

        dx = cad["centre_mm"][0] - hull["centre_mm"][0]
        dy = cad["centre_mm"][1] - hull["centre_mm"][1]
        dyaw = (cad["axis_deg"] - hull["axis_deg"] + 90.0) % 180.0 - 90.0
        delta = {
            "centre_dx_mm": round(dx, 1),
            "centre_dy_mm": round(dy, 1),
            "centre_offset_mm": round(float(np.hypot(dx, dy)), 1),
            "axis_delta_deg": round(dyaw, 2),
            "length_delta_mm": round(cad["length_mm"] - hull["length_mm"], 1),
        }
        result.update({"cad_at_fit": cad, "delta": delta, "fit": os.path.abspath(args.fit)})

        print("\n=== CAD AT THE FITTED POSE ===")
        for k, v in cad.items():
            print("  %-14s %s" % (k, v))
        # Sharper than comparing centroids, which is confounded by the hull's phantom volume.
        # The silhouette is an OUTER bound on the object: every point of the real part projects
        # inside it in every view. So a CAD point landing outside is *provably* misplaced - no
        # assumption about how loosely two views bound the shape. Containment is therefore a
        # lower bound on the error, and the one number here that cannot flatter the fit.
        # Only edges the camera could actually SEE. Without this every back-facing edge counts
        # against the pose - and on this part that is ~80% of them (see app/services/visibility.py).
        vis_mask = None
        mesh_path = os.path.join("outputs/ar_models", os.path.basename(fit.get("mesh", "")))
        if fit.get("mesh") and os.path.exists(mesh_path):
            from app.services import visibility
            tris = visibility.load_stl(mesh_path)
            vis_mask = [visibility.visible_edge_points(tris, mp, fit["rvec"], fit["tvec"], v)
                        for v in views]
            print("  (containment uses visible edges only: %s of %d per view)"
                  % ("/".join(str(int(m.sum())) for m in vis_mask), len(mp)))

        contain = []
        for i_view, (view, sil) in enumerate(zip(views, silhouettes)):
            h, w = view["shape"]
            uv, _ = cv2.projectPoints(world.reshape(-1, 1, 3), view["rvec_cam"],
                                      view["tvec_cam"], view["K"], view["dist"])
            uv = uv.reshape(-1, 2)
            xi = np.round(uv[:, 0]).astype(np.int64)
            yi = np.round(uv[:, 1]).astype(np.int64)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            # Three things would otherwise be miscounted as "the pose is wrong":
            #  1. the board mask cuts a HOLE in the silhouette wherever the part overlaps the
            #     pattern, so the truth there is unknowable, not false;
            #  2. back-facing edges project through the frame's open windows onto bare paper -
            #     correctly outside the silhouette, and nothing to do with alignment;
            #  3. points off the image edge.
            # Excluding all three leaves a number that only moves when the pose actually moves.
            unk = view.get("unknown")
            if unk is not None:
                known = np.ones(len(uv), bool)
                known[ok] = ~(unk[yi[ok], xi[ok]] > 0)
                ok = ok & known
            if vis_mask is not None:
                ok = ok & vis_mask[i_view]
            hit = np.zeros(len(uv), bool)
            hit[ok] = sil[yi[ok], xi[ok]]
            # How far outside? Distance transform of the silhouette's complement gives, for every
            # stray point, the px to the nearest point of the real subject.
            dt = cv2.distanceTransform((sil > 0).astype(np.uint8), cv2.DIST_L2, 5)
            far = np.zeros(len(uv))
            outside = ok & ~hit
            if outside.any():
                dt_out = cv2.distanceTransform((~(sil > 0)).astype(np.uint8), cv2.DIST_L2, 5)
                far[outside] = dt_out[yi[outside], xi[outside]]
            contain.append({
                "view": view["label"],
                "inside_fraction": round(float(hit[ok].mean()), 4) if ok.any() else None,
                "points_judged": int(ok.sum()),
                "median_stray_px": round(float(np.median(far[outside])), 1) if outside.any() else 0.0,
                "max_stray_px": round(float(far.max()), 1),
            })
            if args.out:
                # Look at this before believing the percentage. A CAD point can land outside the
                # silhouette for two very different reasons: the pose is wrong, or the silhouette
                # is INCOMPLETE because the board mask cut a hole in the part where it overlaps
                # the pattern. The picture separates them instantly; the number cannot.
                vis = view["image"].copy()
                vis[sil] = (0.45 * vis[sil] + 0.55 * np.array([0, 90, 0])).astype(np.uint8)
                for (x, y), good in zip(np.c_[xi, yi][ok], hit[ok]):
                    cv2.circle(vis, (int(x), int(y)), 1,
                               (120, 255, 120) if good else (60, 60, 255), -1)
                dbg = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                   os.path.splitext(view["label"])[0] + "_containment.png")
                cv2.imwrite(dbg, vis)
                print("    wrote %s" % os.path.basename(dbg))
        result["containment"] = contain

        # Worst case on the part itself: an axis error pivots the ends, so the tip displacement is
        # what a welder would actually see, not the centroid offset.
        half = cad["length_mm"] / 2.0
        tip = float(np.hypot(dx, dy) + half * abs(np.sin(np.radians(dyaw))))
        delta["worst_end_offset_mm"] = round(tip, 1)

        print("\n=== SEATING ERROR (CAD minus photographs) ===")
        print("  centre off by   %s mm (dx %s, dy %s)"
              % (delta["centre_offset_mm"], delta["centre_dx_mm"], delta["centre_dy_mm"]))
        print("  axis off by     %s deg" % delta["axis_delta_deg"])
        print("  worst end off   %s mm (centre offset + %.1f mm of pivot at the tips)"
              % (delta["worst_end_offset_mm"], half * abs(np.sin(np.radians(dyaw)))))
        print("  length differs  %s mm (hull over-reads: 2 views bound it loosely)"
              % delta["length_delta_mm"])
        print("\n=== CONTAINMENT (CAD points vs measured silhouette) ===")
        print("  A correct pose projects ~entirely INSIDE the silhouette in every view.")
        for c in contain:
            print("  %-42s %5.1f%% inside (%d judged), stray median %s px, max %s px"
                  % (c["view"], 100 * c["inside_fraction"], c["points_judged"],
                     c["median_stray_px"], c["max_stray_px"]))

    if args.self_test and args.fit and vis_mask is not None:
        from app.services import visibility as _vis
        synth = []
        for view in views:
            depth, _sc = _vis.depth_buffer(tris, fit["rvec"], fit["tvec"], view, downscale=1)
            synth.append(depth < _vis.FAR / 2)
            print("  synthetic silhouette %s: %d px" % (view["label"], int(synth[-1].sum())))
        occ = np.ones(len(vox), bool)
        for view, sil in zip(views, synth):
            h, w = view["shape"]
            uv, _ = cv2.projectPoints(vox[occ].reshape(-1, 1, 3), view["rvec_cam"],
                                      view["tvec_cam"], view["K"], view["dist"])
            uv = uv.reshape(-1, 2)
            xi2 = np.round(uv[:, 0]).astype(np.int64)
            yi2 = np.round(uv[:, 1]).astype(np.int64)
            ins = (xi2 >= 0) & (xi2 < w) & (yi2 >= 0) & (yi2 < h)
            hit2 = np.zeros(len(uv), bool)
            hit2[ins] = sil[yi2[ins], xi2[ins]]
            j = np.flatnonzero(occ)
            occ[j[~hit2]] = False
        if occ.any():
            sh = footprint_stats(vox[occ][:, :2])
            bx = sh["centre_mm"][0] - cad["centre_mm"][0]
            by = sh["centre_mm"][1] - cad["centre_mm"][1]
            bias = float(np.hypot(bx, by))
            result["self_test"] = {
                "synthetic_hull": sh, "bias_mm": round(bias, 1),
                "bias_dx_mm": round(bx, 1), "bias_dy_mm": round(by, 1),
                "bias_axis_deg": round((sh["axis_deg"] - cad["axis_deg"] + 90) % 180 - 90, 2),
                "length_inflation_mm": round(sh["length_mm"] - cad["length_mm"], 1),
            }
            print("=== SELF-TEST: this tool's own bias ===")
            print("  A hull built from the CAD's OWN silhouettes, versus the CAD's known centre.")
            print("  centre bias     %.1f mm (dx %.1f, dy %.1f)" % (bias, bx, by))
            print("  axis bias       %s deg" % result["self_test"]["bias_axis_deg"])
            print("  length inflated %s mm (the phantom volume, quantified)"
                  % result["self_test"]["length_inflation_mm"])
            print("  => a measured seating error must exceed %.1f mm to mean anything." % bias)

            # The bias is systematic, not noise: it is where a hull of THIS shape seen by THESE
            # cameras puts its centroid. So subtract it. A perfectly placed CAD would read as
            # offset by exactly the bias, not by zero - which is why the raw figure above is not
            # the answer, and on this capture is not even the right SIGN.
            # Signs matter here and are easy to get backwards. dx is (CAD - real hull); bx is
            # (synthetic hull - CAD). A perfectly placed CAD would make the real hull sit where
            # the synthetic one does, so it would READ as dx = -bx, not dx = 0. The error is
            # therefore how far dx departs from -bx, i.e. dx + bx.
            cdx, cdy = dx + bx, dy + by
            cax = (dyaw + result["self_test"]["bias_axis_deg"] + 90.0) % 180.0 - 90.0
            corr = {
                "centre_dx_mm": round(cdx, 1), "centre_dy_mm": round(cdy, 1),
                "centre_offset_mm": round(float(np.hypot(cdx, cdy)), 1),
                "axis_delta_deg": round(cax, 2),
                "worst_end_offset_mm": round(
                    float(np.hypot(cdx, cdy)) + half * abs(np.sin(np.radians(cax))), 1),
            }
            result["delta_corrected"] = corr
            print("")
            print("=== SEATING ERROR, BIAS-CORRECTED (use this one) ===")
            print("  centre off by   %s mm (dx %s, dy %s)"
                  % (corr["centre_offset_mm"], corr["centre_dx_mm"], corr["centre_dy_mm"]))
            print("  axis off by     %s deg" % corr["axis_delta_deg"])
            print("  worst end off   %s mm" % corr["worst_end_offset_mm"])

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
