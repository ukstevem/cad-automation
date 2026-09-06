#!/usr/bin/env python3
"""
Does adding stereo pairs improve noise and coverage as predicted, or not at all?

This is the measurement that decides whether the production cell is worth building. The proof of
concept runs one stereo pair, which gives ~9.6 mm cloud noise and sees ~18% of the model's edges -
neither good enough for a 10 mm go/no-go. Both are expected to improve with more pairs: noise as
1/sqrt(N) for the independent component, coverage because more viewpoints see more faces.

That expectation is the thing to test, because the two possible outcomes lead opposite ways.

  If error is largely INDEPENDENT between pairs, it averages down, coverage climbs, and six pairs
  reach the tolerance. Buy the cameras.

  If error is largely SYSTEMATIC - calibration bias, a consistent lean in the reconstruction,
  segmentation cutting the same silhouette short every time - then averaging changes nothing, and
  six pairs are six times the cost for the same answer. Far better to learn that on one pair.

Pairs are placed around the working volume at the rig's own elevation, each a narrow baseline of
its own. Camera poses are taken as exact rather than solved from the board: that isolates the
scaling question from board-detection error, which was separately measured at ~0.5 mm and would
add on top of everything here.

    docker compose run --rm --no-deps api python tools/scaling_test.py \\
        --rig outputs/ar_captures/turn90 --mesh outputs/ar_models/mainframe_default_1to5.stl \\
        --fit outputs/ar_fits/turn90 --pairs 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import charuco, image_edges, multiview_fit as MVF, visibility as VIS  # noqa: E402

import synth_capture as SC  # noqa: E402
from stereo_preview import projector_pose, speckle_scene  # noqa: E402
from real_stereo import rectified_cloud  # noqa: E402


def ring_of_pairs(n, centre, radius_mm, height_mm, baseline_mm, width, height, K, dist):
    """
    *n* stereo pairs spaced evenly around the working volume, all looking at its centre.

    Each pair is two cameras *baseline_mm* apart, aimed together - the narrow-baseline arrangement
    that matches well. The pairs themselves are far apart, which is what buys new viewpoints.
    """
    pairs = []
    for i in range(n):
        th = 2.0 * np.pi * i / n
        eye = centre + np.array([radius_mm * np.cos(th), radius_mm * np.sin(th), -height_mm])
        z = centre - eye
        z = z / np.linalg.norm(z)
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(z @ up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=0)
        views = []
        for side in (-0.5, +0.5):
            c = eye + R.T @ np.array([side * baseline_mm, 0.0, 0.0])
            rvec, _ = cv2.Rodrigues(R)
            views.append({"rvec_cam": rvec, "tvec_cam": (-R @ c).reshape(3, 1),
                          "width": width, "height": height, "K": K, "dist": dist,
                          "tag": "p%dc%d" % (i, 0 if side < 0 else 1)})
        pairs.append(views)
    return pairs


def fuse(cloud, voxel_mm=3.0):
    """
    Average measurements that land on the same bit of surface, rather than merely piling them up.

    Merging clouds is a union: N pairs give N times the points, each still carrying its own error,
    so the per-point noise is unchanged. Averaging is what actually reduces it - and it only
    happens where pairs OVERLAP, seeing the same surface from different sides. A voxel grid is the
    cheap way to associate them: points falling in the same cell are measurements of the same
    place, so replace them with their mean.

    This distinction decides the whole question. The 1/sqrt(N) prediction is about averaging; a
    test that merges instead measures nothing about it.
    """
    if not len(cloud):
        return cloud, 0.0
    key = np.floor(cloud / voxel_mm).astype(np.int64)
    _u, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    summed = np.zeros((len(_u), 3))
    np.add.at(summed, inv, cloud)
    fused = summed / counts[:, None]
    return fused, float(counts.mean())


def metrics(cloud, tris, rvec, tvec, tol_mm=10.0, sample=60000, targets=None):
    """
    Noise (cloud -> geometry) and coverage (geometry -> cloud), both in the board frame.

    *targets* are the object-frame points coverage is judged against. Pass the model's EDGE points
    to answer the requirement as written - "every observable edge within 10 mm" - rather than the
    easier question the surface asks. Edges are harder: they lie on depth discontinuities and
    occlusion boundaries, which is precisely where a stereo matcher is least reliable, so surface
    coverage flatters edge coverage.
    """
    from scipy.spatial import cKDTree

    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t = np.asarray(tvec, np.float64).reshape(3, 1)
    # NOISE is judged against the SURFACE and COVERAGE against the TARGETS, and mixing them makes
    # the first meaningless: most cloud points lie on faces, so their distance to the nearest EDGE
    # measures where the edges are, not how noisy the measurement is.
    surf = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    if len(surf) > sample:
        surf = surf[np.random.default_rng(0).choice(len(surf), sample, replace=False)]
    world = (R @ surf.T + t).T
    tgt = targets if targets is not None else surf
    if len(tgt) > sample:
        tgt = tgt[np.random.default_rng(2).choice(len(tgt), sample, replace=False)]
    tgt_world = (R @ tgt.T + t).T

    c = cloud
    if len(c) > sample:
        c = c[np.random.default_rng(1).choice(len(c), sample, replace=False)]
    # NOISE: how far measured points sit from the true surface. Points more than 40 mm out are
    # background or mismatches, not noise, and would swamp the statistic.
    d_ct, _ = cKDTree(world).query(c)
    near = d_ct < 40.0
    noise = float(np.median(d_ct[near])) if near.any() else float("nan")
    # COVERAGE: what fraction of the model has any measurement near it. This is the quantity the
    # go/no-go depends on - an edge with nothing observed near it cannot be judged at all.
    d_tc, _ = cKDTree(c).query(tgt_world)
    cover = float((d_tc < tol_mm).mean())
    return noise, cover, int(near.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--baseline", type=float, default=150.0)
    ap.add_argument("--grain", type=float, default=4.0)
    ap.add_argument("--tol", type=float, default=10.0)
    ap.add_argument("--no-speckle", action="store_true")
    ap.add_argument("--model", default=None,
                    help="AR model JSON; its edge points are what coverage is judged against, "
                         "since the requirement is stated on edges")
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    rig = SC.rig_from_captures(args.rig, profile, board, det)
    for v in rig:
        v["K"], v["dist"] = profile["K"], profile["dist"]
    K = np.asarray(profile["K"], np.float64).reshape(3, 3)
    dist = np.asarray(profile["dist"], np.float64).reshape(-1, 1)
    W, H = int(rig[0]["width"]), int(rig[0]["height"])

    tris = VIS.load_stl(args.mesh)
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)

    # Coverage is judged against the model's EDGES, not its surface - that is what the
    # requirement says, and it is the harder of the two.
    model_path = args.model or os.path.splitext(args.mesh)[0] + ".json"
    edge_pts = None
    if os.path.exists(model_path):
        m = MVF.load_model(model_path)
        edge_pts = MVF.sample_polylines(m["edges"], max_step=3.0)             if hasattr(MVF, "sample_polylines") else             np.vstack([np.asarray(e, np.float64).reshape(-1, 3) for e in m["edges"]])
        print("coverage judged against %d edge points from %s"
              % (len(edge_pts), os.path.basename(model_path)))
    else:
        print("no AR model found - falling back to SURFACE coverage, which flatters the result")

    # Ring geometry taken from where the rig's own cameras actually sit, so this is the same
    # elevation and standoff, not an idealised arrangement.
    R0, _ = cv2.Rodrigues(np.asarray(rig[0]["rvec_cam"], np.float64).reshape(3, 1))
    eye0 = (-R0.T @ np.asarray(rig[0]["tvec_cam"], np.float64).reshape(3, 1)).ravel()
    part = (cv2.Rodrigues(rvec)[0] @ tris.reshape(-1, 3).T + tvec).T
    centre = part.mean(axis=0)
    radius = float(np.hypot(*(eye0[:2] - centre[:2])))
    height = float(abs(eye0[2] - centre[2]))
    print("ring: %d pairs at radius %.0f mm, height %.0f mm above the part, baseline %.0f mm"
          % (args.pairs, radius, height, args.baseline))
    print("speckle: %s" % ("OFF" if args.no_speckle else "%.1f mm grain" % args.grain))
    print("")

    pairs = ring_of_pairs(args.pairs, centre, radius, height, args.baseline, W, H, K, dist)
    proj = projector_pose(rig, profile, board=board)

    clouds = []
    for i, views in enumerate(pairs):
        imgs = []
        for v in views:
            im = SC.render(tris, rvec, tvec, v, board, profile, shadow=0.35, noise=1.5)
            if not args.no_speckle:
                im = speckle_scene(im, tris, rvec, tvec, v, profile, grain_mm=args.grain,
                                   strength=0.55, proj=proj)
            imgs.append(im)
        d, _ = VIS.depth_buffer(tris, rvec, tvec, views[0], downscale=1)
        zh = d[d < VIS.FAR / 2]
        cc, _rec, n = rectified_cloud(imgs[0], imgs[1],
                                      views[0]["rvec_cam"], views[0]["tvec_cam"],
                                      views[1]["rvec_cam"], views[1]["tvec_cam"],
                                      K, dist, K, dist, z_hint=zh if len(zh) else None)
        if cc is None:
            print("  pair %d: matching failed" % i)
            clouds.append(np.zeros((0, 3)))
            continue
        Ra, _ = cv2.Rodrigues(np.asarray(views[0]["rvec_cam"], np.float64).reshape(3, 1))
        ta = np.asarray(views[0]["tvec_cam"], np.float64).reshape(3, 1)
        clouds.append((Ra.T @ (cc.T - ta)).T)
        print("  pair %d rendered and matched: %d points" % (i, len(clouds[-1])))

    print("")
    print("%-8s %10s %9s %11s %13s %13s"
          % ("pairs", "points", "merged", "fused", "EDGE cover", "surface cover"))
    base = None
    for n in range(1, args.pairs + 1):
        merged = np.vstack([c for c in clouds[:n] if len(c)])
        if not len(merged):
            continue
        noise, cover, near = metrics(merged, tris, rvec, tvec, tol_mm=args.tol, targets=edge_pts)
        fused, overlap = fuse(merged, voxel_mm=3.0)
        fnoise, fcover, _ = metrics(fused, tris, rvec, tvec, tol_mm=args.tol, targets=edge_pts)
        _n2, scover, _ = metrics(fused, tris, rvec, tvec, tol_mm=args.tol)
        if base is None:
            base = fnoise
        pred = base / np.sqrt(n)
        print("%-8d %10d %9.2f %11.2f %12.0f%% %12.0f%%"
              % (n, len(merged), noise, fnoise, 100 * fcover, 100 * scover))

    print("")
    print("'merged' is the union - more points, each with its own error. 'fused' averages the")
    print("measurements that land on the same 3 mm of surface, which is where noise can actually")
    print("fall. 'overlap' is how many measurements the average is over: near 1.0 the pairs are")
    print("not seeing the same surface and there is nothing to average.")
    print("")
    print("Fused noise tracking the prediction means the error is independent between pairs and more")
    print("cameras will reach the tolerance. Noise flat while coverage climbs means the error is")
    print("SYSTEMATIC - extra pairs buy visibility but not accuracy, and the fix is calibration or")
    print("reconstruction, not more hardware.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
