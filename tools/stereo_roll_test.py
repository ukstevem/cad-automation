#!/usr/bin/env python3
"""
The decisive test: can a stereo point cloud tell the roll apart, when edges never could?

Roll about the long axis has failed in every test on this project - four controlled physical
trials and every synthetic sweep. tools/depth_discriminability.py showed why that need not be
permanent: under a 180 degree roll, 85-90% of the visible surface moves by more than 2 mm, which a
depth sensor resolves easily even though the outline barely changes.

That measured the IDEAL depth, straight from the renderer. This measures the depth a real stereo
pair would actually recover - speckle, matching, quantisation and all - and then asks the only
question that matters: score the cloud against the CAD at the true roll and at the flipped roll,
and see whether the true one wins.

    docker compose run --rm --no-deps api python tools/stereo_roll_test.py \
        --mesh outputs/ar_models/mainframe_default_1to5.stl \
        --fit outputs/ar_fits/turn90 --rig outputs/ar_captures/turn90 --baseline 150
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
from stereo_preview import offset_camera, speckle  # noqa: E402
from depth_discriminability import rotate_about_own_axis  # noqa: E402


def disparity_range(tris, rvec, tvec, view, profile, baseline_mm, pad=1.35):
    """
    Choose SGBM's search range from the geometry instead of guessing it.

    Disparity is f*B/Z, so the range follows directly from how near and far the part is. Guessing
    is how this test first went wrong: a 256 px range against a true disparity of ~322 px silently
    saturated, put the cloud 380 mm too far away, and made depth look useless. SGBM does not warn -
    it just returns the clipped value.
    """
    depth, _ = VIS.depth_buffer(tris, rvec, tvec, view, downscale=1)
    z = depth[depth < VIS.FAR / 2]
    if not len(z):
        return 0, 256
    f = float(np.asarray(profile["K"], np.float64).reshape(3, 3)[0, 0])
    d_far = f * baseline_mm / float(np.percentile(z, 99))
    d_near = f * baseline_mm / float(np.percentile(z, 1))
    lo = max(0, int(np.floor(d_far / 16.0) * 16) - 16)
    span = int(np.ceil((d_near * pad - lo) / 16.0) * 16)
    return lo, max(16, span)


def stereo_cloud(imgL, imgR, view, profile, baseline_mm, min_disp=0, num_disp=256, block=7):
    """
    Disparity by SGBM, then reprojection to a 3D cloud in the LEFT camera's frame.

    The virtual pair is built with parallel image planes and identical intrinsics, so the images
    are already rectified by construction and Q is the textbook form. A real pair would need
    stereoRectify first, using the relative pose the ChArUco board supplies for free.
    """
    K = np.asarray(profile["K"], np.float64).reshape(3, 3)
    gl = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    valid = disp > (min_disp + 0.5)

    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    ys, xs = np.nonzero(valid)
    d = disp[ys, xs]
    Z = fx * baseline_mm / d
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / K[1, 1]
    return np.stack([X, Y, Z], axis=1), valid, disp


def to_world(pts_cam, view):
    R, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    t = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    return (R.T @ (pts_cam.T - t)).T


def score_against(tris, rvec, tvec, cloud, sample=40000):
    """Median distance from each cloud point to the nearest CAD surface sample. Lower is better."""
    from scipy.spatial import cKDTree

    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t = np.asarray(tvec, np.float64).reshape(3, 1)
    surf = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    if len(surf) > sample:
        surf = surf[np.random.default_rng(0).choice(len(surf), sample, replace=False)]
    world = (R @ surf.T + t).T
    c = cloud
    if len(c) > sample:
        c = c[np.random.default_rng(1).choice(len(c), sample, replace=False)]
    dist, _ = cKDTree(world).query(c)
    # "within 5 mm" is the discriminating statistic: a wrong hypothesis can post a
    # similar median while explaining far less of the surface, because its errors are
    # a few large ones rather than many small ones.
    return float(np.median(dist)), float(np.mean(dist)), len(c), float((dist < 5.0).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--baseline", type=float, default=150.0)
    ap.add_argument("--grain", type=float, default=2.0)
    ap.add_argument("--no-speckle", action="store_true", help="passive stereo, for comparison")
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    views = SC.rig_from_captures(args.rig, profile, board, det)
    for v in views:
        v["K"], v["dist"] = profile["K"], profile["dist"]
    left = views[0]
    right = offset_camera(left, args.baseline)

    tris = VIS.load_stl(args.mesh)
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)

    imgs = []
    for v in (left, right):
        im = SC.render(tris, rvec, tvec, v, board, profile, shadow=0.35, noise=1.5)
        if not args.no_speckle:
            im = speckle(im, tris, rvec, tvec, v, profile, grain_mm=args.grain)
        imgs.append(im)

    lo, span = disparity_range(tris, rvec, tvec, left, profile, args.baseline)
    print("disparity search set from geometry: %d .. %d px" % (lo, lo + span))
    cloud_cam, valid, disp = stereo_cloud(imgs[0], imgs[1], left, profile, args.baseline,
                                          min_disp=lo, num_disp=span)
    print("stereo: %d valid pixels (%.1f%% of frame), disparity %.0f..%.0f px"
          % (valid.sum(), 100 * valid.mean(), disp[valid].min(), disp[valid].max()))
    if len(cloud_cam) < 500:
        print("too few points to judge - matching essentially failed", file=sys.stderr)
        return 1

    # Keep only points ON THE PART, decided from the IMAGE and not from the CAD.
    # A bounding box is not enough: this part is a frame, so paper seen THROUGH its
    # openings falls inside the box and drags every score to ~17 mm alike. Using the
    # CAD's own coverage to select points would be circular - it would favour whichever
    # pose selected them - so the mask comes from Otsu segmentation, which knows
    # nothing about the model.
    subj = image_edges.segment_subject(imgs[0], grow_px=0)
    _ys, _xs = np.nonzero(valid)
    on_part = np.ones(len(_ys), bool) if subj is None else (subj[_ys, _xs] > 0)
    cloud_cam = cloud_cam[on_part]
    print("       %d of %d matched pixels lie on the segmented subject"
          % (int(on_part.sum()), len(on_part)))
    # to_world AFTER the mask, or the cloud is the unfiltered one and every score is background.
    cloud = to_world(cloud_cam, left)
    print("       %d points in the cloud" % len(cloud))
    print("")

    print("%-34s %12s %12s %11s" % ("CAD hypothesis", "median mm", "mean mm", "within 5mm"))
    results = {}
    for label_, axis, deg in (("TRUE pose", None, 0.0),
                              ("rolled 180 about long axis", "long", 180.0),
                              ("rolled  90 about long axis", "long", 90.0),
                              ("flipped end-for-end", "short", 180.0)):
        if axis is None:
            rv, tv = rvec, tvec
        else:
            rv, tv = rotate_about_own_axis(tris, rvec, tvec, axis, deg)
        med, mean, n, near = score_against(tris, rv, tv, cloud)
        results[label_] = med
        print("%-34s %10.2f  %10.2f  %9.1f%%" % (label_, med, mean, 100 * near))

    true_v = results["TRUE pose"]
    best_wrong = min(v for k, v in results.items() if k != "TRUE pose")
    print("\nTRUE pose beats the best wrong hypothesis by %.2f mm (%.1fx)"
          % (best_wrong - true_v, best_wrong / max(true_v, 1e-6)))
    print("For comparison, the EDGE cost separated roll hypotheses by ~0.5 px on a ~23 px")
    print("baseline, which is why roll has never once been recovered from images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
