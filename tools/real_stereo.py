#!/usr/bin/env python3
"""
Run the stereo roll decision on REAL captures from the rig.

Everything proven so far renders its own images. This reads photographs, so Monday's session
produces an answer rather than a folder of PNGs.

CAPTURE PROTOCOL - two shots per pose, nothing moved between them:

    python webcam_capture.py shot roll0_off      # projector OFF
    python webcam_capture.py shot roll0_on       # projector ON

The projector's speckle makes the ChArUco board undecodable, and the board is how every camera
learns where it is. So poses come from the OFF frame and texture from the ON frame. They are
paired by camera serial, which the capture filenames already carry.

WHY THE BOARD STILL DOES THE WORK. The two cameras' relative pose - everything stereoRectify
needs - falls out of each camera solving the board independently. No stereo calibration rig, no
simultaneous exposure, and the pair can be re-aimed between sessions without recalibrating
anything but intrinsics.

    docker compose run --rm --no-deps api python tools/real_stereo.py \
        --captures outputs/ar_captures/mon01 \
        --profile outputs/calibration/RigCam_52FD1B1F.json \
        --cam-profile NEWSERIAL=outputs/calibration/RigCam_new.json \
        --mesh outputs/ar_models/mainframe_default_1to5.stl \
        --pose outputs/ar_fits/turn90/fit.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import charuco, image_edges, multiview_fit as MVF, visibility as VIS  # noqa: E402
from app.services.board_pose import charuco_board_pose  # noqa: E402

from depth_discriminability import rotate_about_own_axis  # noqa: E402
from stereo_roll_test import score_against  # noqa: E402

ROLLS = [0.0, 90.0, 180.0, 270.0]

def rectified_cloud(imA, imB, rvecA, tvecA, rvecB, tvecB, Ka, da, Kb, db, z_hint=None):
    """
    Depth from ANY two calibrated views, via proper rectification.

    The fast path used elsewhere assumes the pair is already rectified - parallel image planes,
    identical intrinsics - which holds for a virtual sideways offset and fails completely for two
    real cameras aimed inward at 50 degrees. This does it properly: relative pose from the two
    board solutions, stereoRectify, remap, SGBM, reproject. Returns points in the frame of camera
    A, or None if matching found nothing.
    """
    h, w = imA.shape[:2]
    Ra, _ = cv2.Rodrigues(np.asarray(rvecA, np.float64).reshape(3, 1))
    Rb, _ = cv2.Rodrigues(np.asarray(rvecB, np.float64).reshape(3, 1))
    ta = np.asarray(tvecA, np.float64).reshape(3, 1)
    tb = np.asarray(tvecB, np.float64).reshape(3, 1)
    R_rel = Rb @ Ra.T
    T_rel = tb - R_rel @ ta
    baseline = float(np.linalg.norm(T_rel))

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(Ka, da, Kb, db, (w, h), R_rel, T_rel,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1x, m1y = cv2.initUndistortRectifyMap(Ka, da, R1, P1, (w, h), cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(Kb, db, R2, P2, (w, h), cv2.CV_32FC1)
    recA = cv2.remap(imA, m1x, m1y, cv2.INTER_LINEAR)
    recB = cv2.remap(imB, m2x, m2y, cv2.INTER_LINEAR)

    fx_r = float(P1[0, 0])
    if z_hint is None:
        lo, span = 0, 256
    else:
        d_far = fx_r * baseline / float(np.percentile(z_hint, 99))
        d_near = fx_r * baseline / float(np.percentile(z_hint, 1))
        lo = max(0, int(np.floor(d_far / 16.0) * 16) - 16)
        span = max(16, int(np.ceil((d_near * 1.35 - lo) / 16.0) * 16))
    blk = 7
    sgbm = cv2.StereoSGBM_create(minDisparity=lo, numDisparities=span, blockSize=blk,
                                 P1=8 * blk * blk, P2=32 * blk * blk, disp12MaxDiff=1,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(cv2.cvtColor(recA, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(recB, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
    valid = disp > (lo + 0.5)
    subj = image_edges.segment_subject(recA, grow_px=0)
    if subj is not None:
        valid &= subj > 0
    if valid.sum() < 300:
        return None, recA, int(valid.sum())
    pts3 = cv2.reprojectImageTo3D(disp, Q)[valid].reshape(-1, 3)
    ok = np.isfinite(pts3).all(axis=1) & (np.abs(pts3[:, 2]) < 1e5)
    cloud_cam = (R1.T @ pts3[ok].T).T                  # rectified -> camera A frame
    return cloud_cam, recA, int(ok.sum())



def camera_labels(path=None):
    """
    Serial -> human label (A / B / C). Printing serials invites mis-pairing on a busy day.

    The tracked copy lives beside the tools; outputs/calibration/ is gitignored, so a map left
    only there would be lost on a fresh clone. A local file still wins if present, so the rig can
    override without a commit.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ([path] if path else []) + ["outputs/calibration/cameras.json",
                                            os.path.join(here, "cameras.json")]:
        try:
            with open(cand, "r", encoding="utf-8") as fh:
                return {k: v.get("label", k)
                        for k, v in json.load(fh).get("cameras", {}).items()}
        except Exception:
            continue
    return {}


def serial_of(filename: str):
    """
    The camera serial out of a capture filename, without mistaking the date for it.

    Capture names look like ``roll0_off_9B17236C_20260907_120000.png``. A naive search for eight
    hex characters also matches ``20260907`` - every digit is a valid hex digit - so on a name
    where the date came first it would key the whole pipeline on the timestamp, silently see one
    camera instead of three, and pick the wrong stereo pair.

    Both known cameras happen to fit the same shape: the C920s report 52FD1B1F and B68DE55F, the
    Brio 9B17236C. So the rule is eight hex characters that are NOT a plausible date.
    """
    for tok in re.split(r"[_.\-]", os.path.basename(filename)):
        if len(tok) != 8 or not re.fullmatch(r"[0-9A-Fa-f]{8}", tok):
            continue
        if tok.isdigit() and tok[:2] in ("19", "20"):     # 20260907 - a date, not a serial
            continue
        return tok.upper()
    return None


def collect(capdir: str):
    """Pair each camera's projector-OFF and projector-ON frames, keyed by serial."""
    cams: dict = {}
    for path in sorted(glob.glob(os.path.join(capdir, "*"))):
        base = os.path.basename(path)
        if not base.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if any(k in base for k in ("overlay", "endcheck", "containment")):
            continue
        lit = "_on_" in base or "_on." in base
        off = "_off_" in base or "_off." in base
        if not (lit or off):
            continue
        serial = serial_of(base)
        if not serial:
            continue
        cams.setdefault(serial, {})["on" if lit else "off"] = path
    return {k: v for k, v in cams.items() if "on" in v and "off" in v}


def board_pose_of(path, profile, board, detector, min_corners=6):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None
    cor, ids, _m, _i = charuco.detect_board_detailed(detector, image_edges.to_gray(img))
    if ids is None or len(ids) < min_corners:
        return None, None, img
    rc, tc, _n = charuco_board_pose(cor, ids, board, profile["K"], profile["dist"],
                                    min_corners=min_corners)
    return rc, tc, img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--profile", required=True, help="default calibration profile")
    ap.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH",
                    help="per-camera profile; the THIRD camera needs its own intrinsics")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--pose", required=True, help="fit.json giving the part's pose to test around")
    ap.add_argument("--use-off", action="store_true",
                    help="match on the projector-OFF frames instead, as the passive control")
    ap.add_argument("--out", default=None, help="write the point cloud as PLY")
    args = ap.parse_args()

    overrides = [(s.split("=", 1)[0], MVF.load_profile(s.split("=", 1)[1]))
                 for s in args.cam_profile]
    default_profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(default_profile["board"])
    det = charuco.make_detector(board)

    labels = camera_labels()

    def name(serial):
        return "%s (%s)" % (labels.get(serial, "?"), serial)

    cams = collect(args.captures)
    if len(cams) < 2:
        print("need at least two cameras with BOTH _off_ and _on_ frames in %s" % args.captures,
              file=sys.stderr)
        print("found: %s" % (", ".join("%s(%s)" % (k, "+".join(sorted(v))) for k, v in cams.items())
                             or "nothing"), file=sys.stderr)
        return 2

    solved = {}
    for serial, paths in cams.items():
        prof = default_profile
        for sub, p in overrides:
            if sub in serial or sub in os.path.basename(paths["off"]):
                prof = p
                break
        rc, tc, _img = board_pose_of(paths["off"], prof, board, det)
        if rc is None:
            print("  %s: board not found in the projector-OFF frame - skipped" % name(serial))
            continue
        solved[serial] = {"rvec": rc, "tvec": tc, "profile": prof, "paths": paths}
        print("  %s: board solved, %.0f mm from board origin"
              % (name(serial), float(np.linalg.norm(tc))))
    if len(solved) < 2:
        print("fewer than two cameras solved the board", file=sys.stderr)
        return 2

    # Pick the NARROWEST pair: stereo matching wants a small angular separation, unlike the pose
    # fit which wants a wide one. With three cameras the rig provides both at once.
    best, keys = None, list(solved)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            Ri, _ = cv2.Rodrigues(solved[keys[i]]["rvec"])
            Rj, _ = cv2.Rodrigues(solved[keys[j]]["rvec"])
            ci = -Ri.T @ solved[keys[i]]["tvec"]
            cj = -Rj.T @ solved[keys[j]]["tvec"]
            sep = float(np.linalg.norm(ci - cj))
            ang = float(np.degrees(np.arccos(np.clip(
                abs(float((Ri.T @ np.array([[0.], [0.], [1.]])).ravel()
                          @ (Rj.T @ np.array([[0.], [0.], [1.]])).ravel())), 0, 1))))
            if best is None or ang < best[0]:
                best = (ang, sep, keys[i], keys[j])
    ang, baseline, ka, kb = best
    print("\nstereo pair: %s + %s   baseline %.0f mm, optical axes %.1f deg apart"
          % (ka, kb, baseline, ang))
    if ang > 25:
        print("  WARNING: %.0f deg is wide for correlation matching. Expect sparse depth;" % ang)
        print("  a third camera closer to one of the existing pair would fix it.")

    A, B = solved[ka], solved[kb]
    Ka = np.asarray(A["profile"]["K"], np.float64).reshape(3, 3)
    Kb = np.asarray(B["profile"]["K"], np.float64).reshape(3, 3)
    da = np.asarray(A["profile"]["dist"], np.float64).reshape(-1, 1)
    db = np.asarray(B["profile"]["dist"], np.float64).reshape(-1, 1)
    Ra, _ = cv2.Rodrigues(A["rvec"])
    Rb, _ = cv2.Rodrigues(B["rvec"])
    # Relative pose straight from the two board solutions - this is the step that removes the
    # need for any stereo calibration.
    R_rel = Rb @ Ra.T
    T_rel = np.asarray(B["tvec"], np.float64).reshape(3, 1) - R_rel @ np.asarray(
        A["tvec"], np.float64).reshape(3, 1)

    key = "off" if args.use_off else "on"
    imA = cv2.imread(A["paths"][key], cv2.IMREAD_COLOR)
    imB = cv2.imread(B["paths"][key], cv2.IMREAD_COLOR)
    h, w = imA.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(Ka, da, Kb, db, (w, h), R_rel, T_rel,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1x, m1y = cv2.initUndistortRectifyMap(Ka, da, R1, P1, (w, h), cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(Kb, db, R2, P2, (w, h), cv2.CV_32FC1)
    recA = cv2.remap(imA, m1x, m1y, cv2.INTER_LINEAR)
    recB = cv2.remap(imB, m2x, m2y, cv2.INTER_LINEAR)

    fx_r = float(P1[0, 0])
    tris = VIS.load_stl(args.mesh)
    with open(args.pose, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)

    # Disparity range from where the part actually is, not guessed - guessing it silently
    # saturated SGBM once already and put the cloud 380 mm too far away.
    world = (cv2.Rodrigues(rvec)[0] @ tris.reshape(-1, 3).T + tvec)
    zc = (Ra @ world + np.asarray(A["tvec"], np.float64).reshape(3, 1))[2]
    d_far = fx_r * baseline / float(np.percentile(zc, 99))
    d_near = fx_r * baseline / float(np.percentile(zc, 1))
    lo = max(0, int(np.floor(d_far / 16.0) * 16) - 16)
    span = max(16, int(np.ceil((d_near * 1.35 - lo) / 16.0) * 16))
    print("disparity search %d .. %d px" % (lo, lo + span))

    blk = 7
    sgbm = cv2.StereoSGBM_create(minDisparity=lo, numDisparities=span, blockSize=blk,
                                 P1=8 * blk * blk, P2=32 * blk * blk, disp12MaxDiff=1,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(cv2.cvtColor(recA, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(recB, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
    pts3 = cv2.reprojectImageTo3D(disp, Q)
    valid = disp > (lo + 0.5)

    # Subject mask from the IMAGE. Using the CAD's own coverage would favour whichever pose
    # selected the points, which is the hypothesis under test.
    subj = image_edges.segment_subject(recA, grow_px=0)
    if subj is not None:
        valid &= subj > 0
    print("matched %d px; %d on the segmented subject" % (int((disp > lo + 0.5).sum()),
                                                          int(valid.sum())))
    if valid.sum() < 500:
        print("too few points - check the projector is on and covering the part", file=sys.stderr)
        return 1

    # rectified-left -> original left camera -> board frame
    cloud_rect = pts3[valid].reshape(-1, 3)
    good = np.isfinite(cloud_rect).all(axis=1) & (np.abs(cloud_rect[:, 2]) < 1e5)
    cloud_cam = (R1.T @ cloud_rect[good].T).T
    cloud = (Ra.T @ (cloud_cam.T - np.asarray(A["tvec"], np.float64).reshape(3, 1))).T
    print("cloud: %d points" % len(cloud))

    print("")
    print("%-22s %12s %12s" % ("roll hypothesis", "median mm", "within 5mm"))
    scores = {}
    for d in ROLLS:
        if d == 0.0:
            rv, tv = rvec, tvec
        else:
            rv, tv = rotate_about_own_axis(tris, rvec, tvec, "long", d)
        med, _mean, _n, near = score_against(tris, rv, tv, cloud)
        scores[d] = near
        print("%-22s %10.2f %11.1f%%" % ("%.0f deg" % d, med, 100 * near))
    win = max(scores, key=scores.get)
    others = [v for k, v in scores.items() if k != win]
    print("")
    print("best: %.0f deg, margin %.4f over the next" % (win, scores[win] - max(others)))
    if scores[win] - max(others) < 1e-3:
        print("TOO CLOSE TO CALL - the data does not separate these; ask the operator.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(cloud))
            fh.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in cloud:
                fh.write("%.2f %.2f %.2f\n" % (p[0], p[1], p[2]))
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
