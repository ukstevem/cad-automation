#!/usr/bin/env python3
"""
Render synthetic captures of a known part at a KNOWN pose, using the real rig's geometry.

The point is ground truth at scale. Every question left open about orientation - where the
end-on limit actually lies, whether a differently shaped part behaves the same, how far the
seating error goes - needs many trials with the answer known in advance. Physically that means
printing parts and moving them by hand, which is slow and gives one data point at a time. Here it
is a loop.

Faithful in the ways that matter for this question, and deliberately so:

* the CAMERAS are the real ones - intrinsics from the calibration profile, board->camera poses
  solved from an actual capture, so the baseline, the elevation and the lens distortion are the
  rig's, not an idealisation;
* the PART is rendered as a solid with hidden surfaces removed (painter's algorithm), so a
  camera sees only what it could really see. The existing synthetic test in tests/ projects every
  CAD edge including the far side, which is exactly the input that made the cost nearly blind;
* the BOARD is drawn from the same ChArUco definition the detector uses, marker bit patterns and
  all, so the whole pipeline runs for real - board detection, masking, segmentation, Canny.

What it does NOT reproduce: shadows, specular highlights, focus blur, sensor noise, the paper's
folds and texture, or a part whose print differs from its CAD. So a fit that fails here will fail
on real photographs, but one that succeeds here has only cleared the easier bar. Synthetic
results are a CEILING - use them to rule things out, and to compare conditions against each
other, not to promise field accuracy.

Which is why the harness must be anchored before it is trusted: render the Main Frame at the
poses actually shot on the rig and check it reproduces the real outcomes, including the near
end-on FAILURE. A harness that passes everything the rig failed is measuring its own optimism.

    docker compose run --rm --no-deps api python tools/synth_capture.py \
        --rig outputs/ar_captures/turn90 \
        --mesh outputs/ar_models/mainframe_default_1to5.stl \
        --pose 200,100,30 --out outputs/ar_captures/synth01
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
from app.services.visibility import load_stl  # noqa: E402

FAR = 1.0e9


def rig_from_captures(capdir: str, profile: dict, board, detector, min_corners: int = 6):
    """Borrow real camera poses: solve board->camera from each photo in a real capture set."""
    views = []
    for path in sorted(glob.glob(os.path.join(capdir, "*"))):
        base = os.path.basename(path)
        if "overlay" in base or "endcheck" in base or "containment" in base:
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cor, ids, _m, _i = charuco.detect_board_detailed(detector, image_edges.to_gray(img))
        if ids is None or len(ids) < min_corners:
            continue
        rc, tc, _n = charuco_board_pose(cor, ids, board, profile["K"], profile["dist"],
                                        min_corners=min_corners)
        h, w = img.shape[:2]
        tag = base.split("_")[1] if "_" in base else os.path.splitext(base)[0]
        views.append({"rvec_cam": rc, "tvec_cam": tc, "width": w, "height": h, "tag": tag})
    if len(views) < 2:
        raise SystemExit("need >=2 usable rig photos in %s" % capdir)
    return views


def seat_on_board(tris: np.ndarray, x: float, y: float, yaw_deg: float, roll_deg: float = 0.0):
    """
    Place the part flat on the board at (x, y) with the given yaw, and optionally rolled.

    The part's own long axis is laid along board X and its thinnest axis made vertical, which is
    how a part actually rests; yaw then turns it in plan and roll turns it about its own length.
    Height is not a free parameter - it is whatever makes the part touch the board.

    Board Z points AWAY from the camera, so a part resting on the board occupies NEGATIVE Z (see
    CLAUDE.md). Seating therefore means the part's LARGEST z sits at 0.
    """
    pts = np.vstack([tris.reshape(-1, 3), tris.mean(axis=1)])
    c = pts.mean(axis=0)
    d = pts - c
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    order = np.argsort(evals)[::-1]
    long_a, mid_a, thin_a = (evecs[:, order[i]] for i in range(3))
    # object -> board: long axis to X, mid to Y, thin to Z (right-handed)
    base = np.stack([long_a, mid_a, np.cross(long_a, mid_a)], axis=0)

    Rroll, _ = cv2.Rodrigues(np.array([[np.radians(roll_deg)], [0.0], [0.0]]))
    Ryaw, _ = cv2.Rodrigues(np.array([[0.0], [0.0], [np.radians(yaw_deg)]]))
    R = Ryaw @ Rroll @ base

    # world = R*p + t. Two constraints fix t completely: the centroid lands at (x, y), and the
    # part touches the board - so its greatest z is 0, since it occupies the negative side.
    Rc = (R @ c.reshape(3, 1)).ravel()
    world_z = (R @ pts.T)[2]
    t = np.array([[x - Rc[0]], [y - Rc[1]], [-float(world_z.max())]])
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t


def render(tris: np.ndarray, rvec_obj, tvec_obj, view: dict, board, profile: dict,
           light=(0.3, -0.4, -0.86), noise: float = 2.0, blur: int = 3) -> np.ndarray:
    """One synthetic photograph: white paper, the ChArUco board, and the shaded part."""
    w, h = int(view["width"]), int(view["height"])
    K = np.asarray(profile["K"], np.float64).reshape(3, 3)
    dist = np.asarray(profile["dist"], np.float64).reshape(-1, 1)
    rc = np.asarray(view["rvec_cam"], np.float64).reshape(3, 1)
    tc = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)

    img = np.full((h, w, 3), 238, np.uint8)                      # paper

    # ── board ────────────────────────────────────────────────────────────────
    # Drawn from the same definition the detector uses. Squares as projected polygons; each
    # marker's bit pattern warped onto its own quad, which keeps distortion right locally.
    corners = np.asarray(board.getChessboardCorners(), np.float64).reshape(-1, 3)
    xs = np.unique(np.round(corners[:, 0], 6))
    ys = np.unique(np.round(corners[:, 1], 6))
    pitch = float(min(np.min(np.diff(xs)), np.min(np.diff(ys))))
    nx, ny = len(xs) + 1, len(ys) + 1
    for iy in range(ny):
        for ix in range(nx):
            if (ix + iy) % 2 == 0:                                # black squares only
                x0, y0 = ix * pitch, iy * pitch
                quad = np.array([[x0, y0, 0], [x0 + pitch, y0, 0],
                                 [x0 + pitch, y0 + pitch, 0], [x0, y0 + pitch, 0]], np.float64)
                uv, _ = cv2.projectPoints(quad.reshape(-1, 1, 3), rc, tc, K, dist)
                cv2.fillConvexPoly(img, np.round(uv.reshape(-1, 2)).astype(np.int32),
                                   (28, 28, 28), lineType=cv2.LINE_AA)
    try:
        obj = board.getObjPoints()
        dic = board.getDictionary()
        ids = np.asarray(board.getIds()).ravel()
        for k, quad in enumerate(obj):
            quad = np.asarray(quad, np.float64).reshape(-1, 3)
            uv, _ = cv2.projectPoints(quad.reshape(-1, 1, 3), rc, tc, K, dist)
            dst = np.round(uv.reshape(-1, 2)).astype(np.float32)
            side = 64
            marker = cv2.aruco.generateImageMarker(dic, int(ids[k]), side)
            src = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR), H, (w, h),
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT,
                                         dst=None)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, np.round(dst).astype(np.int32), 255)
            img[mask > 0] = warped[mask > 0]
    except Exception as exc:                                      # pragma: no cover
        print("  warning: markers not drawn (%s)" % exc, file=sys.stderr)

    # ── part: painter's algorithm, so only visible surfaces are drawn ────────
    R_obj, _ = cv2.Rodrigues(np.asarray(rvec_obj, np.float64).reshape(3, 1))
    t_obj = np.asarray(tvec_obj, np.float64).reshape(3, 1)
    R_cam, _ = cv2.Rodrigues(rc)
    world = (R_obj @ tris.reshape(-1, 3).T + t_obj)
    cam = (R_cam @ world + tc).T.reshape(-1, 3, 3)
    z = cam[:, :, 2]
    keep = np.all(z > 1e-6, axis=1)
    cam, z = cam[keep], z[keep]
    if len(cam):
        wtri = world.T.reshape(-1, 3, 3)[keep]
        n = np.cross(wtri[:, 1] - wtri[:, 0], wtri[:, 2] - wtri[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.maximum(ln, 1e-9)
        l = np.asarray(light, np.float64)
        l = l / np.linalg.norm(l)
        # Grey paint, like the print: ambient plus a soft diffuse term, no specular.
        shade = 70.0 + 105.0 * np.abs(n @ l)
        uv, _ = cv2.projectPoints(wtri.reshape(-1, 1, 3), rc, tc, K, dist)
        uv = uv.reshape(-1, 3, 2)
        for i in np.argsort(-z.mean(axis=1)):                     # far first
            poly = np.round(uv[i]).astype(np.int32)
            if (poly[:, 0].max() < 0 or poly[:, 1].max() < 0
                    or poly[:, 0].min() >= w or poly[:, 1].min() >= h):
                continue
            v = float(shade[i])
            cv2.fillConvexPoly(img, poly, (v, v, v), lineType=cv2.LINE_AA)

    if blur > 1:
        img = cv2.GaussianBlur(img, (blur | 1, blur | 1), 0)
    if noise > 0:
        img = np.clip(img.astype(np.float32)
                      + np.random.default_rng(0).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", required=True, help="a real capture dir, to borrow camera poses")
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--pose", default=None, help="x,y,yaw[,roll] in board mm and degrees")
    ap.add_argument("--from-fit", default=None,
                    help="render at the pose in this fit.json instead of --pose. This is how the "
                         "harness is ANCHORED: render at a pose verified on the real rig, then "
                         "check the pipeline recovers it. Use the pose known to be CORRECT - for "
                         "turn45 that is turn45_alt, not the chosen fit.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", type=float, default=2.0)
    ap.add_argument("--blur", type=int, default=3)
    args = ap.parse_args()

    if not args.pose and not args.from_fit:
        print("need --pose or --from-fit", file=sys.stderr)
        return 2
    x = y = yaw = roll = 0.0
    if args.pose:
        vals = [float(v) for v in args.pose.split(",")]
        if len(vals) == 3:
            vals.append(0.0)
        x, y, yaw, roll = vals

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    views = rig_from_captures(args.rig, profile, board, det)
    tris = load_stl(args.mesh)
    if args.from_fit:
        src = args.from_fit
        if os.path.isdir(src):
            src = os.path.join(src, "fit.json")
        with open(src, "r", encoding="utf-8") as fh:
            f = json.load(fh)
        rvec = np.asarray(f["rvec"], np.float64).reshape(3, 1)
        tvec = np.asarray(f["tvec"], np.float64).reshape(3, 1)
        print("  rendering at the pose from %s" % src)
    else:
        rvec, tvec = seat_on_board(tris, x, y, yaw, roll)

    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.out))
    written = []
    for v in views:
        img = render(tris, rvec, tvec, v, board, profile, noise=args.noise, blur=args.blur)
        p = os.path.join(args.out, "%s_%s_synth.png" % (name, v["tag"]))
        cv2.imwrite(p, img)
        written.append(p)
        print("  wrote %s" % p)

    truth = {
        "rvec": np.asarray(rvec).reshape(3).tolist(),
        "tvec": np.asarray(tvec).reshape(3).tolist(),
        "pose_request": ({"from_fit": args.from_fit} if args.from_fit
                         else {"x": x, "y": y, "yaw_deg": yaw, "roll_deg": roll}),
        "mesh": os.path.basename(args.mesh),
        "rig": os.path.abspath(args.rig),
        "images": [os.path.basename(p) for p in written],
    }
    with open(os.path.join(args.out, "truth.json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)
    print("  wrote %s/truth.json  (ground-truth pose)" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
