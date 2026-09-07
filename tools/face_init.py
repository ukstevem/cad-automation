#!/usr/bin/env python3
"""
Initialise a pose from operator clicks: a point on the model, the same point on the photo.

WHY THIS EXISTS. The automatic search fails in a particular and dangerous way. It cannot represent
discrete placements - a plate lying face-down, a frame turned over - so when one of those is the
truth it returns the best of a set of wrong answers, with a plausible score and no indication that
the right answer was never a candidate. On the bench that cost us: an exhaustive sweep of eighteen
refined yaw starts topped out at 44% on a plate that scored 99% once flipped, and the flip was not
in the search space at all.

An operator resolves that in one glance, because clicking THIS face against THAT face carries the
handedness intrinsically - it is information the search does not have and cannot derive.

WHAT THE CLICKS HAVE TO ACHIEVE, AND IT IS LESS THAN YOU WOULD THINK. They are an INITIALISER, not
an answer. `pose_refine` takes over and converges to sub-millimetre, so the clicks only need to
land inside its capture radius, measured on this rig at roughly 15 mm and 5 degrees. Clicking a
face carries real slop - a point in the middle of a large flat face is barely localised along that
face - but errors across several clicks average down, and `--simulate` below measures exactly how
many clicks and how much slop the chain tolerates.

    # solve from real clicks
    python tools/face_init.py --clicks clicks.json --captures <dir> --mesh <stl> --out <dir>

    # measure how accurate the clicking has to be, using a known-good pose as truth
    python tools/face_init.py --simulate --fit outputs/ar_fits/plateH1_flip \\
        --captures outputs/ar_captures/plate01 --mesh outputs/ar_models/plateH1_flipped.stl
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

from app.services import charuco, multiview_fit as MVF, visibility as VIS  # noqa: E402
import pose_refine as PR  # noqa: E402


def pose_from_clicks(pairs, view):
    """
    Object pose in the BOARD frame from correspondences in one view.

    solvePnP returns the object relative to the CAMERA; the board is the world frame here, so the
    result is composed with the known board->camera pose. Doing it the other way round - solving
    in the camera frame and forgetting to compose - produces a pose that looks sane in one view
    and is nonsense in the other, which is a confusing failure to debug.
    """
    obj = np.asarray([p["model"] for p in pairs], np.float64).reshape(-1, 3)
    img = np.asarray([p["image"] for p in pairs], np.float64).reshape(-1, 2)
    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    dist = np.asarray(view["dist"], np.float64).ravel()
    if len(obj) < 4:
        raise ValueError("need at least 4 clicks per view; SQPNP is unreliable below that")
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError("solvePnP failed on these correspondences")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)

    R_oc, _ = cv2.Rodrigues(rvec)                      # object -> camera
    R_wc, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    t_wc = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    R_ow = R_wc.T @ R_oc                               # object -> world (board)
    t_ow = R_wc.T @ (tvec.reshape(3, 1) - t_wc)
    return cv2.Rodrigues(R_ow)[0].ravel(), t_ow.ravel()


def sample_face_points(mesh, n, rng):
    """
    Points an operator could plausibly click: spread over the model, on face interiors.

    Picked far apart on purpose. Clicks bunched together leave the pose poorly conditioned in
    rotation for the same reason a short baseline does in stereo - and an operator asked for
    "three or four faces" will naturally choose ones that face different ways, which is exactly
    what conditions it well.
    """
    centres = mesh.mean(axis=1)
    pick = [int(rng.integers(len(centres)))]
    for _ in range(n - 1):
        d = np.min(np.linalg.norm(centres[:, None, :] - centres[pick][None, :, :], axis=2), axis=1)
        pick.append(int(np.argmax(d)))
    return centres[pick]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH")
    ap.add_argument("--clicks", default=None,
                    help='JSON: [{"view": "<filename substring>", "model": [x,y,z], '
                         '"image": [u,v]}, ...]')
    ap.add_argument("--simulate", action="store_true",
                    help="no clicks: synthesise them from --fit with known error, and measure how "
                         "much click slop the chain tolerates")
    ap.add_argument("--fit", default=None, help="known-good pose, for --simulate")
    ap.add_argument("--counts", default="3,4,6,8", help="clicks per view to try, for --simulate")
    ap.add_argument("--errors", default="5,10,20,40", help="click error in px, for --simulate")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(base["board"])
    det = charuco.make_detector(board)
    overrides = [(s.split("=", 1)[0], MVF.load_profile(s.split("=", 1)[1]))
                 for s in args.cam_profile]
    mesh = VIS.load_stl(args.mesh)

    views = []
    for path in sorted(glob.glob(os.path.join(args.captures, "*"))):
        b = os.path.basename(path)
        if any(k in b for k in ("overlay", "endcheck", "deviation", "critical", "linecheck")):
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        prof = next((p for sub, p in overrides if sub in b), base)
        v = MVF.build_view(img, prof, board, det, label=b)
        v.update({"K": prof["K"], "dist": prof["dist"], "image": img, "tag": b})
        views.append(v)
    if not views:
        print("no usable captures", file=sys.stderr)
        return 2

    if args.simulate:
        if not args.fit:
            print("--simulate needs --fit (the pose to treat as truth)", file=sys.stderr)
            return 2
        src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
        truth = json.load(open(src, "r", encoding="utf-8"))
        rv_t = np.asarray(truth["rvec"], np.float64).ravel()
        tv_t = np.asarray(truth["tvec"], np.float64).ravel()
        R_t, _ = cv2.Rodrigues(rv_t.reshape(3, 1))
        rng = np.random.default_rng(0)
        print("How accurate does the clicking have to be? Clicks are synthesised from a known-good")
        print("pose, displaced by the stated error, then solved and handed to pose_refine.")
        print("'recovered' is distance from the true pose AFTER refinement - under a millimetre")
        print("means the chain worked and the click error was absorbed entirely.")
        print("")
        print("%8s %10s %14s %14s" % ("clicks", "click err", "before refine", "after refine"))
        for n in [int(x) for x in args.counts.split(",")]:
            for e in [float(x) for x in args.errors.split(",")]:
                b_all, a_all = [], []
                for _t in range(args.trials):
                    pairs_by_view, ok = {}, True
                    obj_pts = sample_face_points(mesh, n, rng)
                    world = (R_t @ obj_pts.T).T + tv_t
                    for v in views:
                        p2, _ = cv2.projectPoints(world.reshape(-1, 1, 3), v["rvec_cam"],
                                                  v["tvec_cam"],
                                                  np.asarray(v["K"], np.float64).reshape(3, 3),
                                                  np.asarray(v["dist"], np.float64))
                        p2 = p2.reshape(-1, 2) + rng.normal(0, e, (len(world), 2))
                        pairs_by_view[v["tag"]] = [{"model": o, "image": q}
                                                   for o, q in zip(obj_pts, p2)]
                    # solve in the view with the most clicks, as the UI would
                    v0 = views[0]
                    try:
                        rv0, tv0 = pose_from_clicks(pairs_by_view[v0["tag"]], v0)
                    except ValueError:
                        ok = False
                    if not ok:
                        continue
                    b_all.append(float(np.linalg.norm(tv0 - tv_t)))
                    r, t = PR.refine(mesh, rv0, tv0, views,
                                     schedule=(40., 20., 10., 5., 3., 2.), iters=3,
                                     dof="seated", verbose=False)
                    a_all.append(float(np.linalg.norm(np.asarray(t) - tv_t)))
                if b_all:
                    print("%8d %7.0f px %11.1f mm %11.1f mm%s"
                          % (n, e, np.median(b_all), np.median(a_all),
                             "   <- recovered" if np.median(a_all) < 2.0 else ""))
        return 0

    if not args.clicks:
        print("give --clicks, or --simulate to size the requirement", file=sys.stderr)
        return 2
    with open(args.clicks, "r", encoding="utf-8") as fh:
        clicks = json.load(fh)
    by_view = {}
    for c in clicks:
        v = next((x for x in views if c["view"] in x["tag"]), None)
        if v is None:
            print("no capture matching view '%s'" % c["view"], file=sys.stderr)
            return 2
        by_view.setdefault(v["tag"], (v, []))[1].append(c)
    tag, (v, pairs) = max(by_view.items(), key=lambda kv: len(kv[1][1]))
    print("solving from %d clicks on %s" % (len(pairs), tag))
    rv0, tv0 = pose_from_clicks(pairs, v)
    print("clicked pose  t = [%.1f, %.1f, %.1f] mm" % tuple(tv0))
    before = PR.score(mesh, rv0, tv0, views)
    r, t = PR.refine(mesh, rv0, tv0, views, schedule=(40., 20., 10., 5., 3., 2.), iters=4,
                     dof="seated", verbose=True)
    after = PR.score(mesh, r, t, views)
    print("")
    print("confirmed %.0f%% -> %.0f%%   silhouette %.0f%% -> %.0f%%"
          % (before[0], after[0], before[1], after[1]))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "fit.json"), "w", encoding="utf-8") as fh:
            json.dump({"rvec": [float(x) for x in np.asarray(r).ravel()],
                       "tvec": [float(x) for x in np.asarray(t).ravel()],
                       "mesh": os.path.basename(args.mesh),
                       "init": "operator face clicks",
                       "clicks": len(pairs)}, fh, indent=2)
        print("wrote %s" % os.path.join(args.out, "fit.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
