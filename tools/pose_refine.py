#!/usr/bin/env python3
"""
Refine a pose until the model's lines sit on the real ones. RAPiD, with the pieces built this week.

WHY THE EXISTING FIT IS NOT ENOUGH. The multiview fit scores a pose by chamfer distance from
tens of thousands of CAD points to the nearest detected edge pixel. On a busy part the nearest
edge is a few pixels away almost regardless of where the model is, so the objective saturates: it
finds roughly the right place and then stops caring. Measured on the rig, that leaves the pose
5-10 mm out, which is fatal for a 10 mm check - and it was masquerading as a lighting problem,
because a model 5 mm from the truth finds no line inside a 2 mm window and reports "no contrast"
on edges that are plainly visible.

WHAT REPLACES IT. Two things that did not exist when that fit was written:

  * `mesh_feature_edges` derives edges from mesh TOPOLOGY - boundary, crease by dihedral angle,
    silhouette by facing - so the targets are real edges rather than rasterisation artefacts, and
    a tessellation diagonal on a flat plate cannot be reported at all.
  * `search_along_normals` asks, for each edge point, a ONE-dimensional question along that
    point's own normal, bounded by the tolerance. Bounded and one-dimensional is what stops the
    saturation: it cannot answer with a different edge across the part.

Each found point then gives one linear constraint on the pose, and a rigid body has six degrees of
freedom, so thousands of constraints are hugely overdetermined. That is the classical RAPiD update
and it converges where chamfer plateaus.

COARSE TO FINE. The search window starts wide and shrinks. Opening at 2 mm around a pose that is
10 mm out would find nothing; opening at 20 mm and closing to 2 mm walks the model in. The window
is the trust region.

    docker compose run --rm --no-deps api python tools/pose_refine.py \\
        --captures outputs/ar_captures/exp_new --fit outputs/ar_fits/rot01 \\
        --mesh outputs/ar_models/mainframe_default_1to5.stl --out outputs/ar_fits/rot01_refined
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import charuco, multiview_fit as MVF, visibility as VIS  # noqa: E402
from critical_edges import visible_feature_edges  # noqa: E402
from line_check import gradient_field, search_along_normals  # noqa: E402


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def pose_jacobian(world, view, rvec, tvec, dof="seated", delta=2.0):
    """
    How each 3D point's projection moves per unit of pose change, as a (N, 2, P) array.

    The perturbation is applied in the WORLD frame about the object's own translation:

        world' = exp(w) (world - t) + t + d   ->   world' ~= world + w x (world - t) + d

    so the rotation part is a cross product with the lever arm from the object origin. Measured by
    projecting the perturbed points rather than differentiating the projection, which keeps lens
    distortion exact and is cheap at six parameters.

    *dof* picks which freedoms exist. 'seated' is the physically honest one for a part lying on a
    table: it can slide in x and y, turn about the board normal, and sit at a height that is not
    known to the millimetre - but it cannot tilt. Handing the solve tilt as well measurably hurts
    (silhouette confirmation 80% -> 62% on this rig, with 12 degrees of rotation invented to
    explain noise), which is what an over-parameterised fit does with freedoms reality does not
    have.
    """
    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    dist = np.asarray(view["dist"], np.float64).ravel()
    rc = np.asarray(view["rvec_cam"], np.float64).reshape(3, 1)
    tc = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    t = np.asarray(tvec, np.float64).ravel()

    base, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rc, tc, K, dist)
    base = base.reshape(-1, 2)

    X, Y, Z = np.eye(3)
    axes = [("tx", X, None), ("ty", Y, None)]
    if dof in ("seated", "full"):
        axes.append(("tz", Z, None))
    if dof == "full":
        axes += [("rx", None, X), ("ry", None, Y)]
    axes.append(("rz", None, Z))

    J = np.empty((len(world), 2, len(axes)))
    lever = world - t
    for k, (_name, dt, dw) in enumerate(axes):
        if dt is not None:
            moved = world + dt * delta
        else:
            # small-angle rotation about the object origin, in radians
            ang = np.radians(delta)
            moved = world + np.cross(dw * ang, lever)
        p2, _ = cv2.projectPoints(moved.reshape(-1, 1, 3), rc, tc, K, dist)
        step = delta if dt is not None else np.radians(delta)
        J[:, :, k] = (p2.reshape(-1, 2) - base) / step
    return J


def apply_update(rvec, tvec, upd, dof="seated"):
    """Compose a pose increment onto (rvec, tvec), rotation applied about the object origin."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t = np.asarray(tvec, np.float64).ravel()
    if dof == "planar":
        d, w = np.array([upd[0], upd[1], 0.0]), np.array([0.0, 0.0, upd[2]])
    elif dof == "seated":
        d, w = np.asarray(upd[:3], np.float64), np.array([0.0, 0.0, upd[3]])
    else:
        d, w = np.asarray(upd[:3], np.float64), np.asarray(upd[3:], np.float64)
    dR, _ = cv2.Rodrigues(w.reshape(3, 1))
    return cv2.Rodrigues(dR @ R)[0].ravel(), (t + d)


def refine(mesh, rvec, tvec, views, schedule=(20.0, 10.0, 5.0, 3.0, 2.0), iters=4,
           dof="seated", step_px=2.0, verbose=True):
    rvec = np.asarray(rvec, np.float64).ravel()
    tvec = np.asarray(tvec, np.float64).ravel()
    grads = {id(v): gradient_field(v["image"], None) for v in views}
    if verbose:
        print("%8s %5s %8s %9s %10s %10s" % ("window", "iter", "lines", "used", "rms px", "move mm"))
    for tol in schedule:
        for it in range(iters):
            A, b = [], []
            n_pts = 0
            for v in views:
                pts, tan, z, world = visible_feature_edges(mesh, rvec.reshape(3, 1),
                                                           tvec.reshape(3, 1), v,
                                                           step_px=step_px, with_world=True)
                if not len(pts):
                    continue
                n_pts += len(pts)
                fx = float(np.asarray(v["K"], np.float64).reshape(3, 3)[0, 0])
                off, found, _pk, _ba = search_along_normals(grads[id(v)], pts, tan, z, fx,
                                                            tol_mm=tol, reach=1.0)
                if not found.any():
                    continue
                m = found
                J = pose_jacobian(world[m], v, rvec, tvec, dof=dof)
                nrm = np.stack([-tan[m][:, 1], tan[m][:, 0]], axis=1)
                A.append(np.einsum("ij,ijk->ik", nrm, J))
                # the residual is how far the real line sits along the normal, in PIXELS
                b.append(off[m] * fx / np.maximum(z[m], 1e-6))
            if not A:
                break
            A = np.vstack(A)
            b = np.concatenate(b)

            # Robust, because a minority of points will have locked onto the wrong line and a
            # least-squares fit hands those the same authority as the rest.
            upd = np.zeros(A.shape[1])
            for _ in range(5):
                r = A @ upd - b
                s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
                w = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
                Aw = A * w[:, None]
                try:
                    upd = np.linalg.solve(Aw.T @ A + 1e-6 * np.eye(A.shape[1]), Aw.T @ b)
                except np.linalg.LinAlgError:
                    upd = np.zeros(A.shape[1])
                    break
            rvec, tvec = apply_update(rvec, tvec, upd, dof=dof)
            rms = float(np.sqrt(np.mean(b ** 2)))
            move = float(np.linalg.norm(upd[:2 if dof == "planar" else 3]))
            if verbose:
                print("%6.0f mm %5d %8d %9d %9.2f %9.3f" % (tol, it + 1, n_pts, len(b), rms, move))
            if move < 0.05 and abs(upd[-1]) < 1e-4:
                break
    return rvec, tvec


def score(mesh, rvec, tvec, views, tol_mm=2.0, step_px=2.0):
    """Confirmed fraction overall and on SILHOUETTE edges, which cannot lack contrast."""
    conf, sil_conf = [], []
    for v in views:
        pts, tan, z = visible_feature_edges(mesh, np.asarray(rvec).reshape(3, 1),
                                            np.asarray(tvec).reshape(3, 1), v, step_px=step_px)
        if not len(pts):
            continue
        fx = float(np.asarray(v["K"], np.float64).reshape(3, 3)[0, 0])
        _off, found, _p, _bq = search_along_normals(gradient_field(v["image"], None), pts, tan, z,
                                                    fx, tol_mm=tol_mm, reach=1.0)
        depth, _ = VIS.depth_buffer(mesh, np.asarray(rvec).reshape(3, 1),
                                    np.asarray(tvec).reshape(3, 1), v, downscale=1)
        solid = (depth < VIS.FAR / 2).astype(np.float32)
        nx, ny = -tan[:, 1], tan[:, 0]
        from line_check import sample_at
        both = np.stack([sample_at(solid, (pts[:, 0] + s * nx * 4).astype(np.float32),
                                          (pts[:, 1] + s * ny * 4).astype(np.float32))
                         for s in (1, -1)], axis=1)
        sil = ~(both > 0.9).all(axis=1)
        conf.append(100.0 * found.mean())
        if sil.any():
            sil_conf.append(100.0 * found[sil].mean())
    return (float(np.mean(conf)) if conf else 0.0,
            float(np.mean(sil_conf)) if sil_conf else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--fit", required=True, help="starting pose (a fit.json or its directory)")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH")
    ap.add_argument("--out", default=None, help="write the refined fit here")
    ap.add_argument("--dof", choices=["planar", "seated", "full"], default="seated",
                    help="which freedoms to solve. 'planar' is x, y and turn - the part pinned to "
                         "an assumed table height. 'seated' (default) adds z, because the part "
                         "lies FLAT but its height is not known to a millimetre and seating error "
                         "is real. 'full' adds tilt, and on this rig it makes things worse: given "
                         "tilt to play with, the solve explains noise with 12 degrees of rotation "
                         "a part lying on a table cannot have.")
    ap.add_argument("--schedule", default="20,10,5,3,2",
                    help="search windows in mm, wide to narrow; the window is the trust region")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--step-px", type=float, default=2.0)
    args = ap.parse_args()

    base = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(base["board"])
    det = charuco.make_detector(board)
    overrides = []
    for spec in args.cam_profile:
        sub, path = spec.split("=", 1)
        overrides.append((sub, MVF.load_profile(path)))

    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    mesh_path = args.mesh or os.path.join("outputs/ar_models",
                                          os.path.basename(fit.get("mesh") or ""))
    mesh = VIS.load_stl(mesh_path)
    rvec = np.asarray(fit["rvec"], np.float64).ravel()
    tvec = np.asarray(fit["tvec"], np.float64).ravel()

    views = []
    for path in sorted(glob.glob(os.path.join(args.captures, "*"))):
        b = os.path.basename(path)
        if any(k in b for k in ("overlay", "endcheck", "containment", "deviation", "critical",
                                "linecheck")):
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

    before = score(mesh, rvec, tvec, views)
    print("start   confirmed %.0f%%   silhouette %.0f%%   (silhouettes cannot lack contrast, so"
          % before)
    print("        anything they miss is pose error, not lighting)")
    print("")
    rv, tv = refine(mesh, rvec, tvec, views,
                    schedule=tuple(float(x) for x in args.schedule.split(",")),
                    iters=args.iters, dof=args.dof, step_px=args.step_px)
    after = score(mesh, rv, tv, views)

    R0, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    R1, _ = cv2.Rodrigues(np.asarray(rv).reshape(3, 1))
    dr, _ = cv2.Rodrigues(R1 @ R0.T)
    print("")
    print("moved   %.2f mm and %.2f deg from the starting pose"
          % (np.linalg.norm(np.asarray(tv) - tvec), np.degrees(np.linalg.norm(dr))))
    print("        confirmed %.0f%% -> %.0f%%   silhouette %.0f%% -> %.0f%%"
          % (before[0], after[0], before[1], after[1]))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        out = dict(fit)
        out.update({"rvec": [float(x) for x in np.asarray(rv).ravel()],
                    "tvec": [float(x) for x in np.asarray(tv).ravel()],
                    "mesh": os.path.basename(mesh_path),
                    "refined_from": os.path.abspath(src),
                    "refine": {"schedule": args.schedule, "iters": args.iters,
                               "dof": args.dof,
                               "confirmed_before": before[0], "confirmed_after": after[0],
                               "silhouette_before": before[1], "silhouette_after": after[1]}})
        with open(os.path.join(args.out, "fit.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print("        wrote %s" % os.path.join(args.out, "fit.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
