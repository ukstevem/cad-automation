"""
The two rig cameras as ONE instrument (bd ghn).

Each camera used to find the ChArUco board on its own, so every capture carried two independent board
poses. Photographing the board where the tower sits and then in its usual place showed they disagree
by 3.55 mm there (2.30 mm after recalibration, bd dan) - and the overlay is fitted against both views at
once, so that disagreement became overlay error no pose could remove. A single camera cannot see it: it
absorbs its own calibration error into a slightly wrong board pose and still reprojects the board to a
third of a pixel.

Here camera B's position relative to camera A is calibrated once, from simultaneous board photographs,
and every capture then gets ONE board pose solved from both cameras' corners together. The two views
can no longer disagree about where the board is.

    calibrate(pairs, ...)          camera B relative to camera A
    joint_board_pose(...)          one board pose for a capture, from both cameras' corners
    apply(views, rig, board)       give a loaded capture's two views that shared pose

The lens calibrations the rig was solved against are recorded, and applying it to views whose lens
calibration differs is refused: re-calibrating a lens moves its extrinsics too, silently.
"""
from __future__ import annotations

import json
import os

import numpy as np

import cv2

from app.services import charuco

SCHEMA = "PSS-RigStereo/0.1"


def serial_of_profile(profile_name: str) -> str:
    """``RigCam_52FD1B1F`` -> ``52FD1B1F``; photographs carry the same serial in their filename."""
    return str(profile_name).rsplit("_", 1)[-1]


def _rot(v):
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def _vec(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def detect_corners(image, board_cfg):
    """ChArUco corners in one photograph: ``(ids, board points Nx3, image points Nx2)`` or None."""
    board = charuco.build_board_from_config(board_cfg)
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _mc, _mi = charuco.detect_board_detailed(charuco.make_detector(board), grey)
    if corners is None or ids is None or len(ids) == 0:
        return None
    ids = ids.ravel().astype(int)
    obj = np.asarray(board.getChessboardCorners(), np.float64)[ids]
    return ids, obj, corners.reshape(-1, 2).astype(np.float64)


def common_corners(obs_a, obs_b):
    """The corners both cameras found, in the same order: ``(board Nx3, image A Nx2, image B Nx2)``."""
    ids = np.intersect1d(obs_a[0], obs_b[0])
    ia = {int(i): k for k, i in enumerate(obs_a[0])}
    ib = {int(i): k for k, i in enumerate(obs_b[0])}
    return (obs_a[1][[ia[int(i)] for i in ids]], obs_a[2][[ia[int(i)] for i in ids]],
            obs_b[2][[ib[int(i)] for i in ids]])


def calibrate(pairs, Ka, Da, Kb, Db, image_size, floor_px=1.0, factor=2.5, rounds=4, min_common=12):
    """Camera B relative to camera A, lens calibrations held fixed.

    ``pairs`` is ``[(name, board Nx3, image A Nx2, image B Nx2)]`` on common corners. The rig shoots
    its two cameras one after the other, so a board that moved between the two exposures makes a pair
    that no fixed transform fits; pairs whose worst camera error exceeds ``max(floor_px, factor x
    median)`` are dropped and the solve repeated. Returns R, T (A to B), rms and the pairs used."""
    keep = [p for p in pairs if len(p[1]) >= min_common]
    if len(keep) < 3:
        raise ValueError("need at least 3 photo pairs with %d+ common corners, have %d" % (min_common, len(keep)))
    log = []
    while True:
        obj = [p[1].reshape(-1, 1, 3).astype(np.float32) for p in keep]
        ia = [p[2].reshape(-1, 1, 2).astype(np.float32) for p in keep]
        ib = [p[3].reshape(-1, 1, 2).astype(np.float32) for p in keep]
        res = cv2.stereoCalibrateExtended(
            obj, ia, ib, np.asarray(Ka, np.float64), np.asarray(Da, np.float64),
            np.asarray(Kb, np.float64), np.asarray(Db, np.float64), tuple(int(x) for x in image_size),
            np.eye(3), np.zeros((3, 1)), flags=cv2.CALIB_FIX_INTRINSIC)
        rms, R, T = float(res[0]), np.asarray(res[5], np.float64), np.asarray(res[6], np.float64).ravel()
        worst = np.asarray(res[-1], np.float64).reshape(-1, 2).max(axis=1)
        limit = max(floor_px, factor * float(np.median(worst)))
        bad = [keep[i][0] for i in range(len(keep)) if worst[i] > limit]
        log.append({"pairs": len(keep), "rms_px": round(rms, 4), "median_pair_px": round(float(np.median(worst)), 3),
                    "worst_pair_px": round(float(worst.max()), 3), "dropped": bad})
        if not bad or len(log) >= rounds or len(keep) - len(bad) < 3:
            break
        keep = [p for p in keep if p[0] not in bad]
    return {"R": R, "T": T, "rms_px": rms, "pairs_used": [p[0] for p in keep],
            "per_pair_px": {p[0]: round(float(e), 3) for p, e in zip(keep, worst)}, "rounds": log}


def joint_board_pose(obs_a, obs_b, Ka, Da, Kb, Db, R_ab, t_ab):
    """ONE board pose (board to camera A) from both cameras' corners under the fixed A-to-B transform.

    ``obs_x`` is ``(board Nx3, image Nx2)`` for that camera, or None if it did not see the board, in
    which case the other camera's pose is used alone. Returns ``(rvec_a, tvec_a, rms_a, rms_b)``."""
    from scipy.optimize import least_squares

    Ka, Kb = np.asarray(Ka, np.float64), np.asarray(Kb, np.float64)
    Da, Db = np.asarray(Da, np.float64), np.asarray(Db, np.float64)
    R_ab, t_ab = np.asarray(R_ab, np.float64), np.asarray(t_ab, np.float64).ravel()
    if obs_a is None and obs_b is None:
        raise ValueError("neither camera saw the board")
    if obs_b is None or (obs_a is not None and len(obs_a[0]) >= len(obs_b[0])):
        _ok, rv, tv = cv2.solvePnP(obs_a[0], obs_a[1], Ka, Da)
        rv, tv = rv.ravel(), tv.ravel()
    else:
        _ok, rvb, tvb = cv2.solvePnP(obs_b[0], obs_b[1], Kb, Db)
        rv = _vec(R_ab.T @ _rot(rvb))
        tv = R_ab.T @ (tvb.ravel() - t_ab)

    def residual(x):
        out = []
        if obs_a is not None:
            pa, _ = cv2.projectPoints(obs_a[0], x[:3], x[3:], Ka, Da)
            out.append((pa.reshape(-1, 2) - obs_a[1]).ravel())
        if obs_b is not None:
            pb, _ = cv2.projectPoints(obs_b[0], _vec(R_ab @ _rot(x[:3])), R_ab @ x[3:] + t_ab, Kb, Db)
            out.append((pb.reshape(-1, 2) - obs_b[1]).ravel())
        return np.concatenate(out)

    sol = least_squares(residual, np.concatenate([rv, tv]), method="lm")
    r = residual(sol.x).reshape(-1, 2)
    na = 0 if obs_a is None else len(obs_a[0])
    rms = lambda e: float(np.sqrt((e ** 2).sum(axis=1).mean())) if len(e) else float("nan")
    return sol.x[:3], sol.x[3:], rms(r[:na]), rms(r[na:])


def camera_b_pose(rvec_a, tvec_a, R_ab, t_ab):
    """Board to camera B, from board to camera A and the rig's fixed A-to-B transform."""
    R_ab = np.asarray(R_ab, np.float64)
    return _vec(R_ab @ _rot(rvec_a)), R_ab @ np.asarray(tvec_a, np.float64).ravel() + np.asarray(t_ab, np.float64).ravel()


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        rig = json.load(fh)
    if rig.get("schema") != SCHEMA:
        raise ValueError("%s is not a %s file" % (path, SCHEMA))
    rig["R"] = np.asarray(rig["R"], np.float64).reshape(3, 3)
    rig["T"] = np.asarray(rig["T"], np.float64).ravel()
    rig["path"] = path
    return rig


def check_intrinsics(rig, which, K, tol_px=0.5):
    """Refuse a lens calibration other than the one the rig was solved against."""
    rec = rig["intrinsics"][which]
    K = np.asarray(K, np.float64).reshape(3, 3)
    got = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    want = np.array([rec["fx"], rec["fy"], rec["cx"], rec["cy"]])
    if np.abs(got - want).max() > tol_px:
        raise ValueError("camera %s (%s): the stereo rig %s was solved against fx/fy/cx/cy %s, but the lens "
                         "calibration in use has %s. Re-run tools/stereo_calibrate.py after any recalibration."
                         % (which, rig["cameras"][which], os.path.basename(rig.get("path", "")),
                            np.round(want, 1).tolist(), np.round(got, 1).tolist()))


def apply_observations(view_a, view_b, obs_a, obs_b, rig):
    """Replace two views' independent board poses with the joint one. ``obs_x`` is ``(board Nx3,
    image Nx2)`` or None. The independent poses are kept as ``*_independent`` for diagnostics."""
    ra, ta, rms_a, rms_b = joint_board_pose(obs_a, obs_b, view_a["K"], view_a["dist"],
                                            view_b["K"], view_b["dist"], rig["R"], rig["T"])
    rb, tb = camera_b_pose(ra, ta, rig["R"], rig["T"])
    # how far the joint pose moves the board, seen from each camera, at the observed corners
    shift = {}
    for key, v, rv, tv, obs in (("A", view_a, ra, ta, obs_a), ("B", view_b, rb, tb, obs_b)):
        if obs is not None and v.get("rvec_cam") is not None:
            old = (_rot(v["rvec_cam"]) @ obs[0].T).T + np.asarray(v["tvec_cam"], np.float64).ravel()
            new = (_rot(rv) @ obs[0].T).T + tv
            shift[key] = round(float(np.linalg.norm(old - new, axis=1).mean()), 3)
        v["rvec_cam_independent"], v["tvec_cam_independent"] = v.get("rvec_cam"), v.get("tvec_cam")
        v["rvec_cam"] = np.asarray(rv, np.float64).reshape(3, 1)
        v["tvec_cam"] = np.asarray(tv, np.float64).reshape(3, 1)
    info = {"applied": True, "rms_px": {"A": round(rms_a, 3), "B": round(rms_b, 3)}, "moved_mm": shift}
    view_a["stereo"] = view_b["stereo"] = info
    return info


def apply(views, rig, board_cfg):
    """Give a capture's two views the board pose solved from both cameras together."""
    a, b = rig["cameras"]["A"], rig["cameras"]["B"]
    va = [v for v in views if a in v["tag"]]
    vb = [v for v in views if b in v["tag"]]
    if len(va) != 1 or len(vb) != 1:
        return {"applied": False, "reason": "need exactly one photograph from each rig camera, have %d and %d"
                % (len(va), len(vb))}
    check_intrinsics(rig, "A", va[0]["K"])
    check_intrinsics(rig, "B", vb[0]["K"])
    obs = []
    for v in (va[0], vb[0]):
        d = detect_corners(v["image"], board_cfg)
        obs.append(None if d is None else (d[1], d[2]))
    return apply_observations(va[0], vb[0], obs[0], obs[1], rig)


def describe(info):
    if not info.get("applied"):
        return "not applied - %s" % info.get("reason")
    return ("one board pose from both cameras: reprojection A %.2f px, B %.2f px; board moved %s mm from each "
            "camera's own estimate" % (info["rms_px"]["A"], info["rms_px"]["B"],
                                       ", ".join("%s %.2f" % kv for kv in sorted(info["moved_mm"].items()))))
