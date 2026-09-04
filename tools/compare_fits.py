#!/usr/bin/env python3
"""
Compare two fits of the same part and say, in physical terms, how the pose changed.

Written to answer one question honestly: when the part is physically turned end-for-end, does
the solver follow it? Two traps make that harder than it sounds.

FIRST, the primary/flipped LABEL in fit.json is not evidence. It names hypotheses relative to
wherever the coarse search happened to land, not relative to the world, so it can change without
the pose changing and vice versa.

SECOND, a single "180 degrees about axis A" summary conflates the two rotations this project has
to keep apart. For a part lying on the board with its length along X:

    180 about Z (board normal)  = turned end-for-end, same face up   <- well determined
    180 about X (its own axis)  = rolled over, same end forward      <- poorly determined (3.4%
                                                                       of geometry discriminates)
    180 about Y                 = BOTH at once, since Rz(180).Rx(180) == Ry(180)

So a bare "180 about Y" reads as a failure when it may be a correct end-for-end call plus a flip
on the axis already known to be unreliable. This tool decomposes the relative rotation into those
two components and reports them separately.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402


def load(p):
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def rot(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    return R


def long_axis(model_path: str) -> np.ndarray:
    """The part's own length direction, in the object frame, by PCA of its edge points."""
    m = load(model_path)
    pts = np.vstack([np.asarray(e, np.float64).reshape(-1, 3) for e in m["edges"]])
    d = pts - pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    return evecs[:, int(np.argmax(evals))]


def align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector *a* onto unit vector *b* (Rodrigues)."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Antiparallel: any perpendicular axis gives the 180 turn.
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        R, _ = cv2.Rodrigues((axis * np.pi).reshape(3, 1))
        return R
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit_a")
    ap.add_argument("fit_b")
    ap.add_argument("--model", default="outputs/ar_models/mainframe_default_1to5.json")
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--captures-a", default=None,
                    help="capture dir for fit_a. Supplying both capture dirs switches the "
                         "comparison into the CAMERA frame, which is required whenever the BOARD "
                         "was moved between the two shots: the board is the world frame, so a "
                         "board-frame answer reports (part turn - board turn), not the part's "
                         "turn. The cameras are fixed, so the camera frame measures what the part "
                         "actually did in the room.")
    ap.add_argument("--captures-b", default=None)
    ap.add_argument("--cam", default="B68DE55F", help="which camera to reference")
    ap.add_argument("--expect-plan", type=float, default=None,
                    help="assert the part swung this many degrees in plan (e.g. 90)")
    ap.add_argument("--expect-endforend", action="store_true",
                    help="assert the part was physically turned end-for-end between the two")
    args = ap.parse_args()

    a, b = load(args.fit_a), load(args.fit_b)
    Ra, Rb = rot(a["rvec"]), rot(b["rvec"])
    e = long_axis(args.model)
    normal = np.array([0.0, 0.0, 1.0])          # board normal, in the board frame
    frame = "board"

    if args.captures_a and args.captures_b:
        import glob
        from app.services import charuco, image_edges, multiview_fit as MVF
        from app.services.board_pose import charuco_board_pose
        prof = MVF.load_profile(args.profile)
        boardobj = charuco.build_board_from_config(prof["board"])
        det = charuco.make_detector(boardobj)

        def board_to_cam(capdir):
            hits = [f for f in glob.glob(os.path.join(capdir, "*"))
                    if args.cam in os.path.basename(f) and "overlay" not in f]
            if not hits:
                raise SystemExit("no capture for camera %s in %s" % (args.cam, capdir))
            img = cv2.imread(hits[0], cv2.IMREAD_COLOR)
            cor, ids, _m, _i = charuco.detect_board_detailed(det, image_edges.to_gray(img))
            rc, tc, _n = charuco_board_pose(cor, ids, boardobj, prof["K"], prof["dist"])
            R, _ = cv2.Rodrigues(np.asarray(rc, np.float64).reshape(3, 1))
            return R

        Ca, Cb = board_to_cam(args.captures_a), board_to_cam(args.captures_b)
        # Board moved? Quantify it, because it is the thing that would otherwise corrupt the answer.
        Rboard = Cb @ Ca.T
        rvb, _ = cv2.Rodrigues(Rboard)
        print("  (board itself moved %.1f deg between the two shots - hence the camera frame)"
              % float(np.degrees(np.linalg.norm(rvb))))
        Ra, Rb = Ca @ Ra, Cb @ Rb          # part pose expressed in the fixed camera
        normal = Ca @ np.array([0.0, 0.0, 1.0])
        frame = "camera (board motion removed)"
    print("  frame: %s" % frame)

    da, db = Ra @ e, Rb @ e                       # the part's length direction, in world, each time
    # End-for-end is a question about direction IN PLAN, so compare the board-plane projections.
    n = normal / max(np.linalg.norm(normal), 1e-9)
    pa = da - n * float(np.dot(da, n))
    pb = db - n * float(np.dot(db, n))
    pa /= max(np.linalg.norm(pa), 1e-9)
    pb /= max(np.linalg.norm(pb), 1e-9)
    plan = float(np.degrees(np.arccos(np.clip(np.dot(pa, pb), -1, 1))))

    # Roll: take out the rotation that merely re-aims the long axis; what is left turns the part
    # about that axis, which is precisely the roll.
    R_rel = Rb @ Ra.T
    R_roll = R_rel @ align(da, db).T
    rv, _ = cv2.Rodrigues(R_roll)
    roll = float(np.degrees(np.linalg.norm(rv)))
    if roll > 1e-6:
        # Sign it against the part's own axis so "rolled the other way" is not reported as 0.
        if float(np.dot(rv.ravel() / np.linalg.norm(rv), db)) < 0:
            roll = -roll
    tilt = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(db, n))), 0, 1))))

    print("\n=== HOW THE POSE CHANGED ===")
    print("  %s  ->  %s" % (os.path.basename(os.path.dirname(args.fit_a)),
                            os.path.basename(os.path.dirname(args.fit_b))))
    print("  long axis swung IN PLAN by   %6.1f deg   (180 = turned end-for-end)" % plan)
    print("  rolled about its own axis by %6.1f deg   (0 = same face up)" % roll)
    print("  long axis out of board plane %6.1f deg   (should be ~90 for a part lying flat)" % tilt)

    print("\n=== DECISIVENESS (discriminating-geometry tie-break) ===")
    for name, d in ((args.fit_a, a), (args.fit_b, b)):
        oc = d.get("orientation_choice") or {}
        sc = oc.get("scores", {})
        print("  %-22s margin %5.2f px   %s"
              % (os.path.basename(os.path.dirname(name)), oc.get("margin_px", float("nan")),
                 ", ".join("%s %.2f" % (k.replace(" 180 deg end-for-end", ""), v)
                           for k, v in sc.items())))

    if args.expect_plan is not None:
        print("")
        print("=== VERDICT ===")
        want = float(args.expect_plan)
        if abs(plan - want) <= 20.0:
            print("  PLAN TURN: PASS - expected ~%.0f deg, measured %.1f deg." % (want, plan))
        else:
            print("  PLAN TURN: FAIL - expected ~%.0f deg, measured %.1f deg." % (want, plan))
        print("  ROLL:      %.1f deg (separate axis, separately unreliable)" % roll)

    if args.expect_endforend:
        print("\n=== VERDICT ===")
        ok_plan = abs(plan - 180.0) <= 20.0
        ok_roll = abs(abs(roll) - 0.0) <= 25.0 or abs(abs(roll) - 360.0) <= 25.0
        if ok_plan:
            print("  END-FOR-END: PASS - the solver swung the part %.1f deg in plan, i.e. it"
                  % plan)
            print("               tracked the physical turn rather than staying put.")
        else:
            print("  END-FOR-END: FAIL - expected ~180 deg in plan, got %.1f." % plan)
        if ok_roll:
            print("  ROLL:        unchanged (%.1f deg), as it should be if the part was not"
                  % roll)
            print("               turned over.")
        else:
            print("  ROLL:        CHANGED by %.1f deg. If the part was not physically rolled,"
                  % roll)
            print("               this is the solver flipping on the axis already known to be")
            print("               poorly determined - independent of the end-for-end result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
