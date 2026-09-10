#!/usr/bin/env python3
"""
Put the welds on the photograph: model weld paths, through the fitted pose, onto the steel.

This is the point of the whole exercise. `connection_detector` finds where solids meet and returns
the weld runs as 3D polylines in model coordinates; `pose_refine` puts model coordinates into the
board frame to better than a millimetre. Nothing joined those two until now.

WHAT THE OPERATOR GETS. Each weld drawn where it belongs on the part, numbered, with its position
in the board frame - which is the frame the fixturing already works in - and its length. The brief
asked for weld centrepoints to +/-25mm; the pose is roughly two orders better than that, so the
limit is the model's own fidelity rather than the measurement.

A CENTREPOINT IS NOT ENOUGH, AND THE SIDECAR CARRIES THE PATH. For a 40mm tack the midpoint is a
fair instruction. For a 300mm fillet it says where the middle is, not where to run, so the path
travels intact and a centrepoint is derived only where something asks for one.

TRUST BEFORE POSITION. A weld position derived from a wrong pose is not approximately right, it is
entirely wrong - a part flipped or turned end-for-end puts every weld somewhere plausible and
useless. So the silhouette confirmation is carried alongside and printed with the results, and a
low figure means the positions should be discarded rather than adjusted.

    # extract the welds a detection run found, into a sidecar
    python tools/weld_locate.py extract --analysis "outputs/analysis/<file>.json" \\
        --node 0:1:1:1:1 --scope within-part --out outputs/welds/mainframe.json

    # project them onto a fitted capture
    python tools/weld_locate.py project --welds outputs/welds/mainframe.json \\
        --captures outputs/ar_captures/<run> --fit outputs/ar_fits/<run> \\
        --mesh outputs/ar_models/<part>.stl --out outputs/welds/overlay.png
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


def cmd_extract(args) -> int:
    """Pull the weld runs out of a stored detection result into a portable sidecar."""
    with open(args.analysis, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    # The sidecar holds ONE detection under "connections", tagged with the node and scope it was
    # run at - not a dict keyed by node|scope as the saver's naming suggests. Check the tags match
    # what was asked for rather than silently measuring a different scope's result.
    results = doc.get("connections") or {}
    if not results.get("connections"):
        print("no detection stored in %s - run one first" % args.analysis, file=sys.stderr)
        return 2
    got = (results.get("node_id") or "", results.get("scope") or "")
    want = (args.node or "", args.scope)
    if got != want:
        print("stored detection is node=%r scope=%r, but you asked for node=%r scope=%r"
              % (got[0], got[1], want[0], want[1]), file=sys.stderr)
        return 2

    welds = []
    for c in results.get("connections") or []:
        if c.get("type") != "welded":
            continue
        for path in (c.get("weld_paths") or []):
            pts = np.asarray(path, np.float64).reshape(-1, 3)
            if len(pts) < 2:
                continue
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            if seg < args.min_length:
                continue
            sa, sb = c.get("solid_a") or {}, c.get("solid_b") or {}
            welds.append({
                "weld_number": "W%03d" % (len(welds) + 1),
                "joins": ["%s:s%s" % (sa.get("node_id"), sa.get("solid_index")),
                          "%s:s%s" % (sb.get("node_id"), sb.get("solid_index"))],
                "method": c.get("weld_method"),
                "length_mm": round(float(seg), 1),
                "path": [[round(float(v), 2) for v in p] for p in pts],
            })
    if not welds:
        print("detection found no weld paths above %.0f mm" % args.min_length, file=sys.stderr)
        return 1

    total = sum(w["length_mm"] for w in welds)
    out = {"source": os.path.basename(args.analysis), "node": args.node, "scope": args.scope,
           "frame": "model coordinates as stored by the detector - the same frame the pose maps "
                    "from, so no further transform is applied downstream",
           "weld_count": len(welds), "total_length_mm": round(total, 1), "welds": welds}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("%d weld runs, %.0f mm total -> %s" % (len(welds), total, args.out))
    print("longest: %s" % ", ".join("%s %.0fmm" % (w["weld_number"], w["length_mm"])
                                    for w in sorted(welds, key=lambda w: -w["length_mm"])[:5]))
    return 0


def _hull_depth(mesh, rvec, tvec, view):
    """Depth buffer of the part's CONVEX HULL - the surface it presents to this camera."""
    from scipy.spatial import ConvexHull
    pts = mesh.reshape(-1, 3)
    h = ConvexHull(pts)
    tris = pts[h.simplices].reshape(-1, 3, 3)
    d, _ = VIS.depth_buffer(tris, rvec, tvec, view, downscale=1)
    return d


def cmd_project(args) -> int:
    with open(args.welds, "r", encoding="utf-8") as fh:
        wd = json.load(fh)
    welds = wd["welds"]
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)
    mesh_path = args.mesh or os.path.join("outputs/ar_models",
                                          os.path.basename(fit.get("mesh") or ""))
    mesh = VIS.load_stl(mesh_path)

    base = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(base["board"])
    det = charuco.make_detector(board)
    overrides = [(s.split("=", 1)[0], MVF.load_profile(s.split("=", 1)[1]))
                 for s in args.cam_profile]
    views = []
    for path in sorted(glob.glob(os.path.join(args.captures, "*"))):
        b = os.path.basename(path)
        if any(k in b for k in ("overlay", "linecheck", "endcheck", "weld")):
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

    # The trust signal, printed before any position. Silhouette confirmation cannot fail for want
    # of contrast, so a low figure means the pose is wrong rather than the picture being poor.
    conf, sil = PR.score(mesh, rvec.ravel(), tvec.ravel(), views)
    print("pose trust: %.0f%% confirmed, %.0f%% silhouette" % (conf, sil))
    if sil < args.min_silhouette:
        print("")
        print("REFUSING to report weld positions. Silhouette confirmation is %.0f%%, below the %.0f%%"
              % (sil, args.min_silhouette))
        print("required. A weld position from a wrong pose is not approximately right, it is")
        print("somewhere plausible and wrong, which is worse than no answer. Re-check the placement.")
        return 1
    print("")

    R, _ = cv2.Rodrigues(rvec)
    if args.min_weld > 0:
        welds = [w for w in welds if w["length_mm"] * args.scale >= args.min_weld]

    # GROUP BY JOINT. The boolean intersection returns several path fragments per contact, so one
    # T-joint arrives as eight separate runs and numbering each stacks eight labels on the same
    # place. An operator inspects a JOINT - "the cleat to the rail" - not a fragment of its
    # perimeter, so the fragments are gathered by the pair of solids they join, their lengths
    # summed, and one number issued per joint. That is also how a weld would be specified on a
    # drawing, which matters when this eventually carries a WPS reference.
    groups = {}
    for w in welds:
        key = tuple(sorted(w.get("joins") or [w["weld_number"]]))
        g = groups.setdefault(key, {"paths": [], "length": 0.0, "method": w.get("method")})
        g["paths"].append(np.asarray(w["path"], np.float64).reshape(-1, 3) * args.scale)
        g["length"] += w["length_mm"] * args.scale

    rows, drawn = [], []
    for n, (key, g) in enumerate(sorted(groups.items(), key=lambda kv: -kv[1]["length"]), start=1):
        num = "W%03d" % n
        worlds = [(R @ p.T).T + tvec.ravel() for p in g["paths"]]
        centre = np.vstack(worlds).mean(axis=0)
        rows.append((num, centre, g["length"], g["method"], len(g["paths"])))
        drawn.append((num, worlds))

    print("%-6s %28s %9s %10s %5s" % ("weld", "centre in the board frame (mm)", "length",
                                       "method", "runs"))
    for num, c, L, meth, nseg in rows[:args.list_max]:
        print("%-6s %8.1f %8.1f %8.1f %7.0f mm %10s %5d"
              % (num, c[0], c[1], c[2], L, (meth or "-")[:10], nseg))
    if len(rows) > args.list_max:
        print("... and %d more" % (len(rows) - args.list_max))
    print("")
    print("%d joints, %.0f mm of weld in total" % (len(rows), sum(r[2] for r in rows)))

    if args.out:
        panels = []
        per_view = {}
        for v in views:
            depth, _ = VIS.depth_buffer(mesh, rvec, tvec, v, downscale=1)
            near = cv2.erode(depth, np.ones((3, 3), np.uint8))
            out = v["image"].copy()
            K = np.asarray(v["K"], np.float64).reshape(3, 3)
            dist = np.asarray(v["dist"], np.float64)
            Rc, _ = cv2.Rodrigues(np.asarray(v["rvec_cam"], np.float64).reshape(3, 1))
            tc = np.asarray(v["tvec_cam"], np.float64).reshape(3, 1)
            h, wd_ = depth.shape
            # PRESENTED FACES ONLY, tested against the CONVEX HULL.
            #
            # Occlusion alone is not enough: on an open frame a weld on the far web is genuinely
            # visible through the openings, so nothing occludes it and it draws. But splitting the
            # part's depth range in half is also wrong - it cuts along the viewing direction, so a
            # joint on the TOP face near the far end lands on the wrong side of the cut for one
            # camera and not the other, and the two views disagree about a face they can both see.
            #
            # The hull settles it. Every weld on a face the camera is presented - top, or the near
            # side - lies on the hull's own near surface. A weld on the far web sits deep behind
            # that surface even with a clear line of sight to it. So both cameras agree about the
            # top face, and differ only about the sides, which is the physical situation.
            hull_depth = _hull_depth(mesh, rvec, tvec, v)
            shown = 0
            seen_here = []
            for num, worlds in drawn:
                world = np.vstack(worlds)
                p2, _ = cv2.projectPoints(world.reshape(-1, 1, 3), v["rvec_cam"], v["tvec_cam"],
                                          K, dist)
                p2 = p2.reshape(-1, 2)
                cz = (Rc @ world.T + tc)[2]
                keep = []
                for (x, y), z_ in zip(p2, cz):
                    xi, yi = int(round(x)), int(round(y))
                    if not (0 <= xi < wd_ and 0 <= yi < h):
                        continue
                    if (z_ - near[yi, xi]) >= 8.0:      # something nearer is in the way
                        continue
                    if (z_ - hull_depth[yi, xi]) > args.shell_mm:   # behind the presented face
                        continue
                    keep.append((xi, yi))
                for a, b in zip(keep[:-1], keep[1:]):
                    if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 60:
                        cv2.line(out, a, b, (40, 230, 255), 3, lineType=cv2.LINE_AA)
                if keep:
                    shown += 1
                    seen_here.append(num)
                    m = keep[len(keep) // 2]
                    cv2.putText(out, num, (m[0] + 6, m[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, (30, 30, 30), 3, lineType=cv2.LINE_AA)
                    cv2.putText(out, num, (m[0] + 6, m[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, (40, 230, 255), 1, lineType=cv2.LINE_AA)
            cv2.rectangle(out, (0, 0), (out.shape[1], 46), (26, 26, 26), -1)
            cv2.putText(out, "%s   %d joints on this face   pose %.0f%% silhouette"
                        % (v["tag"][:26], shown, sil),
                        (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2)
            per_view[v["tag"]] = seen_here
            panels.append(out)
        hh = min(p.shape[0] for p in panels)
        row = np.hstack([cv2.resize(p, (int(p.shape[1] * hh / p.shape[0]), hh)) for p in panels])
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cv2.imwrite(args.out, row)
        print("wrote %s" % args.out)
        # Which joints each camera presents, and which they agree on. Both cameras look down on
        # the same upper face, so the joints there should appear in BOTH lists; only the side
        # faces, each turned away from one camera, should be exclusive. A shared set much smaller
        # than either view's total means the presented-face test is cutting the common face.
        tags = list(per_view)
        if len(tags) == 2:
            a, b = set(per_view[tags[0]]), set(per_view[tags[1]])
            print("")
            print("%-28s %6d joints" % (tags[0][:28], len(a)))
            print("%-28s %6d joints" % (tags[1][:28], len(b)))
            print("%-28s %6d joints  (should be the face both cameras look down on)"
                  % ("seen by BOTH", len(a & b)))
            print("%-28s %6d / %d  (the two side faces)"
                  % ("exclusive to one", len(a ^ b), len(a | b)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="detection result -> weld sidecar")
    e.add_argument("--analysis", required=True)
    e.add_argument("--node", default="")
    e.add_argument("--scope", default="within-part")
    e.add_argument("--min-length", type=float, default=10.0,
                   help="ignore runs shorter than this; short fragments are usually the boolean "
                        "intersection finding a corner rather than a weld")
    e.add_argument("--out", required=True)
    p = sub.add_parser("project", help="weld sidecar + pose -> positions and an overlay")
    p.add_argument("--welds", required=True)
    p.add_argument("--captures", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--mesh", default=None)
    p.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    p.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH")
    p.add_argument("--scale", type=float, default=1.0,
                   help="scale the weld paths into the fitted model's units. The detector works "
                        "in the assembly's own coordinates while the fitted mesh may be a scaled "
                        "copy - the 1:5 test article needs 0.2. This is the coordinate contract "
                        "between the two halves and it has to be stated, not guessed: a wrong "
                        "scale puts every weld somewhere plausible and wrong.")
    p.add_argument("--min-silhouette", type=float, default=60.0)
    p.add_argument("--shell-mm", type=float, default=25.0,
                   help="how far behind the convex hull's near surface a weld may sit and still "
                        "count as presented to this camera. Roughly the depth of the members the "
                        "operator can reach into.")
    p.add_argument("--min-weld", type=float, default=0.0,
                   help="ignore runs shorter than this, in model units. 269 numbered welds is an "
                        "illegible display, and the short ones are mostly tacks and corner "
                        "fragments rather than runs an operator is sent to inspect.")
    p.add_argument("--list-max", type=int, default=15)
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    return {"extract": cmd_extract, "project": cmd_project}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
