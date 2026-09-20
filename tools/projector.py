#!/usr/bin/env python3
"""
Project the placement outline onto the table with an ordinary video projector (bd c5o).

Industrial laser template projectors (Virtek, LAP, SL-Laser) draw the outline of a part onto the steel so the
operator can lay it down without measuring. That is a five-figure purchase a head, and most of what it buys is CAD
import, closed-loop referencing and certification - none of which is needed here, because the outline already
exists: tools/place_guide.py works it out and draws it on a screen. This puts the same outline on the table.

THE SHORT CUT. A full projector calibration wants its intrinsics and its pose, solved from a plane moved through
several positions. None of that is needed: what is projected lies ON THE TABLE, and one plane maps onto another
through a homography. So ONE photograph of a projected pattern is the whole calibration, and it is exact for
anything lying flat - which the footprint outline and the end marker both are. (A ghost standing up in the air
would need the full solve. Later question.)

    projector.py pattern                     a grid of dots to show full screen on the projector
    projector.py calibrate --captures ...    one photo of it on the table, board in shot -> projector.json
    projector.py show --plan <dir>           the outline to display full screen, in projector pixels

Neither the projector nor the cameras may move after calibration. The board may: the outline is fixed to the
cameras, not to the board.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402


# ---------------------------------------------------------------------------------------------------- geometry

def ray_to_table(points_px, view, z_mm=0.0):
    """Where these pixels' rays meet the table, in the board frame.

    The camera's pose comes from the board, so this is the step that ties what the projector drew to the world the
    part is placed in. Distortion is removed first, because a ray is only a straight line once it is."""
    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    dist = np.asarray(view["dist"], np.float64)
    pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)           # x/z, y/z in the camera frame
    Rc = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))[0]
    tc = np.asarray(view["tvec_cam"], np.float64).ravel()
    origin = -Rc.T @ tc                                                # the camera centre, board frame
    dirs = np.hstack([norm, np.ones((len(norm), 1))]) @ Rc             # each row is Rc.T @ d
    s = (z_mm - origin[2]) / dirs[:, 2]
    return origin + dirs * s[:, None]


def fit_homography(projector_px, table_xy):
    """Projector pixels to table millimetres, with the error it leaves behind."""
    src = np.asarray(projector_px, np.float64).reshape(-1, 1, 2)
    dst = np.asarray(table_xy, np.float64)[:, :2].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        raise ValueError("could not fit a projector-to-table mapping from these points")
    back = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = np.linalg.norm(back - dst.reshape(-1, 2), axis=1)
    return H, float(np.median(err)), float(err.max()), int(mask.sum())


def to_projector(H, table_xy):
    """Table millimetres to projector pixels."""
    pts = np.asarray(table_xy, np.float64)[:, :2].reshape(-1, 1, 2).astype(np.float64)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)


def dot_grid(width, height, cols, rows, radius, inset=0.12):
    """The dot centres, in projector pixels, row by row."""
    xs = np.linspace(width * inset, width * (1 - inset), cols)
    ys = np.linspace(height * inset, height * (1 - inset), rows)
    return np.array([[x, y] for y in ys for x in xs], float)


# ---------------------------------------------------------------------------------------------------- commands

def cmd_pattern(args) -> int:
    """A grid of dots, white on black: the projector shows it, a camera finds it, and that is the calibration."""
    centres = dot_grid(args.width, args.height, args.cols, args.rows, args.radius)
    img = np.zeros((args.height, args.width, 3), np.uint8)
    for i, (x, y) in enumerate(centres):
        r = args.radius * (1.7 if i == 0 else 1.0)                     # the first dot is bigger, to fix the order
        cv2.circle(img, (int(round(x)), int(round(y))), int(round(r)), (255, 255, 255), -1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, img)
    spec = {"width": args.width, "height": args.height, "cols": args.cols, "rows": args.rows,
            "radius": args.radius, "centres_px": centres.tolist()}
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=1)
    print("wrote %s (%d dots, %d x %d)" % (args.out, len(centres), args.cols, args.rows))
    print("show it FULL SCREEN on the projector, aimed at the table, with the board still in both camera views")
    return 0


def find_dots(image, spec, min_area=40.0):
    """The projected dots in a photograph, in the pattern's own order.

    Ordered by grid position, not by detection order: the photograph is taken from an angle, the blobs arrive in
    whatever order the labeller pleases, and a wrong pairing quietly ruins the fit instead of failing."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    grey = cv2.GaussianBlur(grey, (0, 0), 2.0)
    thr = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(thr, 8)
    blobs = [(cents[i], float(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    want = spec["cols"] * spec["rows"]
    if len(blobs) < want:
        return None, "found %d dots of %d - is the whole pattern on the table and in view?" % (len(blobs), want)
    blobs.sort(key=lambda b: -b[1])
    pts = np.array([b[0] for b in blobs[:want]], np.float64)
    areas = np.array([b[1] for b in blobs[:want]], float)
    big = pts[int(np.argmax(areas))] if areas.max() > 1.8 * np.median(areas) else None

    # Put them in grid order by mapping the four CORNER DOTS onto the pattern's grid. Corner dots, not the corners
    # of a bounding rectangle: seen from an angle the grid is a quadrilateral and the rectangle's corners sit out in
    # space where no dot is. And taken from the hull's own quadrilateral, not from extremes of x+y and x-y: with the
    # grid turned in frame those two extremes can be the SAME dot, which collapses the mapping to a point.
    hull = cv2.convexHull(pts.astype(np.float32))
    peri = cv2.arcLength(hull, True)
    corners = None
    for frac in (0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, frac * peri, True).reshape(-1, 2)
        if len(approx) == 4:
            corners = approx.astype(np.float32)
            break
    if corners is None:
        return None, "the dots do not form a quadrilateral - is part of the pattern off the table?"
    grid = np.array([[0, 0], [spec["cols"] - 1, 0], [spec["cols"] - 1, spec["rows"] - 1], [0, spec["rows"] - 1]],
                    np.float32)
    best = None
    for k in range(4):
        H = cv2.getPerspectiveTransform(np.roll(corners, k, axis=0), grid)
        g = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        snapped = np.rint(g)
        if len({(int(a), int(b)) for a, b in snapped}) != want:
            continue
        err = float(np.abs(g - snapped).max())
        if best is None or err < best[0]:
            best = (err, snapped)
    if best is None:
        return None, "the dots did not form a %d x %d grid" % (spec["cols"], spec["rows"])
    ordered = pts[np.lexsort((best[1][:, 0], best[1][:, 1]))]
    if big is not None and np.linalg.norm(ordered[0] - big) > np.linalg.norm(ordered[-1] - big):
        ordered = ordered[::-1]                                        # the big dot marks the first cell
    return ordered, None


def cmd_calibrate(args) -> int:
    import weld_locate as WL

    spec = json.load(open(args.pattern, encoding="utf-8"))
    plan = json.load(open(os.path.join(args.plan, "plan.json"), encoding="utf-8")) if args.plan else {}
    profile = args.profile or plan.get("profile") or "outputs/calibration/RigCam_52FD1B1F.json"
    views = WL.load_views(args.captures, profile, args.cam_profile or plan.get("cam_profile") or [],
                          stereo=args.stereo or plan.get("stereo"))
    if not views:
        print("no usable photographs in %s" % args.captures, file=sys.stderr)
        return 2

    proj_px, table_xy, used = [], [], []
    for v in views:
        dots, why = find_dots(v["image"], spec)
        if dots is None:
            print("  %s: %s" % (v["tag"], why))
            continue
        pts3 = ray_to_table(dots, v)
        proj_px.append(np.asarray(spec["centres_px"], np.float64))
        table_xy.append(pts3)
        used.append(v["tag"])
        print("  %s: %d dots on the table, spanning %.0f x %.0f mm"
              % (v["tag"], len(dots), np.ptp(pts3[:, 0]), np.ptp(pts3[:, 1])))
    if not proj_px:
        print("the pattern was not found in any view", file=sys.stderr)
        return 3

    table = np.vstack(table_xy)
    frame = "calibration board"
    if args.plan and plan.get("home_cameras"):
        import place_guide as PG
        homes = {PG._camera_key(k): val for k, val in plan["home_cameras"].items()}
        found = [(v, s) for v in views for s in homes if s in os.path.basename(v["tag"])]
        if found:
            ref, serial = found[0]
            M, m = PG.home_to_capture(homes[serial], ref)
            table = (table - m) @ M                                    # the plan's own frame, so outlines land right
            frame = "home board (the plan's frame)"
    H, med, worst, inliers = fit_homography(np.vstack(proj_px), table)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"schema": "PSS-Projector/0.1", "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                   "H_projector_to_table": H.tolist(), "frame": frame, "pattern": os.path.abspath(args.pattern),
                   "captures": os.path.abspath(args.captures), "views": used, "inliers": inliers,
                   "residual_mm": {"median": med, "max": worst}, "size": [spec["width"], spec["height"]]}, fh, indent=2)
    print("projector mapped to the %s: %d dots, %.2f mm median error, %.2f mm worst" % (frame, inliers, med, worst))
    print("wrote %s" % args.out)
    if med > 3.0:
        print("WARNING loose for placing a part - check nothing moved and that the whole pattern lies flat")
    return 0


def outline_image(H, polygons, size):
    """Black, with each outline drawn where the projector must light it."""
    w, h = int(size[0]), int(size[1])
    img = np.zeros((h, w, 3), np.uint8)
    for poly, colour, thick in polygons:
        px = to_projector(H, np.asarray(poly, np.float64))
        cv2.polylines(img, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], True, colour, thick, cv2.LINE_AA)
    return img


def cmd_show(args) -> int:
    import place_guide as PG
    from app.services import visibility as VIS
    from weld_faces import article_frame

    proj = json.load(open(args.projector, encoding="utf-8"))
    H = np.asarray(proj["H_projector_to_table"], np.float64)
    plan = json.load(open(os.path.join(args.plan, "plan.json"), encoding="utf-8"))
    polys = []
    if plan.get("band_mm") and not args.no_band:
        polys.append((plan["band_mm"], (110, 110, 110), max(2, args.thickness - 2)))   # the slack, dim
    polys.append((plan["outline_mm"], (255, 255, 255), args.thickness))
    img = outline_image(H, polys, proj["size"])

    # the master end, as a bar across that end of the outline
    mesh = VIS.load_stl(plan["mesh"])
    fr = article_frame(mesh)
    Rm = PG.FP._rot(plan["target"]["rvec"])
    t = np.asarray(plan["target"]["tvec"], float)
    s_end = float(fr["lo"][0] if plan["master"]["end"] == "lo" else fr["hi"][0])
    bar = np.array([fr["centre"] + s_end * fr["axes"][:, 0] + q * fr["axes"][:, 1]
                    for q in (fr["lo"][1], fr["hi"][1])]) @ Rm.T + t
    a, b = np.round(to_projector(H, bar)).astype(int)
    cv2.line(img, tuple(a), tuple(b), (0, 140, 255), args.thickness + 4, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, img)
    print("wrote %s (%d x %d) - show it FULL SCREEN on the projector" % (args.out, proj["size"][0], proj["size"][1]))
    print("white: where the part goes | grey: the slack | orange: the %s" % plan["master"]["name"])
    print("display it at the projector's native size, no scaling, or the mapping is wrong by that scale factor")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pattern", help="write the dot grid to show on the projector")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--radius", type=int, default=18)
    p.add_argument("--out", default="outputs/projector/pattern.png")
    c = sub.add_parser("calibrate", help="one photo of the projected pattern gives the projector's mapping")
    c.add_argument("--captures", required=True, help="capture folder: the pattern on the table, board in shot")
    c.add_argument("--pattern", default="outputs/projector/pattern.json")
    c.add_argument("--plan", default=None, help="a placement plan: map into ITS frame, so its outline lands right")
    c.add_argument("--profile", default=None)
    c.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH")
    c.add_argument("--stereo", default=None)
    c.add_argument("--out", default="outputs/projector/projector.json")
    s = sub.add_parser("show", help="the outline to display, in projector pixels")
    s.add_argument("--plan", required=True)
    s.add_argument("--projector", default="outputs/projector/projector.json")
    s.add_argument("--thickness", type=int, default=5)
    s.add_argument("--no-band", action="store_true", help="the outline only, without the tolerance band")
    s.add_argument("--out", default="outputs/projector/show.png")
    args = ap.parse_args()
    return {"pattern": cmd_pattern, "calibrate": cmd_calibrate, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
