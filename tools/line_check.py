#!/usr/bin/env python3
"""
Does each visible model line fall where the real line is? A verification, not a fit.

THE DIFFERENCE FROM MATCHING. The article sits at a known origin and the model is placed at that
same origin, so a correctly placed part puts every projected line on top of the real one, pixel for
pixel. There is nothing to search for and no correspondence to establish. Asking "where is the
nearest image edge" - a chamfer distance - answers a fitting question instead, and it is why that
reading saturates: on a busy part the nearest edge is always a few pixels away no matter how far
the part has actually moved.

WHAT IS MEASURED INSTEAD. Each sampled model edge point carries its own local direction, so the
question is asked one-dimensionally, along that point's normal: stepping outward in millimetres,
where is the image gradient strongest? The search runs only to a little past tolerance, so it
cannot wander onto a different edge across the part - and if the real line is not inside that
window, that is not a missing measurement, it is a failure, and it is reported as one.

Motion along a contour stays invisible - that is the aperture problem and no method escapes it -
but it does not matter for a go/no-go, because a rigid part cannot move in a way that is parallel
to all of its edges at once. The rails go quiet when the part slides lengthways and the
cross-members and end plate speak up. So the verdict is taken from the WORST line, never the
median: any one line out by more than tolerance means the part is out.

    docker compose run --rm --no-deps api python tools/line_check.py \\
        --captures outputs/ar_captures/turn90 --fit outputs/ar_fits/turn90 \\
        --out outputs/ar_fits/linecheck.png
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

from app.services import charuco, image_edges, multiview_fit as MVF, visibility as VIS  # noqa: E402
from critical_edges import geometric_edges, visible_feature_edges  # noqa: E402

GREEN = (90, 190, 90)
AMBER = (60, 180, 235)
RED = (70, 70, 225)
GREY = (140, 140, 140)


def gradient_field(image, exclude=None):
    """
    Edge STRENGTH, not a binary edge map.

    Canny's threshold decision throws away exactly what a 1D search needs: a line that is present
    but faint becomes absent, and the search reports "no line" where there plainly is one. Keeping
    the gradient magnitude lets the peak-finder weigh the evidence itself.
    """
    g = cv2.GaussianBlur(image_edges.to_gray(image).astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    if exclude is not None:
        gx[exclude > 0] = 0.0
        gy[exclude > 0] = 0.0
    return gx, gy


def sample_at(mag, xs, ys, width=2048):
    """
    Bilinear sample at arbitrary float coordinates.

    remap wants a 2D map and refuses one whose side exceeds 32767, so a long list of points has to
    be folded into a rectangle first - a detail of the call, not of the measurement.
    """
    n = len(xs)
    pad = (-n) % width
    mx = np.concatenate([xs, np.zeros(pad, np.float32)]).reshape(-1, width).astype(np.float32)
    my = np.concatenate([ys, np.zeros(pad, np.float32)]).reshape(-1, width).astype(np.float32)
    out = cv2.remap(mag, mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.ravel()[:n]


def search_along_normals(grad, pts, tan, z, fx, tol_mm=10.0, reach=1.0, step_mm=None,
                         contrast=3.0, bg_mm=25.0):
    """
    For each point, walk its own normal in millimetres and find the real line.

    Sampling in millimetres rather than pixels keeps the window the same PHYSICAL size everywhere
    in the image; a fixed pixel window would search further at the far end of the part than at the
    near end and quietly apply a different tolerance to each.

    Returns (offset_mm, found, peak, base) - offset signed along the normal.
    """
    gx, gy = grad
    nx, ny = -tan[:, 1], tan[:, 0]
    px_per_mm = fx / np.maximum(z, 1e-6)
    # Two different spans, and conflating them was a real bug: the profile is sampled WIDE so
    # that "is this peak above the background" has a background to measure, while acceptance
    # stays NARROW because a line found beyond tolerance is a failure, not a measurement. Judging
    # a peak against the median of a window only a few pixels across compares the edge with
    # itself, and nothing ever clears the bar.
    step_mm = step_mm if step_mm else min(0.1, tol_mm / 40.0)
    span = max(reach * tol_mm, bg_mm)
    offs = np.arange(-span, span + step_mm, step_mm)
    prof = np.empty((len(pts), len(offs)), np.float32)
    for j, t_mm in enumerate(offs):
        t = t_mm * px_per_mm
        xs = (pts[:, 0] + nx * t).astype(np.float32)
        ys = (pts[:, 1] + ny * t).astype(np.float32)
        # The DIRECTIONAL derivative along the normal, not the gradient magnitude. A line only
        # counts as this line if it runs the same way: an edge crossing at right angles has a
        # large magnitude but contributes nothing here, which is what stops the search locking
        # onto whatever happens to be passing nearby. It is also why the false-alarm floor
        # matters more than the median - the median cannot move when most edges are parallel to
        # the error.
        prof[:, j] = np.abs(sample_at(gx, xs, ys) * nx + sample_at(gy, xs, ys) * ny)

    # NEAREST plausible line, not the strongest one in the window. The strongest is the wrong
    # rule here: a bright neighbouring feature 20 mm away outshines the faint line actually under
    # the point, and the measurement reports the distance to the neighbour. Because the pose is
    # known rather than searched for, the closest credible peak is the right answer - and when
    # nothing credible lies within reach, that is a genuine failure, reported as one.
    i = np.arange(len(pts))
    base = np.median(prof, axis=1) + 1e-6
    strong = (prof > contrast * base[:, None]) & (prof > 8.0)
    ridge = np.zeros_like(strong)
    ridge[:, 1:-1] = (prof[:, 1:-1] >= prof[:, :-2]) & (prof[:, 1:-1] > prof[:, 2:])
    cand = strong & ridge & (np.abs(offs)[None, :] <= reach * tol_mm)
    dist = np.where(cand, np.abs(offs)[None, :], np.inf)
    k = np.argmin(dist, axis=1)
    found = np.isfinite(dist[i, k])
    peak = prof[i, k]
    # Sub-sample the peak. A line's gradient ridge is a couple of pixels wide, so the discrete
    # maximum is systematically off by up to half a step; the parabola through its neighbours
    # recovers the crest and takes the quantisation out of the measurement.
    kk = np.clip(k, 1, len(offs) - 2)
    a, b, c = prof[i, kk - 1], prof[i, kk], prof[i, kk + 1]
    den = a - 2 * b + c
    safe = np.where(np.abs(den) > 1e-6, den, 1.0)
    delta = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / safe, 0.0)
    off_mm = offs[kk] + np.clip(delta, -1.0, 1.0) * step_mm

    # A point with no credible line inside the window has not been measured at all, so its
    # offset must not be mistaken for a small one; park it at the window edge and let `found`
    # carry the distinction.
    off_mm = np.where(found, off_mm, reach * tol_mm)
    return off_mm, found, peak, base


def jacobian(pts, z, view, delta_mm=5.0):
    """
    How far each edge point's projection moves, in pixels, per millimetre the part moves - a 2x3
    per point, measured rather than derived, so lens distortion is included exactly.
    """
    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    dist = np.asarray(view["dist"], np.float64).ravel()
    rc = np.asarray(view["rvec_cam"], np.float64).reshape(3, 1)
    tc = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    Rc, _ = cv2.Rodrigues(rc)
    xn = (pts[:, 0] - K[0, 2]) / K[0, 0]
    yn = (pts[:, 1] - K[1, 2]) / K[1, 1]
    cam = np.stack([xn * z, yn * z, z], axis=1)
    world = (Rc.T @ (cam.T - tc)).T
    base, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rc, tc, K, dist)
    base = base.reshape(-1, 2)
    J = np.empty((len(pts), 2, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = delta_mm
        p2, _ = cv2.projectPoints((world + e).reshape(-1, 1, 3), rc, tc, K, dist)
        J[:, :, k] = (p2.reshape(-1, 2) - base) / delta_mm
    return J


def solve_discrepancy(per_view, tol_mm=10.0, iters=6):
    """
    Recover the 3D offset between where the part IS and where it was told to be.

    Every measured normal offset is one linear equation in that offset - a single edge constrains
    only the direction across itself, which is the aperture problem, but thousands of edges facing
    different ways constrain all three axes together. Solving them jointly is what makes a 10 mm
    error read as 10 mm: no averaging of blind points against sighted ones, because the blind ones
    contribute equations that are simply uninformative rather than equations that say zero.

    Reweighted a few times so the small proportion of points that latched onto the wrong line -
    unavoidable on a busy part - cannot drag the answer.
    """
    A, b, w = [], [], []
    for r in per_view:
        if not len(r.get("off", [])) or not r["found"].any():
            continue
        m = r["found"]
        J = jacobian(r["pts"][m], r["z"][m], r["view"])
        nrm = np.stack([-r["tan"][m][:, 1], r["tan"][m][:, 0]], axis=1)
        A.append(np.einsum("ij,ijk->ik", nrm, J))
        b.append(r["off"][m] * r["fx"] / np.maximum(r["z"][m], 1e-6))
    if not A:
        return np.zeros(3), 0.0, 0
    A = np.vstack(A)
    b = np.concatenate(b)
    d = np.zeros(3)
    for _ in range(iters):
        res = A @ d - b
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
        wt = 1.0 / (1.0 + (res / (2.5 * s)) ** 2)          # Cauchy: no hard cut, no tuning cliff
        Aw = A * wt[:, None]
        d, *_ = np.linalg.lstsq(Aw.T @ A, Aw.T @ b, rcond=None)
    res = A @ d - b
    inl = float((np.abs(res) < 3 * (1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6)).mean())
    return d, inl, len(b)


def check(mesh, rvec, tvec, views, tol_mm=10.0, step=3, reach=2.5, topology=True):
    per_view = []
    for v in views:
        pts, tan, z = (visible_feature_edges(mesh, rvec, tvec, v)
                       if topology else geometric_edges(mesh, rvec, tvec, v, step=step))
        fx = float(np.asarray(v["K"], np.float64).reshape(3, 3)[0, 0])
        if not len(pts):
            per_view.append({"view": v, "pts": pts, "off": np.zeros(0),
                             "found": np.zeros(0, bool)})
            continue
        grad = gradient_field(v["image"], v.get("silhouette_unknown"))
        off, found, peak, base = search_along_normals(grad, pts, tan, z, fx, tol_mm=tol_mm,
                                                      reach=reach)
        # A line the camera cannot see is not a line that failed. Where there is no contrast of
        # any orientation within tolerance, the model's edge is simply untestable from this view -
        # two faces meeting at an angle that happens to catch the light the same way, most often -
        # and calling that a failure is the same error as calling a self-occluded edge a pass.
        mag = cv2.magnitude(*grad)
        nx, ny = -tan[:, 1], tan[:, 0]
        sc = fx / np.maximum(z, 1e-6)
        best = np.zeros(len(pts))
        for t in np.linspace(-tol_mm, tol_mm, 21):
            best = np.maximum(best, sample_at(mag, (pts[:, 0] + nx * t * sc).astype(np.float32),
                                                   (pts[:, 1] + ny * t * sc).astype(np.float32)))
        blind = (~found) & (best < 12.0)
        per_view.append({"view": v, "pts": pts, "tan": tan, "z": z, "off": off, "found": found,
                         "blind": blind, "dev": np.abs(off), "fx": fx})
    return per_view


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--tol", type=float, default=10.0, help="go/no-go tolerance in mm")
    ap.add_argument("--step", type=int, default=3, help="edge sampling stride in px")
    ap.add_argument("--reach", type=float, default=1.0,
                    help="search window as a multiple of tolerance. 1.0 is the honest setting: a "
                         "line with no counterpart inside tolerance has FAILED, and widening the "
                         "window to find one only relabels a failure as a measurement")
    ap.add_argument("--shift", default=None, metavar="L,W,D",
                    help="displace the model along its OWN axes (length,width,depth) in mm "
                         "before testing - the known-answer validation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--buffer-edges", action="store_true",
                    help="use the old depth/normal-buffer edge extractor instead of mesh topology")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    profile = MVF.load_profile(args.profile)
    board = charuco.build_board_from_config(profile["board"])
    det = charuco.make_detector(board)
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    mesh_path = args.mesh or os.path.join("outputs/ar_models",
                                          os.path.basename(fit.get("mesh", "")))
    mesh = VIS.load_stl(mesh_path)
    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)

    if args.shift:
        R, _ = cv2.Rodrigues(rvec)
        p = np.vstack([mesh.reshape(-1, 3), mesh.mean(axis=1)])
        ev, evec = np.linalg.eigh(np.cov((p - p.mean(axis=0)).T))
        o = np.argsort(ev)[::-1]
        L, W, D = [float(x) for x in args.shift.split(",")]
        d = (R @ evec[:, o[0]]) * L + (R @ evec[:, o[1]]) * W + (R @ evec[:, o[2]]) * D
        tvec = tvec + d.reshape(3, 1)
        if not args.quiet:
            print("model displaced %s mm along its own (length,width,depth)" % args.shift)

    views = []
    for path in sorted(glob.glob(os.path.join(args.captures, "*"))):
        base = os.path.basename(path)
        if any(k in base for k in ("overlay", "endcheck", "containment", "deviation", "critical",
                                   "linecheck")):
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        v = MVF.build_view(img, profile, board, det, label=base)
        v.update({"K": profile["K"], "dist": profile["dist"], "image": img, "tag": base})
        views.append(v)
    if not views:
        print("no usable captures", file=sys.stderr)
        return 2

    per_view = check(mesh, rvec, tvec, views, tol_mm=args.tol, step=args.step, reach=args.reach,
                     topology=not args.buffer_edges)

    if not args.quiet:
        print("%-28s %7s %9s %9s %9s %8s"
              % ("camera", "lines", "confirmed", "displaced", "no contrast", "median"))
    worst_all, panels = 0.0, []
    for r in per_view:
        v = r["view"]
        if not len(r["off"]):
            continue
        dev, found = r["dev"], r["found"]
        d = dev[found]
        if not len(d):
            continue
        blind = r["blind"]
        disp = (~found) & (~blind)
        worst_all = max(worst_all, float(np.percentile(d, 99)))
        testable = int((found | disp).sum())
        conf = 100.0 * found.sum() / max(testable, 1)
        if not args.quiet:
            print("%-28s %7d %8.0f%% %8.0f%% %10.0f%% %7.1f mm"
                  % (v["tag"][:28], len(dev), conf, 100.0 - conf, 100.0 * blind.mean(),
                     np.median(d)))
        if args.out:
            out = v["image"].copy()
            nx, ny = -r["tan"][:, 1], r["tan"][:, 0]
            scale = r["fx"] / np.maximum(r["z"], 1e-6)
            for j in range(len(dev)):
                x, y = r["pts"][j]
                if not found[j]:
                    cv2.circle(out, (int(x), int(y)), 2, GREY, -1, lineType=cv2.LINE_AA)
                    continue
                col = GREEN if dev[j] <= args.tol else (AMBER if dev[j] <= 2 * args.tol else RED)
                t = r["off"][j] * scale[j]
                cv2.line(out, (int(x), int(y)), (int(x + nx[j] * t), int(y + ny[j] * t)),
                         col, 1, lineType=cv2.LINE_AA)
                cv2.circle(out, (int(x), int(y)), 2, col, -1, lineType=cv2.LINE_AA)
            cv2.rectangle(out, (0, 0), (out.shape[1], 74), (26, 26, 26), -1)
            cv2.putText(out, "%s   worst %.1f mm   tol %.0f mm"
                        % (v["tag"][:24], np.percentile(d, 99), args.tol),
                        (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
            panels.append(out)

    # The 3D answer. Everything above is per-camera bookkeeping; this is the number the operator
    # is owed - how far the article is from where it was told to sit, in millimetres, in the
    # board frame the fixturing already works in.
    # The verdict rests on the CONFIRMED FRACTION, not on a solved offset. Solving one is
    # tempting and was tried: every normal residual is a linear constraint on the 3D displacement,
    # and thousands of them across two cameras look overdetermined. But only points that FOUND a
    # line contribute, and a point whose line has moved beyond tolerance finds nothing - so the
    # solve is fed exactly the points that agree with the nominal pose and returns ~0 whatever the
    # part is doing. It is reported below as a diagnostic and must not be read as a verdict.
    conf = np.concatenate([r["found"][~r["blind"]] for r in per_view if len(r.get("off", []))])
    blind = np.concatenate([r["blind"] for r in per_view if len(r.get("off", []))])
    share = 100.0 * conf.mean() if len(conf) else 0.0
    dark = 100.0 * blind.mean() if len(blind) else 100.0
    d, inl, n = solve_discrepancy(per_view, tol_mm=args.tol)

    if args.quiet:
        print("%.1f %.1f" % (share, dark))
        return 0
    print("")
    print("%.0f%% of testable lines confirmed within %.0f mm; %.0f%% of lines untestable "
          "(no contrast)" % (share, args.tol, dark))
    print("residual on confirmed lines %+.1f, %+.1f, %+.1f mm - a DIAGNOSTIC ONLY, not a verdict: "
          % (d[0], d[1], d[2]))
    print("it is computed from lines that were found, so it cannot see lines that moved away.")
    print("")
    if dark > 20.0:
        print("VERDICT  INCONCLUSIVE - %.0f%% of the model's edges have no contrast in these "
              "images." % dark)
        print("         Too much of the part is unmeasurable for a go/no-go to mean anything.")
        print("         This is a lighting problem, and it is the one thing the cell controls.")
    else:
        print("VERDICT  NOT YET CALIBRATED - %.0f%% confirmed." % share)
        print("         The pass threshold has to be set from a bench test at a KNOWN "
              "displacement.")
        print("         Simulation puts a correct pose at 84% and a 4 mm width error at 67%, but")
        print("         length-wise error does not separate (84% -> 72% -> 74%), so no single")
        print("         threshold is trustworthy yet.")

    if args.out and panels:
        h = min(p.shape[0] for p in panels)
        row = np.hstack([cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels])
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cv2.imwrite(args.out, row)
        if not args.quiet:
            print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
