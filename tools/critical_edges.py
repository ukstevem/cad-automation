#!/usr/bin/env python3
"""
Which edges of a model actually constrain its pose, and in which direction.

Testing every edge equally is wasteful and, worse, misleading: most of an elongated part's outline
runs ALONG its length, so those points see nothing when the part slides that way. Averaging them in
buries the few points that do see it. That is the aperture problem, and it is why a silhouette test
reads 6 mm across the part and 4 mm along it for the same 10 mm displacement.

The fix is to ask, per edge point, how far its projection moves per millimetre of pose error - and
to count only the component PERPENDICULAR to the local contour direction, because motion along a
contour is invisible. That number is the edge's sensitivity, and it is a property of the model and
the viewpoint alone: no photograph is needed, so it can be computed for any part before it is
made.

What it produces is a map of where to look. On a long weldment the ends carry nearly all the
sensitivity to length-wise error while the flanks carry none, which is a useful thing to know
before deciding where a camera goes or which edges a go/no-go should weigh.

    docker compose run --rm --no-deps api python tools/critical_edges.py \\
        --captures outputs/ar_captures/turn90 --fit outputs/ar_fits/turn90 \\
        --out outputs/ar_fits/critical.png
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


def normal_buffer(tris, rvec, tvec, view):
    """Per-pixel surface normal, painter's algorithm, same convention as the depth buffer."""
    w, h = int(view["width"]), int(view["height"])
    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    Rc, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    tc = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    Ro, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    to = np.asarray(tvec, np.float64).reshape(3, 1)

    world = (Ro @ tris.reshape(-1, 3).T + to)
    cam = (Rc @ world + tc).T.reshape(-1, 3, 3)
    keep = np.all(cam[:, :, 2] > 1e-6, axis=1)
    cam = cam[keep]
    if not len(cam):
        return np.zeros((h, w, 3), np.float32)
    wt = world.T.reshape(-1, 3, 3)[keep]
    n = np.cross(wt[:, 1] - wt[:, 0], wt[:, 2] - wt[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    # Cull back faces. A painter's algorithm orders triangles by their MEAN depth, which is not
    # the order they occupy any particular pixel, so on a thin plate seen obliquely the far side
    # of the plate wins patches of pixels from the near side. Its normal is exactly reversed, and
    # the boundary of every such patch then reads as a 180 degree crease - which is why the false
    # edges traced the tessellation rather than the part: the patches are triangle-shaped.
    # On a closed solid a triangle facing away from the camera is never visible, so it can go.
    nc = (Rc @ n.T).T
    ctr = cam.mean(axis=1)
    front = np.sum(nc * ctr, axis=1) < 0
    cam, n, wt = cam[front], n[front], wt[front]
    if not len(cam):
        return np.zeros((h, w, 3), np.float32)
    uv = (K @ (cam.reshape(-1, 3) / cam.reshape(-1, 3)[:, 2:3]).T).T[:, :2].reshape(-1, 3, 2)
    buf = np.zeros((h, w, 3), np.float32)
    for i in np.argsort(-cam[:, :, 2].mean(axis=1)):
        poly = np.round(uv[i]).astype(np.int32)
        if (poly[:, 0].max() < 0 or poly[:, 1].max() < 0
                or poly[:, 0].min() >= w or poly[:, 1].min() >= h):
            continue
        cv2.fillConvexPoly(buf, poly, (float(n[i, 0]), float(n[i, 1]), float(n[i, 2])),
                           lineType=cv2.LINE_8)
    return buf


def mesh_feature_edges(tris, crease_deg=25.0, quant=1e-3):
    """
    The model's real edges, from mesh TOPOLOGY rather than from a rendered buffer.

    Every buffer-based test we tried failed the same way: a threshold on a rasterised image cannot
    tell a genuine discontinuity from an artefact of how the surface was rasterised. The worst case
    is a plate seen nearly edge-on, where half a pixel of vertex rounding at a triangle boundary
    becomes tens of millimetres of depth - indistinguishable, by magnitude alone, from a real step.
    That is what drew "edges" along the tessellation of flat webs: lines which do not exist.

    Topology has no such ambiguity. A pair of triangles either meets at an angle or it does not,
    and two triangles splitting one flat quad meet at exactly zero degrees, so the diagonal between
    them can never be reported. Three kinds of edge survive:

      BOUNDARY   an edge used by a single triangle - an open edge of the sheet
      CREASE     two triangles whose dihedral angle exceeds *crease_deg*
      SILHOUETTE two triangles, one facing the camera and one facing away (view-dependent, so it
                 is resolved per view in `visible_feature_edges`)

    Returns (edges, faces, normals): edge vertex pairs as (M,2,3), the one or two faces each edge
    belongs to as (M,2) with -1 for none, and per-triangle unit normals.
    """
    tri = np.asarray(tris, np.float64).reshape(-1, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    # Weld vertices by position: an STL has no shared indices, so the same corner arrives as many
    # separate copies and the topology has to be recovered before anything can be asked of it.
    v = tri.reshape(-1, 3)
    key = np.round(v / quant).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    idx = inv.reshape(-1, 3)

    ea = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]], axis=0)
    ea = np.sort(ea, axis=1)
    face = np.tile(np.arange(len(tri)), 3)
    order = np.lexsort((ea[:, 1], ea[:, 0]))
    ea, face = ea[order], face[order]
    uniq, start, count = np.unique(ea, axis=0, return_index=True, return_counts=True)

    faces = np.full((len(uniq), 2), -1, np.int64)
    faces[:, 0] = face[start]
    two = count >= 2
    faces[two, 1] = face[start[two] + 1]

    pos = np.zeros((int(idx.max()) + 1, 3))
    pos[idx.reshape(-1)] = v
    edges = np.stack([pos[uniq[:, 0]], pos[uniq[:, 1]]], axis=1)
    return edges, faces, n


def visible_feature_edges(tris, rvec, tvec, view, crease_deg=25.0, step_px=2.0, tol_mm=2.0,
                          cache={}, with_world=False):
    """
    Project the model's feature edges into one view and keep the parts the camera can actually see.

    Visibility is settled against the depth buffer - a sampled point counts as seen when the buffer
    agrees with its own depth - so self-occlusion is handled without trusting the buffer to LOCATE
    an edge, which is the job it was bad at.
    """
    ck = id(tris)
    if ck not in cache:
        cache[ck] = mesh_feature_edges(tris, crease_deg=crease_deg)
    edges, faces, n = cache[ck]

    Ro, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    to = np.asarray(tvec, np.float64).reshape(3, 1)
    rc = np.asarray(view["rvec_cam"], np.float64).reshape(3, 1)
    tc = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    Rc, _ = cv2.Rodrigues(rc)
    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    dist = np.asarray(view["dist"], np.float64).ravel()

    nw = (Ro @ n.T).T
    ctr = (Ro @ np.asarray(tris, np.float64).reshape(-1, 3, 3).mean(axis=1).T + to).T
    eye = (-Rc.T @ tc).ravel()
    facing = np.sum(nw * (ctr - eye), axis=1) < 0

    f0, f1 = faces[:, 0], faces[:, 1]
    lone = f1 < 0
    dot = np.ones(len(faces))
    dot[~lone] = np.sum(nw[f0[~lone]] * nw[f1[~lone]], axis=1)
    crease = (~lone) & (np.degrees(np.arccos(np.clip(dot, -1, 1))) > crease_deg)
    sil = (~lone) & (facing[f0] != facing[np.where(lone, 0, f1)])
    seen = lone | crease | sil
    # A crease is only worth testing if the camera is on the right side of it.
    seen &= facing[f0] | np.where(lone, False, facing[np.where(lone, 0, f1)])
    e = edges[seen]
    if not len(e):
        z = np.zeros(0)
        return np.zeros((0, 2)), np.zeros((0, 2)), z

    a = (Ro @ e[:, 0].T + to).T
    b = (Ro @ e[:, 1].T + to).T
    pa, _ = cv2.projectPoints(a.reshape(-1, 1, 3), rc, tc, K, dist)
    pb, _ = cv2.projectPoints(b.reshape(-1, 1, 3), rc, tc, K, dist)
    pa, pb = pa.reshape(-1, 2), pb.reshape(-1, 2)
    npx = np.maximum(1, np.ceil(np.linalg.norm(pb - pa, axis=1) / step_px)).astype(int)

    depth, _ = VIS.depth_buffer(tris, rvec, tvec, view, downscale=1)
    h, w = depth.shape
    pts, tan, zs, w3 = [], [], [], []
    for i in range(len(e)):
        t = np.linspace(0, 1, npx[i] + 1)[:, None]
        p3 = a[i] * (1 - t) + b[i] * t
        p2, _ = cv2.projectPoints(p3.reshape(-1, 1, 3), rc, tc, K, dist)
        p2 = p2.reshape(-1, 2)
        cz = (Rc @ p3.T + tc)[2]
        x, y = np.round(p2[:, 0]).astype(int), np.round(p2[:, 1]).astype(int)
        ok = (x >= 0) & (y >= 0) & (x < w) & (y < h)
        if not ok.any():
            continue
        # Visible means nothing is NEARER here - not that the buffer agrees. A silhouette point
        # sits exactly on the boundary and rounds to a background pixel as often as not, so an
        # equality test culls precisely the edges that matter most.
        ok[ok] &= (cz[ok] - depth[y[ok], x[ok]]) < tol_mm
        if not ok.any():
            continue
        d = pb[i] - pa[i]
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        pts.append(p2[ok])
        tan.append(np.tile(d / L, (int(ok.sum()), 1)))
        zs.append(cz[ok])
        w3.append(p3[ok])
    if not pts:
        empty = (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
        return empty + (np.zeros((0, 3)),) if with_world else empty
    out = (np.vstack(pts), np.vstack(tan), np.concatenate(zs))
    # The 3D point behind each sample, which a pose refinement needs to form its Jacobian.
    return out + (np.vstack(w3),) if with_world else out


def geometric_edges(tris, rvec, tvec, view, step=4, depth_step_mm=4.0, crease_deg=25.0,
                    with_kind=False, kinds="all"):
    """
    Every edge of the model that would produce a brightness change in the image, with tangents.

    The outline alone is too coarse for a frame: it catches the silhouette and the holes and
    nothing else, discarding every member that passes in front of another and every crease along a
    box section - which is most of what the camera actually sees. Rendering the geometry buffers
    and differencing them recovers all of it in one pass:

      DEPTH steps  -> occluding contours anywhere in the image, not just at the mask boundary
      NORMAL steps -> creases, where two faces of the same solid meet at an angle

    Both are properties of the model at this pose, so the set is known before any photograph.
    """
    depth, _ = VIS.depth_buffer(tris, rvec, tvec, view, downscale=1)
    solid = depth < VIS.FAR / 2
    d = np.where(solid, depth, 0).astype(np.float32)

    # Depth discontinuity, judged on INVERSE depth. A plane projects to an image where 1/z is
    # exactly linear, at any slant, so its second derivative vanishes - whereas a first-derivative
    # test on depth itself fires across every steeply-inclined surface, because a real plane seen
    # edge-on genuinely does change depth by hundreds of millimetres per pixel. That is what used
    # to flag half the part as "edge". The threshold comes from the image's own robust spread, so
    # it needs no constant tied to the part's size or its distance from the camera.
    inv = np.where(solid, 1.0 / np.maximum(depth, 1e-3), 0).astype(np.float32)
    lap = np.abs(cv2.Laplacian(inv, cv2.CV_32F, ksize=3))
    body = lap[solid]
    spread = 1.4826 * np.median(np.abs(body - np.median(body))) + 1e-12
    depth_edge = (lap > 6.0 * spread) & solid

    # Crease: the ACTUAL angle between neighbouring surface normals. The previous test compared a
    # Sobel response on unit-vector components - a dimensionless number carrying the kernel's own
    # gain - against an angle in radians, so it read a median of 87 degrees against a 25 degree
    # threshold and marked two thirds of the part as creased.
    nb = normal_buffer(tris, rvec, tvec, view)
    turn = np.zeros(depth.shape, np.float32)
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        sh = np.roll(np.roll(nb, -dy, axis=0), -dx, axis=1)
        dot = np.clip(np.sum(nb * sh, axis=2), -1.0, 1.0)
        both = solid & np.roll(np.roll(solid, -dy, axis=0), -dx, axis=1)
        turn = np.maximum(turn, np.where(both, np.arccos(dot), 0.0))
    crease = (turn > np.radians(crease_deg)) & solid

    # the mask boundary itself, which neither gradient reliably lands on
    outline = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_GRADIENT,
                               np.ones((3, 3), np.uint8)) > 0

    # An occluding edge is a break in the surface and always shows in the image. A crease is two
    # faces of one solid meeting at an angle, and shows only if the light happens to fall
    # differently on each - so the two are not equally trustworthy and are kept apart here.
    occl = depth_edge | outline
    crease = crease & ~occl
    if kinds == "occluding":
        crease = np.zeros_like(crease)
    elif kinds == "crease":
        occl = np.zeros_like(occl)
    emap = (occl | crease)
    emap = cv2.morphologyEx(emap.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    emap = cv2.ximgproc.thinning(emap * 255) > 0 if hasattr(cv2, "ximgproc") else emap > 0
    emap = emap.astype(np.uint8)
    ys, xs = np.nonzero(emap)
    if not len(xs):
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    sel = np.arange(0, len(xs), max(1, step))
    pts = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float64)

    # Local tangent from the structure tensor of the edge map - works for any edge, not just a
    # traced contour, and motion along the tangent is what cannot be seen.
    em = emap.astype(np.float32)
    ex = cv2.Sobel(em, cv2.CV_32F, 1, 0, ksize=5)
    ey = cv2.Sobel(em, cv2.CV_32F, 0, 1, ksize=5)
    jxx = cv2.GaussianBlur(ex * ex, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(ey * ey, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(ex * ey, (0, 0), 2.0)
    xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
    a, b, c = jxx[yi, xi], jxy[yi, xi], jyy[yi, xi]
    theta = 0.5 * np.arctan2(2 * b, a - c)              # gradient direction
    tan = np.stack([-np.sin(theta), np.cos(theta)], axis=1)   # tangent is perpendicular to it
    z = depth[yi, xi]
    ok = z < VIS.FAR / 2
    if with_kind:
        kind = np.where(occl[yi, xi], 0, 1)
        return pts[ok], tan[ok], z[ok], kind[ok]
    return pts[ok], tan[ok], z[ok]


def contour_with_tangents(mesh, rvec, tvec, view, step=4):
    """Kept for comparison: the outline only, which is too coarse on a framed part."""
    return geometric_edges(mesh, rvec, tvec, view, step=step)


def sensitivity(mesh, rvec, tvec, view, axes, delta_mm=10.0, step=4):
    """
    Per-edge-point sensitivity: how far a point's own image moves per mm the part moves.

    Computed by EXACT correspondence, not by matching. Each edge pixel is back-projected to the 3D
    point it sees, that same point is projected again at the displaced pose, and the difference is
    the apparent motion. Only the component perpendicular to the local edge direction is counted,
    because motion along an edge is invisible.

    The first version of this matched the original edge map to the displaced one by nearest
    neighbour, and reported 0.03 - a dense edge map always has something nearby, so it measured
    the same saturation that defeats chamfer fitting rather than the sensitivity it claimed. The
    correspondence was available for free the whole time: a back-projected point knows where it
    went.
    """
    pts, tan, z = geometric_edges(mesh, rvec, tvec, view, step=step)
    if not len(pts):
        return pts, tan, z, {}

    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    dist = np.asarray(view["dist"], np.float64).reshape(-1, 1)
    Rc, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    tc = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)

    # pixel + depth -> the 3D point in the board frame
    xn = (pts[:, 0] - K[0, 2]) / K[0, 0]
    yn = (pts[:, 1] - K[1, 2]) / K[1, 1]
    cam = np.stack([xn * z, yn * z, z], axis=1)
    world = (Rc.T @ (cam.T - tc)).T

    base, _ = cv2.projectPoints(world.reshape(-1, 1, 3), view["rvec_cam"], view["tvec_cam"],
                                K, dist)
    base = base.reshape(-1, 2)

    out = {}
    for name, axis in axes.items():
        moved = world + np.asarray(axis, np.float64).reshape(1, 3) * delta_mm
        p2, _ = cv2.projectPoints(moved.reshape(-1, 1, 3), view["rvec_cam"], view["tvec_cam"],
                                  K, dist)
        vec = p2.reshape(-1, 2) - base
        # perpendicular component only - the aperture problem, in one line
        perp = np.abs(vec[:, 0] * -tan[:, 1] + vec[:, 1] * tan[:, 0])
        out[name] = perp / delta_mm            # px of visible motion per mm of real motion
    return pts, tan, z, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--out", required=True)
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

    # The part's own axes, not the board's - "along the length" is what matters, and it is a
    # property of the article rather than of the table it sits on.
    R, _ = cv2.Rodrigues(rvec)
    pts_o = np.vstack([mesh.reshape(-1, 3), mesh.mean(axis=1)])
    d = pts_o - pts_o.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(d.T))
    order = np.argsort(evals)[::-1]
    axes = {"along length": R @ evecs[:, order[0]],
            "across width": R @ evecs[:, order[1]],
            "through depth": R @ evecs[:, order[2]]}

    views = []
    for path in sorted(glob.glob(os.path.join(args.captures, "*"))):
        base = os.path.basename(path)
        if any(k in base for k in ("overlay", "endcheck", "containment", "deviation", "critical")):
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

    panels = []
    print("%-34s %14s %14s %14s" % ("camera", "along length", "across width", "through depth"))
    for v in views:
        pts, tan, z, sens = sensitivity(mesh, rvec, tvec, v, axes, step=args.step)
        if not len(pts):
            continue
        fx = float(np.asarray(v["K"], np.float64).reshape(3, 3)[0, 0])
        # px per mm -> a dimensionless fraction: 1.0 means the outline moves as far as the part
        frac = {k: val * z / fx for k, val in sens.items()}
        print("%-34s %13.2f %14.2f %14.2f"
              % (v["tag"][:34], np.median(frac["along length"]),
                 np.median(frac["across width"]), np.median(frac["through depth"])))

        # colour by sensitivity to LENGTH-WISE motion, the direction everything is blind to
        s = frac["along length"]
        hot = np.clip(s / 0.6, 0, 1)
        out = v["image"].copy()
        for (x, y), h_ in zip(pts, hot):
            col = (int(60 + 40 * (1 - h_)), int(60 + 110 * (1 - h_)), int(70 + 175 * h_))
            cv2.circle(out, (int(x), int(y)), 3, col, -1, lineType=cv2.LINE_AA)
        top = np.argsort(s)[-max(1, len(s) // 12):]
        for i in top:
            cv2.circle(out, (int(pts[i, 0]), int(pts[i, 1])), 7, (60, 235, 255), 2,
                       lineType=cv2.LINE_AA)
        cv2.rectangle(out, (0, 0), (out.shape[1], 68), (26, 26, 26), -1)
        cv2.putText(out, "%s   sensitivity to LENGTH-WISE error" % v["tag"][:26],
                    (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
        panels.append(out)
        frac_useful = float((s > 0.3).mean())
        print("%-34s   %.0f%% of outline points carry real length-wise sensitivity"
              % ("", 100 * frac_useful))

    h = min(p.shape[0] for p in panels)
    grid = np.hstack([p[:h] for p in panels])
    bar = np.full((120, grid.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, "YELLOW ringed = the most informative 8% for length-wise error",
                (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (225, 225, 225), 2)
    cv2.putText(bar, "GREEN = blind (edge runs along the motion)      RED = sensitive to it",
                (24, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, np.vstack([grid, bar]))
    print("")
    print("wrote %s" % args.out)
    print("A figure near 1.0 means the outline moves as far as the part does - fully observable.")
    print("Near 0 means that direction is invisible from this view however good the measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
