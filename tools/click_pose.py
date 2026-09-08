#!/usr/bin/env python3
"""
Place a part by hand: pick how it is lying, click three points, get a pose.

The whole operator workflow, without waiting for the frontend. Two steps.

    # 1. prepare - renders the model in each recorded resting orientation and writes a page
    python tools/click_pose.py prepare --model outputs/ar_models/mainframe_default_1to5.json \\
        --mesh outputs/ar_models/mainframe_default_1to5.stl \\
        --captures outputs/ar_captures/exp_new --out outputs/ar_click

    #    open outputs/ar_click/click.html, choose the orientation, click three pairs,
    #    press Save and put the file next to it as clicks.json

    # 2. solve
    python tools/click_pose.py solve --dir outputs/ar_click

WHY THE CLICKS CAN BE ROUGH. Once you have said which face is down, the part has three unknowns -
where it sits and which way it is turned - so three clicks give six equations against three. That
is why this asks for a rough click on a face rather than a magnified click on a hole: measured on
the bench, three clicks tolerate about thirty pixels of slop and still land inside the refiner's
capture radius, after which it converges to a couple of tenths of a millimetre. Solving instead for
all six freedoms - asking the clicks to rediscover that the part is neither tilted nor floating -
needed six clicks at five pixels, and that is the version that would not have worked in a workshop.

The render carries a per-pixel 3D lookup, so a click on the model IS a point on the model. No
anchors to hunt for, no numbering to keep track of.
"""
from __future__ import annotations

import argparse
import base64
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
from face_init import pose_seated  # noqa: E402

VIEW_W = 900


def render_with_lookup(tris, R, size=VIEW_W, cam_R=None):
    """
    A shaded view of the part lying on the chosen face, plus the model point behind every pixel.

    The lookup is the whole trick: it turns "click somewhere on that face" into an exact 3D point
    without the operator having to identify a named feature. Painter's algorithm, nearest last, so
    the lookup holds the surface actually seen.
    """
    P = (R @ tris.reshape(-1, 3).T).T.reshape(-1, 3, 3)
    # View from where the CAMERA actually is. A fixed three-quarter view shows the part from one
    # side while the photograph looks from another, so the two do not correspond and only the
    # faces they happen to share can be clicked. The camera's direction is known from the board
    # pose even though the part's yaw is not, so this at least puts the operator on the right side
    # of the part; the remaining turn is offered as separate variants below.
    if cam_R is not None:
        M = np.asarray(cam_R, np.float64).copy()
        M[1] *= -1.0          # image y runs down; keep the preview the same way up as the photo
        M[2] *= -1.0
    else:
        look, _ = cv2.Rodrigues(np.array([-1.05, 0.0, 0.0]))
        spin, _ = cv2.Rodrigues(np.array([0.0, 0.0, -0.6]))
        M = look @ spin
    V = (M @ P.reshape(-1, 3).T).T.reshape(-1, 3, 3)
    flat = V.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) or 1.0
    s = (size - 40) / span
    h = int((hi[1] - lo[1]) * s + 40)
    img = np.full((h, size, 3), 255, np.uint8)
    lut = np.full((h, size, 3), np.nan, np.float32)
    zbuf = np.full((h, size), -1e9, np.float32)

    # Fill the triangles to get two things at once: the lookup that turns a click into a model
    # point, and a depth buffer to hide the lines behind the part. Neither fill is ever shown.
    order = np.argsort(V[:, :, 2].mean(axis=1))
    for i in order:
        q = ((V[i, :, :2] - lo[:2]) * s + 20).astype(np.int32)
        q[:, 1] = h - q[:, 1]
        m = np.zeros((h, size), np.uint8)
        cv2.fillConvexPoly(m, q, 1)
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        lut[ys, xs] = tris.reshape(-1, 3, 3)[i].mean(axis=0)
        # Depth INTERPOLATED across the triangle, not its mean. A long thin member is one long
        # triangle, and testing an edge against the average depth of the triangle it lies on puts
        # it tens of millimetres out at the ends - which culls the edge in dashes along its own
        # surface. Screen-space depth is planar, so three corners determine it.
        A = np.array([[q[0, 0], q[0, 1], 1.0], [q[1, 0], q[1, 1], 1.0], [q[2, 0], q[2, 1], 1.0]])
        try:
            coef = np.linalg.solve(A, V[i, :, 2])
        except np.linalg.LinAlgError:
            zbuf[ys, xs] = float(V[i, :, 2].mean())
            continue
        zbuf[ys, xs] = coef[0] * xs + coef[1] * ys + coef[2]

    # A wireframe with hidden lines removed - an engineering view, not a see-through one. Every
    # feature edge is sampled along its length and each sample kept only where nothing nearer is
    # drawn over it, so an edge disappears behind the part and reappears the other side rather
    # than being drawn or dropped whole.
    from critical_edges import mesh_feature_edges
    edges, faces, fn = mesh_feature_edges(tris, crease_deg=25.0)
    if len(edges):
        f0, f1 = faces[:, 0], faces[:, 1]
        lone = f1 < 0
        dot = np.ones(len(faces))
        dot[~lone] = np.sum(fn[f0[~lone]] * fn[f1[~lone]], axis=1)
        keep = lone | (np.degrees(np.arccos(np.clip(dot, -1, 1))) > 25.0)
        E = (M @ (R @ edges[keep].reshape(-1, 3).T)).T.reshape(-1, 2, 3)
        for e in E:
            a = np.array([(e[0, 0] - lo[0]) * s + 20, h - ((e[0, 1] - lo[1]) * s + 20), e[0, 2]])
            b = np.array([(e[1, 0] - lo[0]) * s + 20, h - ((e[1, 1] - lo[1]) * s + 20), e[1, 2]])
            n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1])))
            t = np.linspace(0, 1, n)[:, None]
            pts = a * (1 - t) + b * t
            xi = np.clip(pts[:, 0].astype(int), 0, size - 1)
            yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
            # visible where this edge is at least as near as whatever filled that pixel
            vis = pts[:, 2] >= zbuf[yi, xi] - 0.5
            run = None
            for k in range(n):
                if vis[k] and run is None:
                    run = k
                elif not vis[k] and run is not None:
                    if k - run > 1:
                        cv2.line(img, (xi[run], yi[run]), (xi[k - 1], yi[k - 1]),
                                 (45, 45, 45), 1, lineType=cv2.LINE_AA)
                    run = None
            if run is not None and n - run > 1:
                cv2.line(img, (xi[run], yi[run]), (xi[n - 1], yi[n - 1]),
                         (45, 45, 45), 1, lineType=cv2.LINE_AA)
    return img, lut


def b64(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


PAGE = """<!doctype html><meta charset=utf-8><title>Place the part</title>
<style>
 body{font:14px system-ui,sans-serif;margin:0;background:#f6f6f6;color:#222}
 header{padding:10px 16px;background:#222;color:#eee}
 header b{font-size:15px}
 .rests{display:flex;gap:8px;padding:10px 16px;flex-wrap:wrap;background:#ececec}
 .rests figure{margin:0;cursor:pointer;border:3px solid transparent;border-radius:4px;background:#fff}
 .rests figure.sel{border-color:#2a7}
 .rests img{display:block;width:260px}
 .rests figure:hover{border-color:#89b}
 .rests figcaption{text-align:center;font-size:12px;padding:2px}
 .panes{display:flex;gap:12px;padding:12px 16px;align-items:flex-start;flex-wrap:wrap}
 .pane{background:#fff;border:1px solid #ddd;border-radius:4px}
 .pane h3{margin:0;padding:6px 10px;font-size:13px;background:#fafafa;border-bottom:1px solid #eee}
 canvas{display:block;cursor:crosshair;max-width:100%}
 .bar{padding:10px 16px;display:flex;gap:12px;align-items:center}
 button{padding:6px 14px;border:1px solid #888;background:#fff;border-radius:3px;cursor:pointer}
 button:disabled{opacity:.4;cursor:default}
 #out{font-family:ui-monospace,monospace;font-size:11px;white-space:pre;background:#fff;
   border:1px solid #ddd;padding:8px;margin:0 16px 16px;max-height:180px;overflow:auto}
 .hint{opacity:.75}
</style>
<header><b>Place the part</b> &mdash; pick how it is lying, then click three pairs:
a point on the model, then the same point on the photo.
<span class=hint>Rough is fine &mdash; about 30&nbsp;px of slop still converges. The orientations
below are drawn from this camera's direction and differ by a roll and a turn, so compare where the
holes and brackets sit. Hidden lines are removed, so this reads like the photograph.</span></header>
<div class=rests id=rests>__RESTS__</div>
<div class=panes>
  <div class=pane><h3>the model &mdash; click a point</h3><canvas id=cm></canvas></div>
  <div class=pane><h3 id=ph>the photo &mdash; click the same point</h3><canvas id=cp></canvas></div>
</div>
<div class=bar>
  <span id=status>choose an orientation above</span>
  <button id=undo disabled>Undo last</button>
  <button id=save disabled>Save clicks.json</button>
</div>
<pre id=out></pre>
<script>
const DATA = __DATA__;
let rest = null, pending = null, pairs = [];
const cm = document.getElementById('cm'), cp = document.getElementById('cp');
const xm = cm.getContext('2d'), xp = cp.getContext('2d');
const photo = new Image(); photo.src = DATA.photo;
const models = {};
function draw() {
  if (rest === null) return;
  const im = models[rest];
  cm.width = im.width; cm.height = im.height; xm.drawImage(im, 0, 0);
  cp.width = photo.width; cp.height = photo.height; xp.drawImage(photo, 0, 0);
  pairs.forEach((p, i) => { mark(xm, p.mx, p.my, i + 1); mark(xp, p.px, p.py, i + 1); });
  if (pending) mark(xm, pending.mx, pending.my, pairs.length + 1, true);
  const n = pairs.length;
  document.getElementById('status').textContent =
    n >= 3 ? n + ' pairs - enough, Save when ready'
           : (pending ? 'now click the same point on the photo'
                      : (n + ' of 3 pairs - click a point on the model'));
  document.getElementById('save').disabled = n < 3;
  document.getElementById('undo').disabled = !n && !pending;
  document.getElementById('out').textContent = n ? JSON.stringify(payload(), null, 1) : '';
}
function mark(ctx, x, y, n, faint) {
  ctx.strokeStyle = faint ? '#e8a' : '#e33'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x, y, 7, 0, 6.284); ctx.stroke();
  ctx.fillStyle = faint ? '#e8a' : '#e33'; ctx.font = 'bold 13px system-ui';
  ctx.fillText(n, x + 10, y - 8);
}
function at(cv, ev) {
  const r = cv.getBoundingClientRect();
  return [Math.round((ev.clientX - r.left) * cv.width / r.width),
          Math.round((ev.clientY - r.top) * cv.height / r.height)];
}
cm.addEventListener('click', ev => {
  if (rest === null || !luts[rest]) return;
  const [x, y] = at(cm, ev);
  const d = luts[rest].getImageData(Math.floor(x / 2), Math.floor(y / 2), 1, 1).data;
  if (d[3] < 128) { document.getElementById('status').textContent = 'that is off the part - click on it'; return; }
  const r = DATA.rests[rest];
  // the lookup image encodes xyz over the model bounding box, 8 bits a channel
  pending = {mx: x, my: y, model: [r.lo[0] + d[0]/255*r.rng[0],
                                   r.lo[1] + d[1]/255*r.rng[1],
                                   r.lo[2] + d[2]/255*r.rng[2]]};
  draw();
});
cp.addEventListener('click', ev => {
  if (!pending) return;
  const [x, y] = at(cp, ev);
  pairs.push({...pending, px: x, py: y}); pending = null; draw();
});
document.getElementById('undo').onclick = () => { if (pending) pending = null; else pairs.pop(); draw(); };
function payload() {
  return {resting_index: DATA.rests[rest].index, rvec: DATA.rests[rest].rvec,
          view: DATA.view,
          clicks: pairs.map(p => ({model: p.model, image: [p.px, p.py]}))};
}
document.getElementById('save').onclick = () => {
  const b = new Blob([JSON.stringify(payload(), null, 1)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'clicks.json'; a.click();
};
const luts = {};
DATA.rests.forEach((r, i) => {
  const im = new Image(); im.src = r.png; models[i] = im;
  const li = new Image();
  li.onload = () => {
    const c = document.createElement('canvas');
    c.width = li.width; c.height = li.height;
    const g = c.getContext('2d', {willReadFrequently: true});
    g.drawImage(li, 0, 0); luts[i] = g;
  };
  li.src = r.lut;
});
document.querySelectorAll('#rests figure').forEach((f, i) => {
  f.onclick = () => {
    document.querySelectorAll('#rests figure').forEach(g => g.classList.remove('sel'));
    f.classList.add('sel'); rest = i; pairs = []; pending = null; draw();
  };
});
photo.onload = draw;
</script>
"""


def cmd_prepare(args) -> int:
    with open(args.model, "r", encoding="utf-8") as fh:
        model = json.load(fh)
    rests = model.get("resting_faces")
    if not rests:
        print("no resting_faces in %s - run tools/resting_faces.py first" % args.model,
              file=sys.stderr)
        return 2
    tris = VIS.load_stl(args.mesh)

    photos = [p for p in sorted(glob.glob(os.path.join(args.captures, "*")))
              if not any(k in os.path.basename(p) for k in ("overlay", "linecheck", "endcheck"))]
    if not photos:
        print("no photos in %s" % args.captures, file=sys.stderr)
        return 2
    shot = next((p for p in photos if args.view and args.view in os.path.basename(p)), photos[0])
    img = cv2.imread(shot, cv2.IMREAD_COLOR)
    scale = min(1.0, 1100.0 / img.shape[1])
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img

    os.makedirs(args.out, exist_ok=True)
    # the board->camera rotation of the photo being clicked, so the preview matches it
    prof0 = next((p for sub, p in [(s_.split("=", 1)[0], MVF.load_profile(s_.split("=", 1)[1]))
                                   for s_ in args.cam_profile]
                  if sub in os.path.basename(shot)), MVF.load_profile(args.profile))
    board0 = charuco.build_board_from_config(prof0["board"])
    v0 = MVF.build_view(img, prof0, board0, charuco.make_detector(board0),
                        label=os.path.basename(shot))
    cam_R, _ = cv2.Rodrigues(np.asarray(v0["rvec_cam"], np.float64).reshape(3, 1))

    entries, figs = [], []
    variants = []
    for i, r in enumerate(rests):
        R0, _ = cv2.Rodrigues(np.asarray(r["rvec"], np.float64).reshape(3, 1))
        for k, yaw in enumerate((0, 90, 180, 270)):
            Ry, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(yaw)]).reshape(3, 1))
            variants.append((i, yaw, Ry @ R0))
    for j, (i, yaw, R) in enumerate(variants):
        png, lut = render_with_lookup(tris, R, cam_R=cam_R)
        # The lookup travels as an IMAGE, not as JSON. Written out per pixel it came to 35 MB for
        # six orientations; as an 8-bit RGB encoding of xyz over the model's bounding box it is a
        # few hundred kilobytes, and 8 bits across a 433 mm part is 1.7 mm - far finer than the
        # 30 mm of click slop the solve absorbs, so the precision costs nothing.
        lo = np.nanmin(lut.reshape(-1, 3), axis=0)
        hi = np.nanmax(lut.reshape(-1, 3), axis=0)
        rng = np.maximum(hi - lo, 1e-6)
        lut = cv2.resize(lut, (lut.shape[1] // 2, lut.shape[0] // 2),
                         interpolation=cv2.INTER_NEAREST)
        enc = np.zeros((lut.shape[0], lut.shape[1], 4), np.uint8)
        on = ~np.isnan(lut[:, :, 0])
        enc[:, :, 3] = on.astype(np.uint8) * 255          # alpha marks "on the part"
        q = np.clip((lut - lo) / rng, 0, 1)
        enc[:, :, 2] = np.nan_to_num(q[:, :, 0]) * 255    # PNG channel order is B,G,R for cv2
        enc[:, :, 1] = np.nan_to_num(q[:, :, 1]) * 255
        enc[:, :, 0] = np.nan_to_num(q[:, :, 2]) * 255
        entries.append({"index": rests[i]["index"], "yaw": yaw,
                        "rvec": [float(x) for x in cv2.Rodrigues(R)[0].ravel()],
                        "png": b64(png), "lut": b64(enc),
                        "lo": [float(x) for x in lo], "rng": [float(x) for x in rng]})
        tw = 260
        figs.append('<figure><img src="%s"><figcaption>lying %d &middot; turned %d&deg;'
                    '</figcaption></figure>'
                    % (b64(cv2.resize(png, (tw, int(png.shape[0] * tw / png.shape[1])),
                                      interpolation=cv2.INTER_AREA)), i, yaw))

    data = {"photo": b64(small), "view": os.path.basename(shot), "scale": scale,
            "rests": entries}
    page = PAGE.replace("__RESTS__", "".join(figs)).replace("__DATA__", json.dumps(data))
    dest = os.path.join(args.out, "click.html")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(page)
    with open(os.path.join(args.out, "session.json"), "w", encoding="utf-8") as fh:
        json.dump({"mesh": os.path.abspath(args.mesh), "captures": os.path.abspath(args.captures),
                   "profile": args.profile, "cam_profile": args.cam_profile,
                   "photo": os.path.basename(shot), "scale": scale}, fh, indent=2)
    print("wrote %s" % dest)
    print("Open it, pick how the part is lying, click three pairs, Save, and drop clicks.json")
    print("into %s. Then:  python tools/click_pose.py solve --dir %s" % (args.out, args.out))
    return 0


def cmd_solve(args) -> int:
    with open(os.path.join(args.dir, "session.json"), "r", encoding="utf-8") as fh:
        sess = json.load(fh)
    cpath = os.path.join(args.dir, "clicks.json")
    if not os.path.exists(cpath):
        print("no clicks.json in %s - save it from click.html first" % args.dir, file=sys.stderr)
        return 2
    with open(cpath, "r", encoding="utf-8") as fh:
        clicks = json.load(fh)

    base = MVF.load_profile(sess["profile"])
    board = charuco.build_board_from_config(base["board"])
    det = charuco.make_detector(board)
    overrides = [(s.split("=", 1)[0], MVF.load_profile(s.split("=", 1)[1]))
                 for s in (sess.get("cam_profile") or [])]
    mesh = VIS.load_stl(sess["mesh"])

    views = []
    for path in sorted(glob.glob(os.path.join(sess["captures"], "*"))):
        b = os.path.basename(path)
        if any(k in b for k in ("overlay", "linecheck", "endcheck")):
            continue
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            continue
        prof = next((p for sub, p in overrides if sub in b), base)
        v = MVF.build_view(im, prof, board, det, label=b)
        v.update({"K": prof["K"], "dist": prof["dist"], "image": im, "tag": b})
        views.append(v)

    tag = clicks.get("view") or sess["photo"]
    v = next((x for x in views if tag in x["tag"]), None)
    if v is None:
        print("the clicked photo %s is not in %s" % (tag, sess["captures"]), file=sys.stderr)
        return 2
    # clicks were made on a scaled-down photo; put them back in full-resolution pixels
    s = float(sess.get("scale") or 1.0)
    pairs = [{"model": c["model"], "image": [c["image"][0] / s, c["image"][1] / s],
              "_tag": v["tag"]} for c in clicks["clicks"]]
    print("%d clicks on %s, part lying as orientation %s"
          % (len(pairs), tag, clicks.get("resting_index")))

    rest_R, _ = cv2.Rodrigues(np.asarray(clicks["rvec"], np.float64).reshape(3, 1))
    rv, tv = pose_seated(pairs, {v["tag"]: v}, rest_R, z0=None)
    print("clicked pose  t = [%.1f, %.1f, %.1f] mm" % tuple(tv))
    before = PR.score(mesh, rv, tv, views)
    r, t = PR.refine(mesh, rv, tv, views, schedule=(40., 20., 10., 5., 3., 2.), iters=4,
                     dof="seated", verbose=True)
    after = PR.score(mesh, r, t, views)
    print("")
    print("confirmed %.0f%% -> %.0f%%   silhouette %.0f%% -> %.0f%%"
          % (before[0], after[0], before[1], after[1]))
    print("")
    print("Silhouette confirmation is the trust signal. High means the placement you chose is the")
    print("one on the table. Low means it is not, and no amount of refinement will rescue it -")
    print("go back and pick a different orientation rather than believing the pose.")
    out = os.path.join(args.dir, "fit.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rvec": [float(x) for x in np.asarray(r).ravel()],
                   "tvec": [float(x) for x in np.asarray(t).ravel()],
                   "mesh": os.path.basename(sess["mesh"]),
                   "init": "operator clicks", "clicks": len(pairs),
                   "resting_index": clicks.get("resting_index"),
                   "confirmed": after[0], "silhouette": after[1]}, fh, indent=2)
    print("wrote %s" % out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="render the orientations and write the clicking page")
    p.add_argument("--model", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--captures", required=True)
    p.add_argument("--view", default=None, help="substring picking which photo to click on")
    p.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    p.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH")
    p.add_argument("--out", default="outputs/ar_click")
    s = sub.add_parser("solve", help="solve and refine from the saved clicks")
    s.add_argument("--dir", default="outputs/ar_click")
    args = ap.parse_args()
    return {"prepare": cmd_prepare, "solve": cmd_solve}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
