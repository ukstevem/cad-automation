#!/usr/bin/env python3
"""
Placement guide (bd 6et): the system shows where, which face down and which way round the part goes, and says
when the operator has it there.

Two steps.

    plan   once per part and rig. The target pose - a chosen resting face, which end points which way, where on
           the table - is fixed in the frame of a reference shot of the board (``--home``). The cameras do not
           move, so the target can be carried into any later shot through camera A's pose, wherever the board
           has been put since.

    check  per shot. Find the part near the target with tools/fast_pose.py, BOTH ways round, and decide which
           way round it is from the master part (the end plate on the tower): the whole-model score barely tells
           an end-for-end flip apart on a symmetric frame, the master part does (77.5% vs 60.0% confirmed on
           tower09). Then say what to do, in the operator's terms - an arrow on the photographs, a distance, a
           turn clockwise or anticlockwise seen from above - and write a page that refreshes itself.

Status, most serious first:

    wrong_way  the part fits better turned end for end
    not_found  no pose near the target the photographs support (wrong face down, outside the search, no board)
    move       found, but not close enough to the target or not inside the well-calibrated zone
    in_place   found, the right way round, within tolerance of the target and inside the zone - green

    python tools/place_guide.py plan --model outputs/ar_models/mainframe_default_1to5.json \\
        --mesh outputs/ar_models/mainframe_default_1to5.stl --rest 3 --end 1 \\
        --target outputs/ar_fits/placement_target.json --zone outputs/ar_fits/best_zone.json \\
        --home outputs/ar_captures/board_home --stereo outputs/calibration/RigStereo_52FD1B1F_B68DE55F.json \\
        --cam-profile B68DE55F=outputs/calibration/RigCam_B68DE55F.json --master-name "end plate" \\
        --out outputs/ar_fits/place/tower
    python tools/place_guide.py check --plan outputs/ar_fits/place/tower --captures outputs/ar_captures/tower09

The board must be in the shot: each check places the cameras from it.
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

import fast_pose as FP  # noqa: E402

STATUS_COLOUR = {"in_place": "#1f8a4c", "move": "#b7791f", "uncertain": "#8a5a1f", "not_found": "#b83232",
                 "wrong_way": "#b83232"}


# ---------------------------------------------------------------------------------------------------- geometry

def master_region(mesh, fr, end="auto", depth_mm=15.0):
    """Which end holds the master part, as a test on a model point's position along the length.

    ``auto`` takes the end with the larger cross-section - the end plate on the tower. That is a stand-in for the
    CAD main part (Tekla: the part the others are welded to), which is the right source once the sidecar says."""
    L, c = fr["axes"][:, 0], fr["centre"]
    lo, hi = float(fr["lo"][0]), float(fr["hi"][0])
    if end == "auto":
        tri_s = (np.asarray(mesh).reshape(-1, 3, 3).mean(axis=1) - c) @ L

        def section(sel):
            p = (np.asarray(mesh).reshape(-1, 3, 3)[sel].reshape(-1, 3) - c) @ fr["axes"]
            return np.ptp(p[:, 1]) * np.ptp(p[:, 2]) if len(p) else 0.0
        end = "lo" if section(tri_s < lo + depth_mm) >= section(tri_s > hi - depth_mm) else "hi"
    return end, (lambda s: s < lo + depth_mm) if end == "lo" else (lambda s: s > hi - depth_mm)


def target_pose(mesh, fr, rest_rvec, end_sign, centre_xy, axis_deg):
    """The part on the table in a chosen resting face, its length along ``axis_deg`` (the master end toward -axis for
    ``end_sign`` +1), centred on ``centre_xy`` and resting on the board plane."""
    R0 = FP._rot(rest_rvec)
    L0 = R0 @ fr["axes"][:, 0]
    L0[2] = 0.0
    L0 /= np.linalg.norm(L0)
    want = end_sign * np.array([np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg)), 0.0])
    yaw = np.arctan2(L0[0] * want[1] - L0[1] * want[0], L0 @ want)
    R = FP._rot([0.0, 0.0, yaw]) @ R0
    P = np.asarray(mesh).reshape(-1, 3) @ R.T
    t = np.array([centre_xy[0], centre_xy[1], 0.0]) - R @ fr["centre"]
    t[2] = -P[:, 2].max()                                   # board z points down: lowest point on the plane
    return R, t


def home_to_capture(home_cam, live_view):
    """(M, m) with x_capture = M x_home + m, through one camera that has not moved between the two shots."""
    Rh, th = FP._rot(home_cam["rvec_cam"]), np.asarray(home_cam["tvec_cam"], float).ravel()
    Rl, tl = FP._rot(live_view["rvec_cam"]), np.asarray(live_view["tvec_cam"], float).ravel()
    return Rl.T @ Rh, Rl.T @ (th - tl)


def turned_end_for_end(R, t, centre_model):
    """The same pose turned half a revolution about the vertical through the part's centre."""
    c = R @ centre_model + t
    Rz = FP._rot([0.0, 0.0, np.pi])
    return Rz @ R, Rz @ (t - c) + c


def yaw_between(R_from, R_to, length_model):
    """Turn about the vertical, in degrees, taking one pose's length direction onto the other's (sign: +z)."""
    a, b = R_from @ length_model, R_to @ length_model
    a[2] = b[2] = 0.0
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def turn_words(deg):
    """A turn about board +z in plain words. The board's z points INTO the table, so a positive turn about it looks
    CLOCKWISE to someone above the table (tests/test_place_guide.py checks this against a camera looking down)."""
    if abs(deg) < 0.5:
        return "no turn needed"
    return "turn %.0f deg %s (seen from above)" % (abs(deg), "clockwise" if deg > 0 else "anticlockwise")


def footprint_inside(mesh, R, t, poly_xy, to_home=None):
    """Is the part's outline on the table (every hull vertex, in the zone's frame) inside the zone polygon?"""
    P = np.asarray(mesh).reshape(-1, 3) @ R.T + t
    if to_home is not None:
        M, m = to_home
        P = (P - m) @ M                                     # x_home = M^T (x - m)
    hull = cv2.convexHull(P[:, :2].astype(np.float32)).reshape(-1, 2)
    poly = np.asarray(poly_xy, np.float32).reshape(-1, 1, 2)
    worst = min(cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in hull)
    return worst >= 0.0, float(worst)


def table_outline(mesh, R, t):
    """The part's outline on the table: the convex hull of everything, flattened. A flat shape on a flat surface is
    what an operator can line a part up with - a wireframe hanging in the air is not."""
    P = np.asarray(mesh).reshape(-1, 3) @ R.T + t
    return cv2.convexHull(P[:, :2].astype(np.float32)).reshape(-1, 2).astype(float)


def grown(poly_xy, margin_mm):
    """``poly_xy`` pushed out by ``margin_mm``: every edge offset along its outward normal, corners cut square.

    This is the tolerance made visible. Green has to agree with something the operator can see, so rather than a
    tighter number the page draws the slack as a second outline - aim at the inner one, green anywhere inside the
    outer one (bd 6et, 2026-09-18: Steve wanted the looser tolerance back, and a number alone could not be judged)."""
    poly = np.asarray(poly_xy, float).reshape(-1, 2)
    if len(poly) < 3 or margin_mm <= 0:
        return poly
    inner = poly if cv2.contourArea(poly.astype(np.float32)) > 0 else poly[::-1]
    out = []
    for i in range(len(inner)):
        a, b = inner[i], inner[(i + 1) % len(inner)]
        e = b - a
        n = np.array([e[1], -e[0]])                       # outward for an anticlockwise ring in x-y
        n /= max(np.linalg.norm(n), 1e-9)
        out += [a + n * margin_mm, b + n * margin_mm]
    return cv2.convexHull(np.asarray(out, np.float32)).reshape(-1, 2).astype(float)


def fraction_within(R, t, samples, maps, px=2.5):
    """Share of model edge points with a same-direction photo edge within ``px`` - the fast stand-in for the
    accurate 'confirmed' score, cheap enough to use on a region."""
    hit, n = 0, 0
    for s, m in zip(samples, maps):
        if not len(s["model"]):
            continue
        c = (s["model"] @ R.T + t) @ m["Rc"].T + m["tc"]
        u = m["K"][0, 0] * c[:, 0] / c[:, 2] + m["K"][0, 2]
        v = m["K"][1, 1] * c[:, 1] / c[:, 2] + m["K"][1, 2]
        ok = (c[:, 2] > 1) & (u >= 0) & (v >= 0) & (u < m["w"] - 1) & (v < m["h"] - 1)
        for k in range(FP.N_BINS):
            a, b = s["start"][k], s["start"][k + 1]
            sel = np.nonzero(ok[a:b])[0] + a
            if len(sel):
                hit += int((FP._bilinear(m["dt"][k], u[sel], v[sel]) <= px).sum())
        n += len(s["model"])
    return 100.0 * hit / max(n, 1)


# ---------------------------------------------------------------------------------------------------- commands

def _views(plan, captures):
    import weld_locate as WL
    return WL.load_views(captures, plan["profile"], plan["cam_profile"], stereo=plan.get("stereo"))


def cmd_plan(args) -> int:
    from app.services import visibility as VIS
    from weld_faces import article_frame

    model = json.load(open(args.model, encoding="utf-8"))
    rests = model.get("resting_faces") or []
    if not 0 <= args.rest < len(rests):
        print("the model has %d resting faces; --rest %d is not one of them" % (len(rests), args.rest), file=sys.stderr)
        return 2
    mesh = VIS.load_stl(args.mesh)
    fr = article_frame(mesh)
    tgt = json.load(open(args.target, encoding="utf-8"))
    zone = json.load(open(args.zone, encoding="utf-8"))
    R, t = target_pose(mesh, fr, rests[args.rest]["rvec"], args.end, tgt["centre_mm"], tgt["long_axis_deg"])
    plan = {"profile": args.profile, "cam_profile": args.cam_profile, "stereo": args.stereo}
    views = _views(plan, args.home)
    if not views:
        print("no board shots in %s" % args.home, file=sys.stderr)
        return 2
    end, test = master_region(mesh, fr, args.master_end, args.master_mm)
    inside, margin = footprint_inside(mesh, R, t, zone["poly_mm"])
    outline = table_outline(mesh, R, t)
    band = grown(outline, args.tol_mm)
    plan.update({
        "schema": "PSS-PlaceGuide/0.1", "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": os.path.abspath(args.model), "mesh": os.path.abspath(args.mesh), "rest": args.rest, "end": args.end,
        "target": {"rvec": FP._vec(R).tolist(), "tvec": t.tolist(), "frame": "home board",
                   "centre_mm": tgt["centre_mm"], "long_axis_deg": tgt["long_axis_deg"]},
        "zone_mm": zone["poly_mm"], "home": os.path.abspath(args.home),
        "outline_mm": outline.tolist(), "band_mm": band.tolist(),
        "home_cameras": {v["tag"]: {"rvec_cam": np.ravel(v["rvec_cam"]).tolist(),
                                    "tvec_cam": np.ravel(v["tvec_cam"]).tolist()} for v in views},
        "master": {"end": end, "depth_mm": args.master_mm, "name": args.master_name},
        "tolerance": {"mm": args.tol_mm, "deg": args.tol_deg, "min_silhouette": args.min_silhouette,
                      "min_present": args.min_present, "zone_slack_mm": args.zone_slack_mm},
    })
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "plan.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2)
    print("target: resting face %d, %s at the %s end, centre (%.0f, %.0f) mm, length at %.0f deg on the home board"
          % (args.rest, args.master_name, end, tgt["centre_mm"][0], tgt["centre_mm"][1], tgt["long_axis_deg"]))
    print("the target's outline is %s the zone (%.0f mm margin)" % ("inside" if inside else "NOT inside", margin))
    print("wrote %s" % os.path.join(args.out, "plan.json"))
    return 0


def _camera_key(tag):
    base = os.path.basename(tag)
    parts = base.split("_")
    return next((p for p in parts if len(p) == 8 and all(ch in "0123456789ABCDEF" for ch in p)), base)


def check(plan, views, mesh, fr):
    """Everything the page shows, for one shot."""
    L = fr["axes"][:, 0]
    homes = {_camera_key(k): v for k, v in plan["home_cameras"].items()}
    match = next(((v, serial) for v in views for serial in homes if serial in os.path.basename(v["tag"])), None)
    if match is None:
        return {"status": "not_found", "say": "The cameras in this shot are not the ones the plan was made with."}
    ref, serial = match
    to_live = home_to_capture(homes[serial], ref)
    M, m = to_live
    Rt_home, tt_home = FP._rot(plan["target"]["rvec"]), np.asarray(plan["target"]["tvec"], float)
    R_tgt, t_tgt = M @ Rt_home, M @ tt_home + m

    maps = [FP.edge_maps(v) for v in views]
    end, in_master = master_region(mesh, fr, plan["master"]["end"], plan["master"]["depth_mm"])
    tol = plan["tolerance"]
    def attempt(R0, t0, accurate=True):
        res = FP.finish(mesh, views, FP._vec(R0), t0, polish=False, maps=maps, accurate=accurate)
        R, t = FP._rot(res["rvec"]), np.asarray(res["tvec"])
        samples = FP.prepare_samples(mesh, res["rvec"], t, views)
        keep = [in_master((s["model"] - fr["centre"]) @ L) for s in samples]
        res["master_pct"] = fraction_within(R, t, FP.subset(samples, keep), maps)
        return res

    # The end-for-end try costs a second search, and the master part settles it on its own: it confirms 60-65%
    # the right way round and 9-18% turned (rig, 2026-09-18). So only spend it when the master part is unhappy.
    p = attempt(R_tgt, t_tgt)
    q = attempt(*turned_end_for_end(R_tgt, t_tgt, fr["centre"]), accurate=False) if p["master_pct"] < 35.0 else None
    wrong_way = bool(q and q["master_pct"] > p["master_pct"] + 15.0 and q["share_pct"] >= p["share_pct"] - 5.0)
    if wrong_way and np.isnan(q["silhouette"]):
        import pose_refine as PR
        q["confirmed"], q["silhouette"] = PR.score(mesh, q["rvec"], np.asarray(q["tvec"]), views)
    use = q if wrong_way else p
    R, t = FP._rot(use["rvec"]), np.asarray(use["tvec"])
    # what the operator must do: take the found part onto the target
    c_found, c_tgt = R @ fr["centre"] + t, R_tgt @ fr["centre"] + t_tgt
    d = c_tgt - c_found
    L_t = R_tgt @ L
    L_t[2] = 0.0
    L_t /= np.linalg.norm(L_t)
    master_dir = L_t if end == "hi" else -L_t
    A_t = np.array([-L_t[1], L_t[0], 0.0])
    turn = yaw_between(R, R_tgt, L) if not wrong_way else None
    dist = float(np.hypot(d[0], d[1]))
    _strict, margin = footprint_inside(mesh, R, t, plan["zone_mm"], to_home=to_live)
    inside = margin >= -tol.get("zone_slack_mm", 15.0)
    band = plan.get("band_mm") or grown(table_outline(mesh, R_tgt, t_tgt), tol["mm"])
    in_band, band_margin = footprint_inside(mesh, R, t, band, to_home=to_live if plan.get("band_mm") else None)
    found = use["silhouette"] >= tol["min_silhouette"] and not use["at_bound"]

    # ONE action at a time, the biggest first: a turn and a slide at once is hard to act on, and the next
    # shot will ask for the other half anyway.
    if wrong_way:
        status = "wrong_way"
        say = "Turn the part end for end: the %s goes at the marked end." % plan["master"]["name"]
    elif use["silhouette"] < tol.get("min_present", 45.0):
        # nothing there at all: with the part off the table this reads about 29%, against 94% for a good lock. The
        # search always returns SOME pose, so a low score is the only thing that says "no part", and it must not be
        # dressed up as "found" (rig, 2026-09-18).
        status = "not_found"
        say = ("No part where the outline is - only %.0f%% of an outline matches. Put it inside the blue outline, "
               "%s at the orange end." % (use["silhouette"], plan["master"]["name"]))
    elif not found:
        status = "uncertain"
        say = ("The part is roughly there, but only %.0f%% of its outline matches. Check nothing is resting on it or "
               "in front of it, and that it is lying on the face shown." % use["silhouette"])
    elif in_band and inside:
        status = "in_place"
        say = "In place."
    else:
        status = "move"
        out_by = max(0.0, -band_margin)
        if abs(turn) > tol["deg"] and abs(turn) >= dist / 10.0:
            say = "%s." % turn_words(turn).capitalize()
        else:
            say = "Slide it %.0f mm along the arrows, into the outer outline." % max(out_by, dist - tol["mm"])
            if not inside:
                say += " Part of it is outside the well-calibrated area."
    lo_hi = [fr["centre"] + float(fr[k][0]) * L for k in ("lo", "hi")]
    return {
        "status": status, "say": say,
        "distance_mm": dist, "along_mm": float(d @ master_dir), "across_mm": float(d @ A_t),
        "turn_deg": turn, "inside_zone": inside, "zone_margin_mm": margin,
        "inside_band": in_band, "band_margin_mm": band_margin,
        "silhouette": use["silhouette"], "confirmed": use["confirmed"],
        # the turned try is only run when the master part is unhappy, so it is often not there at all
        "master_pct": {"planned": p["master_pct"], "turned": q["master_pct"] if q else None},
        "silhouette_both": {"planned": p["silhouette"], "turned": q["silhouette"] if q else None},
        "tilt_deg": use["tilt_deg"], "height_mm": use["height_mm"], "at_bound": use["at_bound"],
        "found": {"rvec": use["rvec"], "tvec": use["tvec"]}, "target": {"rvec": FP._vec(R_tgt).tolist(), "tvec": t_tgt.tolist()},
        "share_pct": use.get("share_pct"),
        "_draw": {"R_tgt": R_tgt, "t_tgt": t_tgt, "R": R, "t": t, "c_found": c_found, "c_tgt": c_tgt,
                  "in_master": in_master, "to_live": to_live, "ends_model": lo_hi, "master_end": end},
    }


def draw(view, mesh, fr, result, plan, scale=0.5):
    """One camera's photograph with flat outlines on the table: where the part goes, the slack around it, where it
    is now, and the end-plate end.

    Flat outlines only, no wireframe. A wireframe of the target and the outline of where it goes are the same
    colour over the same lines, and on the rig it was impossible to tell which was which (2026-09-18) - and a
    shape lying on the table is what a person can line a part up with anyway."""
    img = view["image"].copy()
    dr = result.get("_draw")
    if dr is None:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    K, dist = np.asarray(view["K"], float).reshape(3, 3), np.asarray(view["dist"], float)
    rc, tc = np.asarray(view["rvec_cam"], float).reshape(3, 1), np.asarray(view["tvec_cam"], float).reshape(3, 1)
    M, m = dr["to_live"]

    def project(pts_xyz):
        uv, _ = cv2.projectPoints(np.asarray(pts_xyz, float), rc, tc, K, dist)
        return uv.reshape(-1, 2)

    def flat(poly_xy, frame="home"):
        pts = np.hstack([np.asarray(poly_xy, float), np.zeros((len(poly_xy), 1))])
        return (pts @ M.T + m) * [1, 1, 0] if frame == "home" else pts * [1, 1, 0]

    def ring(pts_xyz, colour, thick):
        uv = project(pts_xyz).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [uv], True, (0, 0, 0), thick + 5, cv2.LINE_AA)
        cv2.polylines(img, [uv], True, colour, thick, cv2.LINE_AA)

    if plan.get("band_mm"):
        # SHADED, not another outline: a second thin line beside the first is exactly what made the page unreadable.
        # Anywhere in the shading is green.
        band_uv = project(flat(plan["band_mm"])).reshape(-1, 1, 2).astype(np.int32)
        shade = img.copy()
        cv2.fillPoly(shade, [band_uv], (235, 205, 160))
        cv2.addWeighted(shade, 0.22, img, 0.78, 0, img)
        cv2.polylines(img, [band_uv], True, (235, 205, 160), 2, cv2.LINE_AA)
    target = flat(plan["outline_mm"]) if plan.get("outline_mm") else flat(table_outline(mesh, dr["R_tgt"], dr["t_tgt"]), "live")
    ring(target, (235, 170, 60), 5)                                           # where the part goes

    if result["status"] != "not_found":
        # magenta for "where it is now": amber sat too close to the orange end marker to tell apart on the rig
        colour = {"in_place": (80, 200, 60), "move": (220, 60, 200)}.get(result["status"], (60, 60, 235))
        ring(flat(table_outline(mesh, dr["R"], dr["t"]), "live"), colour, 3)  # where the part is now
        # one arrow at EACH end, from where that end is to where it should be: together they show the slide and the
        # turn at once, which a single arrow at the centre cannot. Only while there is something to do.
        if result["status"] == "move":
            top = float((np.asarray(mesh).reshape(-1, 3) @ dr["R"].T + dr["t"])[:, 2].min())
            for e in dr["ends_model"]:
                a3, b3 = dr["R"] @ e + dr["t"], dr["R_tgt"] @ e + dr["t_tgt"]
                step = (b3 - a3) * [1, 1, 0]
                if np.linalg.norm(step) < 1.0:
                    continue
                step *= max(1.0, 70.0 / np.linalg.norm(step))                 # at least 70 mm, so it can be seen
                ends = np.vstack([a3, a3 + step])
                ends[:, 2] = top
                a, b = project(ends).astype(int)
                cv2.arrowedLine(img, tuple(a), tuple(b), (0, 0, 0), 14, cv2.LINE_AA, tipLength=0.3)
                cv2.arrowedLine(img, tuple(a), tuple(b), (255, 255, 255), 6, cv2.LINE_AA, tipLength=0.3)

    # last, so nothing covers it: the end the master part goes at, as a bar right across that end of the target,
    # taken from the part's own width rather than from two hull vertices that happen to be neighbours
    s_end = float(fr["lo"][0] if dr["master_end"] == "lo" else fr["hi"][0])
    corners = np.array([fr["centre"] + s_end * fr["axes"][:, 0] + w * fr["axes"][:, 1]
                        for w in (fr["lo"][1], fr["hi"][1])])
    bar = corners @ dr["R_tgt"].T + dr["t_tgt"]
    bar[:, 2] = 0.0
    a, b = project(bar).astype(int)
    cv2.line(img, tuple(a), tuple(b), (0, 0, 0), 15, cv2.LINE_AA)
    cv2.line(img, tuple(a), tuple(b), (0, 140, 255), 8, cv2.LINE_AA)

    ring(flat(plan["zone_mm"]), (120, 120, 120), 2)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="__REFRESH__">
<title>Placement guide</title>
<style>
:root{--bg:#f4f5f3;--ink:#1d2320;--muted:#5d6661;--card:#fff;--line:#d9ddd8;--state:__COLOUR__}
@media (prefers-color-scheme:dark){:root{--bg:#141816;--ink:#e6ebe7;--muted:#9aa49e;--card:#1d2320;--line:#2e3632}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;padding:0 16px 32px}
.banner{background:var(--state);color:#fff;margin:0 -16px;padding:18px 24px}
.banner .state{font:600 13px/1 system-ui;letter-spacing:.08em;text-transform:uppercase;opacity:.9}
.banner .say{font-size:clamp(22px,3.2vw,34px);font-weight:650;margin-top:6px;text-wrap:balance}
.banner form{margin-top:14px}
.banner button{font:600 18px/1 system-ui;background:#fff;color:#1d2320;border:0;border-radius:6px;
padding:14px 34px;cursor:pointer}
.banner button:hover{background:#f0f0ee}.banner .note{opacity:.85;font-size:14px;margin-top:10px}
.facts{display:flex;flex-wrap:wrap;gap:8px 22px;margin:14px 0;color:var(--muted);font-variant-numeric:tabular-nums}
.facts b{color:var(--ink);font-weight:600}
.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.views figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.views img{display:block;width:100%;height:auto}
.views figcaption{padding:6px 10px;color:var(--muted);font-size:13px}
.key{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:13px;margin-top:10px}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
</style></head><body>
<div class="banner"><div class="state">__STATE__</div><div class="say">__SAY__</div>__SET__</div>
<div class="facts">__FACTS__</div>
<div class="views">__VIEWS__</div>
<div class="key"><span><i class="sw" style="background:#3caaeb"></i>where the part goes</span>
<span><i class="sw" style="background:#a0cdeb"></i>close enough (shaded)</span>
<span><i class="sw" style="background:#ff8c00"></i>__MASTER__ end</span>
<span><i class="sw" style="background:#dc3cc8"></i>where it is now</span>
<span><i class="sw" style="background:#3cc850"></i>in place</span>
<span><i class="sw" style="background:#787878"></i>well-calibrated area</span></div>
</body></html>"""

STATE_LABEL = {"in_place": "In place", "move": "Adjust", "uncertain": "Check the part", "not_found": "Not found",
               "wrong_way": "Wrong way round"}


def write_page(out, result, plan, images, capture, seconds, refresh=5, plan_dir=""):
    os.makedirs(out, exist_ok=True)
    figs = []
    for name, img in images:
        fn = "view_%s.jpg" % name
        cv2.imwrite(os.path.join(out, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        figs.append('<figure><img src="%s?t=%d" alt="camera %s"><figcaption>camera %s</figcaption></figure>'
                    % (fn, int(time.time()), html.escape(name), html.escape(name)))
    facts = []
    if result.get("distance_mm") is not None:
        facts.append("off target <b>%.0f mm</b>" % result["distance_mm"])
    if result.get("turn_deg") is not None:
        facts.append("turn <b>%+.1f&deg;</b>" % result["turn_deg"])
    if result.get("inside_zone") is not None:
        facts.append("calibrated area <b>%s</b>" % ("inside" if result["inside_zone"] else "outside"))
    if result.get("silhouette") is not None:
        facts.append("outline found <b>%.0f%%</b>" % result["silhouette"])
    if result.get("master_pct"):
        turned = result["master_pct"]["turned"]
        facts.append("%s matched <b>%.0f%%</b>%s" % (html.escape(plan["master"]["name"]), result["master_pct"]["planned"],
                                                    "" if turned is None else " (turned round <b>%.0f%%</b>)" % turned))
    if result.get("tilt_deg") is not None:
        facts.append("tilt <b>%.1f&deg;</b>, height <b>%+.0f mm</b>" % (result["tilt_deg"], result["height_mm"]))
    facts.append("shot <b>%s</b> checked in %.0f s at %s" % (html.escape(os.path.basename(os.path.normpath(capture))), seconds,
                                                             datetime.datetime.now().strftime("%H:%M:%S")))
    # the operator is at the rig with the page in front of them, so the decision to accept a placement belongs
    # here, not at the terminal. Only offered once it is actually in place.
    if result["status"] == "in_place":
        setter = ('<form method="post" action="/api/v1/place/set"><input type="hidden" name="plan" value="%s">'
                  '<button type="submit">Set this placement</button></form>' % html.escape(plan_dir))
    else:
        setter = '<div class="note">The Set button appears once the part is in place.</div>'
    page = (PAGE.replace("__REFRESH__", str(refresh)).replace("__COLOUR__", STATUS_COLOUR[result["status"]])
            .replace("__SET__", setter)
            .replace("__STATE__", STATE_LABEL[result["status"]]).replace("__SAY__", html.escape(result["say"]))
            .replace("__FACTS__", "".join("<span>%s</span>" % f for f in facts)).replace("__VIEWS__", "".join(figs))
            .replace("__MASTER__", html.escape(plan["master"]["name"])))
    with open(os.path.join(out, "guide.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    status = {k: v for k, v in result.items() if not k.startswith("_")}
    status.update({"capture": os.path.abspath(capture), "seconds": seconds,
                   "checked_at": datetime.datetime.now().isoformat(timespec="seconds")})
    with open(os.path.join(out, "status.json"), "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)


def cmd_check(args) -> int:
    from app.services import visibility as VIS
    from weld_faces import article_frame

    t0 = time.perf_counter()
    plan = json.load(open(os.path.join(args.plan, "plan.json"), encoding="utf-8"))
    mesh = VIS.load_stl(plan["mesh"])
    fr = article_frame(mesh)
    try:
        views = _views(plan, args.captures)
    except Exception as exc:                                    # no board, wrong resolution, ...
        views, why = [], str(exc)
    else:
        why = "no usable photographs in %s" % args.captures
    if not views:
        result = {"status": "not_found", "say": "Cannot see the board: %s" % why}
    else:
        result = check(plan, views, mesh, fr)
    images = [(_camera_key(v["tag"]), draw(v, mesh, fr, result, plan)) for v in views]
    seconds = time.perf_counter() - t0
    write_page(args.out or args.plan, result, plan, images, args.captures, seconds, refresh=args.refresh,
               plan_dir=args.plan.replace("\\", "/").strip("/"))
    print("%s: %s" % (STATE_LABEL[result["status"]].upper(), result["say"]))
    if result.get("distance_mm") is not None:
        print("  off target %.1f mm (along %+.1f toward the %s end, across %+.1f), turn %s, zone %s (margin %.0f mm)"
              % (result["distance_mm"], result["along_mm"], plan["master"]["name"], result["across_mm"],
                 "n/a" if result["turn_deg"] is None else "%+.2f deg" % result["turn_deg"],
                 "inside" if result["inside_zone"] else "OUTSIDE", result["zone_margin_mm"]))
        turned = result["master_pct"]["turned"]
        print("  silhouette %.0f%%; %s confirmed %.0f%%%s; tilt %.2f deg, height %+.1f mm"
              % (result["silhouette"], plan["master"]["name"], result["master_pct"]["planned"],
                 "" if turned is None else " (turned round: %.0f%%)" % turned,
                 result["tilt_deg"], result["height_mm"]))
    print("  %.1f s; page %s" % (seconds, os.path.join(args.out or args.plan, "guide.html")))
    return 0


def cmd_set(args) -> int:
    """The operator says this placement is the one: measure it properly and write the fit the AR view needs.

    The guide's own checks skip the accurate refinement to stay quick; this does not. It starts from the pose the
    last check found - already within a millimetre or two - so the extra work is the measurement, not a search."""
    from app.services import visibility as VIS
    from weld_faces import article_frame

    t0 = time.perf_counter()
    plan = json.load(open(os.path.join(args.plan, "plan.json"), encoding="utf-8"))
    mesh = VIS.load_stl(plan["mesh"])
    fr = article_frame(mesh)
    views = _views(plan, args.captures)
    if not views:
        print("no usable photographs in %s" % args.captures, file=sys.stderr)
        return 2
    prev = os.path.join(args.plan, "status.json")
    start = None
    if os.path.exists(prev):
        st = json.load(open(prev, encoding="utf-8"))
        if st.get("found") and os.path.abspath(st.get("capture", "")) == os.path.abspath(args.captures):
            start = (st["found"]["rvec"], np.asarray(st["found"]["tvec"], float))
    if start is None:                                        # no usable check to build on: find it again
        res = check(plan, views, mesh, fr)
        if "_draw" not in res:
            print("cannot find the part: %s" % res["say"], file=sys.stderr)
            return 3
        start = (res["found"]["rvec"], np.asarray(res["found"]["tvec"], float))

    res = FP.finish(mesh, views, start[0], start[1], along=15.0, across=15.0, yaw=4.0, polish=True)
    if res["silhouette"] < plan["tolerance"].get("min_present", 45.0):
        print("refusing to set: only %.0f%% of the outline matches - that is not the part in place"
              % res["silhouette"], file=sys.stderr)
        return 3
    out = args.out or os.path.join(args.plan, "fit")
    os.makedirs(out, exist_ok=True)
    fit = {"rvec": res["rvec"], "tvec": res["tvec"], "mesh": os.path.basename(plan["mesh"]),
           "init": "placement guide (operator pressed Set)", "confirmed": res["confirmed"],
           "silhouette": res["silhouette"], "captures": os.path.abspath(args.captures),
           "plan": os.path.abspath(args.plan), "rest": plan["rest"], "end": plan["end"],
           "set_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "finish": {k: res[k] for k in ("tilt_deg", "height_mm", "moved_mm", "at_bound", "share_pct")}}
    with open(os.path.join(out, "fit.json"), "w", encoding="utf-8") as fh:
        json.dump(fit, fh, indent=2)
    print("set: confirmed %.0f%%, silhouette %.0f%% (tilt %.2f deg, height %+.1f mm) in %.1f s"
          % (res["confirmed"], res["silhouette"], res["tilt_deg"], res["height_mm"], time.perf_counter() - t0))
    print("wrote %s" % os.path.join(out, "fit.json"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="fix the target pose against a reference shot of the board")
    p.add_argument("--model", required=True, help="model JSON with resting_faces")
    p.add_argument("--mesh", required=True)
    p.add_argument("--rest", type=int, required=True, help="which resting face goes down (position in resting_faces)")
    p.add_argument("--end", type=int, choices=(1, -1), default=1, help="which way round along the target axis")
    p.add_argument("--target", required=True, help="JSON with centre_mm and long_axis_deg on the home board")
    p.add_argument("--zone", required=True, help="JSON with poly_mm, the well-calibrated area on the home board")
    p.add_argument("--home", required=True, help="capture folder: the reference shot of the board")
    p.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    p.add_argument("--cam-profile", action="append", default=[], metavar="SUBSTR=PATH")
    p.add_argument("--stereo", default=None)
    p.add_argument("--master-end", choices=("auto", "lo", "hi"), default="auto")
    p.add_argument("--master-mm", type=float, default=15.0, help="how far in from its end the master part reaches")
    p.add_argument("--master-name", default="master part")
    p.add_argument("--tol-mm", type=float, default=10.0,
                   help="in place within this distance of the target. Keep it tight enough that green agrees with the "
                        "drawn outline: at 25 mm the guide called a part green while it sat visibly outside the box, "
                        "which costs the operator's trust in every later green")
    p.add_argument("--tol-deg", type=float, default=2.0, help="and within this turn")
    p.add_argument("--min-silhouette", type=float, default=75.0, help="found only when this much outline is confirmed")
    p.add_argument("--min-present", type=float, default=45.0,
                   help="below this much outline there is no part there at all, rather than a part that fits badly")
    p.add_argument("--zone-slack-mm", type=float, default=15.0,
                   help="how far past the zone's edge still counts as inside: the zone is traced on a 10 mm grid, and "
                        "the calibration does not fall off a cliff at its line")
    p.add_argument("--out", required=True)
    c = sub.add_parser("check", help="find the part in a shot and say what to do")
    c.add_argument("--plan", required=True, help="directory holding plan.json")
    c.add_argument("--captures", required=True)
    c.add_argument("--out", default=None, help="where the page goes (default: the plan directory)")
    c.add_argument("--refresh", type=int, default=5, help="page refresh, seconds")
    t = sub.add_parser("set", help="the operator accepted the placement: measure it and write the fit")
    t.add_argument("--plan", required=True)
    t.add_argument("--captures", required=True, help="the shot the operator was looking at")
    t.add_argument("--out", default=None, help="where fit.json goes (default: <plan>/fit)")
    args = ap.parse_args()
    return {"plan": cmd_plan, "check": cmd_check, "set": cmd_set}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
