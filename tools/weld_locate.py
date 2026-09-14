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
import datetime
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.services import charuco, multiview_fit as MVF, visibility as VIS  # noqa: E402
import pose_refine as PR  # noqa: E402


_IFC_B64 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"


def ifc_guid(seed: str) -> str:
    """
    An IfcGloballyUniqueId DERIVED from the weld's identity, not minted fresh.

    A random GUID per run would be exactly as unstable as the numbering this file used to have -
    re-run the detector and every weld becomes a new object, so nothing downstream could ever
    match a weld to the record it already has. Hashing the identity instead means the same weld
    on the same part resolves to the same GlobalId forever, which is what the id is for.

    22 characters from IFC's own alphabet: one leading 2-bit group then 21 of 6 bits.
    """
    n = int.from_bytes(hashlib.sha1(seed.encode("utf-8")).digest()[:16], "big")
    out = []
    for _ in range(21):
        out.append(_IFC_B64[n & 63])
        n >>= 6
    out.append(_IFC_B64[n & 3])
    return "".join(reversed(out))


def _piece_mark(doc, node, override):
    """
    The namespace that makes a weld number unique across a JOB rather than a run.

    W001 restarting for every part collides the moment two parts are analysed, and the number is
    the key a weld map joins on, so the collision is not cosmetic. A piece mark already identifies
    the part uniquely within a job, so borrowing it inherits that guarantee without a registry.

    Derived where possible and stated in the output either way - a namespace nobody can see is
    worse than no namespace, because it looks unique without being checkable.
    """
    if override:
        return override, "given on the command line"
    name = (doc.get("cnc_member_names") or {}).get(node)
    if name:
        src = "cnc_member_names[%s]" % node
    else:
        # Fall back to the part the solids belong to. Weldment nodes carry no member name of their
        # own, but every solid in one names its parent, and they agree.
        names = [(c.get("solid_a") or {}).get("name")
                 for c in ((doc.get("connections") or {}).get("connections") or [])]
        names = [x for x in names if x]
        name = max(set(names), key=names.count) if names else node
        src = "the part name carried on the solids" if names else "the node id, for want of anything better"
    mark = re.sub(r"<[^>]*>", "", str(name))                 # drop "<As Machined>" and friends
    # "_Default" is the CAD system's configuration name, not part of anyone's piece mark. Stripped
    # by name rather than by pattern, because a real mark may well contain an underscore.
    mark = re.sub(r"_Default\b", "", mark, flags=re.I)
    mark = re.sub(r"[^A-Za-z0-9]+", "", mark).upper()[:16]
    return (mark or "PART"), src


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

    # ONE WELD PER JOINT. The detector already returns one connection per pair of solids that
    # meet, with the boolean intersection's several path fragments gathered inside it. This used
    # to flatten those fragments into separate welds, which turned 64 joints into 269 "welds" and
    # stacked eight labels on one T-joint. An operator inspects a JOINT - the cleat to the rail -
    # and a drawing specifies one, so the connection is the unit that carries a number.
    joints = []
    for c in results.get("connections") or []:
        if c.get("type") != "welded":
            continue
        segs, length = [], 0.0
        for path in (c.get("weld_paths") or []):
            pts = np.asarray(path, np.float64).reshape(-1, 3)
            if len(pts) < 2:
                continue
            length += float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            segs.append(pts)
        if not segs:
            continue
        # The detector's own figure where it has one: it measures the contact, while summing the
        # fragments measures the polylines that approximate it, and the two differ by ~3%.
        length = float(c.get("weld_length_mm") or length)
        sa, sb = c.get("solid_a") or {}, c.get("solid_b") or {}
        ids = ["%s:s%s" % (sa.get("node_id"), sa.get("solid_index")),
               "%s:s%s" % (sb.get("node_id"), sb.get("solid_index"))]
        joints.append({"ids": sorted(ids), "segs": segs, "length": length,
                       "method": c.get("weld_method"),
                       "centre": np.vstack(segs).mean(axis=0)})
    if not joints:
        print("detection found no welded joints above %.0f mm" % args.min_length, file=sys.stderr)
        return 1

    # NUMBERED BY POSITION, NOT BY ITERATION ORDER. This is what makes the identifier survive a
    # re-run: a joint's centroid does not move when --min-length changes, so the joints present in
    # both runs keep their numbers, and only the ones that appear or vanish disturb the sequence.
    # Sorting by length - which the projection stage used to do - reshuffles everything below any
    # weld that crosses the filter. X leads because it is the extrusion axis by convention here,
    # so the numbers run along the member the way a welder walks it. The solid-id pair breaks ties
    # deterministically, since two joints can share a rounded centroid but never an id pair.
    q = max(args.sort_tol, 1e-6)
    joints.sort(key=lambda j: (round(j["centre"][0] / q), round(j["centre"][1] / q),
                               round(j["centre"][2] / q), tuple(j["ids"])))

    mark, mark_src = _piece_mark(doc, args.node, args.piece_mark)
    project = str(doc.get("cnc_project_number") or "").strip()

    # A NUMBER, ONCE ISSUED, IS ISSUED. The first attempt at this numbered the joints that survived
    # --min-length, and its own stability test failed at 0 of 50: an ordinal is a RANK, so dropping
    # fourteen short welds shifted every number above them. Two changes make it hold.
    #
    # First, number the whole set and filter afterwards, so a display threshold cannot renumber
    # anything. The sequence then has gaps where welds were filtered out, which is correct - weld
    # maps gain gaps at every revision, and a gap is honest where a renumber is not.
    #
    # Second, carry previous assignments forward by identity. Geometric order still shifts if the
    # DETECTOR finds a joint it missed before, because a new weld inserts into the middle of the
    # order. Real fabrication does not renumber for that; it keeps the numbers already issued and
    # allocates new ones at the end. --carry-forward does exactly that.
    prior = {}
    if args.carry_forward and os.path.exists(args.carry_forward):
        with open(args.carry_forward, "r", encoding="utf-8") as fh:
            for w in (json.load(fh).get("welds") or []):
                prior[tuple(w.get("ConnectedTo") or [])] = w.get("Name")

    used = set(prior.values())
    nxt = 1
    for j in joints:
        key = tuple(j["ids"])
        if key in prior:
            j["name"] = prior[key]
            continue
        while ("%s-W%03d" % (mark, nxt)) in used:
            nxt += 1
        j["name"] = "%s-W%03d" % (mark, nxt)
        used.add(j["name"])
        nxt += 1
    carried = sum(1 for j in joints if tuple(j["ids"]) in prior)

    joints = [j for j in joints if j["length"] >= args.min_length]
    if not joints:
        print("every joint was shorter than --min-length %.0f" % args.min_length, file=sys.stderr)
        return 1

    # WHICH FACE OF THE ARTICLE each weld is on (bd bn5), worked out once here from the geometry so
    # every consumer - the AR view, a weld map, a drawing - agrees about it. It needs the article's
    # mesh, and the detector's coordinates are the assembly's, so --scale states the contract
    # between the two (the 1:5 tower needs 0.2). Without --mesh the sidecar simply carries no faces,
    # and a consumer computes them from the same function instead.
    frame = band = None
    scale = float(getattr(args, "scale", None) or 1.0)
    if getattr(args, "mesh", None):
        import weld_faces as WF
        frame = WF.article_frame(VIS.load_stl(args.mesh))
        band = (float(args.face_band) if getattr(args, "face_band", None)
                else WF.default_band(frame))

    welds = []
    for j in joints:
        name = j["name"]
        # Only what we actually know. A null Process or a null throat thickness would assert that
        # the weld was specified as nothing, where absence correctly says nobody has specified it
        # yet - see docs/weld-identification-and-ifc.md section 4.
        pset = {}
        if j["method"]:
            pset["Type1"] = j["method"]
        geometry = {
            "MeasuredLengthMm": round(j["length"], 1),
            "SegmentCount": len(j["segs"]),
            "CentroidMm": [round(float(v), 2) for v in j["centre"]],
        }
        if frame is not None:
            geometry["ArticleFaces"] = WF.faces_of(np.vstack(j["segs"]) * scale, frame, band)
        welds.append({
            # Seeded from WHAT the weld is, never from where it sits in a list. The first version
            # included the ordinal and so inherited every renumber - a GlobalId that changes is
            # not an identifier, it is a serial number for the run.
            "GlobalId": ifc_guid("|".join([project, mark] + j["ids"])),
            "Name": name,
            "Tag": name,
            "PredefinedType": "WELD",
            "Description": "solid %s to solid %s" % (j["ids"][0].rsplit(":s", 1)[-1],
                                                     j["ids"][1].rsplit(":s", 1)[-1]),
            "ConnectedTo": j["ids"],
            "Pset_FastenerWeld": pset,
            # OUR MEASUREMENTS, kept out of Pset_FastenerWeld deliberately. That Pset describes a
            # weld somebody specified; this describes one we found. Writing the measured length
            # into `l` would be the tempting shortcut and would be wrong - `l` is the length of a
            # single weld ELEMENT, so on any intermittent weld it would state something false.
            # ArticleFaces belongs here for the same reason: it is measured, not specified.
            "Pset_PSS_WeldGeometry": geometry,
            "Representation": {
                "type": "Polyline",
                "segments": [[[round(float(v), 2) for v in p] for p in s] for s in j["segs"]],
            },
        })

    total = sum(w["Pset_PSS_WeldGeometry"]["MeasuredLengthMm"] for w in welds)
    out = {
        # The schema version is not decoration: IFC4X3 renamed the single-letter ISO 2553 measures
        # (`l` -> WeldElementLength and the rest), so a reader has to know which naming applies.
        # IFC2X3 cannot carry this at all - it has IfcFastener but no IfcFastenerTypeEnum and no
        # Pset_FastenerWeld, so a WELD there needs ObjectType text and a custom Pset.
        "schema": "IFC4",
        "generator": "cad-automation weld_locate",
        "generated": datetime.datetime.now(datetime.timezone.utc)
                             .replace(microsecond=0).isoformat(),
        "project": project or None,
        "steel_grade": doc.get("cnc_steel_grade"),
        "piece_mark": {"value": mark, "derived_from": mark_src},
        "source": {"analysis": os.path.basename(args.analysis), "node": args.node,
                   "scope": args.scope},
        "placement": {
            "frame": "model",
            "units": "mm",
            "note": "model coordinates as stored by the detector - the same frame the pose maps "
                    "from, so no further transform is applied downstream. An IFC export must "
                    "place these in the project coordinate system explicitly rather than assume "
                    "the two agree.",
            "to_project": None,
        },
        # How the ArticleFaces codes were assigned, so a reader can check them rather than trust
        # them: the article's axes, extents and the band a weld had to fall within.
        "article_frame": (WF.frame_to_json(frame, band, os.path.basename(args.mesh), scale)
                          if frame is not None else None),
        "summary": {"weld_count": len(welds), "total_length_mm": round(total, 1)},
        "welds": welds,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    typed = sum(1 for w in welds if w["Pset_FastenerWeld"].get("Type1"))
    print("%d welded joints, %.0f mm total -> %s" % (len(welds), total, args.out))
    print("piece mark %s (%s), project %s" % (mark, mark_src, project or "unknown"))
    if args.carry_forward:
        print("carried %d weld numbers forward from %s, issued %d new"
              % (carried, os.path.basename(args.carry_forward), len(joints) - carried))
    print("joint type known for %d of %d (%.0f%%) - the rest carry no Type1, which is absence "
          "rather than a null" % (typed, len(welds), 100.0 * typed / len(welds)))
    if frame is not None:
        per_face, on_none = {}, 0
        for w in welds:
            codes = w["Pset_PSS_WeldGeometry"]["ArticleFaces"]
            on_none += not codes
            for code in codes:
                per_face[code] = per_face.get(code, 0) + 1
        print("article faces (band %.1f mm): %s%s"
              % (band, ", ".join("%s %d" % (k, per_face[k]) for k in WF.FACES if k in per_face),
                 "  - %d welds on NO face, so no camera will show them" % on_none if on_none else ""))
    print("longest: %s" % ", ".join(
        "%s %.0fmm" % (w["Name"], w["Pset_PSS_WeldGeometry"]["MeasuredLengthMm"])
        for w in sorted(welds, key=lambda w: -w["Pset_PSS_WeldGeometry"]["MeasuredLengthMm"])[:4]))
    return 0


def load_views(captures, profile, cam_profile=(),
               skip=("overlay", "linecheck", "endcheck", "weld", "ar_view")):
    """A board-in-shot view for every photograph in a capture set, each with its own camera's
    intrinsics. Shared by `project` and tools/ar_view.py, so the two can never read one capture set
    two different ways."""
    base = MVF.load_profile(profile)
    board = charuco.build_board_from_config(base["board"])
    det = charuco.make_detector(board)
    overrides = [(s.split("=", 1)[0], MVF.load_profile(s.split("=", 1)[1]))
                 for s in cam_profile]
    views = []
    for path in sorted(glob.glob(os.path.join(captures, "*"))):
        b = os.path.basename(path)
        if any(k in b for k in skip):
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        prof = next((p for sub, p in overrides if sub in b), base)
        v = MVF.build_view(img, prof, board, det, label=b)
        v.update({"K": prof["K"], "dist": prof["dist"], "image": img, "tag": b})
        views.append(v)
    return views


def place_welds(welds, rvec, tvec, scale):
    """Every joint's segments put into the board frame through the pose.

    Returns ``[(name, [Nx3 segment, ...], row)]``. The row carries what a table needs - centre,
    length, type and run count - in the fitted model's units after ``scale``."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t = np.asarray(tvec, np.float64).ravel()
    placed = []
    for w in welds:
        segs = [np.asarray(s, np.float64).reshape(-1, 3) * scale
                for s in w["Representation"]["segments"]]
        worlds = [(R @ p.T).T + t for p in segs]
        placed.append((w["Name"], worlds, {
            "name": w["Name"],
            "centre": np.vstack(worlds).mean(axis=0),
            "length": w["Pset_PSS_WeldGeometry"]["MeasuredLengthMm"] * scale,
            "type": w["Pset_FastenerWeld"].get("Type1"),
            "runs": len(segs)}))
    return placed


def cmd_project(args) -> int:
    with open(args.welds, "r", encoding="utf-8") as fh:
        wd = json.load(fh)
    if not wd.get("schema"):
        print("%s is an old-format sidecar, written before welds carried stable identifiers."
              % args.welds, file=sys.stderr)
        print("Re-run `weld_locate.py extract` to produce one - the numbers in the old file are "
              "not the numbers this stage would have used, which was the bug.", file=sys.stderr)
        return 2
    welds = wd["welds"]
    src = args.fit if not os.path.isdir(args.fit) else os.path.join(args.fit, "fit.json")
    with open(src, "r", encoding="utf-8") as fh:
        fit = json.load(fh)
    rvec = np.asarray(fit["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(fit["tvec"], np.float64).reshape(3, 1)
    mesh_path = args.mesh or os.path.join("outputs/ar_models",
                                          os.path.basename(fit.get("mesh") or ""))
    mesh = VIS.load_stl(mesh_path)

    views = load_views(args.captures, args.profile, args.cam_profile)
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

    # THE NUMBERS ARE READ, NEVER MINTED. Both stages used to number independently and the two
    # disagreed, so a weld's label depended on which file you were looking at. `extract` owns the
    # identifier now - it is the stage that knows the piece mark - and everything downstream
    # carries it through unchanged. Filtering happens on whole joints, so a short weld is dropped
    # or kept entire rather than losing part of itself.
    if args.min_weld > 0:
        welds = [w for w in welds
                 if w["Pset_PSS_WeldGeometry"]["MeasuredLengthMm"] * args.scale >= args.min_weld]
    if not welds:
        print("every joint was filtered out by --min-weld %.0f" % args.min_weld, file=sys.stderr)
        return 1

    placed = place_welds(welds, rvec, tvec, args.scale)
    rows = [r for _name, _worlds, r in placed]

    # WHICH WELDS EACH CAMERA SHOWS: the ones on the article faces it looks at (bd bn5), using the
    # face codes the sidecar carries or, where it has none, the same function computing them here.
    import weld_faces as WF
    frame = WF.article_frame(mesh)
    band = getattr(args, "face_band", None) or WF.default_band(frame)
    faces_by_weld, computed = WF.faces_for_welds(welds, frame, args.scale, band)
    min_cos = getattr(args, "min_face_cos", None) or 0.2
    occlusion_mm = getattr(args, "occlusion_mm", None) or 8.0
    print("faces: %s" % ("from the sidecar" if not computed
                         else "computed here for %d welds, band %.1f mm" % (computed, band)))

    print("piece %s, project %s, %s"
          % (wd.get("piece_mark", {}).get("value", "?"), wd.get("project") or "unknown",
             wd.get("schema")))
    print("")
    print("%-14s %26s %9s %10s %5s" % ("weld", "centre in the board frame (mm)", "length",
                                       "type", "runs"))
    for r in sorted(rows, key=lambda r: -r["length"])[:args.list_max]:
        c = r["centre"]
        print("%-14s %8.1f %8.1f %8.1f %7.0f mm %10s %5d"
              % (r["name"], c[0], c[1], c[2], r["length"], (r["type"] or "-")[:10], r["runs"]))
    if len(rows) > args.list_max:
        print("... and %d more" % (len(rows) - args.list_max))
    print("")
    print("%d joints, %.0f mm of weld in total" % (len(rows), sum(r["length"] for r in rows)))

    if args.out:
        panels = []
        per_view = {}
        for v in views:
            out = v["image"].copy()
            shown, status, presented, _cos = WF.visible_weld_points(
                placed, faces_by_weld, mesh, rvec, tvec, v, frame, min_cos, occlusion_mm)
            tally = {}
            for s_ in status.values():
                tally[s_] = tally.get(s_, 0) + 1
            print("%-28s faces presented %s | welds %s"
                  % (v["tag"][:28], ", ".join(WF.face_name(frame, rvec, c_)
                                              for c_ in WF.FACES if c_ in presented) or "none",
                     ", ".join("%s %d" % kv for kv in sorted(tally.items()))))
            for num, lines in shown.items():
                for line in lines:
                    pts = [(int(round(x)), int(round(y))) for x, y in line]
                    for a, b in zip(pts[:-1], pts[1:]):
                        cv2.line(out, a, b, (40, 230, 255), 3, lineType=cv2.LINE_AA)
                longest = max(lines, key=len)
                mx, my = longest[len(longest) // 2]
                # Just the ordinal on the image - every weld in shot shares the piece mark, so
                # repeating it 60 times costs legibility and says nothing. The full identifier is
                # in the header and in the sidecar.
                short = num.rsplit("-", 1)[-1]
                cv2.putText(out, short, (int(mx) + 6, int(my) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (30, 30, 30), 3, lineType=cv2.LINE_AA)
                cv2.putText(out, short, (int(mx) + 6, int(my) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (40, 230, 255), 1, lineType=cv2.LINE_AA)
            cv2.rectangle(out, (0, 0), (out.shape[1], 46), (26, 26, 26), -1)
            cv2.putText(out, "%s   %d joints on this face   pose %.0f%% silhouette"
                        % (v["tag"][:26], len(shown), sil),
                        (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2)
            per_view[v["tag"]] = list(shown)
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
                   help="ignore joints shorter than this; the short ones are usually the boolean "
                        "intersection finding a corner rather than a weld")
    e.add_argument("--piece-mark", default=None,
                   help="the namespace that makes weld numbers unique across a JOB. Derived from "
                        "the member or part name when omitted, and recorded in the output either "
                        "way so it is never ambiguous which was used.")
    e.add_argument("--sort-tol", type=float, default=1.0,
                   help="millimetres to round joint centroids to before ordering them. Coarse "
                        "enough that solver noise cannot reorder two welds, fine enough that two "
                        "genuinely different joints do not collide - and where they do, the pair "
                        "of solid ids breaks the tie deterministically.")
    e.add_argument("--carry-forward", default=None, metavar="PREVIOUS.json",
                   help="a previous sidecar for this part. Weld numbers already issued there are "
                        "kept for the same joints and new joints are allocated fresh numbers at "
                        "the end, which is how a weld map survives a revision. Without it a joint "
                        "the detector newly finds inserts into the geometric order and shifts "
                        "every number above it.")
    e.add_argument("--mesh", default=None,
                   help="the article's mesh. Given, each weld is assigned the article faces it is "
                        "on (ArticleFaces) and the frame they were measured in is recorded - see "
                        "tools/weld_faces.py. Omitted, the sidecar carries no faces.")
    e.add_argument("--scale", type=float, default=1.0,
                   help="detector coordinates into the mesh's units, for the face assignment - "
                        "the 1:5 tower needs 0.2")
    e.add_argument("--face-band", type=float, default=None,
                   help="mm in from a face plane a weld may sit and still be on that face; "
                        "default 20%% of the smaller cross-section extent")
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
    p.add_argument("--face-band", type=float, default=None,
                   help="mm in from a face plane a weld may sit and still be on that face, used "
                        "only where the sidecar carries no ArticleFaces")
    p.add_argument("--min-face-cos", type=float, default=0.2,
                   help="how squarely a camera must face an article face to be shown its welds; "
                        "grazing faces are excluded")
    p.add_argument("--occlusion-mm", type=float, default=8.0,
                   help="how far behind the surface in front of it a weld point may sit and still "
                        "count as visible")
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
