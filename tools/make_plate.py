#!/usr/bin/env python3
"""
Build a flat-plate model - STL plus the AR model JSON - from a 2D profile and a thickness.

A cut plate is fully described by its outline, its holes and how thick it is, so it does not need
a CAD package to model. This takes the profile from a DXF (the usual output of a profile drawing)
or from points typed off a drawing, extrudes it, and writes both the mesh the pose tools use and
the edge JSON the multiview fit expects.

The plate is built with its underside on z=0 and centred on the origin, because that is how it
lies on the table: the pose search assumes the part rests on the board plane, and a model whose
origin floats somewhere else makes the seating term meaningless.

    # from a DXF profile (outline plus circles for the holes)
    python tools/make_plate.py --dxf plate.dxf --thickness 8 --name plate01

    # or straight off a drawing, corners clockwise or anticlockwise, holes as x,y,diameter
    python tools/make_plate.py --outline "0,0 124,0 124,120 0,120" --thickness 8 \\
        --holes "20,20,14 104,20,14 62,100,14" --name plate01
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_dxf(path, layer=None):
    """Outline polyline and hole circles from a DXF. Longest closed loop wins as the outline."""
    try:
        import ezdxf
    except ImportError:
        sys.exit("ezdxf not available in this image; pass --outline/--holes instead")
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    loops, holes = [], []
    for e in msp:
        if layer and e.dxf.layer != layer:
            continue
        t = e.dxftype()
        if t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            if e.closed and len(pts) >= 3:
                loops.append(np.asarray(pts, float))
        elif t == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if len(pts) >= 3:
                loops.append(np.asarray(pts, float))
        elif t == "CIRCLE":
            holes.append((e.dxf.center.x, e.dxf.center.y, 2.0 * e.dxf.radius))
    if not loops:
        sys.exit("no closed polyline found in %s - is the profile on another layer?" % path)
    outline = max(loops, key=lambda p: abs(_area(p)))
    # A closed loop that is not the outline and sits inside it is a hole cut as a polyline
    for lp in loops:
        if lp is outline:
            continue
        c = lp.mean(axis=0)
        if _inside(outline, c):
            r = np.linalg.norm(lp - c, axis=1).mean()
            holes.append((c[0], c[1], 2.0 * r))
    return outline, holes


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(poly, pt):
    import cv2
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(pt[0]), float(pt[1])), False) > 0


def triangulate(outline, holes, seg=48):
    """
    Triangulate the plate face, holes cut out, by constrained Delaunay over a sampled boundary.

    Holes matter more than their area suggests: on a plain plate they are the only features that
    localise the pose ALONG the plate, everything else being outline that slides.
    """
    import cv2
    pts = [outline]
    for (hx, hy, d) in holes:
        a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        pts.append(np.stack([hx + d / 2 * np.cos(a), hy + d / 2 * np.sin(a)], axis=1))
    allp = np.vstack(pts)
    lo = np.floor(allp.min(axis=0)) - 10
    hi = np.ceil(allp.max(axis=0)) + 10
    sub = cv2.Subdiv2D((int(lo[0]), int(lo[1]), int(hi[0] - lo[0]), int(hi[1] - lo[1])))
    for p in allp:
        sub.insert((float(p[0]), float(p[1])))
    tris = []
    for t in sub.getTriangleList():
        tri = np.asarray(t, float).reshape(3, 2)
        c = tri.mean(axis=0)
        if not _inside(outline, c):
            continue
        if any(np.hypot(c[0] - hx, c[1] - hy) < d / 2 for (hx, hy, d) in holes):
            continue
        tris.append(tri)
    return np.asarray(tris)


def extrude(face_tris, outline, holes, thickness, seg=48):
    """Face triangles into a closed solid: bottom, top, and a wall round every boundary loop."""
    out = []
    for t in face_tris:
        bot = np.column_stack([t, np.zeros(3)])
        top = np.column_stack([t, np.full(3, thickness)])
        out.append(bot[::-1])                     # bottom faces down
        out.append(top)
    loops = [outline]
    for (hx, hy, d) in holes:
        a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        loops.append(np.stack([hx + d / 2 * np.cos(a), hy + d / 2 * np.sin(a)], axis=1))
    for lp in loops:
        for i in range(len(lp)):
            p, q = lp[i], lp[(i + 1) % len(lp)]
            a0 = np.array([p[0], p[1], 0.0]); a1 = np.array([q[0], q[1], 0.0])
            b0 = np.array([p[0], p[1], thickness]); b1 = np.array([q[0], q[1], thickness])
            out.append(np.stack([a0, a1, b1]))
            out.append(np.stack([a0, b1, b0]))
    return np.asarray(out)


def write_stl(tris, path):
    import struct
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            nn = np.linalg.norm(n)
            n = n / nn if nn > 1e-12 else np.zeros(3)
            fh.write(struct.pack("<3f", *n))
            for v in t:
                fh.write(struct.pack("<3f", *v))
            fh.write(b"\0\0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dxf", default=None)
    ap.add_argument("--layer", default=None, help="restrict the DXF read to one layer")
    ap.add_argument("--outline", default=None, help='"x,y x,y ..." in mm')
    ap.add_argument("--holes", default=None, help='"x,y,dia x,y,dia ..." in mm')
    ap.add_argument("--thickness", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", default="outputs/ar_models")
    args = ap.parse_args()

    if args.dxf:
        outline, holes = read_dxf(args.dxf, args.layer)
    elif args.outline:
        outline = np.asarray([[float(v) for v in p.split(",")]
                              for p in args.outline.split()], float)
        holes = [tuple(float(v) for v in h.split(",")) for h in (args.holes or "").split() if h]
    else:
        sys.exit("give either --dxf or --outline")

    # Centre on the origin - and move the holes with it, or they end up describing a different
    # plate from the one the outline describes.
    centre = outline.mean(axis=0)
    outline = outline - centre
    holes = [(h[0] - centre[0], h[1] - centre[1], h[2]) for h in holes]
    face = triangulate(outline, holes)
    if not len(face):
        sys.exit("triangulation produced nothing - check the outline winding and units")
    tris = extrude(face, outline, holes, args.thickness)

    os.makedirs(args.outdir, exist_ok=True)
    stl = os.path.join(args.outdir, "%s.stl" % args.name)
    write_stl(tris, stl)
    ext = outline.max(axis=0) - outline.min(axis=0)
    print("%s: %.1f x %.1f x %.1f mm, %d holes, %d triangles"
          % (args.name, ext[0], ext[1], args.thickness, len(holes), len(tris)))
    print("wrote %s" % stl)

    # The AR model JSON the multiview fit reads: edge polylines in model coordinates.
    edges = []
    for lp, z in ((outline, 0.0), (outline, args.thickness)):
        edges.append([[float(p[0]), float(p[1]), z] for p in lp] +
                     [[float(lp[0][0]), float(lp[0][1]), z]])
    for (hx, hy, d) in holes:
        a = np.linspace(0, 2 * np.pi, 33)
        for z in (0.0, args.thickness):
            edges.append([[float(hx + d / 2 * np.cos(t)), float(hy + d / 2 * np.sin(t)), z]
                          for t in a])
    model = {"name": args.name, "scale": 1.0, "units": "mm", "edges": edges,
             "mesh": os.path.basename(stl),
             "bbox": [[float(x) for x in outline.min(axis=0)] + [0.0],
                      [float(x) for x in outline.max(axis=0)] + [float(args.thickness)]]}
    js = os.path.join(args.outdir, "%s.json" % args.name)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    print("wrote %s (%d edge polylines)" % (js, len(edges)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
