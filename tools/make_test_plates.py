#!/usr/bin/env python3
"""
A family of laser-cut test plates, designed against the failure modes we have actually hit.

Not an arbitrary set of shapes. Each plate answers a question the bench has already raised and
could not settle, and the family is cut from one drawing so the only differences are deliberate.

WHY LASER CUTTING IS THE RIGHT TOOL HERE. The threshold calibration has been blocked on having
nothing genuinely wrong to measure - shimming a good part by hand was the fallback. A cutter will
put a hole 2 mm from where the model says, repeatably, at a known station. That is a real defect
with a known answer, which is what a pass threshold has to be set against.

SIZE MATTERS MORE THAN IT LOOKS. The first steel plate was 125 mm and only ~220 px across in
frame, so five pixels of click error was 2% of the part. At 300 mm it is 525 px and the same click
is under 1%. Everything - click tolerance, hole discrimination, edge measurement - improves with
the part filling more of the frame, so these are drawn as large as the working area allows.

MATERIAL. Mill-scaled steel gave 0% unmeasurable edges against white paper, better than the
3D-printed article managed. Do not paint them, and do not write on them in marker: the pen on the
first plate produced strong edges no model contains, which is a false-edge source we have not
characterised.

    python tools/make_test_plates.py --out outputs/test_plates
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── the base outline ────────────────────────────────────────────────────────
#
# Deliberately handed: a single large clipped corner, which is what made the first plate's flip
# detectable at 37% against 99%. A rectangle would be its own mirror and no fit could ever tell
# which way up it lay.
W, H = 300.0, 220.0
CLIP = 90.0
BASE = [(0, H), (0, CLIP), (CLIP, 0), (W, 0), (W, H)]

# Holes chosen so NO pair is symmetric about either centreline. On the first steel plate two of
# three holes sat near-symmetrically and moved 1 mm under a flip while the third moved 29 mm - so
# the flip hung on a single feature. Here every hole moves substantially.
HOLES = [(55.0, 165.0, 18.0), (150.0, 60.0, 18.0), (245.0, 150.0, 18.0), (200.0, 195.0, 12.0)]

# The hole the calibration family displaces. Chosen away from the outline so its own edges are the
# only thing that moves - a hole near a corner would be partly rescued by the outline beside it.
TARGET = 1


def dxf(path, outline, holes, label=None):
    """
    Cut geometry on layer CUT, identification text on layer ETCH.

    The layers are separate for two reasons. The cutter needs to be told which to etch and which
    to cut through. And the etched text is edges the model does not contain - the same false-edge
    source as the marker pen on the first steel plate, which we still have not characterised - so
    the model builder must be able to ignore it. `make_plate.py --layer CUT` does exactly that.

    The text goes on ONE face, which makes it a ground-truth marker for handedness: if you can
    read it, you are looking at the etched side. That is worth having when the whole point of
    these plates is testing whether the software can tell which way up they are - the answer key
    should not depend on the software.

    Placed in a corner the outline already occupies, away from every hole, so it competes with as
    few real features as possible.
    """
    import ezdxf
    doc = ezdxf.new("R2010")
    doc.layers.add("CUT", color=1)
    doc.layers.add("ETCH", color=3)
    msp = doc.modelspace()
    msp.add_lwpolyline([(x, y) for x, y in outline], close=True, dxfattribs={"layer": "CUT"})
    for (x, y, d) in holes:
        msp.add_circle((x, y), d / 2.0, dxfattribs={"layer": "CUT"})
    if label:
        t = msp.add_text(label, height=14.0,
                         dxfattribs={"layer": "ETCH", "style": "Standard"})
        t.set_placement((22.0, H - 32.0))
    doc.saveas(path)


def preview(outline, holes, label, note, size=560):
    import cv2
    P = np.asarray(outline, np.float64)
    lo, hi = P.min(axis=0), P.max(axis=0)
    s = (size - 60) / max(hi[0] - lo[0], 1e-9)
    h = int((hi[1] - lo[1]) * s + 90)
    img = np.full((h, size, 3), 255, np.uint8)

    def to_px(p):
        q = (np.asarray(p, np.float64) - lo) * s + 30
        return int(q[0]), int(h - q[1] - 30)
    cv2.polylines(img, [np.array([to_px(p) for p in outline], np.int32)], True, (40, 40, 40), 2,
                  lineType=cv2.LINE_AA)
    for i, (x, y, d) in enumerate(holes):
        col = (40, 40, 200) if i == TARGET else (40, 40, 40)
        cv2.circle(img, to_px((x, y)), max(2, int(d / 2 * s)), col, 2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (size, 34), (30, 30, 30), -1)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)
    cv2.putText(img, note, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1)
    # where the etched identification lands
    ex, ey = to_px((22.0, H - 32.0))
    cv2.putText(img, label, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 160, 30), 1,
                lineType=cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/test_plates")
    ap.add_argument("--offsets", default="0,2,5,10,20",
                    help="how far to displace one hole, in mm, one plate each")
    args = ap.parse_args()
    import cv2
    os.makedirs(args.out, exist_ok=True)

    plates = []

    # 1. The calibration family. Identical but for one hole, displaced by a known amount. This is
    #    what sets the pass threshold: the smallest displacement the check reliably calls out.
    for off in [float(x) for x in args.offsets.split(",")]:
        holes = [list(h) for h in HOLES]
        holes[TARGET][0] += off
        name = "cal_%02dmm" % int(round(off))
        note = ("nominal - the reference" if off == 0 else
                "hole %d displaced %.0f mm in x" % (TARGET + 1, off))
        plates.append((name, BASE, [tuple(h) for h in holes], note))

    # 2. Plain: the same outline with no holes at all. The aperture problem in its purest form -
    #    every edge slides along itself, so this measures what the holes are actually worth.
    plates.append(("plain", BASE, [], "no holes - measures what the holes buy"))

    # 3. Near-symmetric: a rectangle with a symmetric hole pattern, so a flip moves almost nothing.
    #    This is the hard case, and the point is to find where discrimination fails rather than to
    #    pass it. If the check cannot tell this one's handedness, that is a limit worth knowing.
    rect = [(0, 0), (W, 0), (W, H), (0, H)]
    sym = [(60.0, 60.0, 18.0), (W - 60.0, 60.0, 18.0),
           (60.0, H - 60.0, 18.0), (W - 60.0, H - 60.0, 18.0)]
    plates.append(("symmetric", rect, sym,
                   "deliberately ambiguous - a flip barely moves anything"))

    # 4. Slot: a long feature whose own length is unconstrained, next to a hole that is not. Tests
    #    whether the check localises ALONG a feature or only across it.
    slot = [tuple(h) for h in HOLES[:2]]
    plates.append(("slotted", BASE, slot + [(230.0, 110.0, 30.0)],
                   "one large hole - tests localisation along a feature"))

    tiles = []
    for name, outline, holes, note in plates:
        p = os.path.join(args.out, "%s.dxf" % name)
        dxf(p, outline, holes, label=name)
        tiles.append(preview(outline, holes, name, note))
        print("%-12s %3d holes  %s" % (name, len(holes), p))

    hgt = max(t.shape[0] for t in tiles)
    row = []
    for t in tiles:
        pad = np.full((hgt, t.shape[1], 3), 255, np.uint8)
        pad[:t.shape[0]] = t
        row.append(pad)
    per = 4
    rows = [np.hstack(row[i:i + per]) for i in range(0, len(row), per)]
    wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], wmax - r.shape[1], 3), 255, np.uint8)])
            if r.shape[1] < wmax else r for r in rows]
    cv2.imwrite(os.path.join(args.out, "family.png"), np.vstack(rows))

    print("")
    print("%d plates, %.0f x %.0f mm. Cut them all from the same sheet and the same programme so"
          % (len(plates), W, H))
    print("the only differences are the intended ones.")
    print("")
    print("8 or 10 mm, mill scale, no paint and no marker pen - the pen on the first plate made")
    print("strong edges no model contains, and we have not characterised that.")
    print("")
    print("Two layers: CUT to cut through, ETCH for the identification text. Build the models with")
    print("  make_plate.py --dxf <file> --layer CUT")
    print("so the etched text is not modelled - it is deliberately NOT part of the geometry under")
    print("test. It is on one face only, so being able to read it tells you which way up the plate")
    print("is lying: an answer key that does not depend on the software being tested.")
    print("")
    print("The cal_ family is the one that matters: it is the only way to answer 'how small an")
    print("error does this catch', which is what a go/no-go threshold has to be set against.")
    print("family.png shows all of them; the displaced hole is marked in red.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
