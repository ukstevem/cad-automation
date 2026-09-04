#!/usr/bin/env python3
"""
Generate a print-exact ChArUco board PDF for the test cell.

A PNG is the wrong deliverable for a calibration target: printers scale, and a board whose
squares are not the size you told the solver they are produces a confidently wrong pose. This
lays the board out at exact millimetre dimensions in a PDF, and prints its own configuration and
a 100 mm verification rule alongside it.

    docker compose run --rm --no-deps api python tools/make_board_pdf.py \\
        --squares-x 9 --squares-y 6 --square-mm 40 --marker-mm 30 \\
        --dictionary DICT_5X5_100 --paper A3 --out outputs/calibration/board.pdf

The printed configuration is not decoration. A board/dictionary mismatch fails **silently** —
the detector simply returns zero corners with no error — and that has already cost a session on
this project. Having it written on the board itself means the physical object always says what
it is.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.services import charuco  # noqa: E402

PAGE = """<!doctype html><meta charset=utf-8>
<style>
  @page {{ size: {paper} {orient}; margin: {margin_mm}mm; }}
  body {{ margin:0; font-family: DejaVu Sans, sans-serif; color:#000; }}
  .board {{ width: {board_w}mm; height: {board_h}mm; display:block; }}
  .meta {{ margin-top: 6mm; font-size: 9pt; line-height: 1.5; }}
  .meta b {{ font-size: 10pt; }}
  .rule {{ margin-top: 4mm; }}
  .rule .bar {{ width: 100mm; height: 4mm; border: 0.4mm solid #000;
                border-left-width: 0.8mm; border-right-width: 0.8mm; }}
  .warn {{ margin-top: 3mm; font-size: 8.5pt; }}
</style>
<img class="board" src="data:image/png;base64,{b64}">
<div class="meta">
  <b>ChArUco {sx}&times;{sy} &nbsp; square {sq}mm &nbsp; marker {mk}mm &nbsp; {dictname}</b><br>
  Board {board_w} &times; {board_h} mm &mdash; generated for the AR test cell.
  These five values must match the calibration profile exactly.
  A wrong dictionary detects <i>nothing</i>, with no error message.
</div>
<div class="rule">
  <div class="bar"></div>
  <div style="font-size:8.5pt">&#8592; this bar is exactly 100 mm. Measure it after printing.</div>
</div>
<div class="warn">
  Print at <b>100% / actual size</b> &mdash; no "fit to page", no scaling. Then measure the bar
  <i>and</i> one square. If they are not {sq}.0 mm and 100.0 mm, either reprint or tell the
  calibration the size you actually measured &mdash; the square size sets the metric scale of
  every pose derived from this board.<br>
  Mount it <b>dead flat</b> on ply or foamboard. A bowed board is a bent world frame.
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares-x", type=int, default=9)
    ap.add_argument("--squares-y", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=40.0)
    ap.add_argument("--marker-mm", type=float, default=30.0)
    ap.add_argument("--dictionary", default="DICT_5X5_100", choices=charuco.DICT_NAMES)
    ap.add_argument("--paper", default="A3", choices=["A3", "A4"])
    ap.add_argument("--orientation", default="landscape", choices=["landscape", "portrait"])
    ap.add_argument("--margin-mm", type=float, default=10.0)
    ap.add_argument("--dpi", type=float, default=600.0, help="raster density of the board image")
    ap.add_argument("--out", default="outputs/calibration/charuco_board.pdf")
    args = ap.parse_args()

    board = charuco.build_board(args.squares_x, args.squares_y, args.square_mm,
                                args.marker_mm, args.dictionary)

    board_w = args.squares_x * args.square_mm
    board_h = args.squares_y * args.square_mm
    limits = {"A3": (420.0, 297.0), "A4": (297.0, 210.0)}[args.paper]
    if args.orientation == "portrait":
        limits = (limits[1], limits[0])
    usable = (limits[0] - 2 * args.margin_mm, limits[1] - 2 * args.margin_mm)
    if board_w > usable[0] or board_h > usable[1]:
        sys.exit(f"Board {board_w:.0f}x{board_h:.0f}mm does not fit {args.paper} "
                 f"{args.orientation} usable area {usable[0]:.0f}x{usable[1]:.0f}mm. "
                 f"Reduce --square-mm or the square counts.")

    px_per_mm = args.dpi / 25.4
    img = board.generateImage(
        (int(round(board_w * px_per_mm)), int(round(board_h * px_per_mm))),
        marginSize=0, borderBits=1,
    )
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        sys.exit("failed to encode board image")

    html = PAGE.format(
        paper=args.paper, orient=args.orientation, margin_mm=args.margin_mm,
        board_w=f"{board_w:g}", board_h=f"{board_h:g}",
        b64=base64.b64encode(buf.tobytes()).decode(),
        sx=args.squares_x, sy=args.squares_y, sq=f"{args.square_mm:g}",
        mk=f"{args.marker_mm:g}", dictname=args.dictionary,
    )

    from weasyprint import HTML
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    HTML(string=html).write_pdf(args.out)

    n_markers = (args.squares_x * args.squares_y) // 2
    n_corners = (args.squares_x - 1) * (args.squares_y - 1)
    print(f"wrote {args.out}")
    print(f"  board      {board_w:g} x {board_h:g} mm on {args.paper} {args.orientation}")
    print(f"  squares    {args.squares_x} x {args.squares_y} @ {args.square_mm:g}mm "
          f"(marker {args.marker_mm:g}mm)")
    print(f"  dictionary {args.dictionary}  ({n_markers} markers used)")
    print(f"  interior corners: {n_corners}")
    print(f"  PRINT AT 100% - then measure the 100mm bar and one square before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
