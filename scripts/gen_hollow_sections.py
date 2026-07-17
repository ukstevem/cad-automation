"""Generate the EN 10210-2 hot-finished hollow-section entries for the library.

Writes CHS / SHS / RHS into ``app/pipeline/data/Shape_classifier_info.json``,
replacing any existing hollow entries and leaving the open profiles (UB/UC/UBP/
RSJ/RSC/PFC/EA/UEA) untouched.  Re-runnable.

Areas are computed, not transcribed, so the whole table is derived from two
formulae and a size list.  ``--check`` validates them against published EN 10210
values before writing (run it if you change anything).

EN 10210-2 hot-finished corner radii: outer 1.5t, inner 1.0t.

    CHS      A = pi * t * (D - t)
    SHS/RHS  A = 2t(B + H) - 4t^2 - (4 - pi)(r_o^2 - r_i^2)
                = 2t(B + H) - 4t^2 - 1.25 * (4 - pi) * t^2

Usage:
    python scripts/gen_hollow_sections.py --check     # validate only
    python scripts/gen_hollow_sections.py --write     # validate then write
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

LIB = Path(__file__).parent.parent / "app" / "pipeline" / "data" / "Shape_classifier_info.json"

# kg per mm^3 for steel — mass (kg/m) = area(mm^2) * 7.85e-3
_DENSITY_KG_PER_M = 7.85e-3

# --- EN 10210-2 sizes -------------------------------------------------------
# CHS: outside diameter -> wall thicknesses
CHS_SIZES = {
    21.3: [2.3, 2.6, 3.2],
    26.9: [2.3, 2.6, 3.2],
    33.7: [2.6, 3.0, 3.2, 4.0],
    42.4: [2.6, 3.0, 3.2, 4.0],
    48.3: [2.6, 3.0, 3.2, 4.0, 5.0],
    60.3: [2.9, 3.2, 4.0, 5.0],
    76.1: [2.9, 3.2, 4.0, 5.0, 6.3],
    88.9: [3.2, 4.0, 5.0, 6.3],
    114.3: [3.2, 3.6, 4.0, 5.0, 6.3],
    139.7: [4.0, 5.0, 6.3, 8.0, 10.0],
    168.3: [4.0, 5.0, 6.3, 8.0, 10.0],
    193.7: [5.0, 6.3, 8.0, 10.0, 12.5],
    219.1: [5.0, 6.3, 8.0, 10.0, 12.5],
    244.5: [5.0, 6.3, 8.0, 10.0, 12.5],
    273.0: [5.0, 6.3, 8.0, 10.0, 12.5],
    323.9: [6.3, 8.0, 10.0, 12.5],
    355.6: [8.0, 10.0, 12.5, 16.0],
    406.4: [8.0, 10.0, 12.5, 16.0],
    457.0: [8.0, 10.0, 12.5, 16.0],
    508.0: [8.0, 10.0, 12.5, 16.0],
}

# SHS: side -> wall thicknesses
SHS_SIZES = {
    20: [2.0],
    25: [2.5, 3.0],
    30: [2.5, 3.0, 3.2],
    40: [2.5, 3.0, 3.2, 4.0, 5.0],
    50: [2.5, 3.0, 3.2, 4.0, 5.0, 6.3],
    60: [3.0, 3.2, 4.0, 5.0, 6.3, 8.0],
    70: [3.0, 3.2, 4.0, 5.0, 6.3, 8.0],
    80: [3.2, 4.0, 5.0, 6.3, 8.0, 10.0],
    90: [3.6, 4.0, 5.0, 6.3, 8.0, 10.0],
    100: [4.0, 5.0, 6.3, 8.0, 10.0, 12.5],
    120: [5.0, 6.3, 8.0, 10.0, 12.5],
    140: [5.0, 6.3, 8.0, 10.0, 12.5],
    150: [5.0, 6.3, 8.0, 10.0, 12.5, 16.0],
    160: [5.0, 6.3, 8.0, 10.0, 12.5, 16.0],
    180: [6.3, 8.0, 10.0, 12.5, 16.0],
    200: [6.3, 8.0, 10.0, 12.5, 16.0],
    250: [6.3, 8.0, 10.0, 12.5, 16.0],
    300: [6.3, 8.0, 10.0, 12.5, 16.0],
    350: [8.0, 10.0, 12.5, 16.0],
    400: [8.0, 10.0, 12.5, 16.0],
}

# RHS: (height, width) -> wall thicknesses
RHS_SIZES = {
    (50, 30): [2.6, 3.2, 4.0, 5.0],
    (60, 40): [2.6, 3.2, 4.0, 5.0],
    (80, 40): [3.2, 4.0, 5.0, 6.3],
    (90, 50): [3.2, 4.0, 5.0, 6.3],
    (100, 50): [3.2, 4.0, 5.0, 6.3],
    (100, 60): [3.6, 4.0, 5.0, 6.3, 8.0],
    (120, 60): [3.6, 4.0, 5.0, 6.3, 8.0],
    (120, 80): [4.0, 5.0, 6.3, 8.0, 10.0],
    (150, 100): [4.0, 5.0, 6.3, 8.0, 10.0, 12.5],
    (160, 80): [4.0, 5.0, 6.3, 8.0, 10.0, 12.5],
    (200, 100): [5.0, 6.3, 8.0, 10.0, 12.5],
    (200, 120): [6.3, 8.0, 10.0, 12.5],
    (250, 150): [5.0, 6.3, 8.0, 10.0, 12.5, 16.0],
    (300, 200): [6.3, 8.0, 10.0, 12.5, 16.0],
    (400, 200): [8.0, 10.0, 12.5, 16.0],
    (450, 250): [8.0, 10.0, 12.5, 16.0],
    (500, 300): [8.0, 10.0, 12.5, 16.0],
}


def chs_area(D: float, t: float) -> float:
    return math.pi * t * (D - t)


def box_area(H: float, B: float, t: float) -> float:
    return 2 * t * (B + H) - 4 * t ** 2 - 1.25 * (4 - math.pi) * t ** 2


def _fmt(v: float) -> str:
    """Designation number: drop a trailing .0 so 3.0 -> 3, matching catalogues."""
    return f"{v:g}"


def build() -> dict:
    chs, shs, rhs = {}, {}, {}

    for D, ts in CHS_SIZES.items():
        for t in ts:
            A = chs_area(D, t)
            chs[f"{_fmt(D)}x{_fmt(t)}"] = {
                "mass": round(A * _DENSITY_KG_PER_M, 2),
                "height": D,
                "width": D,
                "csa": round(A, 1),
                "web_thickness": t,
                "flange_thickness": t,
                # A round tube has no corners; radius fields exist for schema
                # parity with the open profiles.
                "root_radius": 0.0,
                "toe_radius": 0.0,
                "code_profile": "RO",
                "hollow": True,
            }

    for S, ts in SHS_SIZES.items():
        for t in ts:
            A = box_area(S, S, t)
            shs[f"{_fmt(S)}x{_fmt(S)}x{_fmt(t)}"] = {
                "mass": round(A * _DENSITY_KG_PER_M, 2),
                "height": float(S),
                "width": float(S),
                "csa": round(A, 1),
                "web_thickness": t,
                "flange_thickness": t,
                "root_radius": round(1.5 * t, 2),   # outer corner
                "toe_radius": round(1.0 * t, 2),    # inner corner
                "code_profile": "RU",
                "hollow": True,
            }

    for (H, B), ts in RHS_SIZES.items():
        for t in ts:
            A = box_area(H, B, t)
            rhs[f"{_fmt(H)}x{_fmt(B)}x{_fmt(t)}"] = {
                "mass": round(A * _DENSITY_KG_PER_M, 2),
                "height": float(H),
                "width": float(B),
                "csa": round(A, 1),
                "web_thickness": t,
                "flange_thickness": t,
                "root_radius": round(1.5 * t, 2),
                "toe_radius": round(1.0 * t, 2),
                "code_profile": "RU",
                "hollow": True,
            }

    return {"CHS": chs, "SHS": shs, "RHS": rhs}


# Published EN 10210-2 values (area cm^2, mass kg/m) — the formulae must
# reproduce these before the table is trustworthy.
CHECKS = [
    ("CHS", "33.7x3.2", 3.07, 2.41),
    ("CHS", "42.4x3.2", 3.94, 3.09),
    ("CHS", "48.3x3.2", 4.53, 3.56),
    ("CHS", "114.3x5", 17.2, 13.5),
    ("CHS", "168.3x5", 25.7, 20.1),
    ("CHS", "219.1x10", 6.57e1, 51.6),
    ("SHS", "50x50x3", 5.54, 4.35),
    ("SHS", "100x100x5", 18.7, 14.7),
    ("SHS", "150x150x6.3", 35.8, 28.1),
    ("SHS", "200x200x10", 74.9, 58.8),
    ("RHS", "100x50x3.2", 8.97, 7.04),
    ("RHS", "150x100x6.3", 29.5, 23.1),
    ("RHS", "200x100x8", 44.9, 35.3),
]


def check(lib: dict) -> bool:
    print(f"{'section':22s} {'area cm2':>18s}   {'mass kg/m':>16s}")
    ok = True
    for cat, name, want_a, want_m in CHECKS:
        e = lib[cat].get(name)
        if e is None:
            print(f"  {cat} {name:16s} MISSING")
            ok = False
            continue
        got_a = e["csa"] / 100.0
        got_m = e["mass"]
        ea = abs(got_a - want_a) / want_a * 100
        em = abs(got_m - want_m) / want_m * 100
        flag = "" if (ea < 1.5 and em < 1.5) else "   <-- OFF"
        if flag:
            ok = False
        print(f"  {cat} {name:16s} {got_a:7.2f} vs {want_a:6.2f} ({ea:4.1f}%)   "
              f"{got_m:6.2f} vs {want_m:6.2f} ({em:4.1f}%){flag}")
    return ok


def main() -> None:
    lib = build()
    n = sum(len(v) for v in lib.values())
    print(f"generated {n} hollow sections: "
          + ", ".join(f"{k}={len(v)}" for k, v in lib.items()) + "\n")
    ok = check(lib)
    print("\nvalidation:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)

    if "--write" not in sys.argv:
        print("(dry run — pass --write to update the library)")
        return

    existing = json.loads(LIB.read_text(encoding="utf-8"))
    kept = {k: v for k, v in existing.items() if k not in lib}
    merged = {**kept, **lib}
    LIB.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nwrote {LIB}")
    print("  open profiles kept :", ", ".join(f"{k}={len(v)}" for k, v in kept.items()))
    print("  hollow written     :", ", ".join(f"{k}={len(v)}" for k, v in lib.items()))


if __name__ == "__main__":
    main()
