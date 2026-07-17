import json
import math
import re


# Designation matcher: "UB 914x419x388" → ("UB", "914x419x388")
# Accepts optional separators (space, hyphen, slash, dot) between the category
# prefix and the rest of the designation.  Decimal values in the tail (e.g.
# CHS "114.3x6.3") are preserved.
_NAME_RE = re.compile(r"^\s*([A-Za-z]+)[\s\-\/_]*([0-9.xX\-]+)\s*$")


def classify_by_name(type_name, json_path):
    """Look up an entry in the classifier library by its Tekla-style designation.

    Tekla (and most steel detailers) emit section names like ``UB203x102x23`` or
    ``PFC 300x100x46`` in ``IfcTypeProduct.Name``.  When that name is present
    we can skip the four-pass geometric classifier entirely — the name is more
    reliable than dimensions recovered from a tessellated mesh, and orders of
    magnitude faster.

    Returns a match dict shaped like :func:`classify_profile`'s return value,
    or ``None`` if the name can't be parsed or matched.
    """
    if not type_name:
        return None

    m = _NAME_RE.match(str(type_name))
    if not m:
        return None
    category, designation = m.group(1).upper(), m.group(2).lower()

    with open(json_path) as f:
        lib = json.load(f)

    # Direct category match first (UB → UB section)
    bucket = lib.get(category)
    if bucket is None:
        # Some Tekla catalogues use 'ANG' for EA, 'I' for UB, etc.  Keep a
        # small aliases map here — extend as new sources appear.
        aliases = {
            "ANG": "EA",
            "ANGLE": "EA",
            "I": "UB",
            "HEA": "UC",
            "HEB": "UC",
            "IPE": "UB",
            "UPE": "PFC",
            "UPN": "PFC",
        }
        mapped = aliases.get(category)
        if mapped:
            bucket = lib.get(mapped)
            category = mapped
    if bucket is None:
        return None

    entry = bucket.get(designation) or bucket.get(designation.upper())
    if entry is None:
        # Some designations omit leading zeros or use different separators —
        # normalise and compare lowercased.
        norm_target = designation.replace("-", "x").replace("X", "x")
        for key, val in bucket.items():
            if key.lower().replace("-", "x") == norm_target:
                entry = val
                designation = key
                break

    if entry is None:
        return None

    H = float(entry["height"])
    W = float(entry["width"])
    A = float(entry["csa"])
    return {
        "Designation": designation,
        "Category": category,
        "Profile_type": entry["code_profile"],
        "Match_score": 0.0,
        "Requires_rotation": False,
        "Matched_by": "name",
        "Source_name": type_name,
        "JSON": {
            "height": H,
            "width": W,
            "csa": A,
            "length": 0,
            "Mass": 0.0,
            "web_thickness": entry.get("web_thickness", 0.0),
            "flange_thickness": entry.get("flange_thickness", 0.0),
            "root_radius": entry.get("root_radius", 0.0),
            "toe_radius": entry.get("toe_radius", 0.0),
        },
        "STEP": {
            "height": H,
            "width": W,
            "area": A,
            "length": 0.0,
            "mass": 0.0,
            "web_thickness": entry.get("web_thickness", 0.0),
            "flange_thickness": entry.get("flange_thickness", 0.0),
            "root_radius": entry.get("root_radius", 0.0),
            "toe_radius": entry.get("toe_radius", 0.0),
        },
        "Diagnostics": {
            "matched_by": "name",
        },
    }


# Hollow matching tolerances.  CSA is the trustworthy measurement here (see
# _match_hollow_profile), so it carries the match; the bounding box only vetoes.
HOLLOW_TOL_AREA = 0.10      # CSA within 10%
HOLLOW_DIM_SLACK = 1.5      # mm the library section may exceed the measured box
HOLLOW_MAX_OVERSIZE = 1.5   # reject a section < measured/1.5 (absurdly small)
# Composite ceiling (100*area_err + dim_gap_mm).  Real matches score well under
# this even when the box is inflated — the run-761b26b1 tubes come in at 0.6-7.2
# — so it only bites on a candidate that is the best available yet still wrong.
HOLLOW_MAX_SCORE = 30.0


def _match_hollow_profile(lib, H_meas, W_meas, A_meas, L_meas):
    """Match a closed section on CSA, using the bounding box only as a veto.

    Open profiles are matched on height/width with area as a check.  That is the
    wrong way round for tube, because the two measurements have opposite
    reliability:

    - **CSA is sound.**  It comes from volume/length, which no rotation changes.
      Measured against real EN 10210 sizes it lands within ~1-3%.
    - **The OBB cross-section is not.**  A smooth cylinder has no straight edges
      for ``align_by_longest_straight_edge`` to grab, so a raked handrail comes
      out slightly skewed and the box inflates.  Run 761b26b1's measured ODs
      smear 33.7 -> 41.9 with no clustering on any standard size, while their
      CSAs put 56 of them on plain 33.7 tube.

    So score on CSA, and use the box only for what it can be trusted to say: a
    section cannot be *larger* than its own bounding box, so anything that does
    not fit is out.  That resolves the CSA collisions — 42.4x2.3 and 33.7x3.0
    are 290 vs 289 mm2, indistinguishable by area, but a 42.4 tube cannot fit
    inside a 34.9 box.

    The dimension gap still scores, so among candidates that fit and agree on
    area the closest-fitting wins: at OD 120 / CSA 1650, CHS 114.3x5.0 (4% area
    off, 6mm gap) beats 88.9x6.3 (0.9% off, 31mm gap) — the right answer, since
    an 88.9 tube rattling around inside a 120 box is not what was measured.
    """
    meas_lo, meas_hi = min(H_meas, W_meas), max(H_meas, W_meas)
    best, best_score = None, float("inf")

    for cat, ents in lib.items():
        for name, info in ents.items():
            if not info.get("hollow"):
                continue
            H_lib = float(info["height"])
            W_lib = float(info["width"])
            A_lib = float(info["csa"])

            ea = abs(A_meas - A_lib) / max(A_lib, 1e-9)
            if ea > HOLLOW_TOL_AREA:
                continue

            lib_lo, lib_hi = min(H_lib, W_lib), max(H_lib, W_lib)
            # Must fit the measured box (which is an upper bound on the truth).
            if lib_lo > meas_lo + HOLLOW_DIM_SLACK or lib_hi > meas_hi + HOLLOW_DIM_SLACK:
                continue
            # ...but not be absurdly smaller than it either.
            if lib_hi < meas_hi / HOLLOW_MAX_OVERSIZE:
                continue

            dim_gap = max(0.0, meas_lo - lib_lo) + max(0.0, meas_hi - lib_hi)
            score = 100 * ea + dim_gap
            if score < best_score:
                best_score = score
                best = {
                    "Designation": name,
                    "Category": cat,
                    "Profile_type": info["code_profile"],
                    "Match_score": score,
                    "Requires_rotation": False,
                    "JSON": {
                        "height": H_lib,
                        "width": W_lib,
                        "csa": A_lib,
                        "length": 0,
                        "Mass": info.get("mass", 0.0) * L_meas / 1000.0,
                        "web_thickness": info.get("web_thickness", 0.0),
                        "flange_thickness": info.get("flange_thickness", 0.0),
                        "root_radius": info.get("root_radius", 0.0),
                        "toe_radius": info.get("toe_radius", 0.0),
                    },
                    "STEP": {
                        "height": H_meas,
                        "width": W_meas,
                        "area": float(A_meas),
                        "length": L_meas,
                        "mass": info.get("mass", 0.0) * L_meas / 1000.0,
                        "web_thickness": info.get("web_thickness", 0.0),
                        "flange_thickness": info.get("flange_thickness", 0.0),
                        "root_radius": info.get("root_radius", 0.0),
                        "toe_radius": info.get("toe_radius", 0.0),
                    },
                    "Diagnostics": {
                        "hollow": True,
                        "area_err": round(ea, 4),
                        "dim_gap": round(dim_gap, 2),
                        "tol_area_used": HOLLOW_TOL_AREA,
                    },
                }
    if best is not None and best_score > HOLLOW_MAX_SCORE:
        return None
    return best


def classify_profile(cs, json_path, tol_dim=1.0, tol_area=0.05, hollow=False):
    """
    Attempts to classify a structural profile by matching its dimensions and area.
    Keeps API identical for callers.

    Now with:
      - Automatic relaxed retry when strict pass fails
      - Angle-biased pass when fill ratio indicates "not a plate"

    ``hollow`` selects which half of the library is eligible, and the two are
    mutually exclusive: a closed section must never match an open profile, nor
    an open one a tube.  Letting them mix is what matched a 33.7 CHS handrail to
    an EA 35x35x4 and cut a wrong NC1 (bd 5wg) — a round tube's fill ratio sits
    in the same band as an angle's, so the angle-biased pass below happily
    claimed it.  Default False keeps every existing caller on open profiles.

    Hollow sections take a different matching strategy entirely — see
    :func:`_match_hollow_profile`.
    """
    with open(json_path) as f:
        lib = json.load(f)

    H_meas = float(cs["span_web"])
    W_meas = float(cs["span_flange"])
    A_meas = float(cs["area"])
    L_meas = float(cs.get("length", 0.0))

    if hollow:
        return _match_hollow_profile(lib, H_meas, W_meas, A_meas, L_meas)

    # quick "not a plate" heuristic using measured section bounding box
    # plates:   A ≈ H*W  -> fill ~ 1.0
    # angles:   A ≈ t*(H + W - t) -> fill << 1.0
    fill = A_meas / max(H_meas * W_meas, 1e-9)

    def _try_match(height, width, tol_dim_local, tol_area_local, restrict_profile=None):
        best, best_score = None, float('inf')
        for cat, ents in lib.items():
            for name, info in ents.items():
                if info.get("hollow"):
                    continue          # open-profile matching only
                if restrict_profile and info.get("code_profile") != restrict_profile:
                    continue

                H_lib = info["height"]
                W_lib = info["width"]
                A_lib = info["csa"]

                # CSA relative error first (fast reject)
                ea = abs(A_meas - A_lib) / max(A_lib, 1e-9)
                if ea > tol_area_local:
                    continue

                dh = abs(height - H_lib)
                dw = abs(width  - W_lib)
                if dh > tol_dim_local or dw > tol_dim_local:
                    continue

                # keep a simple composite score
                el = 0.0
                if L_meas > 0:
                    el = abs(L_meas - info.get("length", L_meas)) / max(info.get("length", L_meas), 1e-9)
                score = dh + dw + 100*ea + 100*el

                if score < best_score:
                    best_score = score
                    best = {
                        "Designation": name,
                        "Category": cat,
                        "Profile_type": info["code_profile"],
                        "Match_score": score,
                        "Requires_rotation": False,
                        "JSON": {
                            "height": H_lib,
                            "width": W_lib,
                            "csa": A_lib,
                            "length": 0,
                            "Mass": info.get("mass", 0.0) * L_meas / 1000.0,
                            "web_thickness": info.get("web_thickness", 0.0),
                            "flange_thickness": info.get("flange_thickness", 0.0),
                            "root_radius": info.get("root_radius", 0.0),
                            "toe_radius":  info.get("toe_radius", 0.0),
                        },
                        "STEP": {
                            "height": height,
                            "width": width,
                            "area": float(A_meas),
                            "length": info.get("length", L_meas),
                            "mass": info.get("mass", 0.0) * L_meas / 1000.0,
                            "web_thickness": info.get("web_thickness", 0.0),
                            "flange_thickness": info.get("flange_thickness", 0.0),
                            "root_radius": info.get("root_radius", 0.0),
                            "toe_radius":  info.get("toe_radius", 0.0),
                        },
                        "Diagnostics": {
                            "fill_ratio": float(fill),
                            "tol_dim_used": float(tol_dim_local),
                            "tol_area_used": float(tol_area_local),
                        }
                    }
        return best, best_score

    # ---- PASS 1: strict, both orientations ----
    original, score1 = _try_match(H_meas, W_meas, tol_dim, tol_area)
    swapped,  score2 = _try_match(W_meas, H_meas, tol_dim, tol_area)

    best_match = None
    if original and (not swapped or score1 <= score2):
        original["Requires_rotation"] = False
        best_match = original
    elif swapped:
        swapped["Requires_rotation"] = True
        best_match = swapped

    # patch L-sections to mirror your existing logic
    if best_match and best_match.get("Profile_type") == "L":
        wt = best_match["JSON"]["web_thickness"]
        best_match["JSON"]["flange_thickness"] = wt
        best_match["STEP"]["flange_thickness"] = wt

    if best_match:
        return best_match

    # ---- PASS 2: relaxed retry (wider tolerances) ----
    tol_dim_relaxed  = max(2.0, tol_dim * 2.0)
    tol_area_relaxed = max(0.12, tol_area * 2.5)

    original, score1 = _try_match(H_meas, W_meas, tol_dim_relaxed, tol_area_relaxed)
    swapped,  score2 = _try_match(W_meas, H_meas, tol_dim_relaxed, tol_area_relaxed)

    if original and (not swapped or score1 <= score2):
        original["Requires_rotation"] = False
        return original
    if swapped:
        swapped["Requires_rotation"] = True
        return swapped

    # ---- PASS 3: angle-biased fallback if fill ratio says "not a plate" ----
    # Typical thresholds:
    #   plate: fill ~ 0.85–1.00
    #   angle: fill ~ 0.15–0.35  (depends on t/H and t/W)
    if fill < 0.70:
        # Try only 'L' profiles with a slightly more lenient CSA tolerance.
        tol_area_L = max(0.15, tol_area_relaxed)
        L1, s1 = _try_match(H_meas, W_meas, tol_dim_relaxed, tol_area_L, restrict_profile="L")
        L2, s2 = _try_match(W_meas, H_meas, tol_dim_relaxed, tol_area_L, restrict_profile="L")
        cand = None
        if L1 and (not L2 or s1 <= s2):
            cand = L1
            cand["Requires_rotation"] = False
        elif L2:
            cand = L2
            cand["Requires_rotation"] = True

        if cand:
            # same L-thickness mirroring
            wt = cand["JSON"]["web_thickness"]
            cand["JSON"]["flange_thickness"] = wt
            cand["STEP"]["flange_thickness"] = wt
            cand.setdefault("Diagnostics", {})["angle_bias"] = True
            return cand

    # ---- PASS 4: dimension-priority fallback (relaxed dims, tight area) ----
    # When alignment fails the cross-section dims are unreliable, but CSA
    # (from volume / length) is still accurate.  Match on area with a tight
    # tolerance to avoid false positives from plates with similar CSA.
    tol_area_dim = 0.10   # 10% area tolerance — tight enough to reject plates
    tol_dim_dim  = max(2.0, tol_dim * 2.0)

    original, score1 = _try_match(H_meas, W_meas, tol_dim_dim, tol_area_dim)
    swapped,  score2 = _try_match(W_meas, H_meas, tol_dim_dim, tol_area_dim)

    best_match = None
    if original and (not swapped or score1 <= score2):
        original["Requires_rotation"] = False
        best_match = original
    elif swapped:
        swapped["Requires_rotation"] = True
        best_match = swapped

    if best_match:
        if best_match.get("Profile_type") == "L":
            wt = best_match["JSON"]["web_thickness"]
            best_match["JSON"]["flange_thickness"] = wt
            best_match["STEP"]["flange_thickness"] = wt
        best_match.setdefault("Diagnostics", {})["dim_priority"] = True
        return best_match

    # Nothing found; return None so main can try the plate path.
    return None
