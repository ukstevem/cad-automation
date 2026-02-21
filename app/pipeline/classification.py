import json
import math


def classify_profile(cs, json_path, tol_dim=1.0, tol_area=0.05):
    """
    Attempts to classify a structural profile by matching its dimensions and area.
    Keeps API identical for callers.

    Now with:
      - Automatic relaxed retry when strict pass fails
      - Angle-biased pass when fill ratio indicates "not a plate"
    """
    with open(json_path) as f:
        lib = json.load(f)

    H_meas = float(cs["span_web"])
    W_meas = float(cs["span_flange"])
    A_meas = float(cs["area"])
    L_meas = float(cs.get("length", 0.0))

    # quick "not a plate" heuristic using measured section bounding box
    # plates:   A ≈ H*W  -> fill ~ 1.0
    # angles:   A ≈ t*(H + W - t) -> fill << 1.0
    fill = A_meas / max(H_meas * W_meas, 1e-9)

    def _try_match(height, width, tol_dim_local, tol_area_local, restrict_profile=None):
        best, best_score = None, float('inf')
        for cat, ents in lib.items():
            for name, info in ents.items():
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

    # Nothing found; return None so main can try the plate path.
    return None
