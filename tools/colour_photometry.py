#!/usr/bin/env python3
"""
Three coloured lights, one RGB frame, and the surface normal at every pixel.

THE IDEA. Put a red, a green and a blue light at three different positions. The camera's three
channels respond to the three lights nearly independently, so a single ordinary photograph is
three images of the scene under three lighting directions - and three shaded views of a Lambertian
surface determine its normal. This is photometric stereo (Woodham, 1980) in its colour form: one
shot, no moving parts, and it does not care whether the part is static.

WHY IT MATTERS HERE. The line check keeps failing on creases inside dark webs, where both faces
catch the light identically so there is no gradient to find. Two faces at an angle cannot return
the same R:G:B ratio when the lights come from different directions - the ratio encodes the normal.
And it changes what is compared: instead of sparse, ambiguous EDGES, a dense per-pixel NORMAL map,
against a model that already produces exactly that.

WHY THIS TEST IS NOT CIRCULAR. Rendering with a Lambertian model and then inverting a Lambertian
model proves nothing. So the render deliberately includes everything expected to break it -
specular highlights, ambient fill, channel crosstalk between real LEDs and a real Bayer filter,
sensor noise, and 8-bit quantisation - while the solver assumes plain Lambertian reflectance and
knows only where the lights are. The residual is then an estimate of what the bench would give.

THE DESIGN TENSION IT MEASURES. Lights bunched near the camera keep every visible surface lit by
all three, but make the 3x3 system ill-conditioned and amplify noise. Lights spread wide condition
it beautifully but drive surfaces into attached shadow, where with only three channels there is no
spare measurement. The cone half-angle sweep below is the whole engineering decision.

    docker compose run --rm --no-deps api python tools/colour_photometry.py \\
        --captures outputs/ar_captures/e2e_off --fit outputs/ar_fits/e2e_off \\
        --out outputs/ar_fits/photometry.png
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

from app.services import charuco, multiview_fit as MVF, visibility as VIS  # noqa: E402
from critical_edges import normal_buffer  # noqa: E402


def light_ring(view, half_angle_deg=30.0, n=3, roll_deg=0.0):
    """
    *n* light directions in the WORLD frame, evenly spaced round a cone about the camera axis.

    Returned as unit vectors pointing FROM the surface TOWARD each light, which is the convention
    the shading and the solve both use. Rotation only - the lights are treated as distant, so a
    direction is all they have.
    """
    Rc, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    ha = np.radians(half_angle_deg)
    out = []
    for k in range(n):
        az = 2 * np.pi * k / n + np.radians(roll_deg)
        cam = np.array([np.sin(ha) * np.cos(az), np.sin(ha) * np.sin(az), -np.cos(ha)])
        out.append(Rc.T @ (cam / np.linalg.norm(cam)))
    return np.asarray(out)


def render_colour(tris, rvec, tvec, view, lights, albedo=0.55, ambient=0.10, kd=0.85,
                  ks=0.0, shininess=40.0, crosstalk=0.08, noise=1.5, seed=0,
                  albedo_varies=0.0):
    """
    One synthetic RGB frame lit by three coloured lights, with the imperfections that matter.

    Channel order follows OpenCV (B, G, R), so lights[0] is the blue lamp. Every term here except
    the diffuse one is something the solver does not know about, and is present precisely so the
    measured error means something.
    """
    rng = np.random.default_rng(seed)
    depth, _ = VIS.depth_buffer(tris, rvec, tvec, view, downscale=1)
    solid = depth < VIS.FAR / 2
    nb = normal_buffer(tris, rvec, tvec, view)

    K = np.asarray(view["K"], np.float64).reshape(3, 3)
    Rc, _ = cv2.Rodrigues(np.asarray(view["rvec_cam"], np.float64).reshape(3, 1))
    tc = np.asarray(view["tvec_cam"], np.float64).reshape(3, 1)
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # unit vector from each surface point back toward the camera, in the world frame
    ray = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)], axis=2)
    ray /= np.linalg.norm(ray, axis=2, keepdims=True)
    v = -(ray @ Rc)                       # camera -> world, and reversed to point at the camera

    n = nb.copy()
    flip = np.sum(n * v, axis=2) < 0      # face the normals at the camera
    n[flip] *= -1.0

    a = np.full(depth.shape, albedo, np.float64)
    if albedo_varies > 0:
        # mill scale and primer are not uniform; low-frequency variation is the realistic case
        blob = rng.normal(0, 1, (h // 16 + 2, w // 16 + 2))
        blob = cv2.resize(blob, (w, h), interpolation=cv2.INTER_CUBIC)
        a *= 1.0 + albedo_varies * blob / max(np.std(blob), 1e-6)
        a = np.clip(a, 0.05, 1.0)

    img = np.zeros((h, w, 3), np.float64)
    for c, L in enumerate(lights):
        ndl = np.sum(n * L, axis=2)
        lam = np.clip(ndl, 0, None)                      # attached shadow is a hard zero
        val = ambient + kd * a * lam
        if ks > 0:
            hv = L + v
            hv /= np.maximum(np.linalg.norm(hv, axis=2, keepdims=True), 1e-9)
            val += ks * np.clip(np.sum(n * hv, axis=2), 0, None) ** shininess
        img[:, :, c] = val

    if crosstalk > 0:
        # real LEDs have broad spectra and a Bayer filter has overlapping responses, so each
        # channel sees a little of its neighbours' light
        C = np.full((3, 3), crosstalk) + np.eye(3) * (1.0 - crosstalk)
        C /= C.sum(axis=1, keepdims=True)
        img = img.reshape(-1, 3) @ C.T
        img = img.reshape(h, w, 3)

    img = np.clip(img * 255.0, 0, 255)
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)          # 8-bit is what the camera delivers
    img[~solid] = 30
    return img, n, solid


def solve_normals(img, lights, ambient_guess=0.10, min_level=0.04):
    """
    Invert the Lambertian model per pixel: three intensities, one normal.

    The solver is given only the light directions - no specular model, no crosstalk matrix, no
    albedo map. `ok` marks the pixels where all three channels carry real signal; a pixel in
    attached shadow for even one light has too few measurements and is honestly excluded rather
    than solved badly.
    """
    L = np.asarray(lights, np.float64)
    Linv = np.linalg.inv(L)
    I = img.astype(np.float64) / 255.0 - ambient_guess
    g = I.reshape(-1, 3) @ Linv.T
    mag = np.linalg.norm(g, axis=1)
    nrm = g / np.maximum(mag, 1e-9)[:, None]
    ok = (I.reshape(-1, 3) > min_level).all(axis=1) & (mag > 1e-3)
    sh = img.shape[:2]
    return nrm.reshape(*sh, 3), mag.reshape(sh), ok.reshape(sh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--profile", default="outputs/calibration/RigCam_52FD1B1F.json")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--cones", default="10,20,30,40,50,60",
                    help="light-cone half-angles to sweep, in degrees")
    ap.add_argument("--specular", type=float, default=0.10)
    ap.add_argument("--crosstalk", type=float, default=0.08)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--albedo-varies", type=float, default=0.15)
    ap.add_argument("--out", default=None)
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

    view = None
    for path in sorted(glob.glob(os.path.join(args.captures, "*"))):
        img0 = cv2.imread(path, cv2.IMREAD_COLOR)
        if img0 is None:
            continue
        view = MVF.build_view(img0, profile, board, det, label=os.path.basename(path))
        view.update({"K": profile["K"], "dist": profile["dist"], "tag": os.path.basename(path)})
        break
    if view is None:
        print("no usable captures", file=sys.stderr)
        return 2

    print("Rendered WITH specular %.2f, crosstalk %.2f, albedo variation %.0f%%, noise %.1f LSB,"
          % (args.specular, args.crosstalk, 100 * args.albedo_varies, args.noise))
    print("solved WITHOUT knowledge of any of them - so the error below is not self-fulfilling.")
    print("")
    print("%-9s %10s %12s %11s %11s"
          % ("cone", "condition", "lit by all 3", "median err", "p90 err"))

    best, panels = None, []
    for cone in [float(x) for x in args.cones.split(",")]:
        lights = light_ring(view, half_angle_deg=cone)
        img, ntrue, solid = render_colour(mesh, rvec, tvec, view, lights,
                                          ks=args.specular, crosstalk=args.crosstalk,
                                          noise=args.noise, albedo_varies=args.albedo_varies)
        nrec, mag, ok = solve_normals(img, lights)
        use = solid & ok
        if use.sum() < 100:
            print("%-8.0fd %10.1f %11.0f%% %11s %11s"
                  % (cone, np.linalg.cond(lights), 100 * use.sum() / max(solid.sum(), 1),
                     "-", "-"))
            continue
        dot = np.clip(np.sum(nrec[use] * ntrue[use], axis=1), -1, 1)
        err = np.degrees(np.arccos(np.abs(dot)))
        cov = 100.0 * use.sum() / solid.sum()
        print("%-8.0fd %10.1f %11.0f%% %10.1fd %10.1fd"
              % (cone, np.linalg.cond(lights), cov, np.median(err), np.percentile(err, 90)))
        score = np.median(err) + 0.25 * (100 - cov)
        if best is None or score < best[0]:
            best = (score, cone, img, nrec, ntrue, solid, use, err)

    if best and args.out:
        _, cone, img, nrec, ntrue, solid, use, err = best
        vis = np.zeros_like(img)
        vis[solid] = ((nrec[solid] * 0.5 + 0.5) * 255).astype(np.uint8)
        tru = np.zeros_like(img)
        tru[solid] = ((ntrue[solid] * 0.5 + 0.5) * 255).astype(np.uint8)
        emap = np.zeros(img.shape[:2], np.float32)
        emap[use] = np.minimum(err, 30.0)
        heat = cv2.applyColorMap((emap / 30.0 * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat[~use] = 40
        row = np.hstack([img, tru, vis, heat])
        for i, lab in enumerate(["the photograph (one RGB frame)", "true normals",
                                 "recovered normals", "error, 0-30 deg"]):
            x = i * img.shape[1]
            cv2.rectangle(row, (x, 0), (x + img.shape[1], 60), (26, 26, 26), -1)
            cv2.putText(row, lab, (x + 18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (240, 240, 240), 2)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cv2.imwrite(args.out, cv2.resize(row, None, fx=0.5, fy=0.5))
        print("")
        print("wrote %s (best cone %.0f deg)" % (args.out, cone))
    return 0


if __name__ == "__main__":
    sys.exit(main())
