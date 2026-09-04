"""
Multi-view AR fit endpoints — the UI over ``tools/fit_multiview.py`` (bead ...-cg0).

Lists the capture sets, calibration profiles and AR models on disk, runs a fit as a background
subprocess, and serves the result plus overlay images back to the Capture/Fit tab.

The fit runs in a child process (``app/workers/run_ar_fit.py``) rather than inline: a coarse yaw
scan takes minutes and would otherwise stall the event loop for the whole app.

The overlays are the point of this screen, not the RMS. A collapsed or mis-seated pose can score
a low residual — that has happened repeatedly on this rig — so the UI shows the pictures first
and the numbers second, and surfaces the ``degenerate`` and ``visible_fraction`` flags directly.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import List, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services.task_manager import task_manager

logger = structlog.get_logger()

router = APIRouter(prefix="/ar-fit", tags=["ar-fit"])

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,80}$")


def _outputs_dir() -> str:
    return getattr(settings, "OUTPUT_DIR", "/app/outputs")


def _sub(*parts: str) -> str:
    return os.path.join(_outputs_dir(), *parts)


def _safe_name(name: str) -> str:
    """Reject anything that could escape the outputs tree."""
    if not _SAFE.match(name or "") or ".." in name:
        raise HTTPException(status_code=400, detail=f"unsafe name: {name!r}")
    return name


@router.get("/sources")
async def sources():
    """Everything the fit form needs: capture sets, calibration profiles, AR models."""
    captures = []
    root = _sub("ar_captures")
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            imgs = [n for n in os.listdir(d) if n.lower().endswith(IMAGE_EXTS)]
            if imgs:
                captures.append({
                    "name": name,
                    "images": len(imgs),
                    "examples": sorted(imgs)[:4],
                    # A fit wants a capture pair (or a few); a large set is almost certainly a
                    # calibration run, and scanning it is slow and pointless. Flag it so the UI
                    # can warn instead of letting the user start an eight-minute mistake.
                    "looks_like_calibration": len(imgs) > 8,
                })

    profiles = []
    pdir = _sub("calibration")
    if os.path.isdir(pdir):
        for name in sorted(os.listdir(pdir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(pdir, name), encoding="utf-8") as fh:
                    p = json.load(fh)
            except Exception:                          # noqa: BLE001 - skip unreadable
                continue
            profiles.append({
                "file": name,
                "name": p.get("name") or name,
                "image_size": p.get("image_size"),
                "rms": p.get("rms_reproj_error_px"),
                "board": p.get("board"),
            })

    models = []
    mdir = _sub("ar_models")
    if os.path.isdir(mdir):
        for name in sorted(os.listdir(mdir)):
            if name.endswith(".json"):
                models.append({"file": name})

    fits = []
    fdir = _sub("ar_fits")
    if os.path.isdir(fdir):
        for name in sorted(os.listdir(fdir)):
            if os.path.exists(os.path.join(fdir, name, "fit.json")):
                fits.append(name)

    # Smallest first, so a capture pair is the default selection rather than whatever
    # sorts first alphabetically (which was cal_*, the calibration set).
    captures.sort(key=lambda c: (c["looks_like_calibration"], c["images"], c["name"]))
    return {"captures": captures, "profiles": profiles, "models": models, "fits": fits}


class FitRequest(BaseModel):
    captures: str
    profile: str
    cam_profiles: Optional[List[str]] = None       # ["TAG=file.json", ...]
    model: str
    out: Optional[str] = None
    coarse: bool = True
    coarse_step: float = 60.0
    coarse_yaw: float = 10.0
    canny_low: Optional[int] = None
    canny_high: Optional[int] = None
    working_margin: float = 150.0
    full_6dof: bool = False


@router.post("/run")
async def run(req: FitRequest):
    """Kick off a fit in a subprocess and return a task id to poll."""
    captures = _sub("ar_captures", _safe_name(req.captures))
    if not os.path.isdir(captures):
        raise HTTPException(status_code=404, detail=f"no capture set '{req.captures}'")
    profile = _sub("calibration", _safe_name(req.profile))
    if not os.path.exists(profile):
        raise HTTPException(status_code=404, detail=f"no profile '{req.profile}'")
    model = _sub("ar_models", _safe_name(req.model))
    if not os.path.exists(model):
        raise HTTPException(status_code=404, detail=f"no model '{req.model}'")
    out_name = _safe_name(req.out or req.captures)
    out = _sub("ar_fits", out_name)

    cmd = [sys.executable, "-m", "app.workers.run_ar_fit",
           "--captures", captures, "--profile", profile, "--model", model, "--out", out,
           "--working-margin", str(req.working_margin),
           "--coarse-step", str(req.coarse_step), "--coarse-yaw", str(req.coarse_yaw)]
    for spec in (req.cam_profiles or []):
        if "=" not in spec:
            raise HTTPException(status_code=400, detail=f"cam profile must be TAG=file: {spec}")
        tag, fname = spec.split("=", 1)
        cmd += ["--cam-profile", f"{tag}={_sub('calibration', _safe_name(fname))}"]
    if req.coarse:
        cmd.append("--coarse")
    if req.full_6dof:
        cmd.append("--full-6dof")
    if req.canny_low is not None:
        cmd += ["--canny-low", str(req.canny_low)]
    if req.canny_high is not None:
        cmd += ["--canny-high", str(req.canny_high)]

    async def job(progress_callback=None):
        # (completed, total, current_name) — the worker is one opaque step, so report it as
        # such rather than inventing a fake percentage.
        if progress_callback:
            progress_callback(0, 1, "detecting board, masking, fitting")
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            tail = (stderr or b"").decode(errors="replace").strip().splitlines()[-6:]
            raise RuntimeError("ar-fit worker failed: " + " | ".join(tail))
        for line in (stdout or b"").decode(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise RuntimeError("ar-fit worker produced no JSON result")

    task_id = task_manager.submit_async("ar-fit", out_name, job)
    logger.info("ar_fit_started", task_id=task_id, out=out_name)
    return {"task_id": task_id, "out": out_name, "status": "pending"}


@router.get("/status/{task_id}")
async def status(task_id: str):
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="unknown task")
    return task.to_dict()


@router.get("/result/{name}")
async def result(name: str):
    path = _sub("ar_fits", _safe_name(name), "fit.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"no fit result '{name}'")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
