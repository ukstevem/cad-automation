"""
The placement guide's SET button (bd 6et).

``tools/place_guide.py`` writes a page the operator watches while placing a part, and
``tools/place_loop.sh`` / ``.ps1`` keeps it up to date. When the part is in place the operator has to be able to
say so from where they are standing - at the rig, looking at the page, not at the terminal. A static page cannot
write a file, so it posts here, and this drops a request beside the plan for the loop to pick up.

Deliberately the whole of it: no pose solving, no subprocess. The loop already has the photographs, the plan and
the tools; all it lacks is the operator's word.
"""
from __future__ import annotations

import datetime
import json
import os

import structlog
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import HTMLResponse

from app.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/place", tags=["place"])

_PAGE = """<!doctype html><meta charset="utf-8"><title>%(title)s</title>
<meta http-equiv="refresh" content="3;url=%(back)s">
<style>body{margin:0;font:16px/1.5 system-ui,sans-serif;background:%(bg)s;color:#fff;padding:14vh 24px;text-align:center}
h1{font-size:clamp(24px,5vw,40px);margin:0 0 8px}a{color:#fff}</style>
<h1>%(title)s</h1><p>%(detail)s</p><p><a href="%(back)s">back to the guide</a></p>"""


def _plan_dir(plan: str) -> str:
    """A plan directory inside the AR outputs, and nowhere else."""
    root = os.path.realpath(os.path.join(getattr(settings, "OUTPUT_DIR", "/app/outputs"), "ar_fits"))
    full = os.path.realpath(os.path.join("/app", plan) if not os.path.isabs(plan) else plan)
    if not (full == root or full.startswith(root + os.sep)) or not os.path.isdir(full):
        raise HTTPException(status_code=400, detail="not a placement plan directory: %s" % plan)
    if not os.path.exists(os.path.join(full, "plan.json")):
        raise HTTPException(status_code=400, detail="no plan.json in %s" % plan)
    return full


@router.post("/set", response_class=HTMLResponse)
def set_placement(plan: str = Form(...)) -> HTMLResponse:
    """Record that the operator is happy with the placement shown."""
    full = _plan_dir(plan)
    with open(os.path.join(full, "set.request"), "w", encoding="utf-8") as fh:
        json.dump({"requested_at": datetime.datetime.now().isoformat(timespec="seconds")}, fh)
    logger.info("placement set requested", plan=full)
    back = "/outputs/ar_fits/%s/guide.html" % os.path.relpath(
        full, os.path.join(getattr(settings, "OUTPUT_DIR", "/app/outputs"), "ar_fits")).replace(os.sep, "/")
    return HTMLResponse(_PAGE % {"title": "Set", "bg": "#1f8a4c", "back": back,
                                 "detail": "Measuring this placement and building the weld view - about half a minute."})
