"""
Frontend router - serves the web UI
"""
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

TEMPLATE_DIR = Path("/app/frontend/templates")


@router.get("/ui", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page"""
    html_path = TEMPLATE_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)
