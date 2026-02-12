"""
Analysis router - assembly tree inspection for uploaded STEP files
"""
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
import structlog

from app.config import settings
from app.parsers.assembly_analyzer import AssemblyAnalyzer
from app.exceptions import STEPParseError

router = APIRouter()
logger = structlog.get_logger()

STEP_EXTENSIONS = {".step", ".stp"}

# Uploaded files are stored as "<8-hex-chars>_<original_name>"
_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_")


def _original_name(filename: str) -> str:
    """Strip the unique-id prefix to recover the original filename."""
    return _PREFIX_RE.sub("", filename)


@router.get("/analysis/files")
async def list_uploaded_files() -> Dict[str, Any]:
    """List STEP files available for analysis, sorted newest-first."""
    upload_dir = Path(settings.UPLOAD_DIR)
    if not upload_dir.exists():
        return {"files": []}

    files: List[Dict[str, Any]] = []
    for entry in upload_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in STEP_EXTENSIONS:
            stat = entry.stat()
            uploaded_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            files.append({
                "filename": entry.name,
                "display_name": _original_name(entry.name),
                "size_bytes": stat.st_size,
                "uploaded_at": uploaded_at.isoformat(),
            })

    # Newest first
    files.sort(key=lambda f: f["uploaded_at"], reverse=True)

    return {"files": files}


@router.get("/analysis/assembly/{filename}")
async def get_assembly_tree(filename: str) -> Dict[str, Any]:
    """
    Analyse a previously uploaded STEP file and return its assembly tree.

    Args:
        filename: Name of the file in the uploads directory.

    Returns:
        Assembly tree with named nodes and classification metadata.
    """
    file_path = Path(settings.UPLOAD_DIR) / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "File not found",
                "filename": filename,
            },
        )

    try:
        analyzer = AssemblyAnalyzer(str(file_path))
        result = analyzer.analyze()
        return {"success": True, "filename": filename, **result}

    except STEPParseError as e:
        logger.error("assembly_analysis_failed", filename=filename, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Analysis failed",
                "message": str(e),
                "details": e.details,
            },
        )

    except Exception as e:
        logger.error("unexpected_analysis_error", filename=filename, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred during assembly analysis",
                "details": {"error_type": type(e).__name__},
            },
        )
