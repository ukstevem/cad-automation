"""
Parsers package for CAD file processing.

The top-level :func:`get_assembly_parser` factory dispatches on file extension:

* ``.step`` / ``.stp`` → :class:`~app.parsers.assembly_analyzer.AssemblyAnalyzer`
* ``.ifc``             → vendor-specific parser via :func:`app.parsers.ifc.get_ifc_parser`

All parsers expose an ``analyze()`` method returning the XCAF-shaped tree
dict ``{"assembly_tree": [...], "summary": {...}}``.
"""
from pathlib import Path

from app.config import settings
from app.parsers.step_parser import STEPParser
from app.parsers.assembly_analyzer import AssemblyAnalyzer


def get_assembly_parser(file_path: str):
    """Return a parser for ``file_path`` chosen by its extension.

    Raises ValueError for unsupported extensions.  The returned object exposes
    ``analyze()``.
    """
    ext = Path(file_path).suffix.lower()
    if ext in settings.STEP_EXTENSIONS:
        return AssemblyAnalyzer(file_path)
    if ext in settings.IFC_EXTENSIONS:
        # Lazy import — avoids importing ifcopenshell for STEP-only callers
        from app.parsers.ifc import get_ifc_parser
        return get_ifc_parser(file_path)
    raise ValueError(
        f"Unsupported file extension for assembly analysis: {ext!r}. "
        f"Supported: {sorted(settings.STEP_EXTENSIONS | settings.IFC_EXTENSIONS)}"
    )


__all__ = ["STEPParser", "AssemblyAnalyzer", "get_assembly_parser"]
