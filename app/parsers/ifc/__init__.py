"""IFC parser package.

Entry point: :func:`get_ifc_parser` returns a concrete parser instance for an
IFC file based on its header (vendor, schema, view definition).  The main
``app.parsers`` factory dispatches here when the file extension is ``.ifc``.
"""
from app.parsers.ifc.dispatcher import get_ifc_parser
from app.parsers.ifc.base import BaseIFCParser
from app.parsers.ifc.header import IFCHeader, sniff_ifc_header
from app.parsers.ifc.exceptions import (
    IFCHeaderError,
    IFCVendorNotSupportedError,
    IFCParseError,
)

__all__ = [
    "get_ifc_parser",
    "BaseIFCParser",
    "IFCHeader",
    "sniff_ifc_header",
    "IFCHeaderError",
    "IFCVendorNotSupportedError",
    "IFCParseError",
]
