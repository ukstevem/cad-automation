"""Stub parsers for vendors we haven't implemented yet.

Each stub raises :class:`IFCVendorNotSupportedError` with a descriptive message
so the upload / analysis routes can surface a friendly error and log what
header the file carried.  When a real parser is written, replace the stub with
a concrete subclass of :class:`BaseIFCParser` (see ``tekla.py``) and update
the registry.
"""
from __future__ import annotations

from typing import Any, Dict

import structlog

from app.parsers.ifc.base import BaseIFCParser
from app.parsers.ifc.exceptions import IFCVendorNotSupportedError

logger = structlog.get_logger()


class _UnsupportedVendorParser(BaseIFCParser):
    """Base for vendor stubs.  Subclasses set ``vendor_name``."""

    vendor_name: str = "unknown"

    def analyze(self) -> Dict[str, Any]:
        # Log the header so operators know what showed up
        logger.warning(
            "ifc_vendor_not_supported",
            vendor=self.vendor_name,
            originating_system=self.header.originating_system if self.header else "",
            schema=self.header.schema if self.header else "",
            view_definitions=self.header.view_definitions if self.header else [],
            path=str(self.file_path),
        )
        raise IFCVendorNotSupportedError(
            f"IFC files from {self.vendor_name} are not yet supported.",
            details={
                "vendor": self.vendor_name,
                "originating_system": self.header.originating_system if self.header else "",
                "schema": self.header.schema if self.header else "",
                "path": str(self.file_path),
                "hint": (
                    "Only Tekla Structures is supported in v1.  "
                    "Add a vendor parser under app/parsers/ifc/ and register it "
                    "in registry.py to enable support."
                ),
            },
        )

    def _build_tree(self, totals):
        # Unused — analyze() raises before reaching here.
        return []


class RevitIFCParser(_UnsupportedVendorParser):
    vendor_name = "Autodesk Revit"


class AdvanceSteelIFCParser(_UnsupportedVendorParser):
    vendor_name = "Autodesk Advance Steel"


class SDS2IFCParser(_UnsupportedVendorParser):
    vendor_name = "SDS/2"


class AllplanIFCParser(_UnsupportedVendorParser):
    vendor_name = "Allplan"


class BentleyIFCParser(_UnsupportedVendorParser):
    vendor_name = "Bentley (OpenBuildings / AECOsim)"


class ArchiCADIFCParser(_UnsupportedVendorParser):
    vendor_name = "Graphisoft ArchiCAD"


class FreeCADIFCParser(_UnsupportedVendorParser):
    vendor_name = "FreeCAD"


class GenericIFCParser(_UnsupportedVendorParser):
    vendor_name = "Generic / Unknown vendor"
