"""Registry mapping IFC vendor signatures to parser classes.

A ``VendorRule`` compares the sniffed :class:`IFCHeader` against a compiled
regex and, on match, returns a dotted import path to the parser class.

Lazy import means ``ifcopenshell`` is loaded only when a file actually reaches
a parser that needs it — header-only probes (for logging or dispatch diagnostics)
do not pay that cost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import List, Optional, Type

from app.parsers.ifc.header import IFCHeader


@dataclass(frozen=True)
class VendorRule:
    """A single routing rule in the vendor registry."""

    label: str                 # e.g. "Tekla Structures"
    pattern: re.Pattern        # matched against header.originating_system
    parser_path: str           # dotted import path, e.g. "app.parsers.ifc.tekla:TeklaIFCParser"

    def matches(self, header: IFCHeader) -> bool:
        targets = [header.originating_system, header.authoring_tool]
        return any(self.pattern.search(t) for t in targets if t)

    def load(self) -> Type:
        module_path, _, cls_name = self.parser_path.partition(":")
        module = import_module(module_path)
        return getattr(module, cls_name)


# Order matters: first match wins.  Keep the most specific patterns near the
# top.  Case-insensitive throughout.
_VENDOR_RULES: List[VendorRule] = [
    VendorRule(
        label="Tekla Structures",
        pattern=re.compile(r"Tekla\s*Structures", re.IGNORECASE),
        parser_path="app.parsers.ifc.tekla:TeklaIFCParser",
    ),
    VendorRule(
        label="Autodesk Revit",
        pattern=re.compile(r"(Autodesk\s*)?Revit", re.IGNORECASE),
        parser_path="app.parsers.ifc.stubs:RevitIFCParser",
    ),
    VendorRule(
        label="Autodesk Advance Steel",
        pattern=re.compile(r"Advance\s*Steel", re.IGNORECASE),
        parser_path="app.parsers.ifc.stubs:AdvanceSteelIFCParser",
    ),
    VendorRule(
        label="SDS/2",
        pattern=re.compile(r"\bSDS[\s/]?2\b", re.IGNORECASE),
        parser_path="app.parsers.ifc.stubs:SDS2IFCParser",
    ),
    VendorRule(
        label="Allplan",
        pattern=re.compile(r"Allplan", re.IGNORECASE),
        parser_path="app.parsers.ifc.stubs:AllplanIFCParser",
    ),
    VendorRule(
        label="Bentley",
        pattern=re.compile(r"(Bentley|OpenBuildings|AECOsim)", re.IGNORECASE),
        parser_path="app.parsers.ifc.stubs:BentleyIFCParser",
    ),
    VendorRule(
        label="ArchiCAD",
        pattern=re.compile(r"(ArchiCAD|Graphisoft)", re.IGNORECASE),
        parser_path="app.parsers.ifc.stubs:ArchiCADIFCParser",
    ),
    VendorRule(
        label="FreeCAD",
        pattern=re.compile(r"FreeCAD", re.IGNORECASE),
        parser_path="app.parsers.ifc.stubs:FreeCADIFCParser",
    ),
]

_GENERIC_FALLBACK = VendorRule(
    label="Generic IFC",
    pattern=re.compile(r".*"),  # matches anything; used as last resort
    parser_path="app.parsers.ifc.stubs:GenericIFCParser",
)


def match_vendor(header: IFCHeader) -> VendorRule:
    """Return the first rule that matches the header, or the generic fallback."""
    for rule in _VENDOR_RULES:
        if rule.matches(header):
            return rule
    return _GENERIC_FALLBACK


def list_rules() -> List[VendorRule]:
    """Enumerate configured rules — useful for diagnostics and tests."""
    return list(_VENDOR_RULES) + [_GENERIC_FALLBACK]
