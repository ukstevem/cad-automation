"""Vendor dispatcher: sniff the header, pick the parser, instantiate it."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from app.parsers.ifc.header import sniff_ifc_header
from app.parsers.ifc.registry import match_vendor

if TYPE_CHECKING:
    from app.parsers.ifc.base import BaseIFCParser


logger = structlog.get_logger()


def get_ifc_parser(file_path: str | Path) -> "BaseIFCParser":
    """Return a parser instance suited to the IFC file at ``file_path``.

    Reads the Part-21 HEADER section, picks a vendor rule from the registry,
    and lazily imports the concrete parser class.  The returned instance has
    already been bound to the file and to the sniffed header.
    """
    path = Path(file_path)
    header = sniff_ifc_header(path)
    rule = match_vendor(header)

    logger.info(
        "ifc_vendor_dispatched",
        path=str(path),
        vendor=rule.label,
        schema=header.schema,
        originating_system=header.originating_system,
        view_definitions=header.view_definitions,
    )

    parser_cls = rule.load()
    return parser_cls(str(path), header=header, vendor_label=rule.label)
