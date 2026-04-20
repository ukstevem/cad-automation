"""IFC (ISO-10303-21) header sniffer.

Parses the HEADER section of an IFC file with regex only — no ``ifcopenshell``
load required — so that vendor routing can run cheaply even for unsupported
files.

IFC Part 21 HEADER example::

    ISO-10303-21;
    HEADER;
    FILE_DESCRIPTION(('ViewDefinition[CoordinationView_V2.0]', ...), '2;1');
    FILE_NAME('foo.ifc', '2026-04-02T08:15:04', ('author'),
              ('Structural Designer'),
              'IFC Parser Version: 3.14.0',
              'Tekla Structures 2024 Service Pack 10, IFC Export Version: 4.0.0.0', '');
    FILE_SCHEMA(('IFC2X3'));
    ENDSEC;

We only need fields from ``FILE_DESCRIPTION``, ``FILE_NAME``, and
``FILE_SCHEMA`` to route to the right vendor parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from app.config import settings
from app.parsers.ifc.exceptions import IFCHeaderError


# The HEADER section occupies the first few hundred bytes.  Read up to
# ``MAX_IFC_HEADER_SIZE`` from disk (4 KB) to cover generous author fields.
_HEADER_SECTION_RE = re.compile(
    r"HEADER;(?P<body>.*?)ENDSEC;",
    re.IGNORECASE | re.DOTALL,
)
_FILE_DESCRIPTION_RE = re.compile(
    r"FILE_DESCRIPTION\s*\(\s*\((?P<descriptions>.*?)\)\s*,\s*'(?P<impl_level>[^']*)'\s*\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
_FILE_NAME_RE = re.compile(
    # FILE_NAME has 7 positional string fields; we care about #6 (originating system)
    # Each field is either a quoted string '…' (with possible doubled quotes)
    # or a list ('a','b') or '$' for NULL.  We pull the first six commas-out.
    r"FILE_NAME\s*\(\s*(?P<args>.*?)\s*\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
_FILE_SCHEMA_RE = re.compile(
    r"FILE_SCHEMA\s*\(\s*\(\s*'(?P<schema>[^']+)'",
    re.IGNORECASE,
)
# Decode Part-21 string literal: '' → ', and \X\ or \X2\…\X0\ escapes
_DOUBLED_QUOTE_RE = re.compile(r"''")


@dataclass
class IFCHeader:
    """Structured view of an IFC file's Part-21 HEADER section."""

    schema: str                           # e.g. "IFC2X3", "IFC4", "IFC4X3"
    originating_system: str = ""          # FILE_NAME field 6 — the vendor signature
    authoring_tool: str = ""              # FILE_NAME field 5 — preprocessor
    view_definitions: List[str] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)
    file_name: str = ""                   # FILE_NAME field 1
    timestamp: str = ""                   # FILE_NAME field 2
    authors: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)

    def vendor_label(self) -> str:
        """Short human-readable vendor tag for logs + error messages."""
        if self.originating_system:
            return self.originating_system.split(",")[0].strip()
        if self.authoring_tool:
            return self.authoring_tool.strip()
        return "unknown"


def sniff_ifc_header(path: str | Path) -> IFCHeader:
    """Read up to ``MAX_IFC_HEADER_SIZE`` bytes and return an :class:`IFCHeader`.

    Raises :class:`IFCHeaderError` if the file is not ISO-10303-21 or the
    HEADER section is malformed enough that we can't pick out FILE_SCHEMA.
    """
    p = Path(path)
    with p.open("rb") as f:
        raw = f.read(settings.MAX_IFC_HEADER_SIZE)

    text = raw.decode("utf-8", errors="replace")

    if not text.startswith("ISO-10303-21;"):
        raise IFCHeaderError(
            "File does not start with ISO-10303-21; header",
            details={"path": str(p), "found": text[:32]},
        )

    section_match = _HEADER_SECTION_RE.search(text)
    if not section_match:
        raise IFCHeaderError(
            "Missing HEADER;…ENDSEC; block",
            details={"path": str(p)},
        )
    body = section_match.group("body")

    schema_match = _FILE_SCHEMA_RE.search(body)
    if not schema_match:
        raise IFCHeaderError(
            "FILE_SCHEMA not found in IFC header",
            details={"path": str(p), "snippet": body[:300]},
        )

    desc_strings: List[str] = []
    impl_level = ""
    desc_match = _FILE_DESCRIPTION_RE.search(body)
    if desc_match:
        desc_strings = _extract_string_list(desc_match.group("descriptions"))
        impl_level = desc_match.group("impl_level")

    view_definitions = _extract_view_definitions(desc_strings)

    filename = timestamp = preprocessor = originator = ""
    authors: List[str] = []
    orgs: List[str] = []
    fn_match = _FILE_NAME_RE.search(body)
    if fn_match:
        fields = _split_file_name_args(fn_match.group("args"))
        # Positional: [name, timestamp, authors, organizations, preprocessor, originator, authorisation]
        if len(fields) >= 1:
            filename = _first_string(fields[0])
        if len(fields) >= 2:
            timestamp = _first_string(fields[1])
        if len(fields) >= 3:
            authors = _extract_string_list(fields[2])
        if len(fields) >= 4:
            orgs = _extract_string_list(fields[3])
        if len(fields) >= 5:
            preprocessor = _first_string(fields[4])
        if len(fields) >= 6:
            originator = _first_string(fields[5])

    return IFCHeader(
        schema=schema_match.group("schema"),
        originating_system=originator,
        authoring_tool=preprocessor,
        view_definitions=view_definitions,
        descriptions=desc_strings,
        file_name=filename,
        timestamp=timestamp,
        authors=authors,
        organizations=orgs,
    )


# --- Internal helpers ---------------------------------------------------------

def _extract_string_list(raw: str) -> List[str]:
    """Return all single-quoted string literals from a Part-21 string list.

    Handles the '' escape for embedded quotes.
    """
    out: List[str] = []
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] != "'":
            i += 1
            continue
        # found opening quote — scan until the matching (non-doubled) one
        j = i + 1
        buf: List[str] = []
        while j < n:
            if raw[j] == "'":
                if j + 1 < n and raw[j + 1] == "'":
                    buf.append("'")
                    j += 2
                    continue
                break
            buf.append(raw[j])
            j += 1
        out.append("".join(buf))
        i = j + 1
    return out


def _first_string(raw: str) -> str:
    strings = _extract_string_list(raw)
    return strings[0] if strings else ""


def _split_file_name_args(args: str) -> List[str]:
    """Split FILE_NAME positional arguments on top-level commas.

    FILE_NAME args contain nested lists ``('a','b')`` — we must not split on
    commas inside those.  Strings can contain commas inside ''; the Part-21
    grammar guarantees strings are always single-quoted.
    """
    parts: List[str] = []
    depth = 0
    in_string = False
    buf: List[str] = []
    i = 0
    n = len(args)
    while i < n:
        ch = args[i]
        if in_string:
            buf.append(ch)
            if ch == "'":
                if i + 1 < n and args[i + 1] == "'":
                    # escaped quote — consume both
                    buf.append(args[i + 1])
                    i += 2
                    continue
                in_string = False
            i += 1
            continue
        if ch == "'":
            in_string = True
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return parts


_VIEW_DEFINITION_RE = re.compile(r"ViewDefinition\s*\[(?P<views>[^\]]+)\]", re.IGNORECASE)


def _extract_view_definitions(descriptions: Iterable[str]) -> List[str]:
    """Pick ``CoordinationView_V2.0`` etc. out of FILE_DESCRIPTION items."""
    views: List[str] = []
    for d in descriptions:
        m = _VIEW_DEFINITION_RE.search(d)
        if m:
            for v in m.group("views").split(","):
                v = v.strip()
                if v:
                    views.append(v)
    return views
