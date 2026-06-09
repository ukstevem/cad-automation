"""
Known bought-out catalogue products, identifiable by part name (and, later,
signature cross-section shape).

Unlike the general make-or-buy decision (which isn't geometric), specific
proprietary products are *confidently* bought-out — e.g. Unistrut (a strut
channel with inturned rolled lips, always purchased). This module maps a part
name to (category, product) so the classifier can high-confidence route them to
BOUGHT_OUT before the geometric rules run.

Extend ``_RULES`` as new catalogue products show up. Each rule is
``(match_regex, exclude_regex|None, category, product_label)`` — the name must
match ``match_regex`` and NOT match ``exclude_regex``.
"""
import re
from typing import Optional, Tuple

_RULES = [
    # Unistrut strut-channel — but NOT the "UNIST PLATE" support plates, which
    # are fabricated (CNC) plates that merely sit with the strut.
    (re.compile(r"unist", re.I), re.compile(r"plate", re.I), "BOUGHT_OUT", "Unistrut"),
]


def match_catalogue_product(name: Optional[str]) -> Optional[Tuple[str, str]]:
    """Return (category, product) if *name* is a known catalogue product, else None."""
    if not name:
        return None
    for pat, excl, cat, prod in _RULES:
        if pat.search(name) and not (excl and excl.search(name)):
            return cat, prod
    return None
