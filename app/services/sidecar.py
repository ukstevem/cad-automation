"""Safe read/write for the per-file analysis sidecar JSON.

The sidecar (``outputs/analysis/<prefix>_<stem>.json``) holds many independently
written sections — ``analysis``, ``project_state``, ``consolidation``,
``classification``, ``cnc_*``, ``ar_*`` — and several endpoints/workers
read-modify-write the same file. Two failure modes caused real data loss
(run c25bf2c8, 2026-06-17 — the entire ``project_state`` + ``analysis`` tree
were destroyed):

1. **Clobber-on-corrupt.** Writers caught ``JSONDecodeError`` and silently fell
   back to ``existing = {}``, then wrote back *only their own section* — erasing
   every other section. A transient/partial-read corruption thus escalated to
   permanent total loss. Fixed: :func:`load_sidecar` raises
   :class:`SidecarCorruptError` (after moving the bad file aside, so its
   contents are preserved for recovery) instead of returning ``{}``. Callers
   must NOT overwrite when this is raised.

2. **Half-written reads.** A reader could observe a file mid-write. Fixed:
   :func:`atomic_write_json` writes a temp file in the same directory then
   :func:`os.replace` — readers only ever see a complete document.

This does not yet serialise *concurrent section updates* (two writers that both
read-then-write can still lose one section's update — last writer wins). That is
a smaller, non-catastrophic issue tracked separately; atomic writes + the
corrupt-guard eliminate the data-destroying paths.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class SidecarCorruptError(Exception):
    """The sidecar exists but could not be parsed — refusing to overwrite it."""


def load_sidecar(path: Path) -> Dict[str, Any]:
    """Return the parsed sidecar, or ``{}`` if it does not exist yet.

    Raises :class:`SidecarCorruptError` if the file exists but cannot be parsed.
    On a genuine parse error the file is first moved aside to ``<name>.corrupt``
    so its (possibly partially recoverable) contents are not lost and the next
    clean write starts from a fresh path.
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        backup = path.with_name(path.name + ".corrupt")
        try:
            os.replace(path, backup)
        except OSError:
            backup = path  # left in place; still refuse to clobber
        raise SidecarCorruptError(
            f"{path.name}: {e} (preserved as {backup.name})"
        ) from e
    except OSError as e:
        # Transient (locked/partial) — do NOT move the file, just refuse.
        raise SidecarCorruptError(f"{path.name}: {e}") from e


def atomic_write_json(path: Path, data: Any) -> None:
    """Write ``data`` as JSON atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def update_sidecar(
    path: Path,
    mutate: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Load (safely), apply ``mutate``, atomically write, and return the result.

    ``mutate`` receives the existing dict and may mutate it in place (returning
    ``None``) or return a new dict. Propagates :class:`SidecarCorruptError` —
    the caller decides how to surface it; the on-disk data is never clobbered.
    """
    existing = load_sidecar(path)
    result = mutate(existing)
    if result is None:
        result = existing
    atomic_write_json(path, result)
    return result
