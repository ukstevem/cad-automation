"""
Standalone subprocess worker for LIGHTWEIGHT part classification (rw7.10).

Run as:
    python classify_parts.py <file_path> <analysis_json_path> \\
                             <ref_ids_json> [member_ids_json] [steel_grade]

For each ref_id, computes the refined class (features + library match + decision
tree) via ``classify_ref`` — WITHOUT generating any NC1/DXF.  This is the cheap
pass that assists the top-down triage before the heavy CNC analysis runs.

Prints ``{"results": {ref_id: {refined_class, refined_confidence, ...}}}`` to
stdout.  The CALLER (the classify endpoint) persists these into the sidecar's
``classification`` key via the safe, atomic sidecar helpers.

This worker must NEVER write the sidecar itself: as a subprocess it raced with
the FastAPI process's writes and, on a corrupt/partial read, fell back to ``{}``
and wrote back only its own section — destroying ``analysis`` + ``project_state``
(run c2c2ac88, 2026-06-18). Persistence now happens once, in the parent, through
``load_sidecar``/``atomic_write_json``.

Same OCC-subprocess discipline as analyse_cnc_parts.py (stderr-only logging,
project root on sys.path, 64 MB-stack thread).
"""
import faulthandler
import gc
import json
import sys
import threading
from pathlib import Path

faulthandler.enable()

try:
    import structlog
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr)
    )
except Exception:
    pass

import logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: classify_parts.py <file_path> <analysis_json_path> "
              "<ref_ids_json> [member_ids_json] [steel_grade]", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    analysis_json_path = sys.argv[2]
    ref_ids_json = sys.argv[3]
    member_ids_json = sys.argv[4] if len(sys.argv) > 4 else "{}"
    steel_grade = sys.argv[5] if len(sys.argv) > 5 else ""

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    ref_ids = json.loads(ref_ids_json)
    member_ids = json.loads(member_ids_json)

    result_holder: list = [None]
    error_holder: list = [None]

    # NOTE: this worker deliberately does NOT write the sidecar — see the module
    # docstring. The parent endpoint persists the returned results safely.

    def run() -> None:
        try:
            from app.services.cnc_shape_analyser import classify_ref, _read_xcaf

            print(f"Opening STEP file: {file_path}", file=sys.stderr)
            doc, shape_tool = _read_xcaf(file_path)
            print(f"Classifying {len(ref_ids)} ref_ids", file=sys.stderr)

            results = {}
            for i, ref_id in enumerate(ref_ids):
                print(f"[{i+1}/{len(ref_ids)}] classify {ref_id}", file=sys.stderr)
                try:
                    r = classify_ref(file_path, ref_id, member_ids.get(ref_id),
                                     steel_grade or None,
                                     _doc=doc, _shape_tool=shape_tool)
                except Exception as exc:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    r = {"refined_class": None, "error": str(exc)}
                results[ref_id] = r
                gc.collect()

            result_holder[0] = {"results": results}
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc(file=sys.stderr)
            error_holder[0] = exc

    try:
        threading.stack_size(64 * 1024 * 1024)
    except OSError:
        pass

    t = threading.Thread(target=run)
    t.start()
    t.join()

    if error_holder[0] is not None:
        print(str(error_holder[0]), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result_holder[0]))
    sys.exit(0)


if __name__ == "__main__":
    main()
