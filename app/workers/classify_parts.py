"""
Standalone subprocess worker for LIGHTWEIGHT part classification (rw7.10).

Run as:
    python classify_parts.py <file_path> <analysis_json_path> \\
                             <ref_ids_json> [member_ids_json] [steel_grade]

For each ref_id, computes the refined class (features + library match + decision
tree) via ``classify_ref`` — WITHOUT generating any NC1/DXF.  This is the cheap
pass that assists the top-down triage before the heavy CNC analysis runs.

Prints ``{"results": {ref_id: {refined_class, refined_confidence, ...}}}`` to
stdout.  Results are also saved progressively into the sidecar's
``classification`` key so partial progress survives a timeout/crash.

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

    def _save_partial(ref_id: str, result: dict) -> None:
        path = Path(analysis_json_path)
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (json.JSONDecodeError, OSError):
            existing = {}
        existing.setdefault("classification", {})[ref_id] = result
        try:
            path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"WARNING: could not write classification for {ref_id}: {e}", file=sys.stderr)

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
                try:
                    _save_partial(ref_id, r)
                except Exception as save_exc:
                    print(f"WARNING: _save_partial failed for {ref_id}: {save_exc}",
                          file=sys.stderr)

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
