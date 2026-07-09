"""
Standalone subprocess worker for CNC shape analysis.

Run as:
    python analyse_cnc_parts.py <file_path> <analysis_json_path> \\
                                <ref_ids_json> <out_dir> \\
                                [member_ids_json] [parent_names_json] \\
                                [project_number] [steel_grade] \\
                                [instance_counts_json]

For each ref_id in ref_ids_json, opens the STEP file via XCAF, runs the
full plate/section detection pipeline, and emits the results as JSON.

Prints a single JSON object {"results": {ref_id: result_dict, ...}} to stdout
on success.  Prints an error to stderr and exits 1 on failure.

Running in a subprocess means OCP/OCC can hold the GIL without blocking the
FastAPI event loop.  The parent awaits asyncio.create_subprocess_exec() so
uvicorn remains fully responsive while geometry analysis runs.
"""
import faulthandler
import gc
import json
import os
import sys
import threading
from pathlib import Path

# Dump C-level traceback on fatal signals (SIGSEGV, SIGABRT, …)
faulthandler.enable()

# Redirect structlog and stdlib logging to stderr so they never contaminate
# the JSON written to stdout.  Must be done before any 'app.*' import.
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
        print(
            "Usage: analyse_cnc_parts.py <file_path> <analysis_json_path>"
            " <out_dir> [project_number] [steel_grade]"
            "  (ref_ids/member_ids/parent_names/instance_counts JSON on stdin)",
            file=sys.stderr,
        )
        sys.exit(1)

    file_path = sys.argv[1]
    analysis_json_path = sys.argv[2]
    out_dir = Path(sys.argv[3])
    project_number = sys.argv[4] if len(sys.argv) > 4 else ""
    steel_grade = sys.argv[5] if len(sys.argv) > 5 else ""

    # Ensure the project root is on sys.path so 'app' package is importable.
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Large JSON payloads arrive on stdin — passing a few thousand refs via argv
    # exceeds ARG_MAX ("[Errno 7] Argument list too long") on Linux.
    _stdin = sys.stdin.read()
    _payload = json.loads(_stdin) if _stdin.strip() else {}
    ref_ids = _payload.get("ref_ids", [])
    member_ids = _payload.get("member_ids", {})
    parent_names = _payload.get("parent_names", {})
    instance_counts = _payload.get("instance_counts", {}) or {}

    result_holder: list = [None]
    error_holder: list = [None]

    def _save_partial(ref_id: str, result: dict) -> None:
        """Write a single ref_id result into the sidecar JSON immediately.

        Progressive saving ensures partial progress survives a timeout or crash.
        Only the current ref_id is updated; all other sidecar sections are
        preserved (assembly analysis, project state, consolidation, etc.).

        Safety (run c2c2ac88, 2026-06-18): NEVER fall back to ``{}`` on a parse
        error — that rewrites the whole file with only ``cnc_analysis`` and
        destroys ``analysis``/``project_state``. On any unreadable sidecar we
        skip this save (refuse to clobber). Writes are atomic (temp + replace).
        """
        path = Path(analysis_json_path)
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except json.JSONDecodeError as e:
            print(f"WARNING: sidecar unparseable — skipping partial save for {ref_id}: {e}",
                  file=sys.stderr)
            return
        except OSError as e:
            print(f"WARNING: sidecar read failed — skipping partial save for {ref_id}: {e}",
                  file=sys.stderr)
            return
        existing.setdefault("cnc_analysis", {})[ref_id] = result
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            print(f"WARNING: could not write partial result for {ref_id}: {e}", file=sys.stderr)

    def run() -> None:
        try:
            from app.services.cnc_shape_analyser import analyse_ref, _read_xcaf

            # Open the STEP file ONCE and reuse the document for every ref_id.
            # Re-opening for each ref_id causes OCC to accumulate memory until
            # it segfaults (exit -11) on large assemblies with 100+ parts.
            print(f"Opening STEP file: {file_path}", file=sys.stderr)
            doc, shape_tool = _read_xcaf(file_path)
            print(f"STEP file loaded, processing {len(ref_ids)} ref_ids", file=sys.stderr)

            results = {}
            for i, ref_id in enumerate(ref_ids):
                member_id = member_ids.get(ref_id)
                parent_name = parent_names.get(ref_id)
                print(f"[{i+1}/{len(ref_ids)}] Analysing {ref_id}", file=sys.stderr)
                try:
                    inst_count = int(instance_counts.get(ref_id, 1) or 1)
                    result = analyse_ref(file_path, ref_id, out_dir,
                                        member_id, parent_name,
                                        project_number or None,
                                        steel_grade or None,
                                        inst_count,
                                        _doc=doc,
                                        _shape_tool=shape_tool)
                except Exception as exc:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    result = {"type": "unknown", "message": str(exc)}

                results[ref_id] = result
                # Force GC to release OCC shape objects before JSON serialization.
                # Without this, OCC's internal allocator can corrupt heap memory
                # that manifests as a SIGSEGV inside json.dumps on large assemblies.
                gc.collect()
                # Save immediately so a timeout only loses the part currently
                # being processed, not all previously completed parts.
                try:
                    _save_partial(ref_id, result)
                except Exception as save_exc:
                    import traceback as _tb
                    print(f"WARNING: _save_partial failed for {ref_id}: {save_exc}", file=sys.stderr)
                    _tb.print_exc(file=sys.stderr)

            result_holder[0] = {"results": results}
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc(file=sys.stderr)
            error_holder[0] = exc

    # Run in a thread with a larger stack — OCC XCAF reader can recurse deeply
    # for complex assemblies, and BRepGProp can be stack-intensive too.
    try:
        threading.stack_size(64 * 1024 * 1024)  # 64 MB
    except OSError:
        pass  # Platform may not support custom stack size; proceed anyway

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
