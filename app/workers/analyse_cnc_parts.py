"""
Standalone subprocess worker for CNC shape analysis.

Run as:
    python analyse_cnc_parts.py <file_path> <analysis_json_path> \\
                                <ref_ids_json> <out_dir> [member_ids_json]

For each ref_id in ref_ids_json, opens the STEP file via XCAF, runs the
full plate/section detection pipeline, and emits the results as JSON.

Prints a single JSON object {"results": {ref_id: result_dict, ...}} to stdout
on success.  Prints an error to stderr and exits 1 on failure.

Running in a subprocess means OCP/OCC can hold the GIL without blocking the
FastAPI event loop.  The parent awaits asyncio.create_subprocess_exec() so
uvicorn remains fully responsive while geometry analysis runs.
"""
import faulthandler
import json
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
    if len(sys.argv) < 5:
        print(
            "Usage: analyse_cnc_parts.py <file_path> <analysis_json_path>"
            " <ref_ids_json> <out_dir> [member_ids_json]",
            file=sys.stderr,
        )
        sys.exit(1)

    file_path = sys.argv[1]
    analysis_json_path = sys.argv[2]
    ref_ids_json = sys.argv[3]
    out_dir = Path(sys.argv[4])
    member_ids_json = sys.argv[5] if len(sys.argv) > 5 else "{}"
    parent_names_json = sys.argv[6] if len(sys.argv) > 6 else "{}"
    project_number = sys.argv[7] if len(sys.argv) > 7 else ""
    steel_grade = sys.argv[8] if len(sys.argv) > 8 else ""

    # Ensure the project root is on sys.path so 'app' package is importable.
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    ref_ids = json.loads(ref_ids_json)
    member_ids = json.loads(member_ids_json)
    parent_names = json.loads(parent_names_json)

    result_holder: list = [None]
    error_holder: list = [None]

    def run() -> None:
        try:
            from app.services.cnc_shape_analyser import analyse_ref

            results = {}
            for ref_id in ref_ids:
                member_id = member_ids.get(ref_id)
                parent_name = parent_names.get(ref_id)
                try:
                    results[ref_id] = analyse_ref(file_path, ref_id, out_dir,
                                                  member_id, parent_name,
                                                  project_number or None,
                                                  steel_grade or None)
                except Exception as exc:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    results[ref_id] = {"type": "unknown", "message": str(exc)}

            result_holder[0] = {"results": results}
        except Exception as exc:
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
