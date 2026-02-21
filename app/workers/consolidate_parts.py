"""
Standalone subprocess worker for parts consolidation.

Run as:
    python consolidate_parts.py <file_path> <analysis_json_path>

Reads the cached assembly tree from ``analysis_json_path``, opens the STEP
file via XCAF to compute shape fingerprints for every unique prototype, then
groups geometrically identical prototypes into consolidation groups.

Prints a JSON array of group dicts to stdout on success.
Prints an error message to stderr and exits 1 on failure.

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
    if len(sys.argv) != 3:
        print(
            "Usage: consolidate_parts.py <file_path> <analysis_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    file_path = sys.argv[1]
    analysis_json_path = sys.argv[2]

    # Ensure the project root is on sys.path so 'app' package is importable.
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    result_holder: list = [None]
    error_holder: list = [None]

    def run() -> None:
        try:
            sidecar = json.loads(
                Path(analysis_json_path).read_text(encoding="utf-8")
            )
            # The sidecar JSON has {"analysis": {...}, "project_state": {...}}
            # We need the "analysis" section which contains the assembly_tree.
            analysis_section = sidecar.get("analysis", sidecar)

            from app.services.parts_consolidator import PartsConsolidator

            consolidator = PartsConsolidator(file_path, analysis_section)
            result_holder[0] = consolidator.consolidate()
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
