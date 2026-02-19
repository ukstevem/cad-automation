"""
Standalone subprocess worker for STL generation.

Run as:
    python generate_stl.py <mode> <file_path> <output_dir> [<node_id>]

Where mode is one of:
    all       - generate STLs for all root-level items (initial view)
    children  - generate STLs for children of <node_id> (on assembly explode)
    solids    - generate STLs for individual solids in <node_id> (on multi-solid explode)

Prints a JSON array of result dicts to stdout on success.
Prints error message to stderr and exits 1 on failure.

Running in a subprocess means OCP/OCC can hold the GIL without blocking the
FastAPI event loop.  The parent awaits asyncio.create_subprocess_exec() so
uvicorn remains fully responsive (serves poll requests, static files, etc.)
while mesh generation runs.
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
    if len(sys.argv) < 4:
        print(
            "Usage: generate_stl.py <mode> <file_path> <output_dir> [<node_id>]",
            file=sys.stderr,
        )
        sys.exit(1)

    mode = sys.argv[1]          # "all" | "children" | "solids"
    file_path = sys.argv[2]
    output_dir = sys.argv[3]
    node_id = sys.argv[4] if len(sys.argv) > 4 else None

    if mode in ("children", "solids") and not node_id:
        print(f"node_id is required for mode '{mode}'", file=sys.stderr)
        sys.exit(1)

    # Ensure the project root is on sys.path so 'app' package is importable.
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    result_holder: list = [None]
    error_holder: list = [None]

    def run() -> None:
        try:
            from app.services.stl_generator import STLGenerator

            gen = STLGenerator(file_path, output_dir)

            if mode == "all":
                result_holder[0] = gen.generate_all()
            elif mode == "children":
                result_holder[0] = gen.generate_children(node_id)
            elif mode == "solids":
                result_holder[0] = gen.generate_solids(node_id)
            else:
                raise ValueError(f"Unknown mode: {mode!r}")
        except Exception as exc:
            error_holder[0] = exc

    # Run in a thread with a larger stack — OCC mesh generation can recurse
    # deeply for complex shapes.
    try:
        threading.stack_size(64 * 1024 * 1024)  # 64 MB
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
