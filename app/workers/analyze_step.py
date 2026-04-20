"""
Standalone subprocess worker for XCAF assembly analysis.

Run as:
    python analyze_step.py <file_path>

Prints JSON result to stdout on success.
Prints error message to stderr and exits 1 on failure.

Running analysis in a subprocess (rather than a thread) means OpenCascade/OCP
can hold the GIL without blocking the FastAPI event loop.  The parent process
calls subprocess.run() which waits via OS-level I/O (releasing the GIL), so
uvicorn remains fully responsive during the parse.

Stack size note:
    OCC's STEPCAFControl_Reader.Transfer() recurses deeply for complex assemblies.
    The default 8 MB thread stack can overflow (SIGSEGV on the guard page) for
    large files.  We run the analysis in a new thread with a 64 MB stack.

faulthandler note:
    faulthandler.enable() intercepts SIGSEGV/SIGABRT and writes a Python/C
    traceback to stderr before the process exits.  This makes crash diagnosis
    possible when OCC's C++ code segfaults.
"""
import faulthandler
import json
import sys
import threading
from pathlib import Path

# Dump C-level traceback to stderr on any fatal signal (SIGSEGV, SIGABRT …)
faulthandler.enable()

# Redirect structlog (and stdlib logging) output to stderr so that diagnostic
# messages never contaminate the JSON written to stdout.  The parent process
# captures stdout and parses it as JSON; any non-JSON bytes there cause a
# parse failure.  This must be done before importing any 'app.*' module that
# calls structlog.get_logger() at module level.
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
    if len(sys.argv) != 2:
        print("Usage: analyze_step.py <file_path>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]

    # Ensure the project root (/app in Docker) is on the path so we can
    # import from the 'app' package.
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    result_holder: list = [None]
    error_holder: list = [None]

    def run() -> None:
        try:
            from app.parsers import get_assembly_parser

            analyzer = get_assembly_parser(file_path)
            result_holder[0] = analyzer.analyze()
        except Exception as exc:
            error_holder[0] = exc

    # Run XCAF analysis in a dedicated thread with a large stack.
    # OCC's XCAF reader can recurse deeply for complex assemblies; the default
    # 8 MB stack is sometimes insufficient, causing SIGSEGV on the guard page.
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
