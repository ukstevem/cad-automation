"""
Standalone subprocess worker for connection detection.

Run as:
    python detect_connections.py <file_path> <analysis_json_path> [node_id] [scope]

Opens the STEP file via XCAF, extracts all solids from the assembly tree
(optionally scoped to *node_id*), detects weld/bolt connections between
them, and prints a JSON result to stdout.

*scope* is one of ``all`` (default), ``siblings``, ``within-part``.

Prints an error message to stderr and exits 1 on failure.
"""
import faulthandler
import json
import sys
import threading
from pathlib import Path

faulthandler.enable()

# Redirect structlog and stdlib logging to stderr so they never contaminate
# the JSON written to stdout.
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
    if len(sys.argv) < 3:
        print(
            "Usage: detect_connections.py <file_path> <analysis_json_path> [node_id]",
            file=sys.stderr,
        )
        sys.exit(1)

    file_path = sys.argv[1]
    analysis_json_path = sys.argv[2]
    node_id = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    scope = sys.argv[4] if len(sys.argv) > 4 else "all"

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
            analysis_section = sidecar.get("analysis", sidecar)
            tree_nodes = analysis_section.get("assembly_tree", [])

            # Load bought-out classifications from project_state
            project_state = sidecar.get("project_state", {})
            classifications = project_state.get("classifications", {}) or {}

            # If a node_id scope was given, find that subtree
            if node_id:
                subtree = _find_subtree(tree_nodes, node_id)
                if subtree is None:
                    raise ValueError(
                        f"node_id {node_id!r} not found in assembly tree"
                    )
                tree_nodes = subtree

            # Load XCAF document
            from app.services.cnc_shape_analyser import _read_xcaf

            doc, shape_tool = _read_xcaf(file_path)

            if scope == "cross-assembly":
                from app.pipeline.connection_detector import (
                    detect_cross_assembly,
                )
                result_holder[0] = detect_cross_assembly(
                    doc, shape_tool, tree_nodes,
                    classifications=classifications,
                )
            else:
                from app.pipeline.connection_detector import detect_connections
                try:
                    result_holder[0] = detect_connections(
                        doc, shape_tool, tree_nodes, scope=scope,
                        classifications=classifications,
                    )
                except ValueError as exc:
                    if "Too many solids" in str(exc):
                        # Auto-fallback: assembly exceeds MAX_SOLIDS, use
                        # cross-assembly detection (pairwise boundary approach)
                        import logging as _log
                        _log.warning(
                            "Auto-fallback to cross-assembly detection: %s", exc,
                        )
                        from app.pipeline.connection_detector import (
                            detect_cross_assembly,
                        )
                        result = detect_cross_assembly(
                            doc, shape_tool, tree_nodes,
                            classifications=classifications,
                        )
                        result["auto_fallback"] = True
                        result["fallback_reason"] = str(exc)
                        result_holder[0] = result
                    else:
                        raise
        except Exception as exc:
            error_holder[0] = exc

    # 64 MB stack — OCC can be stack-intensive
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


def _find_subtree(nodes: list, target_id: str) -> list | None:
    """Return the children list rooted at *target_id*, or ``None``."""
    for node in nodes:
        if node.get("id") == target_id:
            children = node.get("children", [])
            # Return the node itself wrapped in a list so the detector
            # can iterate its children (or process it as a leaf).
            return [node]
        children = node.get("children", [])
        if children:
            found = _find_subtree(children, target_id)
            if found is not None:
                return found
    return None


if __name__ == "__main__":
    main()
