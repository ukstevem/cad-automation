"""
Standalone subprocess worker: extract a part's CAD edges in WORLD coordinates.

Run as:
    python extract_edges.py <file_path> <analysis_json_path> [node_id]

Opens the STEP via XCAF, extracts solids (optionally scoped to *node_id*) in world
coords using the same machinery as connection detection, samples polylines along
every edge, and prints a JSON payload to stdout:

    {
      "edges":    [ [[x,y,z], ...], ... ],   # world-coord polylines (mm)
      "vertices": [ [x,y,z], ... ],          # unique edge endpoints (corner anchors)
      "bbox":     {"min": [x,y,z], "max": [x,y,z]},
      "summary":  {"solids": n, "edges": m, "vertices": v}
    }

These edges/vertices are in the SAME world frame as the weld paths and world-space
STLs, so a camera pose solved against them projects everything consistently. Used by
the AR alignment overlay (Phase 0 spike).

Prints an error to stderr and exits 1 on failure.
"""
import faulthandler
import json
import sys
import threading
from pathlib import Path

import numpy as np

faulthandler.enable()

try:
    import structlog
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
except Exception:
    pass

import logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def _round_key(pt, q=0.5):
    """Quantise a point for dedup (q mm grid)."""
    return (round(pt[0] / q) * q, round(pt[1] / q) * q, round(pt[2] / q) * q)


def _strip_synthetic_solids(nodes):
    """
    Drop the display-only solid children the frontend embeds on multi-solid parts.

    Those nodes (node_type == "solid", ids like "0:1:1:2:s3") are NOT real XCAF labels —
    they exist only so the UI can show a part's solids. If the extractor recurses into
    them it resolves null labels and finds 0 solids, silently dropping every weldment.
    Removing them makes the part node a leaf, so its own XCAF label is meshed (which
    explodes the real solids via TopExp_Explorer in _extract_leaf).
    """
    for n in nodes:
        kids = [c for c in (n.get("children") or []) if c.get("node_type") != "solid"]
        n["children"] = kids
        _strip_synthetic_solids(kids)
    return nodes


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: extract_edges.py <file_path> <analysis_json_path> [node_id]", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    analysis_json_path = sys.argv[2]
    node_id = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    result_holder: list = [None]
    error_holder: list = [None]

    def run() -> None:
        try:
            sidecar = json.loads(Path(analysis_json_path).read_text(encoding="utf-8"))
            analysis_section = sidecar.get("analysis", sidecar)
            tree_nodes = analysis_section.get("assembly_tree", [])

            project_state = sidecar.get("project_state", {})
            classifications = project_state.get("classifications", {}) or {}

            if node_id:
                from app.workers.detect_connections import _find_subtree
                subtree = _find_subtree(tree_nodes, node_id)
                if subtree is None:
                    raise ValueError(f"node_id {node_id!r} not found in assembly tree")
                tree_nodes = subtree

            # Remove display-only synthetic solid children so multi-solid weldments
            # (e.g. Main Frame_Default) are meshed from their own XCAF label.
            _strip_synthetic_solids(tree_nodes)

            from app.services.cnc_shape_analyser import _read_xcaf
            from app.pipeline.connection_detector import (
                extract_all_solids, _extract_edge_polylines,
            )

            doc, shape_tool = _read_xcaf(file_path)
            solids, _bo = extract_all_solids(doc, shape_tool, tree_nodes, classifications)

            edges: list = []
            for s in solids:
                edges.extend(_extract_edge_polylines(s.shape, n_samples=20))

            # Unique corner anchors = deduped edge endpoints.
            seen = set()
            vertices: list = []
            for poly in edges:
                for p in (poly[0], poly[-1]):
                    k = _round_key(p)
                    if k not in seen:
                        seen.add(k)
                        vertices.append([round(p[0], 2), round(p[1], 2), round(p[2], 2)])

            # World-space AABB from all sampled points.
            xs = [p[0] for poly in edges for p in poly]
            ys = [p[1] for poly in edges for p in poly]
            zs = [p[2] for poly in edges for p in poly]
            bbox = (
                {"min": [round(min(xs), 2), round(min(ys), 2), round(min(zs), 2)],
                 "max": [round(max(xs), 2), round(max(ys), 2), round(max(zs), 2)]}
                if xs else {"min": [0, 0, 0], "max": [0, 0, 0]}
            )

            # Oriented bounding box (PCA) — true element L×W×H, robust to the part
            # being rotated in world space (unlike the axis-aligned bbox). Drives
            # marker sizing: a marker can't be bigger than the face it sits on.
            obb_dims = []
            if xs and len(xs) >= 3:
                pts_all = np.array([p for poly in edges for p in poly], dtype=float)
                c = pts_all.mean(axis=0)
                X = pts_all - c
                _evals, evecs = np.linalg.eigh(np.cov(X.T))
                proj = X @ evecs
                ext = proj.max(axis=0) - proj.min(axis=0)
                obb_dims = sorted(round(float(e), 1) for e in ext)  # [min, mid, max]

            result_holder[0] = {
                "edges": edges,
                "vertices": vertices,
                "bbox": bbox,
                "obb": obb_dims,
                "summary": {
                    "solids": len(solids), "edges": len(edges),
                    "vertices": len(vertices), "obb": obb_dims,
                },
            }
        except Exception as exc:  # noqa: BLE001
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
