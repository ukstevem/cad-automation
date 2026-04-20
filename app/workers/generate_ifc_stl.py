"""Standalone subprocess worker for IFC STL generation.

Run as:
    python generate_ifc_stl.py <mode> <file_path> <output_dir> [<node_id>]

Only ``mode=all`` is supported in v1 — it tessellates every product in the
IFC that has geometry and writes one binary STL per leaf into ``output_dir``.
The output matches the STEP worker's contract (list of ``{node_id, name,
stl_file, size_bytes}``) so the frontend needs no changes.

The ifcopenshell 0.8.x wheel on PyPI does NOT include PyOpenCASCADE bindings,
so we use the triangulation path (``verts`` + ``faces`` flat arrays) and write
binary STL via ``trimesh`` (already a project dependency).  World-space verts
are requested from ifcopenshell directly via the ``use-world-coords`` setting
so the output needs no post-transform.

Node IDs for IFC are synthesised at parse time (e.g. ``0:0:0:0:3:12``) — they
are NOT IFC GlobalIds.  We read the cached analysis sidecar to build a
GUID → (node_id, name) map, then match products to nodes via GlobalId.
"""
import faulthandler
import json
import re
import sys
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


def _safe_filename_part(raw: str, max_len: int = 64) -> str:
    """Strip characters that are illegal / awkward in filenames."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(raw))
    s = s.strip().strip(".")
    return (s or "unnamed")[:max_len]


def _collect_guid_to_node(tree):
    """Walk the cached assembly tree and return {guid: (node_id, name)}.

    Only collects leaves that carry ``ifc_metadata.guid`` and have geometry.
    """
    out: dict = {}

    def walk(nodes):
        for n in nodes:
            meta = n.get("ifc_metadata") or {}
            guid = meta.get("guid")
            if guid and n.get("solid_count", 0) > 0:
                out[guid] = (n["id"], n.get("name") or "part")
            children = n.get("children")
            if children:
                walk(children)

    walk(tree)
    return out


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: generate_ifc_stl.py <mode> <file_path> <output_dir> [<node_id>]",
            file=sys.stderr,
        )
        sys.exit(1)

    mode = sys.argv[1]
    file_path = sys.argv[2]
    output_dir = Path(sys.argv[3])
    # node_id arg is accepted for contract parity but ignored in v1

    if mode != "all":
        print(
            f"IFC STL generation mode {mode!r} is not supported yet. "
            "v1 only generates all leaves on first analysis.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Put project root on sys.path so 'app.*' imports resolve
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from app.config import settings  # noqa: E402

    # Load the cached analysis JSON for GUID → node_id mapping
    upload_path = Path(file_path)
    sidecar_path = Path(settings.ANALYSIS_OUTPUT_DIR) / f"{upload_path.stem}.json"
    if not sidecar_path.exists():
        print(
            f"Analysis sidecar not found at {sidecar_path}; "
            "IFC STL generation requires the assembly tree to have been cached first.",
            file=sys.stderr,
        )
        sys.exit(1)

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    tree = (sidecar.get("analysis") or {}).get("assembly_tree") or []
    guid_to_node = _collect_guid_to_node(tree)
    if not guid_to_node:
        print("No IFC leaves with GUIDs + geometry in cached tree", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import trimesh
    import ifcopenshell
    import ifcopenshell.geom as ifc_geom

    ifc_file = ifcopenshell.open(str(upload_path))
    geom_settings = ifc_geom.settings()
    for key, value in (
        ("weld-vertices", True),
        ("use-world-coords", True),
    ):
        try:
            geom_settings.set(key, value)
        except Exception:
            pass

    results: list = []

    # Iterate products of interest (the GUIDs we already know belong to leaves).
    # by_guid() is O(1) after the model is indexed, so this is much cheaper
    # than walking every product and filtering.
    for guid, (node_id, name) in guid_to_node.items():
        try:
            product = ifc_file.by_guid(guid)
        except Exception:
            continue
        if product is None or not getattr(product, "Representation", None):
            continue

        try:
            shape = ifc_geom.create_shape(geom_settings, product)
        except Exception as exc:
            print(
                f"[tessellate-failed] guid={guid} entity={product.is_a()} error={exc}",
                file=sys.stderr,
            )
            results.append({
                "name": name,
                "node_id": node_id,
                "stl_file": None,
                "size_bytes": 0,
                "error": str(exc),
            })
            continue

        geom_obj = getattr(shape, "geometry", None)
        verts_flat = list(getattr(geom_obj, "verts", []) or [])
        faces_flat = list(getattr(geom_obj, "faces", []) or [])
        if len(verts_flat) < 9 or len(faces_flat) < 3:
            continue

        verts = np.asarray(verts_flat, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces_flat, dtype=np.int64).reshape(-1, 3)

        stl_filename = (
            f"{_safe_filename_part(name)}_{node_id.replace(':', '_')}.stl"
        )
        stl_path = output_dir / stl_filename

        # process=False skips cleanup passes (merge verts, fix winding) that
        # are nice-to-have but slow.  ifcopenshell's output is already welded.
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.export(str(stl_path), file_type="stl")

        results.append({
            "name": name,
            "node_id": node_id,
            "stl_file": stl_filename,
            "size_bytes": stl_path.stat().st_size,
        })

    print(json.dumps(results))
    sys.exit(0)


if __name__ == "__main__":
    main()
