"""
Parse a polyface-mesh DXF and produce a BOM by grouping identical bodies.
"""
import ezdxf
import numpy as np
from collections import defaultdict
from pathlib import Path
import csv, json, sys

DXF_PATH = Path(__file__).parent / "G8702-P5-001 rev 03 3D.dxf"
DIM_TOL = 1.0        # mm tolerance for matching dimensions
VOL_TOL_PCT = 0.05   # 5 % volume tolerance for matching


def mesh_volume_and_area(vertices: np.ndarray, faces: list[tuple]) -> tuple[float, float]:
    """
    Compute approximate volume (signed tetrahedra method) and surface area
    from mesh vertices and face index tuples.
    """
    volume = 0.0
    area = 0.0
    for face_indices in faces:
        pts = vertices[np.array(face_indices)]
        # Triangulate the face (fan from first vertex)
        for i in range(1, len(pts) - 1):
            v0, v1, v2 = pts[0], pts[i], pts[i + 1]
            # Signed volume of tetrahedron with origin
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
            # Triangle area
            area += np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
    return abs(volume), area


def extract_body(entity) -> dict | None:
    """Extract geometry info from a polyface mesh entity."""
    if not entity.is_poly_face_mesh:
        return None

    verts = list(entity.vertices)
    mesh_verts = [v for v in verts if v.is_poly_face_mesh_vertex]
    face_recs = [v for v in verts if v.is_face_record]

    if not mesh_verts or not face_recs:
        return None

    # Build vertex array (1-indexed in DXF face records)
    pts = np.array([v.dxf.location.xyz for v in mesh_verts])

    # Build face list (face records use 1-based indices, 0 means unused)
    faces = []
    for fr in face_recs:
        idx = []
        for attr in ('vtx0', 'vtx1', 'vtx2', 'vtx3'):
            val = getattr(fr.dxf, attr, 0)
            if val != 0:
                idx.append(abs(val) - 1)  # to 0-based
        if len(idx) >= 3:
            faces.append(tuple(idx))

    # Bounding box
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    dims = bbox_max - bbox_min

    # Sort dimensions descending (length, width, thickness)
    sorted_dims = sorted(dims, reverse=True)

    # Volume and surface area
    vol, area = mesh_volume_and_area(pts, faces)

    return {
        'layer': entity.dxf.layer,
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'dims_raw': dims.tolist(),
        'dims_sorted': sorted_dims,
        'volume': vol,
        'surface_area': area,
        'n_vertices': len(mesh_verts),
        'n_faces': len(face_recs),
    }


def fingerprint(body: dict) -> tuple:
    """Create a grouping fingerprint from sorted dims + volume."""
    d = body['dims_sorted']
    return (
        round(d[0] / DIM_TOL) * DIM_TOL,
        round(d[1] / DIM_TOL) * DIM_TOL,
        round(d[2] / DIM_TOL) * DIM_TOL,
        round(body['volume'] / max(body['volume'] * VOL_TOL_PCT, 1.0)),
        body['n_faces'],  # face count as extra discriminator
    )


def main():
    print(f"Reading {DXF_PATH.name} ...")
    doc = ezdxf.readfile(str(DXF_PATH))
    msp = doc.modelspace()

    bodies = []
    for entity in msp:
        if entity.dxftype() == 'POLYLINE':
            body = extract_body(entity)
            if body:
                bodies.append(body)

    print(f"Extracted {len(bodies)} bodies\n")

    # Group by fingerprint
    groups = defaultdict(list)
    for b in bodies:
        fp = fingerprint(b)
        groups[fp].append(b)

    # Sort groups by count descending
    sorted_groups = sorted(groups.values(), key=lambda g: len(g), reverse=True)

    # Print BOM
    print(f"{'#':<4} {'Qty':<5} {'Length':>10} {'Width':>10} {'Thick':>10} {'Volume':>14} {'Layers':<15}")
    print("-" * 75)

    bom_rows = []
    for i, group in enumerate(sorted_groups, 1):
        rep = group[0]
        d = rep['dims_sorted']
        layers = sorted(set(b['layer'] for b in group))
        row = {
            'item': i,
            'qty': len(group),
            'length': round(d[0], 1),
            'width': round(d[1], 1),
            'thickness': round(d[2], 1),
            'volume': round(rep['volume'], 1),
            'surface_area': round(rep['surface_area'], 1),
            'layers': ','.join(layers),
        }
        bom_rows.append(row)
        print(f"{i:<4} {len(group):<5} {d[0]:>10.1f} {d[1]:>10.1f} {d[2]:>10.1f} {rep['volume']:>14.1f} {','.join(layers):<15}")

    print(f"\n{len(sorted_groups)} unique parts, {len(bodies)} total bodies")

    # Write CSV
    csv_path = DXF_PATH.with_suffix('.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=bom_rows[0].keys())
        writer.writeheader()
        writer.writerows(bom_rows)
    print(f"\nBOM written to {csv_path}")

    # Write JSON
    json_path = DXF_PATH.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({'file': DXF_PATH.name, 'total_bodies': len(bodies),
                    'unique_parts': len(sorted_groups), 'bom': bom_rows}, f, indent=2)
    print(f"JSON written to {json_path}")


if __name__ == '__main__':
    main()
