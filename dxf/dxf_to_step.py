"""
Convert a polyface-mesh DXF into a multi-body STEP file.

Each polyface mesh becomes a named solid in the STEP output,
preserving layer info in the part name. Uses XCAF so the
existing CAD automation pipeline can read it with full names.

Usage (runs inside Docker container):
    python3 dxf_to_step.py /app/dxf/input.dxf /app/dxf/output.step
"""
import sys
import json
import struct
import tempfile
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# DXF parsing (ezdxf runs on the HOST because it's installed there)
# This script is designed to run in two stages:
#   Stage 1 (host):  parse DXF → intermediate JSON with mesh data
#   Stage 2 (docker): read JSON → build STEP via OpenCASCADE
#
# OR if ezdxf is available in Docker, it can run as a single script.
# ---------------------------------------------------------------------------


def _extract_polyface_meshes(entities, meshes: list, idx_ref: list):
    """Extract polyface meshes from an iterable of DXF entities."""
    for entity in entities:
        if entity.dxftype() != 'POLYLINE' or not entity.is_poly_face_mesh:
            continue

        verts = list(entity.vertices)
        mesh_verts = [v for v in verts if v.is_poly_face_mesh_vertex]
        face_recs = [v for v in verts if v.is_face_record]

        if not mesh_verts or not face_recs:
            continue

        pts = [[v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]
               for v in mesh_verts]

        faces = []
        for fr in face_recs:
            f = []
            for attr in ('vtx0', 'vtx1', 'vtx2', 'vtx3'):
                val = getattr(fr.dxf, attr, 0)
                if val != 0:
                    f.append(abs(val) - 1)  # 0-based
            if len(f) >= 3:
                faces.append(f)

        layer = entity.dxf.layer
        meshes.append({
            'index': idx_ref[0],
            'layer': layer,
            'vertices': pts,
            'faces': faces,
        })
        idx_ref[0] += 1


def parse_dxf_to_meshes(dxf_path: str) -> list[dict]:
    """Parse DXF and extract polyface meshes as vertex/face arrays.

    Searches both modelspace directly and inside block definitions
    referenced by INSERT entities (handles files where geometry lives
    in blocks rather than directly in modelspace).
    """
    import ezdxf

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    meshes = []
    idx_ref = [0]  # mutable counter shared across calls

    # Pass 1: modelspace directly
    _extract_polyface_meshes(msp, meshes, idx_ref)

    # Pass 2: recurse into block definitions referenced by INSERT entities
    if not meshes:
        visited_blocks = set()
        for entity in msp:
            if entity.dxftype() != 'INSERT':
                continue
            bname = entity.dxf.name
            if bname in visited_blocks:
                continue
            visited_blocks.add(bname)
            blk = doc.blocks.get(bname)
            if blk:
                _extract_polyface_meshes(blk, meshes, idx_ref)

    return meshes


def meshes_to_step(meshes: list[dict], step_path: str):
    """Build a multi-body STEP file from mesh data using XCAF."""
    from OCP.gp import gp_Pnt
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCP.TopoDS import TopoDS_Compound, TopoDS_Shell, TopoDS_Solid
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.Poly import Poly_Triangle, Poly_Triangulation
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.Interface import Interface_Static

    # Set STEP write precision
    Interface_Static.SetCVal_s("write.step.schema", "AP214")

    # Create XCAF document
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    # Group meshes by layer for assembly structure
    by_layer = defaultdict(list)
    for m in meshes:
        by_layer[m['layer']].append(m)

    built = 0
    failed = 0

    for layer, layer_meshes in sorted(by_layer.items()):
        for mesh in layer_meshes:
            verts = mesh['vertices']
            faces = mesh['faces']

            # Sew triangles into a shell, then solidify
            sewing = BRepBuilderAPI_Sewing(1.0)  # 1mm tolerance

            for face in faces:
                # Triangulate quads into triangles
                triangles = []
                if len(face) == 3:
                    triangles.append(face)
                elif len(face) == 4:
                    triangles.append([face[0], face[1], face[2]])
                    triangles.append([face[0], face[2], face[3]])
                else:
                    # Fan triangulation for polygons
                    for i in range(1, len(face) - 1):
                        triangles.append([face[0], face[i], face[i + 1]])

                for tri in triangles:
                    p0 = gp_Pnt(*verts[tri[0]])
                    p1 = gp_Pnt(*verts[tri[1]])
                    p2 = gp_Pnt(*verts[tri[2]])

                    # Skip degenerate triangles
                    if p0.Distance(p1) < 1e-6 or p1.Distance(p2) < 1e-6 or p0.Distance(p2) < 1e-6:
                        continue

                    try:
                        wire = BRepBuilderAPI_MakePolygon(p0, p1, p2, True).Wire()
                        face_shape = BRepBuilderAPI_MakeFace(wire, True).Face()
                        sewing.Add(face_shape)
                    except Exception:
                        continue

            sewing.Perform()
            sewn = sewing.SewedShape()

            # Try to make a solid from the sewn shell
            builder = BRep_Builder()
            solid = TopoDS_Solid()
            try:
                builder.MakeSolid(solid)
                builder.Add(solid, sewn)
            except Exception:
                # If solidification fails, use the sewn shape directly
                solid = sewn

            # Add to XCAF with a name
            label = shape_tool.AddShape(solid)
            name = f"L{layer}_Body_{mesh['index']:03d}"
            TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
            built += 1

    print(f"Built {built} bodies ({failed} failed)")

    # Write STEP
    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, STEPControl_AsIs)
    status = writer.Write(step_path)
    if status == 1:  # IFSelect_RetDone
        print(f"STEP written to {step_path}")
    else:
        print(f"STEP write failed with status {status}")

    return built


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 dxf_to_step.py <input.dxf> <output.step>")
        sys.exit(1)

    dxf_path = sys.argv[1]
    step_path = sys.argv[2]

    print(f"Parsing DXF: {dxf_path}")
    meshes = parse_dxf_to_meshes(dxf_path)
    print(f"Found {len(meshes)} polyface meshes")

    if not meshes:
        print("No polyface meshes found!")
        sys.exit(1)

    # Write intermediate JSON so Docker container can read it
    json_path = step_path + ".meshes.json"
    with open(json_path, 'w') as f:
        json.dump(meshes, f)
    print(f"Mesh data written to {json_path}")

    print(f"Building STEP file...")
    meshes_to_step(meshes, step_path)


if __name__ == '__main__':
    main()
