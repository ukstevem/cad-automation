"""
Generate a PNG thumbnail of a STEP file's full top-level assembly.

Uses OCC tessellation (BRepMesh_IncrementalMesh) at a coarse deflection to keep
triangle count manageable, then renders a shaded isometric view with matplotlib's
non-interactive Agg backend (no display required).

Typical cost: 1–15 seconds depending on assembly complexity.
Result is cached on disk; subsequent calls return immediately.
"""
import threading
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Triangle budget — kept deliberately low; the thumbnail only needs to convey
# the overall size and shape of the assembly, not any surface detail.
_MAX_TRIS = 20_000


def generate_thumbnail(file_path: Path, out_path: Path) -> bool:
    """
    Tessellate the root assembly and render a shaded PNG thumbnail.

    Runs OCC work on a dedicated thread with a larger stack (OCC XCAF reader
    can recurse deeply on complex assemblies).

    Returns True on success, False on any failure.
    """
    result: list = [False]

    def _run() -> None:
        try:
            result[0] = _do_generate(file_path, out_path)
        except Exception as exc:
            import traceback
            logger.error(
                "thumbnail_error",
                file=str(file_path),
                error=str(exc),
                traceback=traceback.format_exc(),
            )

    try:
        threading.stack_size(32 * 1024 * 1024)  # 32 MB
    except OSError:
        pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=120)  # 2-minute ceiling
    return result[0]


def _do_generate(file_path: Path, out_path: Path) -> bool:
    """Core implementation — runs inside the dedicated thread."""
    import os
    import numpy as np

    # Set the Agg backend via env var before any pyplot import.  This is the
    # most reliable way to ensure headless (no-display) rendering even if
    # matplotlib was partially initialised elsewhere in the process.
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    except Exception as exc:
        logger.error("thumbnail_matplotlib_import_failed", error=str(exc))
        return False

    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_LabelSequence
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    # ------------------------------------------------------------------
    # 1. Read XCAF
    # ------------------------------------------------------------------
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(False)
    reader.SetColorMode(False)
    reader.SetLayerMode(False)

    if reader.ReadFile(str(file_path)) != IFSelect_RetDone:
        logger.warning("thumbnail_reader_failed", file=str(file_path))
        return False
    if not reader.Transfer(doc):
        logger.warning("thumbnail_transfer_failed", file=str(file_path))
        return False

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    if roots.Length() == 0:
        return False

    # Combine ALL free shapes into one compound so we always render the
    # complete file — some STEP exports have multiple root shapes rather
    # than a single top-level assembly.
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    added = 0
    for i in range(roots.Length()):
        s = shape_tool.GetShape_s(roots.Value(i + 1))
        if s is not None and not s.IsNull():
            builder.Add(compound, s)
            added += 1
    if added == 0:
        return False
    root_shape = compound

    # ------------------------------------------------------------------
    # 2. Bounding box → adaptive deflection
    # ------------------------------------------------------------------
    box = Bnd_Box()
    BRepBndLib.Add_s(root_shape, box)
    if box.IsVoid():
        return False

    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    max_extent = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)

    # 2 % of the largest dimension — very coarse, fast to compute, enough
    # to show the overall shape and scale of the assembly.
    deflection = max(max_extent * 0.02, 1.0)

    # ------------------------------------------------------------------
    # 3. Tessellate
    # ------------------------------------------------------------------
    mesh = BRepMesh_IncrementalMesh(root_shape, deflection, False, 0.5)
    mesh.Perform()

    # ------------------------------------------------------------------
    # 4. Collect triangles (world-space coordinates)
    #
    # Strategy: take at most _MAX_PER_FACE triangles from each face so
    # every part of the assembly is represented in the thumbnail even
    # when the total face count is very large.  After collecting, if the
    # grand total still exceeds _MAX_TRIS, subsample uniformly at random.
    # ------------------------------------------------------------------
    import random as _random

    _MAX_PER_FACE = 30    # per-face cap — keeps all parts visible at minimal cost

    all_tris: list = []
    explorer = TopExp_Explorer(root_shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, loc)

        if triangulation is not None:
            is_identity = loc.IsIdentity()
            trsf = None if is_identity else loc.Transformation()

            n = triangulation.NbTriangles()
            # Choose which triangle indices to sample from this face
            if n <= _MAX_PER_FACE:
                indices = range(1, n + 1)
            else:
                indices = _random.sample(range(1, n + 1), _MAX_PER_FACE)

            for i in indices:
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                pts = []
                for n_idx in (n1, n2, n3):
                    pt = triangulation.Node(n_idx)
                    if trsf is not None:
                        pt = pt.Transformed(trsf)
                    pts.append([pt.X(), pt.Y(), pt.Z()])
                all_tris.append(pts)

        explorer.Next()

    if not all_tris:
        logger.warning("thumbnail_no_triangles", file=str(file_path))
        return False

    # Final subsample if still over budget
    if len(all_tris) > _MAX_TRIS:
        tris = _random.sample(all_tris, _MAX_TRIS)
    else:
        tris = all_tris

    logger.info("thumbnail_triangles_collected", n=len(tris), file=str(file_path))

    # ------------------------------------------------------------------
    # 5. Shade triangles using face normals
    # ------------------------------------------------------------------
    pts_arr = np.array(tris, dtype=np.float64)   # (N, 3, 3)
    v1 = pts_arr[:, 1] - pts_arr[:, 0]
    v2 = pts_arr[:, 2] - pts_arr[:, 0]
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    normals /= norms

    # Single key light — simple but enough to read the 3D shape
    key_dir = np.array([0.6, 0.4, 0.7])
    key_dir /= np.linalg.norm(key_dir)
    intensity = np.clip(
        0.30 + 0.70 * np.abs(normals @ key_dir), 0.0, 1.0
    )

    # Steel-blue base colour
    base = np.array([0.24, 0.48, 0.71])
    face_colours = np.clip(intensity[:, None] * base, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 6. Render with matplotlib
    # ------------------------------------------------------------------
    bg = "#eef2f7"
    fig = plt.figure(figsize=(5, 4), dpi=80, facecolor=bg)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor(bg)

    poly = Poly3DCollection(
        tris,
        facecolors=face_colours,
        edgecolors="none",
        linewidth=0,
        shade=False,
    )
    ax.add_collection3d(poly)

    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    cz = (zmax + zmin) / 2
    half = max_extent * 0.55

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=225)
    ax.axis("off")

    plt.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out_path),
        dpi=80,
        bbox_inches="tight",
        facecolor=bg,
        pad_inches=0.1,
    )
    plt.close(fig)

    logger.info(
        "thumbnail_generated",
        file=str(file_path),
        path=str(out_path),
        n_triangles=len(tris),
        deflection=round(deflection, 2),
    )
    return True
