"""
Render STL files to PNG thumbnails for BOM export.

Uses trimesh to load the STL mesh and matplotlib for headless rendering.
Produces an isometric-ish view with simple edge shading.
"""
import io
from pathlib import Path
from typing import Optional

import numpy as np


def render_stl_thumbnail(
    stl_path: str | Path,
    width: int = 180,
    height: int = 140,
    dpi: int = 100,
) -> Optional[bytes]:
    """
    Render an STL file to a PNG thumbnail (bytes).

    Returns None if the file doesn't exist or rendering fails.
    """
    stl_path = Path(stl_path)
    if not stl_path.exists():
        return None

    try:
        import trimesh
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh = trimesh.load_mesh(str(stl_path))
        if mesh.is_empty:
            return None

        verts = mesh.vertices
        faces = mesh.faces

        # Centre at origin
        centroid = verts.mean(axis=0)
        verts = verts - centroid

        # Auto-scale to unit cube for consistent sizing
        extent = verts.max(axis=0) - verts.min(axis=0)
        scale = max(extent) or 1.0
        verts = verts / scale

        # Build face polygons
        face_verts = verts[faces]

        # Simple face normals for shading
        v0 = face_verts[:, 1] - face_verts[:, 0]
        v1 = face_verts[:, 2] - face_verts[:, 0]
        normals = np.cross(v0, v1)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        # Light from upper-right-front
        light_dir = np.array([0.4, 0.6, 0.8])
        light_dir = light_dir / np.linalg.norm(light_dir)
        intensity = np.abs(normals @ light_dir)
        intensity = 0.3 + 0.7 * intensity  # ambient + diffuse

        # Face colours: steel blue with shading
        base_colour = np.array([0.55, 0.70, 0.82])
        face_colours = np.outer(intensity, base_colour)
        face_colours = np.clip(face_colours, 0, 1)
        # Add alpha
        alphas = np.ones((len(face_colours), 1))
        face_colours = np.hstack([face_colours, alphas])

        fig = plt.figure(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            facecolor="white",
        )
        ax = fig.add_subplot(111, projection="3d")

        poly = Poly3DCollection(
            face_verts,
            facecolors=face_colours,
            edgecolors=(0.3, 0.3, 0.3, 0.15),
            linewidths=0.2,
        )
        ax.add_collection3d(poly)

        # Set limits
        half = 0.6
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_zlim(-half, half)

        # Isometric-ish view
        ax.view_init(elev=25, azim=-135)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception:
        return None


def render_assembly_thumbnail(
    stl_dir: str | Path,
    width: int = 640,
    height: int = 480,
    dpi: int = 100,
) -> Optional[bytes]:
    """
    Combine all STL files in a directory into a single assembly thumbnail.

    Loads every .stl file, merges them into one scene, and renders an
    isometric view. Returns PNG bytes, or None on failure.
    """
    stl_dir = Path(stl_dir)
    if not stl_dir.is_dir():
        return None

    stl_files = sorted(stl_dir.glob("*.stl"))
    if not stl_files:
        return None

    try:
        import trimesh
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Load and concatenate all meshes
        all_verts = []
        all_faces = []
        vert_offset = 0
        for stl_file in stl_files:
            try:
                mesh = trimesh.load_mesh(str(stl_file))
                if mesh.is_empty:
                    continue
                all_verts.append(mesh.vertices)
                all_faces.append(mesh.faces + vert_offset)
                vert_offset += len(mesh.vertices)
            except Exception:
                continue

        if not all_verts:
            return None

        verts = np.vstack(all_verts)
        faces = np.vstack(all_faces)

        # Centre at origin
        centroid = verts.mean(axis=0)
        verts = verts - centroid

        # Auto-scale to unit cube
        extent = verts.max(axis=0) - verts.min(axis=0)
        scale = max(extent) or 1.0
        verts = verts / scale

        # Build face polygons
        face_verts = verts[faces]

        # Face normals for shading
        v0 = face_verts[:, 1] - face_verts[:, 0]
        v1 = face_verts[:, 2] - face_verts[:, 0]
        normals = np.cross(v0, v1)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        # Light from upper-right-front
        light_dir = np.array([0.4, 0.6, 0.8])
        light_dir = light_dir / np.linalg.norm(light_dir)
        intensity = np.abs(normals @ light_dir)
        intensity = 0.3 + 0.7 * intensity

        # Steel blue with shading
        base_colour = np.array([0.55, 0.70, 0.82])
        face_colours = np.outer(intensity, base_colour)
        face_colours = np.clip(face_colours, 0, 1)
        alphas = np.ones((len(face_colours), 1))
        face_colours = np.hstack([face_colours, alphas])

        fig = plt.figure(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            facecolor="white",
        )
        ax = fig.add_subplot(111, projection="3d")

        poly = Poly3DCollection(
            face_verts,
            facecolors=face_colours,
            edgecolors=(0.3, 0.3, 0.3, 0.08),
            linewidths=0.15,
        )
        ax.add_collection3d(poly)

        half = 0.6
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_zlim(-half, half)

        ax.view_init(elev=25, azim=-135)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception:
        return None
