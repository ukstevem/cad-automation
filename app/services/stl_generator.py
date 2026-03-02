"""
XCAF-based STL generator for assembly items.

Reads a STEP file via STEPCAFControl_Reader (same pattern as AssemblyAnalyzer),
meshes shapes with BRepMesh_IncrementalMesh, and writes binary STL via StlAPI_Writer.

Supports two modes:
  - generate_all(): root-level assembly children (initial analysis)
  - generate_children(parent_node_id): children of a specific assembly node (on explode)
"""
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

import multiprocessing
import os

from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool
from OCP.TDataStd import TDataStd_Name
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.TopLoc import TopLoc_Location

from app.config import settings
from app.exceptions import STLGenerationError

logger = structlog.get_logger()

# Characters not safe for filenames
_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    safe = _UNSAFE_RE.sub("_", name).strip(". ")
    return safe or "unnamed"


def _mesh_to_stl_child(
    shape,
    output_path_str: str,
    linear_deflection: float,
    angular_deflection: float,
) -> None:
    """
    Entry point for a per-item forked child process.

    Runs BRepMesh_IncrementalMesh + StlAPI_Writer for a single shape and
    exits immediately via os._exit() so that Python's atexit / __del__
    machinery never runs in the child (which would corrupt OCC reference
    counts in the parent's address space after fork).

    Exit codes:
        0  success
        1  Python exception during meshing / writing
        2  BRepMesh_IncrementalMesh.IsDone() returned False
        3  StlAPI_Writer.Write() returned False
    """
    try:
        # Give the child a large stack for OCC's recursive mesh algorithms
        # (mirrors the threading.stack_size(64 MB) used in the worker script).
        try:
            import resource as _resource
            _resource.setrlimit(
                _resource.RLIMIT_STACK,
                (64 * 1024 * 1024, _resource.RLIM_INFINITY),
            )
        except Exception:
            pass

        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
        mesh.Perform()
        if not mesh.IsDone():
            os._exit(2)

        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        if not writer.Write(shape, output_path_str):
            os._exit(3)

        os._exit(0)
    except Exception:
        os._exit(1)


class STLGenerator:
    """Generate STL files for root-level items in a STEP assembly."""

    def __init__(self, file_path: str, output_dir: str):
        self.file_path = Path(file_path)
        self.output_dir = Path(output_dir)
        self.linear_deflection = settings.STL_LINEAR_DEFLECTION
        self.angular_deflection = settings.STL_ANGULAR_DEFLECTION

        if not self.file_path.exists():
            raise FileNotFoundError(f"STEP file not found: {file_path}")

    def generate_all(self, progress_callback: Callable = None) -> List[Dict[str, Any]]:
        """
        Generate STL files for all root-level assembly items.

        Args:
            progress_callback: Called as (completed, total, current_name)

        Returns:
            List of dicts with name, stl_file, node_id, size_bytes
        """
        logger.info("stl_generation_starting", file_path=str(self.file_path))

        doc, shape_tool = self._read_xcaf()
        items = self._find_root_items(shape_tool)

        if not items:
            logger.warning("no_root_items_found", file_path=str(self.file_path))
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(items)

        for i, item in enumerate(items):
            name = item["name"]
            if progress_callback:
                progress_callback(i, total, name)

            try:
                stl_path = self.output_dir / f"{_safe_filename(name)}.stl"
                self._generate_stl(item["shape"], stl_path)
                size_bytes = stl_path.stat().st_size
                results.append({
                    "name": name,
                    "stl_file": stl_path.name,
                    "node_id": item["node_id"],
                    "size_bytes": size_bytes,
                })
                logger.info("stl_generated", name=name, size_bytes=size_bytes)
            except Exception as e:
                logger.error("stl_item_failed", name=name, error=str(e))
                results.append({
                    "name": name,
                    "stl_file": None,
                    "node_id": item["node_id"],
                    "error": str(e),
                })

        if progress_callback:
            progress_callback(total, total, "")

        logger.info("stl_generation_complete", total=total, generated=len([r for r in results if r.get("stl_file")]))
        return results

    def generate_children(
        self, parent_node_id: str, progress_callback: Callable = None
    ) -> List[Dict[str, Any]]:
        """
        Generate STL files for direct children of a specific assembly node.

        Args:
            parent_node_id: XCAF label entry string (e.g. "0:1:1:3")
            progress_callback: Called as (completed, total, current_name)

        Returns:
            List of dicts with name, stl_file, node_id, size_bytes
        """
        logger.info(
            "stl_children_generation_starting",
            file_path=str(self.file_path),
            parent_node_id=parent_node_id,
        )

        doc, shape_tool = self._read_xcaf()

        # Find the parent label from its entry string
        parent_label = TDF_Label()
        TDF_Tool.Label_s(doc.GetData(), TCollection_AsciiString(parent_node_id), parent_label)

        if parent_label.IsNull():
            raise STLGenerationError(
                f"Node not found in document: {parent_node_id}",
                details={"parent_node_id": parent_node_id},
            )

        # Resolve references on the parent
        resolved = parent_label
        if XCAFDoc_ShapeTool.IsReference_s(parent_label):
            ref_label = TDF_Label()
            XCAFDoc_ShapeTool.GetReferredShape_s(parent_label, ref_label)
            resolved = ref_label

        # Get children components
        components = TDF_LabelSequence()
        is_assembly = XCAFDoc_ShapeTool.GetComponents_s(resolved, components)

        if not is_assembly or components.Length() == 0:
            logger.warning(
                "parent_not_assembly_or_no_children",
                parent_node_id=parent_node_id,
            )
            return []

        # Extract items for each child
        items = []
        for j in range(components.Length()):
            child_label = components.Value(j + 1)
            item = self._extract_item(child_label, shape_tool)
            if item:
                items.append(item)

        if not items:
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(items)

        for i, item in enumerate(items):
            name = item["name"]
            if progress_callback:
                progress_callback(i, total, name)

            try:
                stl_path = self.output_dir / f"{_safe_filename(name)}.stl"
                self._generate_stl(item["shape"], stl_path)
                size_bytes = stl_path.stat().st_size
                results.append({
                    "name": name,
                    "stl_file": stl_path.name,
                    "node_id": item["node_id"],
                    "size_bytes": size_bytes,
                })
                logger.info("stl_child_generated", name=name, size_bytes=size_bytes)
            except Exception as e:
                logger.error("stl_child_failed", name=name, error=str(e))
                results.append({
                    "name": name,
                    "stl_file": None,
                    "node_id": item["node_id"],
                    "error": str(e),
                })

        if progress_callback:
            progress_callback(total, total, "")

        logger.info(
            "stl_children_generation_complete",
            parent_node_id=parent_node_id,
            total=total,
            generated=len([r for r in results if r.get("stl_file")]),
        )
        return results

    def generate_solids(
        self, node_id: str, progress_callback: Callable = None
    ) -> List[Dict[str, Any]]:
        """
        Generate individual STL files for each solid in a multi-solid part.

        Uses the prototype (referred) label entry as the base for synthetic
        node IDs so that all instances of the same part share the same IDs.

        Args:
            node_id: XCAF label entry string of the part (instance or prototype)
            progress_callback: Called as (completed, total, current_name)

        Returns:
            List of dicts with name, stl_file, node_id (synthetic: <ref_entry>:s<N>), size_bytes
        """
        logger.info(
            "stl_solids_generation_starting",
            file_path=str(self.file_path),
            node_id=node_id,
        )

        doc, shape_tool = self._read_xcaf()

        # Find the label from its entry string
        label = TDF_Label()
        TDF_Tool.Label_s(doc.GetData(), TCollection_AsciiString(node_id), label)

        if label.IsNull():
            raise STLGenerationError(
                f"Node not found in document: {node_id}",
                details={"node_id": node_id},
            )

        # Resolve reference to the prototype
        resolved = label
        if XCAFDoc_ShapeTool.IsReference_s(label):
            ref_label = TDF_Label()
            XCAFDoc_ShapeTool.GetReferredShape_s(label, ref_label)
            resolved = ref_label

        # Use the prototype entry as the base for synthetic IDs (shared across instances)
        base_id = self._label_entry(resolved)

        # Get the compound shape
        shape = XCAFDoc_ShapeTool.GetShape_s(resolved)
        if shape is None or shape.IsNull():
            raise STLGenerationError(
                f"No shape for node: {node_id}",
                details={"node_id": node_id},
            )

        # Get the part name for labeling
        part_name = (
            self._get_label_name(label)
            or self._get_label_name(resolved)
            or "Part"
        )

        # Extract individual solids
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        solids = []
        idx = 0
        while explorer.More():
            solids.append({
                "solid": explorer.Current(),
                "name": f"{part_name} - Solid {idx + 1}",
                "node_id": f"{base_id}:s{idx}",
            })
            explorer.Next()
            idx += 1

        if not solids:
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(solids)

        for i, item in enumerate(solids):
            solid_name = item["name"]
            if progress_callback:
                progress_callback(i, total, solid_name)

            try:
                stl_path = self.output_dir / f"{_safe_filename(solid_name)}.stl"
                self._generate_stl(item["solid"], stl_path)
                size_bytes = stl_path.stat().st_size
                results.append({
                    "name": solid_name,
                    "stl_file": stl_path.name,
                    "node_id": item["node_id"],
                    "size_bytes": size_bytes,
                })
                logger.info("stl_solid_generated", name=solid_name, size_bytes=size_bytes)
            except Exception as e:
                logger.error("stl_solid_failed", name=solid_name, error=str(e))
                results.append({
                    "name": solid_name,
                    "stl_file": None,
                    "node_id": item["node_id"],
                    "error": str(e),
                })

        if progress_callback:
            progress_callback(total, total, "")

        logger.info(
            "stl_solids_generation_complete",
            node_id=node_id,
            base_id=base_id,
            total=total,
            generated=len([r for r in results if r.get("stl_file")]),
        )
        return results

    def generate_solids_world(
        self, node_id: str, progress_callback: Callable = None
    ) -> List[Dict[str, Any]]:
        """
        Generate individual STL files for each solid in a part, in world coordinates.

        Unlike generate_solids() which resolves to the prototype shape,
        this method uses the instance shape directly so that the STL
        geometry is in world space — matching weld path coordinates from
        the connection detector.

        Args:
            node_id: XCAF label entry string of the instance node
            progress_callback: Called as (completed, total, current_name)

        Returns:
            List of dicts with name, stl_file, node_id (synthetic: <node_id>:s<N>), size_bytes
        """
        logger.info(
            "stl_solids_world_generation_starting",
            file_path=str(self.file_path),
            node_id=node_id,
        )

        doc, shape_tool = self._read_xcaf()

        # Find the label from its entry string
        label = TDF_Label()
        TDF_Tool.Label_s(doc.GetData(), TCollection_AsciiString(node_id), label)

        if label.IsNull():
            raise STLGenerationError(
                f"Node not found in document: {node_id}",
                details={"node_id": node_id},
            )

        # Get the instance shape, then strip the node's own world placement
        # so that solids are in the assembly-prototype frame.  The connection
        # detector extracts leaf shapes from prototype child labels which do
        # NOT carry the instance placement — stripping it here keeps the STL
        # and weld-path coordinate systems aligned.
        shape = XCAFDoc_ShapeTool.GetShape_s(label)
        if shape is None or shape.IsNull():
            raise STLGenerationError(
                f"No shape for node: {node_id}",
                details={"node_id": node_id},
            )
        shape = shape.Located(TopLoc_Location())

        # Get the part name — try instance label first, then resolved ref
        part_name = self._get_label_name(label)
        if not part_name and XCAFDoc_ShapeTool.IsReference_s(label):
            ref_label = TDF_Label()
            XCAFDoc_ShapeTool.GetReferredShape_s(label, ref_label)
            part_name = self._get_label_name(ref_label)
        part_name = part_name or "Part"

        # Extract individual solids from the world-space compound
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        solids = []
        idx = 0
        while explorer.More():
            solids.append({
                "solid": explorer.Current(),
                "name": f"{part_name} - Solid {idx + 1}",
                "node_id": f"{node_id}:s{idx}",
            })
            explorer.Next()
            idx += 1

        if not solids:
            return []

        # Connection-preview STLs go in a subdirectory to avoid colliding
        # with prototype-coord STLs from generate_solids() (analysis tab).
        world_dir = self.output_dir / "connpreview"
        world_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(solids)

        for i, item in enumerate(solids):
            solid_name = item["name"]
            if progress_callback:
                progress_callback(i, total, solid_name)

            try:
                stl_path = world_dir / f"{_safe_filename(solid_name)}.stl"
                self._generate_stl(item["solid"], stl_path)
                size_bytes = stl_path.stat().st_size
                results.append({
                    "name": solid_name,
                    "stl_file": stl_path.name,
                    "node_id": item["node_id"],
                    "size_bytes": size_bytes,
                })
                logger.info("stl_solid_world_generated", name=solid_name, size_bytes=size_bytes)
            except Exception as e:
                logger.error("stl_solid_world_failed", name=solid_name, error=str(e))
                results.append({
                    "name": solid_name,
                    "stl_file": None,
                    "node_id": item["node_id"],
                    "error": str(e),
                })

        if progress_callback:
            progress_callback(total, total, "")

        logger.info(
            "stl_solids_world_generation_complete",
            node_id=node_id,
            total=total,
            generated=len([r for r in results if r.get("stl_file")]),
        )
        return results

    def generate_solids_world_batch(
        self, node_ids: List[str], progress_callback: Callable = None
    ) -> List[Dict[str, Any]]:
        """
        Generate world-coordinate STLs for multiple nodes in a single STEP parse.

        This is much faster than calling generate_solids_world() N times because
        the STEP file (often 50-100 MB) is only parsed once.
        """
        logger.info(
            "stl_solids_world_batch_starting",
            file_path=str(self.file_path),
            node_count=len(node_ids),
        )

        doc, shape_tool = self._read_xcaf()

        world_dir = self.output_dir / "connpreview"
        world_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        total_nodes = len(node_ids)

        for ni, nid in enumerate(node_ids):
            # Find the label from its entry string
            label = TDF_Label()
            TDF_Tool.Label_s(doc.GetData(), TCollection_AsciiString(nid), label)

            if label.IsNull():
                logger.warning("stl_batch_node_not_found", node_id=nid)
                continue

            shape = XCAFDoc_ShapeTool.GetShape_s(label)
            if shape is None or shape.IsNull():
                logger.warning("stl_batch_node_no_shape", node_id=nid)
                continue
            shape = shape.Located(TopLoc_Location())

            # Get part name
            part_name = self._get_label_name(label)
            if not part_name and XCAFDoc_ShapeTool.IsReference_s(label):
                ref_label = TDF_Label()
                XCAFDoc_ShapeTool.GetReferredShape_s(label, ref_label)
                part_name = self._get_label_name(ref_label)
            part_name = part_name or "Part"

            # Extract solids
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_SOLID
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            idx = 0
            while explorer.More():
                solid = explorer.Current()
                solid_name = f"{part_name} - Solid {idx + 1}"
                solid_node_id = f"{nid}:s{idx}"

                if progress_callback:
                    progress_callback(
                        len(all_results), -1,
                        f"Node {ni + 1}/{total_nodes}: {solid_name}",
                    )

                try:
                    stl_path = world_dir / f"{_safe_filename(solid_name)}.stl"
                    if not stl_path.exists():
                        self._generate_stl(solid, stl_path)
                    size_bytes = stl_path.stat().st_size
                    all_results.append({
                        "name": solid_name,
                        "stl_file": stl_path.name,
                        "node_id": solid_node_id,
                        "size_bytes": size_bytes,
                    })
                except Exception as e:
                    logger.error(
                        "stl_batch_solid_failed",
                        node_id=nid, solid=solid_name, error=str(e),
                    )

                explorer.Next()
                idx += 1

        logger.info(
            "stl_solids_world_batch_complete",
            node_count=total_nodes,
            total_stls=len(all_results),
        )
        return all_results

    def _read_xcaf(self):
        """Create an XCAF document and populate it from the STEP file."""
        doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(False)
        reader.SetLayerMode(False)

        status = reader.ReadFile(str(self.file_path))
        if status != IFSelect_RetDone:
            raise STLGenerationError(
                "STEPCAFControl_Reader failed to read file",
                details={"file": str(self.file_path), "status": int(status)},
            )

        if not reader.Transfer(doc):
            raise STLGenerationError(
                "STEPCAFControl_Reader transfer failed",
                details={"file": str(self.file_path)},
            )

        shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        return doc, shape_tool

    def _find_root_items(self, shape_tool) -> List[Dict[str, Any]]:
        """
        Find direct children of each free shape (root).

        Returns list of dicts with name, shape, and node_id for each
        root-level item (both assemblies and parts).
        """
        roots = TDF_LabelSequence()
        shape_tool.GetFreeShapes(roots)

        items = []
        for i in range(roots.Length()):
            root_label = roots.Value(i + 1)

            # Check if root itself is an assembly with children
            components = TDF_LabelSequence()
            is_assembly = XCAFDoc_ShapeTool.GetComponents_s(root_label, components)

            # Also resolve references on the root
            resolved = root_label
            if XCAFDoc_ShapeTool.IsReference_s(root_label):
                ref_label = TDF_Label()
                XCAFDoc_ShapeTool.GetReferredShape_s(root_label, ref_label)
                resolved = ref_label
                components = TDF_LabelSequence()
                is_assembly = XCAFDoc_ShapeTool.GetComponents_s(resolved, components)

            if is_assembly and components.Length() > 0:
                # Collect direct children of the root assembly
                for j in range(components.Length()):
                    child_label = components.Value(j + 1)
                    item = self._extract_item(child_label, shape_tool)
                    if item:
                        items.append(item)
            else:
                # Root is itself a part - generate STL for it directly
                item = self._extract_item(root_label, shape_tool)
                if item:
                    items.append(item)

        return items

    def _extract_item(self, label, shape_tool) -> Optional[Dict[str, Any]]:
        """Extract name, shape, and node_id from a label."""
        node_id = self._label_entry(label)

        # Resolve references for naming and shape
        resolved = label
        name = self._get_label_name(label)

        if XCAFDoc_ShapeTool.IsReference_s(label):
            ref_label = TDF_Label()
            XCAFDoc_ShapeTool.GetReferredShape_s(label, ref_label)
            ref_name = self._get_label_name(ref_label)
            if ref_name:
                name = ref_name
            resolved = ref_label

        if not name:
            name = f"Item_{node_id}"

        shape = XCAFDoc_ShapeTool.GetShape_s(resolved)
        if shape is None or shape.IsNull():
            return None

        return {"name": name, "shape": shape, "node_id": node_id}

    def _generate_stl(self, shape, output_path: Path):
        """
        Mesh a shape and write binary STL.

        Each item is meshed in an isolated forked child process so that a
        SIGSEGV (exit -11) from OCC on degenerate geometry only kills that
        one child.  The parent catches the non-zero exit code and raises
        STLGenerationError, which the generation loop already handles by
        recording an error for that item and continuing with the rest.

        Two attempts are made before giving up — intermittent SIGSEGV crashes
        from OCC often do not reproduce on retry because ASLR gives a different
        memory layout, avoiding whatever alignment triggered the original fault.

        Uses 'fork' (Linux/Docker default) so the OCC shape objects are
        available in the child's address space without pickling.
        """
        reasons = {2: "IsDone() False", 3: "StlAPI_Writer failed"}
        last_error: Exception = STLGenerationError(f"No attempts made for {output_path.stem}")

        for attempt in range(2):
            ctx = multiprocessing.get_context("fork")
            p = ctx.Process(
                target=_mesh_to_stl_child,
                args=(shape, str(output_path), self.linear_deflection, self.angular_deflection),
            )
            p.start()
            p.join(timeout=300)  # 5-minute per-item ceiling

            if p.exitcode is None:
                p.kill()
                p.join()
                last_error = STLGenerationError(f"Meshing timed out for {output_path.stem}")
            elif p.exitcode == 0:
                return  # success
            else:
                detail = reasons.get(p.exitcode, f"exit {p.exitcode}")
                last_error = STLGenerationError(
                    f"Meshing subprocess failed ({detail}) for {output_path.stem}"
                )

            if attempt == 0:
                logger.warning(
                    "stl_generation_retrying",
                    stem=output_path.stem,
                    exitcode=p.exitcode,
                )

        raise last_error

    @staticmethod
    def _label_entry(label) -> str:
        from OCP.TDF import TDF_Tool
        from OCP.TCollection import TCollection_AsciiString
        entry = TCollection_AsciiString()
        TDF_Tool.Entry_s(label, entry)
        return entry.ToCString()

    @staticmethod
    def _get_label_name(label) -> Optional[str]:
        name_attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), name_attr):
            return name_attr.Get().ToExtString()
        return None
