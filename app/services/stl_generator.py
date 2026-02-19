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

from app.config import settings
from app.exceptions import STLGenerationError

logger = structlog.get_logger()

# Characters not safe for filenames
_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    safe = _UNSAFE_RE.sub("_", name).strip(". ")
    return safe or "unnamed"


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
        """Mesh a shape and write binary STL."""
        mesh = BRepMesh_IncrementalMesh(shape, self.linear_deflection, False, self.angular_deflection)
        mesh.Perform()

        if not mesh.IsDone():
            raise STLGenerationError(f"Meshing failed for {output_path.stem}")

        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        success = writer.Write(shape, str(output_path))
        if not success:
            raise STLGenerationError(f"StlAPI_Writer failed for {output_path.stem}")

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
