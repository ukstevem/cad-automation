"""
XCAF-based assembly analyzer for extracting named assembly trees from STEP files.

Uses STEPCAFControl_Reader (instead of STEPControl_Reader) to preserve
product structure, part names, and assembly relationships.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog

from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.TDF import TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID

from app.exceptions import STEPParseError

logger = structlog.get_logger()


class AssemblyAnalyzer:
    """Analyze STEP assembly structure using the XCAF framework."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"STEP file not found: {file_path}")

    def analyze(self) -> Dict[str, Any]:
        """Read STEP file via XCAF and return the assembly tree with names."""
        logger.info("analyzing_assembly", file_path=str(self.file_path))

        try:
            doc, shape_tool = self._read_xcaf()
        except Exception as e:
            logger.error("xcaf_read_failed", error=str(e), file=str(self.file_path))
            raise STEPParseError(
                f"Failed to read STEP file with XCAF: {e}",
                details={"file": str(self.file_path), "error": str(e)},
            )

        # Walk the free shapes (top-level roots)
        roots = TDF_LabelSequence()
        shape_tool.GetFreeShapes(roots)

        tree: List[Dict[str, Any]] = []
        totals = {"assemblies": 0, "parts": 0, "solids": 0}

        for i in range(roots.Length()):
            label = roots.Value(i + 1)  # 1-based
            node = self._walk_tree(label, shape_tool, depth=0, totals=totals)
            tree.append(node)

        result = {
            "assembly_tree": tree,
            "summary": {
                "total_assemblies": totals["assemblies"],
                "total_parts": totals["parts"],
                "total_solids": totals["solids"],
            },
        }
        logger.info("assembly_analysis_complete", summary=result["summary"])
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_xcaf(self):
        """Create an XCAF document and populate it from the STEP file."""
        doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(False)
        reader.SetLayerMode(False)

        status = reader.ReadFile(str(self.file_path))
        if status != IFSelect_RetDone:
            raise STEPParseError(
                "STEPCAFControl_Reader failed to read file",
                details={"file": str(self.file_path), "status": int(status)},
            )

        if not reader.Transfer(doc):
            raise STEPParseError(
                "STEPCAFControl_Reader transfer failed",
                details={"file": str(self.file_path)},
            )

        shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        return doc, shape_tool

    def _walk_tree(
        self,
        label,
        shape_tool,
        depth: int,
        totals: Dict[str, int],
    ) -> Dict[str, Any]:
        """Recursively walk an XCAF label tree and build the node dict."""
        entry_str = self._label_entry(label)
        instance_name = self._get_label_name(label)

        # Determine if this label is an assembly (has components)
        components = TDF_LabelSequence()
        is_assembly = XCAFDoc_ShapeTool.GetComponents_s(label, components)

        # If this is a reference, resolve to the referred shape for naming/solids
        referred = label  # default
        ref_name: Optional[str] = None
        if XCAFDoc_ShapeTool.IsReference_s(label):
            from OCP.TDF import TDF_Label
            ref_label = TDF_Label()
            XCAFDoc_ShapeTool.GetReferredShape_s(label, ref_label)
            ref_name = self._get_label_name(ref_label)
            referred = ref_label
            # Re-check components on the referred shape
            components = TDF_LabelSequence()
            is_assembly = XCAFDoc_ShapeTool.GetComponents_s(referred, components)

        # Display name: prefer the reference element name (definition),
        # fall back to instance name, then a placeholder.
        display_name = ref_name or instance_name or f"Unnamed_{entry_str}"

        # Count solids on the shape itself (not recursively through children)
        num_solids = 0
        if not is_assembly:
            shape = XCAFDoc_ShapeTool.GetShape_s(referred)
            if shape and not shape.IsNull():
                num_solids = self._count_solids(shape)

        node_type = self._classify_node(is_assembly, num_solids)

        # Update totals
        if is_assembly:
            totals["assemblies"] += 1
        else:
            totals["parts"] += 1
            totals["solids"] += num_solids

        # ref_id: entry of the referred/prototype label.
        # Two instances of the same shape share the same ref_id.
        ref_id = self._label_entry(referred)

        # Chirality: detect if this instance has a mirror transformation (det = -1).
        is_mirrored = self._is_mirrored(label)

        node: Dict[str, Any] = {
            "id": entry_str,
            "name": display_name,
            "instance_ref": instance_name,
            "ref_id": ref_id,
            "is_mirrored": is_mirrored,
            "node_type": node_type,
            "solid_count": num_solids,
            "children": [],
        }

        if is_assembly:
            for i in range(components.Length()):
                child_label = components.Value(i + 1)
                child_node = self._walk_tree(child_label, shape_tool, depth + 1, totals)
                node["children"].append(child_node)

        return node

    @staticmethod
    def _label_entry(label) -> str:
        """Return the TDF_Label entry as a colon-separated string."""
        from OCP.TDF import TDF_Tool
        from OCP.TCollection import TCollection_AsciiString
        entry = TCollection_AsciiString()
        TDF_Tool.Entry_s(label, entry)
        return entry.ToCString()

    @staticmethod
    def _get_label_name(label) -> Optional[str]:
        """Extract the name string from a TDataStd_Name attribute, if present."""
        name_attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), name_attr):
            return name_attr.Get().ToExtString()
        return None

    @staticmethod
    def _count_solids(shape) -> int:
        """Count the number of TopoDS_SOLID entities in a shape."""
        count = 0
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            count += 1
            explorer.Next()
        return count

    @staticmethod
    def _is_mirrored(label) -> bool:
        """
        Detect if this instance has a mirror transformation (negative determinant).

        Uses BRep-level shape location from XCAFDoc_ShapeTool.GetShape_s() rather
        than reading the XCAFDoc_Location XCAF attribute directly.  The XCAF
        attribute path caused SIGSEGV in OCC 7.7.x for certain labels in complex
        STEP files; this approach is safe because it operates on the Python-level
        TopoDS_Shape and TopLoc_Location objects, not on XCAF attribute nodes.

        Returns True if the transformation determinant is negative (reflected/mirrored).
        Wrapped in try/except so any API mismatch falls back to False rather than
        raising (SIGSEGV cannot be caught, but that path is avoided here).
        """
        try:
            shape = XCAFDoc_ShapeTool.GetShape_s(label)
            if shape.IsNull():
                return False
            loc = shape.Location()
            if loc.IsIdentity():
                return False
            return loc.IsNegative()
        except Exception:
            return False

    @staticmethod
    def _classify_node(is_assembly: bool, num_solids: int) -> str:
        """Return a node_type string based on structure and solid count."""
        if is_assembly:
            return "assembly"
        if num_solids > 1:
            return "part_multi_solid"
        if num_solids == 1:
            return "part_single_solid"
        return "part_no_solid"
