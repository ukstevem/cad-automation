"""Shared helpers for IFC parsers.

Subclass :class:`BaseIFCParser` and override:

* :meth:`_build_tree` — hierarchy synthesis (vendor-specific)
* :meth:`_should_include_product` — filter for noise entities (vendor-specific)

The base class owns:

* File opening (with lazy ``ifcopenshell`` import)
* Geometry extraction via BREP serialisation (OCP-compatible, works regardless
  of which OpenCascade binding ifcopenshell was compiled against)
* Placement resolution to 4×4 column-major matrices (Three.js format)
* Solid counting, mirror detection, node-type classification
* Property-set harvesting
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from app.parsers.ifc.exceptions import IFCParseError

if TYPE_CHECKING:
    from app.parsers.ifc.header import IFCHeader


logger = structlog.get_logger()


# Entities that are almost never physical parts in a structural model.
# Vendors can extend this via ``_should_include_product``.
DEFAULT_EXCLUDED_ENTITIES = frozenset({
    "IfcCovering",          # paint / coating proxies (Tekla surface treatments)
    "IfcOpeningElement",    # door / window cut-outs, not physical
    "IfcSpace",             # functional rooms
    "IfcAnnotation",        # model annotations
    "IfcGrid",              # grid lines
    "IfcGridAxis",
})


class BaseIFCParser:
    """Base class for vendor-specific IFC parsers.

    The public surface mirrors :class:`app.parsers.assembly_analyzer.AssemblyAnalyzer`
    so the worker / router layer can treat either parser interchangeably.
    """

    # Subclasses can extend or override these.
    excluded_entities = DEFAULT_EXCLUDED_ENTITIES

    def __init__(
        self,
        file_path: str,
        header: Optional["IFCHeader"] = None,
        vendor_label: str = "",
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"IFC file not found: {file_path}")
        self.header = header
        self.vendor_label = vendor_label or (header.vendor_label() if header else "unknown")
        self._ifc_file = None

    # ------------------------------------------------------------------
    # Public API — mirrors AssemblyAnalyzer.analyze()
    # ------------------------------------------------------------------

    def analyze(self) -> Dict[str, Any]:
        """Parse the file and return ``{"assembly_tree": [...], "summary": {...}}``."""
        logger.info(
            "ifc_analysis_start",
            path=str(self.file_path),
            vendor=self.vendor_label,
        )

        try:
            self._open()
        except Exception as exc:
            logger.error("ifc_open_failed", error=str(exc), path=str(self.file_path))
            raise IFCParseError(
                f"Failed to open IFC file: {exc}",
                details={"file": str(self.file_path), "error": str(exc)},
            )

        totals = {"assemblies": 0, "parts": 0, "solids": 0}
        tree = self._build_tree(totals)
        native_bom = self._extract_native_bom(tree)

        result = {
            "assembly_tree": tree,
            "native_bom": native_bom,
            "summary": {
                "total_assemblies": totals["assemblies"],
                "total_parts": totals["parts"],
                "total_solids": totals["solids"],
                "total_bom_rows": len(native_bom),
                "source": "ifc",
                "vendor": self.vendor_label,
                "schema": self.header.schema if self.header else "",
            },
        }
        logger.info("ifc_analysis_complete", summary=result["summary"])
        return result

    # ------------------------------------------------------------------
    # Native BOM extraction (data-first, no geometry)
    # ------------------------------------------------------------------

    def _extract_native_bom(self, tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Walk the assembly tree and build a flat BOM from IFC metadata.

        One row per physical part (leaf node with ``ifc_metadata``). All fields
        come from the file — no geometry. Geometric analysis is reserved for
        downstream fabrication outputs (DXF, NC1).
        """
        rows: List[Dict[str, Any]] = []

        def _walk(nodes: List[Dict[str, Any]]) -> None:
            for node in nodes:
                meta = node.get("ifc_metadata")
                if meta:
                    rows.append(self._bom_row_from_node(node, meta))
                children = node.get("children") or []
                if children:
                    _walk(children)

        _walk(tree)
        return rows

    def _bom_row_from_node(
        self, node: Dict[str, Any], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build one BOM row from a tree node + its ifc_metadata payload."""
        tekla = meta.get("tekla_properties") or {}
        quants = meta.get("quantities") or {}
        material = meta.get("material") or ""

        # Material is commonly "STEEL/S275" or "ALUMINIUM/6082-T6" — the suffix
        # after "/" is the grade. Fall back to the full string.
        grade = material.rsplit("/", 1)[-1].strip() if "/" in material else material

        # Some Tekla exports place physical data in Pset_* sets rather than
        # IfcElementQuantity, so quants is empty. Fall back to the flattened
        # tekla_properties (mixed-case keys like "Weight", "Length") in that case.
        def _pick(*keys):
            for k in keys:
                v = quants.get(k)
                if v is not None:
                    return v
            for k in keys:
                v = tekla.get(k)
                if v is not None:
                    return v
            return None

        weight = _pick("NetWeight", "GrossWeight", "Weight")
        length = _pick("Length")
        volume = _pick("NetVolume", "GrossVolume", "Volume")
        area = _pick(
            "NetSurfaceArea", "GrossSurfaceArea", "OuterSurfaceArea",
            "Net surface area", "Gross surface area",
        )

        return {
            "node_id": node.get("id"),
            "guid": meta.get("guid"),
            "type_guid": meta.get("type_guid"),
            "entity": meta.get("entity"),
            "mark": node.get("name"),
            "profile": meta.get("type_name") or "",
            "material": material,
            "grade": grade,
            "length": length,
            "weight": weight,
            "volume": volume,
            "area": area,
            "assembly_mark": tekla.get("AssemblyMark") or tekla.get("AssemblyName"),
            "assembly_position": tekla.get("AssemblyPosition"),
            "pour": tekla.get("Pour") or tekla.get("AssemblyPour"),
            "phase": tekla.get("Phase") or tekla.get("AssemblyPhase"),
            "finish": tekla.get("Finish") or tekla.get("PartFinish"),
            "part_class": tekla.get("Class") or tekla.get("PartClass"),
            "source": "ifc",
        }

    # ------------------------------------------------------------------
    # Extension points for subclasses
    # ------------------------------------------------------------------

    def _build_tree(self, totals: Dict[str, int]) -> List[Dict[str, Any]]:
        """Synthesise the assembly tree.  Must be implemented by subclasses."""
        raise NotImplementedError

    def _should_include_product(self, product) -> bool:
        """Return True if ``product`` should appear in the tree + BOM."""
        return product.is_a() not in self.excluded_entities

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the IFC file using the (lazily imported) ifcopenshell."""
        import ifcopenshell  # lazy — heavy C extension
        self._ifc_file = ifcopenshell.open(str(self.file_path))

    @property
    def ifc(self):
        if self._ifc_file is None:
            self._open()
        return self._ifc_file

    # ---- Geometry --------------------------------------------------------
    #
    # v1 uses the **triangulation** path: ifcopenshell builds a mesh, we record
    # whether it succeeded, and the tree reports solid_count = 1 for successful
    # tessellation, 0 for failure / missing representation.  This is sufficient
    # for the current consumers (tree, BOM, classification metadata) because:
    #
    # * STL generation is still STEP-only (guarded in stl router)
    # * Connection detection is STEP-only (guarded by ifc metadata absence)
    # * BRep-exact solid enumeration isn't needed for single-product instances —
    #   Tekla exports one IfcProduct per physical item when Assemblies:Off.
    #
    # Getting real ``TopoDS_Shape`` objects requires ifcopenshell to be built
    # with pythonocc or OCP support.  The stock PyPI wheel we install (0.8.5)
    # does not include that; a future upgrade (rebuilt wheel, or pythonocc in
    # the conda env) would switch us back to the BREP path without touching
    # anything except ``_tessellate`` below.

    def _geom_settings(self):
        """Build and return ifcopenshell geometry settings for mesh output."""
        import ifcopenshell.geom as ifc_geom
        settings = ifc_geom.settings()
        # All settings in ifcopenshell 0.8.x use hyphenated string keys.
        for key, value in (
            ("weld-vertices", True),
            ("use-world-coords", True),
            ("apply-default-materials", True),
        ):
            try:
                settings.set(key, value)
            except Exception:
                # Best-effort: unrecognised keys are ignored across minor versions
                pass
        return settings

    def _tessellate(self, product) -> bool:
        """Attempt to tessellate ``product``.  Returns True on success.

        Does not return the mesh — the tree only needs the success signal for
        solid-count classification.  Writes diagnostic logs for failures so we
        can see which entities ifcopenshell choked on.
        """
        if not getattr(product, "Representation", None):
            return False

        import ifcopenshell.geom as ifc_geom
        try:
            settings = self._geom_settings()
            tri = ifc_geom.create_shape(settings, product)
        except Exception as exc:
            logger.debug(
                "ifc_tessellate_failed",
                guid=getattr(product, "GlobalId", ""),
                entity=product.is_a(),
                error=str(exc),
            )
            return False

        # Sanity: a successful mesh should have at least 3 vertices (one tri)
        geom = getattr(tri, "geometry", None)
        verts = getattr(geom, "verts", None) if geom is not None else None
        return bool(verts and len(verts) >= 9)

    # ---- Placement -------------------------------------------------------

    def _resolve_placement(self, product) -> Optional[List[float]]:
        """Resolve the product's ObjectPlacement chain to a 4×4 column-major matrix.

        Returns ``None`` for identity placements (matches AssemblyAnalyzer behaviour).
        """
        placement = getattr(product, "ObjectPlacement", None)
        if placement is None:
            return None

        try:
            import ifcopenshell.util.placement as ifc_placement
            mat = ifc_placement.get_local_placement(placement)  # 4x4 numpy array, row-major
        except Exception as exc:
            logger.debug(
                "ifc_placement_failed",
                guid=getattr(product, "GlobalId", ""),
                error=str(exc),
            )
            return None

        return _matrix_to_column_major(mat)

    @staticmethod
    def _is_mirrored_from_matrix(mat_cm: Optional[List[float]]) -> bool:
        """Detect reflection (negative 3×3 determinant) in a column-major 4×4."""
        if not mat_cm:
            return False
        # Column-major: mat_cm[col*4 + row].  Rows/cols 0..2 are the 3×3 rotation block.
        m = mat_cm
        a, b, c = m[0], m[4], m[8]    # row 0
        d, e, f = m[1], m[5], m[9]    # row 1
        g, h, i = m[2], m[6], m[10]   # row 2
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        return det < 0

    # ---- Identity / classification --------------------------------------

    @staticmethod
    def _classify_node(is_assembly: bool, solid_count: int) -> str:
        if is_assembly:
            return "assembly"
        if solid_count > 1:
            return "part_multi_solid"
        if solid_count == 1:
            return "part_single_solid"
        return "part_no_solid"

    @staticmethod
    def _chiral_key(ref_id: str, is_mirrored: bool) -> str:
        return f"{ref_id}:{'M' if is_mirrored else 'N'}"

    # ---- Metadata --------------------------------------------------------

    @staticmethod
    def _get_type_object(product):
        """Return the associated IfcTypeProduct (or None)."""
        for rel in getattr(product, "IsDefinedBy", []) or []:
            if rel.is_a("IfcRelDefinesByType"):
                return rel.RelatingType
        return None

    @staticmethod
    def _get_property_sets(product) -> Dict[str, Dict[str, Any]]:
        """Return ``{pset_name: {prop_name: value}}`` for the product."""
        psets: Dict[str, Dict[str, Any]] = {}
        for rel in getattr(product, "IsDefinedBy", []) or []:
            if not rel.is_a("IfcRelDefinesByProperties"):
                continue
            pset = rel.RelatingPropertyDefinition
            if not pset.is_a("IfcPropertySet"):
                continue
            props: Dict[str, Any] = {}
            for p in pset.HasProperties or []:
                if p.is_a("IfcPropertySingleValue") and p.NominalValue is not None:
                    props[p.Name] = p.NominalValue.wrappedValue
            if props:
                psets[pset.Name or pset.is_a()] = props
        return psets

    @staticmethod
    def _get_quantities(product) -> Dict[str, Dict[str, Any]]:
        """Return ``{qset_name: {quantity_name: value}}`` from IfcElementQuantity.

        Tekla and most IFC exporters attach physical quantities (Length, Weight,
        Volume, NetArea, etc.) via ``IfcElementQuantity`` sets distinct from
        property sets. These are the authoritative values for BOM fields
        like weight and length — no geometric computation required.
        """
        qsets: Dict[str, Dict[str, Any]] = {}
        for rel in getattr(product, "IsDefinedBy", []) or []:
            if not rel.is_a("IfcRelDefinesByProperties"):
                continue
            qset = rel.RelatingPropertyDefinition
            if not qset.is_a("IfcElementQuantity"):
                continue
            quants: Dict[str, Any] = {}
            for q in qset.Quantities or []:
                # IfcPhysicalSimpleQuantity subclasses carry the value on a
                # type-specific attribute (LengthValue, AreaValue, VolumeValue,
                # WeightValue, CountValue). Fall back to common attribute names.
                for attr in ("LengthValue", "AreaValue", "VolumeValue",
                             "WeightValue", "CountValue", "TimeValue"):
                    val = getattr(q, attr, None)
                    if val is not None:
                        quants[q.Name] = val
                        break
            if quants:
                qsets[qset.Name or qset.is_a()] = quants
        return qsets

    @staticmethod
    def _get_material_name(product) -> str:
        """Return the first associated material's name, or empty string."""
        for rel in getattr(product, "HasAssociations", []) or []:
            if not rel.is_a("IfcRelAssociatesMaterial"):
                continue
            mat = rel.RelatingMaterial
            name = getattr(mat, "Name", None)
            if name:
                return name
            # IfcMaterialLayerSet / MaterialProfileSet etc. — dig in
            for attr in ("MaterialLayers", "MaterialProfiles", "Materials"):
                items = getattr(mat, attr, None) or []
                for item in items:
                    inner_name = getattr(getattr(item, "Material", None), "Name", None) \
                        or getattr(item, "Name", None)
                    if inner_name:
                        return inner_name
        return ""


# ---------------------------------------------------------------------------
# Module-level helpers (not bound to the parser instance)
# ---------------------------------------------------------------------------

def _matrix_to_column_major(mat) -> List[float]:
    """Convert a 4×4 row-major (list-of-lists or numpy array) matrix to a
    16-element column-major list suitable for Three.js ``Matrix4.fromArray``.
    """
    # Accept either numpy.ndarray or nested list-of-lists
    rows = list(mat)
    flat = [0.0] * 16
    for col in range(4):
        for row in range(4):
            val = rows[row][col]
            flat[col * 4 + row] = float(val)
    return flat
