"""Tekla Structures IFC parser.

Handles IFC2X3 files produced by Tekla Structures 2020+ with the
``CoordinationView_V2.0`` export preset.  Tekla's export is usually flat
(``Option [Assemblies:Off]``) so this parser synthesises a hierarchy under
each building storey by grouping products into **type buckets** keyed by
``IfcTypeProduct`` — which is where Tekla writes section profile names
(e.g. ``UB203x102x23``, ``PL10``).

Excluded entities include ``IfcCovering`` (surface-treatment proxies) which
Tekla emits for painted/coated parts and which are not physical items.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import structlog

from app.parsers.ifc.base import BaseIFCParser, DEFAULT_EXCLUDED_ENTITIES

logger = structlog.get_logger()


# Entities Tekla emits that should always appear in the tree for a structural
# workflow.  Anything else is filtered through the DEFAULT_EXCLUDED_ENTITIES
# list plus per-vendor extensions.
_TEKLA_INCLUDED_ENTITIES = frozenset({
    "IfcBeam",
    "IfcColumn",
    "IfcMember",
    "IfcPlate",
    "IfcSlab",
    "IfcWall",
    "IfcWallStandardCase",
    "IfcFooting",
    "IfcPile",
    "IfcRoof",
    "IfcStair",
    "IfcRailing",
    "IfcReinforcingBar",
    "IfcReinforcingMesh",
    "IfcTendon",
    "IfcFastener",
    "IfcMechanicalFastener",
    "IfcDiscreteAccessory",
    "IfcBuildingElementProxy",
})


class TeklaIFCParser(BaseIFCParser):
    """Tekla-flavoured IFC assembly tree builder."""

    # Extend the default noise filter with Tekla-specific churn
    excluded_entities = DEFAULT_EXCLUDED_ENTITIES

    def _should_include_product(self, product) -> bool:
        if product.is_a() in self.excluded_entities:
            return False
        # Tekla occasionally emits IfcBuildingElementProxy for misc bits we
        # still want in the BOM; allow any entity on the whitelist even if
        # future defaults would exclude it.
        if product.is_a() in _TEKLA_INCLUDED_ENTITIES:
            return True
        # Any other IfcElement subclass: include by default
        return product.is_a("IfcElement")

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, totals: Dict[str, int]) -> List[Dict[str, Any]]:
        ifc = self.ifc
        projects = ifc.by_type("IfcProject")
        if not projects:
            logger.warning("ifc_no_project", path=str(self.file_path))
            return []

        tree: List[Dict[str, Any]] = []
        for p_idx, project in enumerate(projects):
            project_node = self._make_container_node(
                project, f"{p_idx}", totals
            )
            # Walk IfcRelAggregates to get the spatial hierarchy
            for s_idx, site in enumerate(self._decompose(project)):
                site_node = self._make_container_node(
                    site, f"{p_idx}:{s_idx}", totals
                )
                for b_idx, building in enumerate(self._decompose(site)):
                    building_node = self._make_container_node(
                        building, f"{p_idx}:{s_idx}:{b_idx}", totals
                    )
                    for st_idx, storey in enumerate(self._decompose(building)):
                        storey_id = f"{p_idx}:{s_idx}:{b_idx}:{st_idx}"
                        storey_node = self._make_container_node(
                            storey, storey_id, totals
                        )
                        self._populate_storey_with_type_buckets(
                            storey, storey_node, storey_id, totals
                        )
                        if storey_node["children"]:
                            building_node["children"].append(storey_node)
                    if building_node["children"]:
                        site_node["children"].append(building_node)
                if site_node["children"]:
                    project_node["children"].append(site_node)
            if project_node["children"]:
                tree.append(project_node)

        return tree

    # ------------------------------------------------------------------
    # Helpers — spatial decomposition + container nodes
    # ------------------------------------------------------------------

    def _decompose(self, parent) -> List:
        """Return immediate spatial children of ``parent`` via IfcRelAggregates."""
        children = []
        for rel in getattr(parent, "IsDecomposedBy", []) or []:
            if rel.is_a("IfcRelAggregates"):
                children.extend(rel.RelatedObjects)
        return children

    def _contained_products(self, spatial_element) -> List:
        """Products contained in ``spatial_element`` via IfcRelContainedInSpatialStructure."""
        out: List = []
        for rel in getattr(spatial_element, "ContainsElements", []) or []:
            if rel.is_a("IfcRelContainedInSpatialStructure"):
                out.extend(rel.RelatedElements)
        return out

    def _make_container_node(
        self,
        entity,
        id_: str,
        totals: Dict[str, int],
    ) -> Dict[str, Any]:
        """Build an assembly-type node for a spatial container (Project/Site/…)."""
        totals["assemblies"] += 1
        name = getattr(entity, "Name", None) or entity.is_a()
        ref_id = getattr(entity, "GlobalId", id_)
        placement = self._resolve_placement(entity)
        is_mirrored = self._is_mirrored_from_matrix(placement)
        return {
            "id": id_,
            "name": name,
            "instance_ref": name,
            "ref_id": ref_id,
            "chiral_key": self._chiral_key(ref_id, is_mirrored),
            "is_mirrored": is_mirrored,
            "placement": placement,
            "node_type": "assembly",
            "solid_count": 0,
            "children": [],
        }

    # ------------------------------------------------------------------
    # Type-bucket synthesis under a storey
    # ------------------------------------------------------------------

    def _populate_storey_with_type_buckets(
        self,
        storey,
        storey_node: Dict[str, Any],
        storey_id: str,
        totals: Dict[str, int],
    ) -> None:
        """Group the storey's contained products by IfcTypeProduct and emit buckets.

        Products without a type are grouped under a synthetic "Unclassified" bucket
        (keyed by the product's own ``is_a()``) so they still appear in the tree.
        """
        # Preserve encounter order for deterministic node IDs
        buckets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        for product in self._contained_products(storey):
            if not self._should_include_product(product):
                continue

            type_obj = self._get_type_object(product)
            if type_obj is not None:
                bucket_key = type_obj.GlobalId
                bucket_label = type_obj.Name or type_obj.is_a()
            else:
                bucket_key = f"__untyped__{product.is_a()}"
                bucket_label = f"{product.is_a()} (untyped)"

            bucket = buckets.get(bucket_key)
            if bucket is None:
                bucket = {
                    "key": bucket_key,
                    "label": bucket_label,
                    "type_obj": type_obj,
                    "products": [],
                }
                buckets[bucket_key] = bucket
            bucket["products"].append(product)

        for b_idx, (_bucket_key, bucket) in enumerate(buckets.items()):
            bucket_id = f"{storey_id}:{b_idx}"
            bucket_node = self._make_type_bucket_node(
                bucket, bucket_id, totals
            )

            for p_idx, product in enumerate(bucket["products"]):
                product_node = self._make_product_node(
                    product, f"{bucket_id}:{p_idx}", bucket, totals
                )
                if product_node is not None:
                    bucket_node["children"].append(product_node)

            if bucket_node["children"]:
                storey_node["children"].append(bucket_node)

    def _make_type_bucket_node(
        self,
        bucket: Dict[str, Any],
        bucket_id: str,
        totals: Dict[str, int],
    ) -> Dict[str, Any]:
        """Synthetic assembly node that groups instances of one IfcTypeProduct."""
        totals["assemblies"] += 1
        count = len(bucket["products"])
        label = f"{bucket['label']} ({count})"
        return {
            "id": bucket_id,
            "name": label,
            "instance_ref": label,
            "ref_id": bucket["key"],
            "chiral_key": f"{bucket['key']}:N",
            "is_mirrored": False,
            "placement": None,
            "node_type": "assembly",
            "solid_count": 0,
            "children": [],
        }

    def _make_product_node(
        self,
        product,
        node_id: str,
        bucket: Dict[str, Any],
        totals: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        """Build a leaf node for one IfcProduct, including geometry + metadata."""
        has_geom = self._tessellate(product)
        # Tekla exports one IfcProduct per physical part (Assemblies:Off), so
        # each product with valid geometry maps to exactly one solid for the
        # classification layer.  Multi-solid IFC products are rare in this
        # flow; if they appear later, switch here to a real solid enumeration.
        solid_count = 1 if has_geom else 0

        placement = self._resolve_placement(product)
        is_mirrored = self._is_mirrored_from_matrix(placement)

        # ref_id: per-instance identity. Each Tekla physical part has its own
        # part mark and is classified independently, so we use product.GlobalId
        # rather than the shared IfcTypeProduct.GlobalId (which would cause
        # classifying one instance to classify every sibling sharing the type).
        # The type GUID is preserved in ifc_metadata for BOM-level grouping.
        type_obj = bucket.get("type_obj")
        ref_id = product.GlobalId

        # Display name: use Tekla part mark (product.Name) first
        name = product.Name or (type_obj.Name if type_obj else None) or product.is_a()
        instance_ref = product.Name or f"{product.is_a()}:{product.GlobalId}"

        totals["parts"] += 1
        totals["solids"] += solid_count

        node: Dict[str, Any] = {
            "id": node_id,
            "name": name,
            "instance_ref": instance_ref,
            "ref_id": ref_id,
            "chiral_key": self._chiral_key(ref_id, is_mirrored),
            "is_mirrored": is_mirrored,
            "placement": placement,
            "node_type": self._classify_node(False, solid_count),
            "solid_count": solid_count,
            "children": [],
            # IFC-specific metadata (downstream can ignore it; STEP tree never has it)
            "ifc_metadata": self._harvest_metadata(product, type_obj),
        }
        return node

    # ------------------------------------------------------------------
    # Metadata harvesting
    # ------------------------------------------------------------------

    def _harvest_metadata(self, product, type_obj) -> Dict[str, Any]:
        """Collect Tekla-relevant metadata for classification + BOM.

        Captures the full pset + quantity payload so downstream consumers
        (native BOM, COBie inspector, IFC-out writer) can read authoritative
        data straight from the file rather than recomputing it geometrically.
        """
        profile_name = type_obj.Name if type_obj else ""
        material_name = self._get_material_name(product)

        psets = self._get_property_sets(product)
        quantities = self._get_quantities(product)

        # Flatten Tekla Pset_Tekla_* single values for convenient BOM lookup
        # (AssemblyMark, Pour, Phase, Finish, PartClass, ...).
        tekla_props: Dict[str, Any] = {}
        for pset_name, props in psets.items():
            if pset_name.startswith("Pset_Tekla") or pset_name.startswith("Tekla"):
                for k, v in props.items():
                    tekla_props[k] = v

        # Flatten quantities across all Qto_* sets — names are standardised
        # (Length, NetWeight, GrossWeight, NetVolume, NetSurfaceArea, ...).
        quant_flat: Dict[str, Any] = {}
        for qset_name, quants in quantities.items():
            for k, v in quants.items():
                quant_flat[k] = v

        return {
            "entity": product.is_a(),
            "guid": product.GlobalId,
            "type_guid": type_obj.GlobalId if type_obj is not None else None,
            "type_name": profile_name,
            "material": material_name,
            "tekla_properties": tekla_props,
            "quantities": quant_flat,
            "property_sets": psets,
        }
