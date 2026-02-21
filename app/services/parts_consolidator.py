"""
Parts consolidation service.

Computes a geometric fingerprint for each unique XCAF prototype (ref_id) in the
cached assembly tree, then groups those with identical fingerprints into
consolidation groups.

Two levels of grouping are returned:

part_groups
    Whole-prototype level (existing behaviour).  Each group covers one or more
    ref_ids whose overall compound shape has an identical fingerprint.

solid_groups
    Individual-solid level (new).  For ``part_multi_solid`` prototypes, each
    constituent solid is fingerprinted separately.  Groups that contain solids
    from two or more *different* parent ref_ids represent shared solid geometry
    (e.g. two different multi-solid weldments that both include the same bracket
    body).  These groups let the BOM merge those solid bodies into one row.

intra_solid_groups
    Like solid_groups but for identical solids *within the same ref_id*.  When
    a multi-solid prototype contains two or more geometrically identical solid
    bodies (e.g. four identical gusset plates welded into one weldment), each
    group lists the ref_id and the solid_indices that share the same fingerprint.
    These groups let the BOM collapse duplicate intra-part solid rows into one.

Fingerprint
-----------
(volume, surface_area, dim0, dim1, dim2, inertia0, inertia1, inertia2, chirality)

``volume``, ``surface_area`` and the three sorted bounding-box dimensions are
rounded to _ROUND_DP decimal places.

``inertia0..2`` are the three **principal moments of inertia** (eigenvalues of
the inertia tensor relative to the centre of mass), sorted by magnitude and
rounded to the nearest 1000 mm⁵.  They are invariant under rigid-body
transformations but sensitive to mass distribution, so two shapes with the same
volume/area/bbox but holes at different positions still get different fingerprints.

``chirality`` is +1 or −1: the determinant of the canonical principal-axis
frame, where each axis direction is chosen so that the CoM–BboxCenter offset
projects *positively* onto it.  For a shape and its mirror image the canonical
frames have opposite determinants, so this correctly distinguishes left-hand
from right-hand parts.

For nearly-symmetric shapes (CoM within _CHIRALITY_THR of the bbox centre along
all three principal axes) a lexicographic tiebreaker is used; both a symmetric
part and its mirror image receive the same chirality value (+1 or −1) since
second-order moments cannot distinguish them.

Face count is excluded: STEP files can represent identical geometry with
different face counts depending on how edges/faces are split during export.

The ``is_mirrored`` flag on assembly tree instances (set when the XCAF transform
determinant is negative) handles the complementary case: one prototype placed
both normally and with a mirror transform.

Parts with zero or negligible volume (shells, wires, no solid geometry) are
placed in individual singleton groups and are not fingerprint-matched.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.TDF import TDF_Label, TDF_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID

from app.exceptions import STEPParseError

logger = structlog.get_logger()

# Rounding precision for fingerprint components (mm-scale geometry)
_ROUND_DP = 1  # 0.1 mm / mm² / mm³ resolution

# Parts with |volume| below this threshold are treated as zero-volume
_MIN_VOLUME = 0.001  # mm³

# Chirality: minimum CoM–BboxCenter projection (mm) onto a principal axis
# before using the signed projection to set that axis's canonical direction.
# Below this the axis is considered locally symmetric and a lexicographic
# tiebreaker is used instead.  Value is compared against the raw projection
# distance; 0.01 mm is above floating-point noise but well below any real
# geometric asymmetry that would make a part chiral.
_CHIRALITY_THR = 0.01  # mm


def _canonical_chirality(vol_props, pp, xmin, ymin, zmin, xmax, ymax, zmax) -> int:
    """
    Return +1 or −1: the handedness of the canonical principal-axis frame.

    Each principal axis is given a canonical direction so that the CoM–BboxCenter
    offset projects positively onto it (or, when the projection is below the
    noise threshold, so that the largest-magnitude component is positive).

    The determinant of the 3×3 matrix whose rows are these canonical unit
    vectors is ±1.  Under a geometric reflection it flips sign, so two shapes
    that are mirror images receive opposite chirality values.

    Returns +1 on any failure so that chirality detection degrades gracefully
    to the pre-chirality behaviour (all parts treated as +1, i.e. matched by
    the remaining 8 fingerprint components only).
    """
    try:
        com = vol_props.CentreOfMass()
        delta = (
            com.X() - (xmin + xmax) / 2.0,
            com.Y() - (ymin + ymax) / 2.0,
            com.Z() - (zmin + zmax) / 2.0,
        )

        axes = (
            pp.FirstAxisOfInertia(),
            pp.SecondAxisOfInertia(),
            pp.ThirdAxisOfInertia(),
        )

        rows = []
        for ax in axes:
            v = (float(ax.X()), float(ax.Y()), float(ax.Z()))
            proj = delta[0] * v[0] + delta[1] * v[1] + delta[2] * v[2]
            if abs(proj) > _CHIRALITY_THR:
                # Asymmetric axis: canonical direction aligns with the offset.
                s = 1.0 if proj > 0.0 else -1.0
            else:
                # Nearly-symmetric axis: lexicographic tiebreaker — flip so
                # the component with the largest absolute value is positive.
                max_i = max(range(3), key=lambda i: abs(v[i]))
                s = 1.0 if v[max_i] >= 0.0 else -1.0
            rows.append((s * v[0], s * v[1], s * v[2]))

        (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = rows
        det = (
            a0 * (b1 * c2 - b2 * c1)
            - a1 * (b0 * c2 - b2 * c0)
            + a2 * (b0 * c1 - b1 * c0)
        )
        return 1 if det >= 0.0 else -1

    except Exception:
        return 1  # graceful fallback


class PartsConsolidator:
    """Group geometrically identical parts across XCAF ref_id boundaries."""

    def __init__(self, file_path: str, analysis_data: dict):
        """
        Args:
            file_path:      Path to the STEP file.
            analysis_data:  The ``analysis`` section of the JSON sidecar
                            (dict containing ``assembly_tree`` and ``summary``).
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"STEP file not found: {file_path}")

        # Collect all unique leaf parts (by ref_id) from the cached tree.
        # Each entry stores {name, node_type, instances: [...]}.
        self._parts_by_ref: Dict[str, Dict] = {}
        self._collect_parts(
            analysis_data.get("assembly_tree", []), None, self._parts_by_ref
        )

    def consolidate(self) -> Dict[str, Any]:
        """
        Compute shape fingerprints for all unique prototypes and group those
        with identical geometry.

        Returns a dict::

            {
                "part_groups":       [...],   # whole-prototype consolidation groups
                "solid_groups":      [...],   # cross-ref individual-solid groups
                "intra_solid_groups":[...],   # intra-part identical-solid groups
            }

        Each part_group dict contains::

            canonical_name   – most common name among grouped ref_ids
            ref_ids          – list of ref_ids in this group (length ≥ 1)
            all_node_ids     – flat list of all instance node_ids
            normal_count     – instances where is_mirrored is False
            mirrored_count   – instances where is_mirrored is True
            total_count      – normal_count + mirrored_count
            fingerprint      – list [vol, area, d0, d1, d2, I0, I1, I2, chirality] or null

        Each solid_group dict contains::

            members          – list of {ref_id, solid_index} dicts (≥ 2 refs)
            fingerprint      – list [vol, area, d0, d1, d2, I0, I1, I2, chirality]

        Each intra_solid_group dict contains::

            ref_id           – the parent ref_id
            solid_indices    – list of solid indices sharing the same fingerprint
            fingerprint      – list [vol, area, d0, d1, d2, I0, I1, I2, chirality]

        solid_groups only contains groups with members from ≥ 2 different ref_ids.
        intra_solid_groups only contains groups with ≥ 2 solid indices from the
        same ref_id (i.e. the cases excluded from solid_groups).

        part_groups ordering:
            1. Multi-ref groups first (genuine consolidations)
            2. Within same group size: highest total_count first
            3. Alphabetical by canonical_name
        """
        if not self._parts_by_ref:
            return {"part_groups": [], "solid_groups": []}

        logger.info("consolidating_parts", num_unique_refs=len(self._parts_by_ref))

        doc, shape_tool = self._read_xcaf()

        # ------------------------------------------------------------------
        # Part-level groups (whole-prototype fingerprints)
        # ------------------------------------------------------------------
        fingerprints: Dict[str, Optional[Tuple]] = {}
        for ref_id in self._parts_by_ref:
            fingerprints[ref_id] = self._fingerprint_for_ref(doc, ref_id)

        fp_groups: Dict[Any, List[str]] = {}
        _singleton_key = 0
        for ref_id, fp in fingerprints.items():
            if fp is None:
                fp_groups[_singleton_key] = [ref_id]
                _singleton_key += 1
            else:
                fp_groups.setdefault(fp, []).append(ref_id)

        part_groups: List[Dict[str, Any]] = []
        for fp_key, ref_ids in fp_groups.items():
            all_node_ids: List[str] = []
            normal_count = 0
            mirrored_count = 0
            names: List[str] = []

            for rid in ref_ids:
                info = self._parts_by_ref[rid]
                names.append(info["name"])
                for inst in info["instances"]:
                    all_node_ids.append(inst["node_id"])
                    if inst.get("is_mirrored"):
                        mirrored_count += 1
                    else:
                        normal_count += 1

            canonical_name = max(set(names), key=names.count)
            fp_list = list(fp_key) if isinstance(fp_key, tuple) else None

            part_groups.append(
                {
                    "canonical_name": canonical_name,
                    "ref_ids": ref_ids,
                    "all_node_ids": all_node_ids,
                    "normal_count": normal_count,
                    "mirrored_count": mirrored_count,
                    "total_count": normal_count + mirrored_count,
                    "fingerprint": fp_list,
                }
            )

        part_groups.sort(
            key=lambda g: (
                len(g["ref_ids"]) == 1,
                -g["total_count"],
                g["canonical_name"].lower(),
            )
        )

        # ------------------------------------------------------------------
        # Solid-level groups (individual solid body fingerprints)
        # ------------------------------------------------------------------
        # Only process multi-solid prototypes — single-solid parts are already
        # covered at the part-group level.
        solid_fp_bucket: Dict[Tuple, List[Dict]] = {}
        for ref_id, info in self._parts_by_ref.items():
            if info.get("node_type") != "part_multi_solid":
                continue
            solid_fps = self._solid_fingerprints_for_ref(doc, ref_id)
            logger.info(
                "solid_fingerprints_computed",
                ref_id=ref_id,
                name=info["name"],
                solids=[
                    {"index": idx, "fp": list(fp) if fp else None}
                    for idx, fp in solid_fps
                ],
            )
            for solid_index, fp in solid_fps:
                if fp is None:
                    continue
                solid_fp_bucket.setdefault(fp, []).append(
                    {"ref_id": ref_id, "solid_index": solid_index}
                )

        # Partition solid fp groups: cross-ref (≥ 2 different ref_ids) vs
        # intra-part (all members share the same ref_id, ≥ 2 solid indices).
        solid_groups: List[Dict[str, Any]] = []
        intra_solid_groups: List[Dict[str, Any]] = []
        for fp_key, members in solid_fp_bucket.items():
            unique_refs = {m["ref_id"] for m in members}
            if len(unique_refs) >= 2:
                solid_groups.append(
                    {"members": members, "fingerprint": list(fp_key)}
                )
            elif len(members) >= 2:
                # All solids belong to the same ref_id — intra-part duplicates
                intra_solid_groups.append(
                    {
                        "ref_id": members[0]["ref_id"],
                        "solid_indices": [m["solid_index"] for m in members],
                        "fingerprint": list(fp_key),
                    }
                )

        consolidated = sum(1 for g in part_groups if len(g["ref_ids"]) > 1)
        logger.info(
            "consolidation_complete",
            total_part_groups=len(part_groups),
            consolidated_part_groups=consolidated,
            solid_groups=len(solid_groups),
            intra_solid_groups=len(intra_solid_groups),
            original_refs=len(self._parts_by_ref),
        )
        return {
            "part_groups": part_groups,
            "solid_groups": solid_groups,
            "intra_solid_groups": intra_solid_groups,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_parts(
        nodes: list, parent_name: Optional[str], result: Dict
    ) -> None:
        """Recursively walk the assembly tree, collecting leaf parts by ref_id."""
        for node in nodes:
            if node.get("node_type") == "assembly":
                PartsConsolidator._collect_parts(
                    node.get("children", []), node["name"], result
                )
            else:
                ref_id = node.get("ref_id") or node["id"]
                if ref_id not in result:
                    result[ref_id] = {
                        "name": node["name"],
                        "node_type": node.get("node_type", ""),
                        "instances": [],
                    }
                result[ref_id]["instances"].append(
                    {
                        "node_id": node["id"],
                        "is_mirrored": node.get("is_mirrored", False),
                    }
                )

    def _read_xcaf(self):
        """Open the STEP file via STEPCAFControl_Reader (name mode only)."""
        doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(False)
        reader.SetLayerMode(False)

        status = reader.ReadFile(str(self.file_path))
        if status != IFSelect_RetDone:
            raise STEPParseError(
                "STEPCAFControl_Reader failed",
                details={"file": str(self.file_path), "status": int(status)},
            )

        if not reader.Transfer(doc):
            raise STEPParseError(
                "STEPCAFControl_Reader transfer failed",
                details={"file": str(self.file_path)},
            )

        shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        return doc, shape_tool

    def _fingerprint_shape(self, shape) -> Optional[Tuple]:
        """
        Compute a geometric fingerprint for any TopoDS_Shape.

        Returns ``(volume, surface_area, dim0, dim1, dim2, I0, I1, I2, chirality)``
        or ``None`` if the shape has negligible volume.

        ``I0..I2`` are the sorted principal moments of inertia (eigenvalues of
        the inertia tensor relative to the centre of mass, rounded to the
        nearest 1000 mm⁵).

        ``chirality`` is +1 or −1: the determinant of the canonical principal-
        axis frame built by directing each eigenvector so that the CoM–
        BboxCenter offset projects positively onto it.  Mirror-image shapes
        produce opposite chirality values.  See ``_canonical_chirality`` for
        details and the symmetric-part tiebreaker.
        """
        try:
            vol_props = GProp_GProps()
            BRepGProp.VolumeProperties_s(shape, vol_props)
            volume = round(vol_props.Mass(), _ROUND_DP)

            if abs(volume) < _MIN_VOLUME:
                return None

            surf_props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(shape, surf_props)
            surface = round(surf_props.Mass(), _ROUND_DP)

            # Use SetGap(0) + AddClose for a tight bounding box that is not
            # inflated by per-face tolerances.  Without this, two geometrically
            # identical shapes with different STEP tolerances can produce
            # different bounding-box sizes and therefore different fingerprints.
            bbox = Bnd_Box()
            bbox.SetGap(0.0)
            BRepBndLib.AddClose_s(shape, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            dims = tuple(
                sorted(
                    [
                        round(xmax - xmin, _ROUND_DP),
                        round(ymax - ymin, _ROUND_DP),
                        round(zmax - zmin, _ROUND_DP),
                    ]
                )
            )

            # Principal moments of inertia: eigenvalues of the inertia tensor
            # computed relative to the centre of mass by OCC.  Invariant under
            # rigid-body transformations but sensitive to internal mass
            # distribution (e.g. holes at different positions).
            #
            # Rounded to the nearest 1000 mm⁵ to absorb the small numerical
            # noise (~5 units on 10^13-scale values) that arises when the same
            # geometry is stored with different face topology in STEP.  A
            # realistic engineering difference (10 mm hole moved 5 mm) produces
            # a ~40 000 mm⁵ change — 40× the rounding unit.
            pp = vol_props.PrincipalProperties()
            moments = tuple(sorted(round(v, -3) for v in pp.Moments()))

            chirality = _canonical_chirality(
                vol_props, pp, xmin, ymin, zmin, xmax, ymax, zmax
            )

            return (volume, surface, *dims, *moments, chirality)

        except Exception as exc:
            logger.warning("fingerprint_shape_failed", error=str(exc))
            return None

    def _fingerprint_for_ref(self, doc, ref_id: str) -> Optional[Tuple]:
        """
        Compute a geometric fingerprint for a prototype label entry.

        Looks up the label by ref_id entry string, then delegates to
        ``_fingerprint_shape``.  Returns ``None`` on lookup failure or if the
        shape has negligible volume.
        """
        try:
            label = TDF_Label()
            TDF_Tool.Label_s(doc.GetData(), TCollection_AsciiString(ref_id), label)
            if label.IsNull():
                return None

            shape = XCAFDoc_ShapeTool.GetShape_s(label)
            if shape is None or shape.IsNull():
                return None

            return self._fingerprint_shape(shape)

        except Exception as exc:
            logger.warning("fingerprint_failed", ref_id=ref_id, error=str(exc))
            return None

    def _solid_fingerprints_for_ref(
        self, doc, ref_id: str
    ) -> List[Tuple[int, Optional[Tuple]]]:
        """
        For a multi-solid prototype, fingerprint each constituent solid
        individually.

        Returns a list of ``(solid_index, fingerprint)`` pairs — one per solid
        in the compound, in the order returned by TopExp_Explorer.  The
        solid_index corresponds to the ``:sN`` suffix used in the frontend's
        synthetic solid nodeIds (``"<ref_id>:s0"``, ``"<ref_id>:s1"``, …).
        """
        try:
            label = TDF_Label()
            TDF_Tool.Label_s(doc.GetData(), TCollection_AsciiString(ref_id), label)
            if label.IsNull():
                return []

            shape = XCAFDoc_ShapeTool.GetShape_s(label)
            if shape is None or shape.IsNull():
                return []

            results: List[Tuple[int, Optional[Tuple]]] = []
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            idx = 0
            while exp.More():
                # Strip the accumulated location before fingerprinting so that
                # identical solid geometry always produces the same fingerprint
                # regardless of where the solid sits within its parent compound.
                # Without this, the bounding-box dimensions can differ between
                # two geometrically identical solids that have different
                # orientations/translations in their respective compounds.
                solid = exp.Current().Located(TopLoc_Location())
                fp = self._fingerprint_shape(solid)
                results.append((idx, fp))
                idx += 1
                exp.Next()

            return results

        except Exception as exc:
            logger.warning(
                "solid_fingerprints_failed", ref_id=ref_id, error=str(exc)
            )
            return []
