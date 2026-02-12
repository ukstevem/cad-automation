"""
STEP file parser for geometry extraction
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog
import cadquery as cq
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.TopAbs import TopAbs_ShapeEnum
from OCP.TopoDS import TopoDS_Shape, TopoDS_Iterator
from OCP.TopExp import TopExp_Explorer

from app.exceptions import STEPParseError, GeometryExtractionError

logger = structlog.get_logger()


class STEPParser:
    """Parser for STEP files using OpenCascade"""
    
    def __init__(self, file_path: str):
        """
        Initialize STEP parser
        
        Args:
            file_path: Path to STEP file
        """
        self.file_path = Path(file_path)
        self.reader = STEPControl_Reader()
        self.shapes: List[TopoDS_Shape] = []
        self.metadata: Dict[str, Any] = {}
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"STEP file not found: {file_path}")
    
    def parse(self) -> Dict[str, Any]:
        """
        Parse STEP file and extract metadata
        
        Returns:
            Dict containing parse results and metadata
            
        Raises:
            STEPParseError: If parsing fails
        """
        logger.info("parsing_step_file", file_path=str(self.file_path))
        
        try:
            # Read the STEP file
            status = self.reader.ReadFile(str(self.file_path))
            
            if status != IFSelect_ReturnStatus.IFSelect_RetDone:
                raise STEPParseError(
                    "Failed to read STEP file",
                    details={
                        "file": str(self.file_path),
                        "status": status
                    }
                )
            
            # Transfer roots
            self.reader.TransferRoots()
            
            # Get number of shapes
            num_shapes = self.reader.NbShapes()
            
            logger.info("step_file_parsed", num_shapes=num_shapes)
            
            # Extract shapes
            for i in range(1, num_shapes + 1):
                shape = self.reader.Shape(i)
                self.shapes.append(shape)
            
            # Extract metadata
            self.metadata = self._extract_metadata()
            
            return {
                "success": True,
                "num_shapes": num_shapes,
                "num_solids": self.metadata.get("num_solids", 0),
                "metadata": self.metadata,
                "file_path": str(self.file_path)
            }
            
        except Exception as e:
            logger.error("step_parse_error", error=str(e), file=str(self.file_path))
            raise STEPParseError(
                f"Error parsing STEP file: {str(e)}",
                details={"file": str(self.file_path), "error": str(e)}
            )
    
    @staticmethod
    def _find_solids(shape: TopoDS_Shape) -> List[TopoDS_Shape]:
        """Find all SOLID shapes within a shape hierarchy using TopExp_Explorer"""
        solids = []
        explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_SOLID)
        while explorer.More():
            solids.append(explorer.Current())
            explorer.Next()
        return solids

    @staticmethod
    def _get_direct_children(shape: TopoDS_Shape) -> List[TopoDS_Shape]:
        """Get direct child shapes of a compound"""
        children = []
        it = TopoDS_Iterator(shape)
        while it.More():
            children.append(it.Value())
            it.Next()
        return children

    def _build_structure(self, shape: TopoDS_Shape, depth: int = 0) -> Dict[str, Any]:
        """
        Recursively build the shape hierarchy tree.
        Only recurses into COMPOUND/COMPSOLID nodes.
        """
        shape_type = self._get_shape_type(shape)
        node: Dict[str, Any] = {"type": shape_type}

        if shape_type in ("COMPOUND", "COMPSOLID"):
            children = self._get_direct_children(shape)
            node["children"] = [
                self._build_structure(child, depth + 1) for child in children
            ]
        elif shape_type == "SOLID":
            node["bounding_box"] = self._get_bounding_box(shape)
            try:
                cq_shape = cq.Shape.cast(shape)
                node["volume"] = cq_shape.Volume()
                node["surface_area"] = cq_shape.Area()
            except Exception as e:
                logger.warning("solid_measurement_failed", error=str(e), depth=depth)

        return node

    def _extract_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from parsed STEP file, traversing into compounds
        to find all child solids.
        """
        metadata = {
            "num_top_level_shapes": len(self.shapes),
            "shape_types": {},
            "bounding_box": None,
            "file_size": self.file_path.stat().st_size,
            "num_solids": 0,
        }

        # Count top-level shape types
        for shape in self.shapes:
            shape_type = self._get_shape_type(shape)
            metadata["shape_types"][shape_type] = metadata["shape_types"].get(shape_type, 0) + 1

        # Find all solids across the entire hierarchy
        all_solids = []
        for shape in self.shapes:
            all_solids.extend(self._find_solids(shape))
        metadata["num_solids"] = len(all_solids)

        # Bounding box of the full assembly
        if self.shapes:
            try:
                metadata["bounding_box"] = self._get_bounding_box(self.shapes[0])
            except Exception as e:
                logger.warning("bounding_box_extraction_failed", error=str(e))

        return metadata
    
    @staticmethod
    def _get_shape_type(shape: TopoDS_Shape) -> str:
        """Get human-readable shape type"""
        shape_type_map = {
            TopAbs_ShapeEnum.TopAbs_COMPOUND: "COMPOUND",
            TopAbs_ShapeEnum.TopAbs_COMPSOLID: "COMPSOLID",
            TopAbs_ShapeEnum.TopAbs_SOLID: "SOLID",
            TopAbs_ShapeEnum.TopAbs_SHELL: "SHELL",
            TopAbs_ShapeEnum.TopAbs_FACE: "FACE",
            TopAbs_ShapeEnum.TopAbs_WIRE: "WIRE",
            TopAbs_ShapeEnum.TopAbs_EDGE: "EDGE",
            TopAbs_ShapeEnum.TopAbs_VERTEX: "VERTEX",
        }
        return shape_type_map.get(shape.ShapeType(), "UNKNOWN")
    
    @staticmethod
    def _get_bounding_box(shape: TopoDS_Shape) -> Dict[str, float]:
        """
        Get bounding box of a shape
        
        Returns:
            Dictionary with min/max coordinates
        """
        try:
            # Use CadQuery for easier bounding box extraction
            cq_shape = cq.Shape.cast(shape)
            bb = cq_shape.BoundingBox()
            
            return {
                "xmin": bb.xmin,
                "xmax": bb.xmax,
                "ymin": bb.ymin,
                "ymax": bb.ymax,
                "zmin": bb.zmin,
                "zmax": bb.zmax,
                "length": bb.xmax - bb.xmin,
                "width": bb.ymax - bb.ymin,
                "height": bb.zmax - bb.zmin
            }
        except Exception as e:
            logger.warning("bounding_box_calculation_failed", error=str(e))
            return {}
    
    def extract_geometry(self) -> Dict[str, Any]:
        """
        Extract detailed geometry information, traversing into compounds
        to find and measure all solids.
        """
        logger.info("extracting_geometry", num_shapes=len(self.shapes))

        if not self.shapes:
            raise GeometryExtractionError(
                "No shapes available for geometry extraction",
                details={"file": str(self.file_path)}
            )

        try:
            # Collect all solids from the entire hierarchy
            all_solids = []
            for shape in self.shapes:
                all_solids.extend(self._find_solids(shape))

            geometry_data = {
                "structure": [self._build_structure(s) for s in self.shapes],
                "solids": [],
                "num_solids": len(all_solids),
                "total_volume": 0.0,
                "total_surface_area": 0.0,
            }

            for idx, solid in enumerate(all_solids):
                solid_info = {
                    "index": idx,
                    "type": "SOLID",
                    "bounding_box": self._get_bounding_box(solid),
                }

                try:
                    cq_shape = cq.Shape.cast(solid)
                    try:
                        volume = cq_shape.Volume()
                        solid_info["volume"] = volume
                        geometry_data["total_volume"] += volume
                    except Exception:
                        logger.warning("volume_calc_failed", solid_index=idx)

                    try:
                        area = cq_shape.Area()
                        solid_info["surface_area"] = area
                        geometry_data["total_surface_area"] += area
                    except Exception:
                        logger.warning("area_calc_failed", solid_index=idx)
                except Exception as e:
                    logger.warning("solid_processing_failed", solid_index=idx, error=str(e))

                geometry_data["solids"].append(solid_info)

            logger.info(
                "geometry_extracted",
                num_solids=len(geometry_data["solids"]),
                total_volume=geometry_data["total_volume"],
            )

            return geometry_data

        except Exception as e:
            logger.error("geometry_extraction_error", error=str(e))
            raise GeometryExtractionError(
                f"Failed to extract geometry: {str(e)}",
                details={"file": str(self.file_path), "error": str(e)}
            )
    
    def get_shapes(self) -> List[TopoDS_Shape]:
        """Get list of parsed shapes"""
        return self.shapes
    
    def get_cadquery_assembly(self) -> Optional[cq.Assembly]:
        """
        Convert to CadQuery Assembly for easier manipulation
        
        Returns:
            CadQuery Assembly or None if conversion fails
        """
        try:
            if not self.shapes:
                return None
            
            # Create assembly
            assy = cq.Assembly()
            
            for idx, shape in enumerate(self.shapes):
                cq_shape = cq.Shape.cast(shape)
                assy.add(cq_shape, name=f"part_{idx}")
            
            return assy
            
        except Exception as e:
            logger.error("assembly_conversion_failed", error=str(e))
            return None
