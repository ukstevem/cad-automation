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
from OCP.TopoDS import TopoDS_Shape

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
                "metadata": self.metadata,
                "file_path": str(self.file_path)
            }
            
        except Exception as e:
            logger.error("step_parse_error", error=str(e), file=str(self.file_path))
            raise STEPParseError(
                f"Error parsing STEP file: {str(e)}",
                details={"file": str(self.file_path), "error": str(e)}
            )
    
    def _extract_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from parsed STEP file
        
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            "num_shapes": len(self.shapes),
            "shape_types": {},
            "bounding_box": None,
            "file_size": self.file_path.stat().st_size
        }
        
        # Count shape types
        for shape in self.shapes:
            shape_type = self._get_shape_type(shape)
            metadata["shape_types"][shape_type] = metadata["shape_types"].get(shape_type, 0) + 1
        
        # Get bounding box if shapes exist
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
        Extract detailed geometry information
        
        Returns:
            Dictionary with geometry details
            
        Raises:
            GeometryExtractionError: If extraction fails
        """
        logger.info("extracting_geometry", num_shapes=len(self.shapes))
        
        if not self.shapes:
            raise GeometryExtractionError(
                "No shapes available for geometry extraction",
                details={"file": str(self.file_path)}
            )
        
        try:
            geometry_data = {
                "shapes": [],
                "total_volume": 0.0,
                "total_surface_area": 0.0
            }
            
            for idx, shape in enumerate(self.shapes):
                shape_info = {
                    "index": idx,
                    "type": self._get_shape_type(shape),
                    "bounding_box": self._get_bounding_box(shape)
                }
                
                # Try to get volume and surface area for solids
                if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID:
                    try:
                        cq_shape = cq.Shape.cast(shape)
                        
                        # These operations can fail for invalid geometry
                        try:
                            volume = cq_shape.Volume()
                            shape_info["volume"] = volume
                            geometry_data["total_volume"] += volume
                        except:
                            logger.warning(f"Could not calculate volume for shape {idx}")
                        
                        try:
                            area = cq_shape.Area()
                            shape_info["surface_area"] = area
                            geometry_data["total_surface_area"] += area
                        except:
                            logger.warning(f"Could not calculate area for shape {idx}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing solid shape {idx}: {e}")
                
                geometry_data["shapes"].append(shape_info)
            
            logger.info(
                "geometry_extracted",
                num_shapes=len(geometry_data["shapes"]),
                total_volume=geometry_data["total_volume"]
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
