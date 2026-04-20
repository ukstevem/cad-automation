"""
Upload router with file validation and STEP parsing
"""
import os
import uuid
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
import structlog

from app.config import settings
from app.validators import FileValidator
from app.parsers import STEPParser
from app.exceptions import (
    FileValidationError,
    FileSizeError,
    FileTypeError,
    STEPParseError,
    GeometryExtractionError
)

router = APIRouter()
logger = structlog.get_logger()


@router.post("/upload/")
async def upload_step_file(
    file: UploadFile = File(...),
    parse_geometry: bool = False
) -> Dict[str, Any]:
    """
    Upload and validate STEP file
    
    Args:
        file: Uploaded STEP file
        parse_geometry: Whether to parse geometry (default: False)
        
    Returns:
        JSON response with upload status and file info
        
    Raises:
        HTTPException: On validation or processing errors
    """
    logger.info(
        "upload_request_received",
        filename=file.filename,
        content_type=file.content_type,
        parse_geometry=parse_geometry
    )
    
    try:
        # Validate the uploaded file
        file_ext, file_size = await FileValidator.validate_upload(
            file.file,
            file.filename
        )
        
        # Generate unique filename to prevent collisions
        unique_id = uuid.uuid4().hex[:8]
        safe_filename = f"{unique_id}_{file.filename}"
        file_path = Path(settings.UPLOAD_DIR) / safe_filename
        
        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Save file
        file.file.seek(0)
        content = file.file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(
            "file_saved",
            filename=safe_filename,
            path=str(file_path),
            size=len(content)
        )
        
        # Prepare response
        response_data = {
            "success": True,
            "message": "File uploaded and validated successfully",
            "file_info": {
                "original_filename": file.filename,
                "saved_filename": safe_filename,
                "file_path": str(file_path),
                "extension": file_ext,
                "size_bytes": file_size,
                "size_mb": round(file_size / 1024 / 1024, 2),
                "unique_id": unique_id
            }
        }
        
        # Parse STEP file if requested. IFC uploads skip this legacy path —
        # their geometry is extracted by the main /analysis/assembly pipeline.
        if parse_geometry and file_ext in settings.STEP_EXTENSIONS:
            try:
                parser = STEPParser(str(file_path))
                parse_result = parser.parse()

                # Extract geometry
                geometry_data = parser.extract_geometry()

                response_data["parse_result"] = parse_result
                response_data["geometry"] = geometry_data

                logger.info(
                    "step_file_parsed",
                    filename=safe_filename,
                    num_shapes=parse_result.get("num_shapes", 0)
                )

            except (STEPParseError, GeometryExtractionError) as e:
                logger.error(
                    "parsing_failed",
                    filename=safe_filename,
                    error=str(e),
                    details=e.details
                )
                response_data["parse_error"] = {
                    "message": str(e),
                    "details": e.details
                }
                response_data["success"] = False
        elif parse_geometry and file_ext in settings.IFC_EXTENSIONS:
            response_data["parse_skipped"] = (
                "parse_geometry=true is ignored for IFC uploads; "
                "use /api/v1/analysis/assembly/{filename} to extract geometry"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data
        )
        
    except FileSizeError as e:
        logger.error("file_size_error", error=str(e), details=e.details)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "File too large",
                "message": str(e),
                "details": e.details
            }
        )
    
    except FileTypeError as e:
        logger.error("file_type_error", error=str(e), details=e.details)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error": "Invalid file type",
                "message": str(e),
                "details": e.details
            }
        )
    
    except FileValidationError as e:
        logger.error("file_validation_error", error=str(e), details=e.details)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Validation failed",
                "message": str(e),
                "details": e.details
            }
        )
    
    except Exception as e:
        logger.error("unexpected_upload_error", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred during file upload",
                "details": {"error_type": type(e).__name__}
            }
        )


@router.get("/validate/{filename}")
async def validate_existing_file(filename: str) -> Dict[str, Any]:
    """
    Validate an existing file in the uploads directory
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        Validation results
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "File not found",
                "filename": filename
            }
        )
    
    try:
        with open(file_path, "rb") as f:
            file_ext, file_size = await FileValidator.validate_upload(f, filename)
        
        return {
            "success": True,
            "filename": filename,
            "extension": file_ext,
            "size_bytes": file_size,
            "size_mb": round(file_size / 1024 / 1024, 2),
            "valid": True
        }
        
    except FileValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Validation failed",
                "message": str(e),
                "details": e.details
            }
        )


@router.post("/parse/{filename}")
async def parse_existing_file(filename: str) -> Dict[str, Any]:
    """
    Parse an existing STEP file and extract geometry
    
    Args:
        filename: Name of the file to parse
        
    Returns:
        Parse results with geometry data
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "File not found",
                "filename": filename
            }
        )
    
    try:
        parser = STEPParser(str(file_path))
        parse_result = parser.parse()
        geometry_data = parser.extract_geometry()
        
        return {
            "success": True,
            "filename": filename,
            "parse_result": parse_result,
            "geometry": geometry_data
        }
        
    except (STEPParseError, GeometryExtractionError) as e:
        logger.error("parse_error", filename=filename, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Parse failed",
                "message": str(e),
                "details": e.details
            }
        )
