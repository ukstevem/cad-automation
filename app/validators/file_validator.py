"""
File validation utilities for uploaded STEP files
"""
import os
import magic
from pathlib import Path
from typing import BinaryIO, Tuple
import structlog

from app.config import settings
from app.exceptions import FileSizeError, FileTypeError, FileValidationError

logger = structlog.get_logger()


class FileValidator:
    """Validates uploaded files for STEP processing"""
    
    @staticmethod
    async def validate_file_size(file: BinaryIO, filename: str) -> int:
        """
        Validate file size is within limits
        
        Args:
            file: File object to validate
            filename: Name of the file
            
        Returns:
            int: File size in bytes
            
        Raises:
            FileSizeError: If file exceeds maximum size
        """
        # Get file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        logger.info(
            "validating_file_size",
            filename=filename,
            size_bytes=file_size,
            max_size_bytes=settings.MAX_UPLOAD_SIZE
        )
        
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise FileSizeError(
                f"File size ({file_size} bytes) exceeds maximum allowed size "
                f"({settings.MAX_UPLOAD_SIZE} bytes)",
                details={
                    "filename": filename,
                    "size": file_size,
                    "max_size": settings.MAX_UPLOAD_SIZE,
                    "size_mb": round(file_size / 1024 / 1024, 2),
                    "max_size_mb": round(settings.MAX_UPLOAD_SIZE / 1024 / 1024, 2)
                }
            )
        
        if file_size == 0:
            raise FileSizeError(
                "File is empty",
                details={"filename": filename, "size": 0}
            )
        
        return file_size
    
    @staticmethod
    def validate_file_extension(filename: str) -> str:
        """
        Validate file has an allowed extension
        
        Args:
            filename: Name of the file
            
        Returns:
            str: Validated file extension (lowercase)
            
        Raises:
            FileTypeError: If extension is not allowed
        """
        file_ext = Path(filename).suffix.lower()
        
        logger.info(
            "validating_file_extension",
            filename=filename,
            extension=file_ext,
            allowed=list(settings.ALLOWED_EXTENSIONS)
        )
        
        if not file_ext:
            raise FileTypeError(
                "File has no extension",
                details={
                    "filename": filename,
                    "allowed_extensions": list(settings.ALLOWED_EXTENSIONS)
                }
            )
        
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise FileTypeError(
                f"File extension '{file_ext}' is not allowed",
                details={
                    "filename": filename,
                    "extension": file_ext,
                    "allowed_extensions": list(settings.ALLOWED_EXTENSIONS)
                }
            )
        
        return file_ext
    
    @staticmethod
    async def validate_step_content(file: BinaryIO, filename: str) -> bool:
        """
        Validate file content is actually a STEP file

        Args:
            file: File object to validate
            filename: Name of the file

        Returns:
            bool: True if valid STEP content

        Raises:
            FileTypeError: If content is not a valid STEP file
        """
        # Read header portion for validation
        file.seek(0)
        header = file.read(settings.MAX_STEP_HEADER_SIZE)
        file.seek(0)

        # Check for STEP magic bytes
        if not header.startswith(settings.STEP_MAGIC_BYTES):
            raise FileTypeError(
                "File does not contain valid STEP format header",
                details={
                    "filename": filename,
                    "expected_header": settings.STEP_MAGIC_BYTES.decode('utf-8', errors='ignore'),
                    "found_header": header[:50].decode('utf-8', errors='ignore')
                }
            )

        # Additional STEP format validation
        header_str = header.decode('utf-8', errors='ignore')

        required_sections = ['HEADER;', 'DATA;']
        missing_sections = [s for s in required_sections if s not in header_str]

        if missing_sections:
            logger.warning(
                "step_header_incomplete",
                filename=filename,
                missing_sections=missing_sections
            )

        logger.info(
            "step_content_validated",
            filename=filename,
            header_valid=True
        )

        return True

    @staticmethod
    async def validate_ifc_content(file: BinaryIO, filename: str) -> bool:
        """
        Validate file content is a STEP-encoded IFC file.

        IFC Part 21 files share the ``ISO-10303-21;`` magic bytes with STEP,
        but carry ``FILE_SCHEMA(('IFC...'))`` in the HEADER section rather
        than a STEP application protocol identifier.
        """
        file.seek(0)
        header = file.read(settings.MAX_IFC_HEADER_SIZE)
        file.seek(0)

        if not header.startswith(settings.STEP_MAGIC_BYTES):
            raise FileTypeError(
                "File does not contain valid IFC (ISO-10303-21) header",
                details={
                    "filename": filename,
                    "expected_header": settings.STEP_MAGIC_BYTES.decode('utf-8', errors='ignore'),
                    "found_header": header[:50].decode('utf-8', errors='ignore'),
                }
            )

        header_str = header.decode('utf-8', errors='ignore')
        if "FILE_SCHEMA" not in header_str or "IFC" not in header_str:
            raise FileTypeError(
                "File header is ISO-10303-21 but does not declare an IFC schema",
                details={
                    "filename": filename,
                    "found_header_snippet": header_str[:200],
                }
            )

        logger.info("ifc_content_validated", filename=filename, header_valid=True)
        return True
    
    @staticmethod
    async def validate_upload(file: BinaryIO, filename: str) -> Tuple[str, int]:
        """
        Perform complete validation on uploaded file
        
        Args:
            file: File object to validate
            filename: Name of the file
            
        Returns:
            Tuple[str, int]: Validated extension and file size
            
        Raises:
            FileValidationError: If any validation check fails
        """
        logger.info("starting_file_validation", filename=filename)
        
        try:
            # 1. Validate extension
            file_ext = FileValidator.validate_file_extension(filename)

            # 2. Validate size
            file_size = await FileValidator.validate_file_size(file, filename)

            # 3. Validate content — dispatched by extension
            if file_ext in settings.IFC_EXTENSIONS:
                await FileValidator.validate_ifc_content(file, filename)
            else:
                await FileValidator.validate_step_content(file, filename)

            logger.info(
                "file_validation_complete",
                filename=filename,
                extension=file_ext,
                size=file_size,
                status="success"
            )

            return file_ext, file_size
            
        except (FileSizeError, FileTypeError) as e:
            logger.error(
                "file_validation_failed",
                filename=filename,
                error=str(e),
                details=e.details
            )
            raise
        except Exception as e:
            logger.error(
                "unexpected_validation_error",
                filename=filename,
                error=str(e)
            )
            raise FileValidationError(
                f"Unexpected error during validation: {str(e)}",
                details={"filename": filename, "error_type": type(e).__name__}
            )
