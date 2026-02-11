"""
Tests for file validators
"""
import pytest
import io
from pathlib import Path

from app.validators import FileValidator
from app.exceptions import FileSizeError, FileTypeError, FileValidationError
from app.config import settings


class TestFileValidator:
    """Test suite for FileValidator"""
    
    @pytest.mark.asyncio
    async def test_validate_file_size_valid(self):
        """Test file size validation with valid file"""
        content = b"test content"
        file = io.BytesIO(content)
        
        size = await FileValidator.validate_file_size(file, "test.step")
        
        assert size == len(content)
        assert file.tell() == 0  # File pointer reset
    
    @pytest.mark.asyncio
    async def test_validate_file_size_too_large(self):
        """Test file size validation with oversized file"""
        # Create file larger than max size
        content = b"x" * (settings.MAX_UPLOAD_SIZE + 1)
        file = io.BytesIO(content)
        
        with pytest.raises(FileSizeError) as exc_info:
            await FileValidator.validate_file_size(file, "large.step")
        
        assert "exceeds maximum" in str(exc_info.value)
        assert exc_info.value.details["size"] == len(content)
    
    @pytest.mark.asyncio
    async def test_validate_file_size_empty(self):
        """Test file size validation with empty file"""
        file = io.BytesIO(b"")
        
        with pytest.raises(FileSizeError) as exc_info:
            await FileValidator.validate_file_size(file, "empty.step")
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_validate_file_extension_valid_step(self):
        """Test extension validation with .step file"""
        ext = FileValidator.validate_file_extension("model.step")
        assert ext == ".step"
    
    def test_validate_file_extension_valid_stp(self):
        """Test extension validation with .stp file"""
        ext = FileValidator.validate_file_extension("MODEL.STP")
        assert ext == ".stp"
    
    def test_validate_file_extension_invalid(self):
        """Test extension validation with invalid extension"""
        with pytest.raises(FileTypeError) as exc_info:
            FileValidator.validate_file_extension("model.stl")
        
        assert "not allowed" in str(exc_info.value)
        assert exc_info.value.details["extension"] == ".stl"
    
    def test_validate_file_extension_missing(self):
        """Test extension validation with no extension"""
        with pytest.raises(FileTypeError) as exc_info:
            FileValidator.validate_file_extension("modelfile")
        
        assert "no extension" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_step_content_valid(self):
        """Test STEP content validation with valid file"""
        content = b"ISO-10303-21;\nHEADER;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        file = io.BytesIO(content)
        
        result = await FileValidator.validate_step_content(file, "test.step")
        
        assert result is True
        assert file.tell() == 0  # File pointer reset
    
    @pytest.mark.asyncio
    async def test_validate_step_content_invalid_header(self):
        """Test STEP content validation with invalid header"""
        content = b"INVALID HEADER\nsome content"
        file = io.BytesIO(content)
        
        with pytest.raises(FileTypeError) as exc_info:
            await FileValidator.validate_step_content(file, "invalid.step")
        
        assert "not contain valid STEP format" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_upload_complete_success(self):
        """Test complete upload validation with valid STEP file"""
        content = b"ISO-10303-21;\nHEADER;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        file = io.BytesIO(content)
        
        ext, size = await FileValidator.validate_upload(file, "model.step")
        
        assert ext == ".step"
        assert size == len(content)
    
    @pytest.mark.asyncio
    async def test_validate_upload_invalid_extension(self):
        """Test complete upload validation with invalid extension"""
        content = b"ISO-10303-21;\nHEADER;\n"
        file = io.BytesIO(content)
        
        with pytest.raises(FileTypeError):
            await FileValidator.validate_upload(file, "model.stl")
    
    @pytest.mark.asyncio
    async def test_validate_upload_invalid_content(self):
        """Test complete upload validation with invalid STEP content"""
        content = b"INVALID CONTENT"
        file = io.BytesIO(content)
        
        with pytest.raises(FileTypeError):
            await FileValidator.validate_upload(file, "model.step")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
