"""
Tests for upload router endpoints
"""
import pytest
import os
from fastapi.testclient import TestClient
from pathlib import Path
import io

from app.main import app
from app.config import settings


client = TestClient(app)


class TestUploadEndpoint:
    """Test suite for upload endpoint"""
    
    def setup_method(self):
        """Setup test environment"""
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    def test_upload_valid_step_file(self):
        """Test uploading a valid STEP file"""
        content = b"ISO-10303-21;\nHEADER;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        
        files = {
            "file": ("test_model.step", io.BytesIO(content), "application/step")
        }
        
        response = client.post("/api/v1/upload/", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "file_info" in data
        assert data["file_info"]["extension"] == ".step"
        assert data["file_info"]["size_bytes"] == len(content)
    
    def test_upload_with_parse_geometry(self):
        """Test uploading with geometry parsing enabled"""
        # Note: This test may fail without a real STEP file
        # In production, you'd use a fixture with a real STEP file
        content = b"ISO-10303-21;\nHEADER;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        
        files = {
            "file": ("test_model.step", io.BytesIO(content), "application/step")
        }
        
        response = client.post(
            "/api/v1/upload/",
            files=files,
            params={"parse_geometry": True}
        )
        
        # May succeed or fail depending on STEP validity
        assert response.status_code in [200, 422]
    
    def test_upload_invalid_extension(self):
        """Test uploading file with invalid extension"""
        content = b"some content"
        
        files = {
            "file": ("model.stl", io.BytesIO(content), "application/octet-stream")
        }
        
        response = client.post("/api/v1/upload/", files=files)
        
        assert response.status_code == 415  # Unsupported Media Type
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
    
    def test_upload_invalid_step_content(self):
        """Test uploading file with .step extension but invalid content"""
        content = b"INVALID STEP CONTENT"
        
        files = {
            "file": ("model.step", io.BytesIO(content), "application/step")
        }
        
        response = client.post("/api/v1/upload/", files=files)
        
        assert response.status_code == 415
        data = response.json()
        assert "STEP format" in str(data)
    
    def test_upload_empty_file(self):
        """Test uploading empty file"""
        content = b""
        
        files = {
            "file": ("empty.step", io.BytesIO(content), "application/step")
        }
        
        response = client.post("/api/v1/upload/", files=files)
        
        assert response.status_code == 413  # Request Entity Too Large (or could be 400)
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "directories" in data


class TestValidateEndpoint:
    """Test suite for validate endpoint"""
    
    def test_validate_nonexistent_file(self):
        """Test validating a file that doesn't exist"""
        response = client.get("/api/v1/validate/nonexistent.step")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in str(data).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
