"""
Configuration management for CAD automation system
"""
from pydantic_settings import BaseSettings
from typing import Set


class Settings(BaseSettings):
    """Application settings"""
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "/uploads"
    ALLOWED_EXTENSIONS: Set[str] = {".step", ".stp"}
    
    # STEP file validation
    STEP_MAGIC_BYTES: bytes = b"ISO-10303-21;"
    MAX_STEP_HEADER_SIZE: int = 1024  # bytes to read for validation
    
    # Processing settings
    TEMP_DIR: str = "/tmp/cad-processing"
    OUTPUT_DIR: str = "/outputs"

    # Analysis cache
    ANALYSIS_OUTPUT_DIR: str = "/outputs/analysis"

    # STL generation settings
    STL_OUTPUT_DIR: str = "/outputs/stl"
    STL_LINEAR_DEFLECTION: float = 0.5
    STL_ANGULAR_DEFLECTION: float = 0.5
    
    # API settings
    API_TITLE: str = "CAD Automation API"
    API_VERSION: str = "0.2.0"
    API_DESCRIPTION: str = "STEP to DXF/DWG/NC conversion with BOM generation"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
