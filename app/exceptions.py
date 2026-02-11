"""
Custom exceptions for CAD automation system
"""


class CADAutomationException(Exception):
    """Base exception for CAD automation errors"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class FileValidationError(CADAutomationException):
    """Raised when file validation fails"""
    pass


class FileSizeError(FileValidationError):
    """Raised when file size exceeds limits"""
    pass


class FileTypeError(FileValidationError):
    """Raised when file type is not allowed"""
    pass


class STEPParseError(CADAutomationException):
    """Raised when STEP file parsing fails"""
    pass


class GeometryExtractionError(CADAutomationException):
    """Raised when geometry extraction fails"""
    pass


class ConversionError(CADAutomationException):
    """Raised when file conversion fails"""
    pass
