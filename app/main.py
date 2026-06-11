"""
FastAPI main application with enhanced error handling
"""
import os
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from fastapi.staticfiles import StaticFiles
from app.routers import upload, frontend, analysis, stl, cnc_analysis, connections, projects, calibration
from app.exceptions import CADAutomationException

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("application_starting", version=settings.API_VERSION)
    
    # Ensure required directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.STL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.CALIBRATION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(settings.OUTPUT_DIR, "projects"), exist_ok=True)

    logger.info(
        "directories_initialized",
        upload_dir=settings.UPLOAD_DIR,
        temp_dir=settings.TEMP_DIR,
        output_dir=settings.OUTPUT_DIR
    )
    
    yield

    # Shutdown
    from app.services.task_manager import task_manager
    task_manager.shutdown()
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handlers
@app.exception_handler(CADAutomationException)
async def cad_exception_handler(request: Request, exc: CADAutomationException):
    """Handle CAD automation exceptions"""
    logger.error(
        "cad_automation_error",
        error=str(exc),
        details=exc.details,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        "unexpected_error",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "details": {"error_type": type(exc).__name__}
        }
    )


# Static files for frontend
app.mount("/static", StaticFiles(directory="/app/frontend/static"), name="static")

# Serve generated STL files (ensure dir exists before mount)
os.makedirs(settings.STL_OUTPUT_DIR, exist_ok=True)
app.mount("/outputs/stl", StaticFiles(directory=settings.STL_OUTPUT_DIR), name="stl-outputs")

# Include routers
app.include_router(frontend.router, tags=["frontend"])
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(stl.router, prefix="/api/v1", tags=["stl"])
app.include_router(cnc_analysis.router, prefix="/api/v1", tags=["cnc-analysis"])
app.include_router(connections.router, prefix="/api/v1", tags=["connections"])
app.include_router(projects.router, prefix="/api/v1", tags=["projects"])
app.include_router(calibration.router, prefix="/api/v1", tags=["calibration"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "endpoints": {
            "upload": "/api/v1/upload/",
            "validate": "/api/v1/validate/{filename}",
            "parse": "/api/v1/parse/{filename}",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/api/v1/config")
async def frontend_config():
    """Return environment-driven config for the frontend."""
    return {
        "nesting_base_url": settings.NESTING_BASE_URL,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "directories": {
            "upload": os.path.exists(settings.UPLOAD_DIR),
            "temp": os.path.exists(settings.TEMP_DIR),
            "output": os.path.exists(settings.OUTPUT_DIR)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
