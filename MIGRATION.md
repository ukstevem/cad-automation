# Migration Guide: Stage 1 → Stage 2

This guide helps you upgrade from your existing Stage 1 implementation to the enhanced Stage 2 system.

## 📋 What's New in Stage 2

### Core Enhancements
1. **File Validation System**
   - Comprehensive file size checking
   - Extension validation
   - STEP content validation (magic bytes)
   - Detailed error messages

2. **STEP Parsing Engine**
   - OpenCascade integration
   - Geometry extraction
   - Metadata analysis
   - Volume/area calculation

3. **Structured Error Handling**
   - Custom exception hierarchy
   - HTTP status code mapping
   - Detailed error responses

4. **Testing Infrastructure**
   - Unit tests for validators
   - API integration tests
   - Async test support

5. **Enhanced Logging**
   - Structured JSON logging with structlog
   - Request/response tracking
   - Error context

## 🔄 Migration Steps

### Step 1: Backup Your Current Code
```bash
# In your existing project
git add .
git commit -m "Stage 1 complete - before Stage 2 migration"
git tag stage-1
```

### Step 2: Update Dependencies
Replace your `requirements.txt` with the new one that includes:
- `OCP==7.7.2.1` (OpenCascade)
- `cadquery==2.4.0`
- `ezdxf==1.1.3`
- `structlog==24.1.0`
- `python-magic==0.4.27`
- `pytest` and `pytest-asyncio`

### Step 3: Update Dockerfile
The new Dockerfile includes system dependencies for OpenCascade:
```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxi6 \
    libxrender1 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libfreetype6-dev \
    libffi-dev \
    libmagic1
```

### Step 4: Add New Modules

#### Create `app/config.py`
This centralizes configuration management. Copy from Stage 2.

#### Create `app/exceptions.py`
Custom exceptions for better error handling. Copy from Stage 2.

#### Create `app/validators/` package
```
app/validators/
├── __init__.py
└── file_validator.py
```

#### Create `app/parsers/` package
```
app/parsers/
├── __init__.py
└── step_parser.py
```

### Step 5: Update `app/routers/upload.py`

**Key Changes:**
1. Import new validators and parsers
2. Add `parse_geometry` parameter
3. Use structured error handling
4. Add file validation before saving
5. Return detailed JSON responses

**Before (Stage 1):**
```python
@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = Path(UPLOAD_DIR) / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename}
```

**After (Stage 2):**
```python
@router.post("/upload/")
async def upload_step_file(
    file: UploadFile = File(...),
    parse_geometry: bool = False
):
    # Validate file
    file_ext, file_size = await FileValidator.validate_upload(
        file.file, file.filename
    )
    
    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    
    # Save file
    # ... save logic ...
    
    # Parse if requested
    if parse_geometry:
        parser = STEPParser(str(file_path))
        parse_result = parser.parse()
        geometry_data = parser.extract_geometry()
        # ... include in response ...
```

### Step 6: Update `app/main.py`

**Add:**
1. Lifespan context manager
2. Custom exception handlers
3. Structured logging configuration
4. CORS middleware
5. Health check endpoint

### Step 7: Add Testing Infrastructure

Create `tests/` directory with:
- `test_validators.py`
- `test_upload.py`
- `pytest.ini`

### Step 8: Update Docker Compose

Ensure your `docker-compose.yml` has:
- Volume mounts for uploads/outputs
- Environment variables
- Health check configuration

### Step 9: Create Environment File
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Step 10: Rebuild and Test

```bash
# Stop existing containers
docker-compose down

# Rebuild with new dependencies
docker-compose up --build

# In another terminal, run tests
docker-compose exec api pytest -v

# Test the API
python examples.py
```

## 🔍 API Changes

### Upload Endpoint

**Stage 1:**
```bash
curl -X POST "http://localhost:8000/upload/" -F "file=@model.step"
```

**Stage 2:**
```bash
# Basic upload with validation
curl -X POST "http://localhost:8000/api/v1/upload/" -F "file=@model.step"

# Upload with geometry parsing
curl -X POST "http://localhost:8000/api/v1/upload/?parse_geometry=true" \
  -F "file=@model.step"
```

### Response Format

**Stage 1:**
```json
{
  "filename": "model.step"
}
```

**Stage 2:**
```json
{
  "success": true,
  "message": "File uploaded and validated successfully",
  "file_info": {
    "original_filename": "model.step",
    "saved_filename": "abc12345_model.step",
    "extension": ".step",
    "size_bytes": 524288,
    "size_mb": 0.5,
    "unique_id": "abc12345"
  },
  "parse_result": { ... },  // if parse_geometry=true
  "geometry": { ... }        // if parse_geometry=true
}
```

## 🆕 New Endpoints

Stage 2 adds:
- `GET /api/v1/validate/{filename}` - Validate existing file
- `POST /api/v1/parse/{filename}` - Parse existing file
- `GET /health` - Health check

## ⚠️ Breaking Changes

1. **URL Prefix**: Endpoints now use `/api/v1/` prefix
2. **Response Format**: All responses use structured JSON
3. **Error Codes**: Specific HTTP codes for different errors
4. **File Naming**: Files saved with unique IDs to prevent collisions

## 🔧 Configuration Changes

### Environment Variables

**New in Stage 2:**
```bash
MAX_UPLOAD_SIZE=104857600  # File size limit
UPLOAD_DIR=/uploads
OUTPUT_DIR=/outputs
TEMP_DIR=/tmp/cad-processing
```

### Settings Access

**Before:**
```python
UPLOAD_DIR = "/uploads"
```

**After:**
```python
from app.config import settings
settings.UPLOAD_DIR
settings.MAX_UPLOAD_SIZE
```

## 📊 Monitoring & Debugging

### Structured Logs

Stage 2 uses structured JSON logging:
```bash
docker-compose logs api | jq
```

Example log entry:
```json
{
  "event": "file_validation_complete",
  "filename": "model.step",
  "extension": ".step",
  "size": 524288,
  "status": "success",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### Error Tracking

All errors include:
- Error type/code
- Detailed message
- Context (filename, size, etc.)
- HTTP status code

## ✅ Validation Checklist

After migration:
- [ ] All dependencies installed
- [ ] Docker container builds successfully
- [ ] API starts without errors
- [ ] Health check returns 200 OK
- [ ] Can upload valid STEP files
- [ ] Invalid files are rejected with proper errors
- [ ] Tests pass (`pytest -v`)
- [ ] Logs are structured JSON
- [ ] Existing files are validated
- [ ] Geometry parsing works

## 🐛 Common Issues

### Issue: OCP Import Errors
**Cause**: Missing system dependencies
**Fix**: Rebuild container with updated Dockerfile

### Issue: Tests Fail
**Cause**: Old test files or missing pytest dependencies
**Fix**: 
```bash
docker-compose exec api pip install pytest pytest-asyncio httpx
docker-compose exec api pytest -v
```

### Issue: Files Not Saving
**Cause**: Volume permissions
**Fix**:
```bash
docker-compose down
docker volume rm cad-automation_uploads
docker-compose up --build
```

## 📚 Learning Resources

- FastAPI docs: https://fastapi.tiangolo.com
- OpenCascade docs: https://dev.opencascade.org
- CadQuery docs: https://cadquery.readthedocs.io
- pytest docs: https://docs.pytest.org

## 🎯 Next Steps

After successful migration:
1. Review all endpoints in the API docs (`/docs`)
2. Run the example scripts (`python examples.py`)
3. Familiarize with the STEP parser API
4. Prepare for Stage 3: DXF/DWG generation

## 💬 Support

If you encounter issues:
1. Check logs: `docker-compose logs api`
2. Run tests: `docker-compose exec api pytest -v`
3. Verify health: `curl http://localhost:8000/health`
4. Review README.md for troubleshooting

---

**Migration Time Estimate**: 30-60 minutes

**Recommended Approach**: 
1. Keep Stage 1 running
2. Set up Stage 2 in parallel
3. Test thoroughly
4. Switch when confident
