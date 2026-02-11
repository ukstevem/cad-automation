# CAD Automation System - Stage 2

A FastAPI-based system for processing STEP assemblies to generate DXF/DWG profiles, NC files (DSTV standard), and Bills of Materials.

## 🎯 Stage 2 Features

### ✅ Implemented
- **File Validation**
  - File size limits (configurable, default 100MB)
  - Extension validation (.step, .stp)
  - STEP format content validation (magic bytes check)
  - Structured error handling with detailed feedback

- **STEP File Parsing**
  - OpenCascade-based STEP parser
  - Geometry extraction and analysis
  - Bounding box calculation
  - Volume and surface area computation
  - Shape type identification

- **Enhanced API**
  - Upload endpoint with validation
  - Parse endpoint for existing files
  - Validate endpoint for file checking
  - Health check endpoint
  - Structured JSON responses with detailed error messages

- **Testing**
  - Comprehensive test suite for validators
  - API endpoint tests
  - Async test support with pytest-asyncio

## 🏗️ Architecture

```
cad-automation-stage2/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with error handlers
│   ├── config.py            # Configuration management
│   ├── exceptions.py        # Custom exceptions
│   ├── routers/
│   │   ├── __init__.py
│   │   └── upload.py        # Upload/validation endpoints
│   ├── validators/
│   │   ├── __init__.py
│   │   └── file_validator.py  # File validation logic
│   └── parsers/
│       ├── __init__.py
│       └── step_parser.py   # STEP parsing with OCP
├── tests/
│   ├── __init__.py
│   ├── test_validators.py
│   └── test_upload.py
├── uploads/                 # User uploads (volume)
├── outputs/                 # Generated files (volume)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── .env.example
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo>
cd cad-automation-stage2
cp .env.example .env
```

### 2. Build and Run with Docker
```bash
docker-compose up --build
```

### 3. Access the API
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## 📝 API Endpoints

### Upload STEP File
```bash
POST /api/v1/upload/
```

**Parameters:**
- `file`: STEP file upload (multipart/form-data)
- `parse_geometry`: boolean (optional, default: false)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload/?parse_geometry=true" \
  -F "file=@model.step"
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "message": "File uploaded and validated successfully",
  "file_info": {
    "original_filename": "model.step",
    "saved_filename": "abc12345_model.step",
    "file_path": "/uploads/abc12345_model.step",
    "extension": ".step",
    "size_bytes": 524288,
    "size_mb": 0.5,
    "unique_id": "abc12345"
  },
  "parse_result": {
    "success": true,
    "num_shapes": 12,
    "metadata": { ... }
  },
  "geometry": {
    "shapes": [ ... ],
    "total_volume": 15000.5,
    "total_surface_area": 3500.2
  }
}
```

**Error Response (413 Request Entity Too Large):**
```json
{
  "detail": {
    "error": "File too large",
    "message": "File size (150000000 bytes) exceeds maximum allowed size (104857600 bytes)",
    "details": {
      "filename": "large_model.step",
      "size": 150000000,
      "max_size": 104857600,
      "size_mb": 143.05,
      "max_size_mb": 100.0
    }
  }
}
```

**Error Response (415 Unsupported Media Type):**
```json
{
  "detail": {
    "error": "Invalid file type",
    "message": "File extension '.stl' is not allowed",
    "details": {
      "filename": "model.stl",
      "extension": ".stl",
      "allowed_extensions": [".step", ".stp"]
    }
  }
}
```

### Validate Existing File
```bash
GET /api/v1/validate/{filename}
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/validate/abc12345_model.step"
```

### Parse Existing File
```bash
POST /api/v1/parse/{filename}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/parse/abc12345_model.step"
```

### Health Check
```bash
GET /health
```

## 🧪 Testing

### Run All Tests
```bash
docker-compose exec api pytest
```

### Run Specific Test File
```bash
docker-compose exec api pytest tests/test_validators.py -v
```

### Run Tests with Coverage
```bash
docker-compose exec api pytest --cov=app --cov-report=html
```

## ⚙️ Configuration

Edit `.env` file to customize:

```bash
# File size limit (bytes)
MAX_UPLOAD_SIZE=104857600  # 100MB

# Allowed extensions
ALLOWED_EXTENSIONS=.step,.stp

# Directories
UPLOAD_DIR=/uploads
OUTPUT_DIR=/outputs
TEMP_DIR=/tmp/cad-processing
```

## 🔍 Error Handling

The API uses structured error responses with HTTP status codes:

- **200 OK**: Success
- **400 Bad Request**: Validation error (general)
- **404 Not Found**: File not found
- **413 Request Entity Too Large**: File size exceeds limit
- **415 Unsupported Media Type**: Invalid file type/format
- **422 Unprocessable Entity**: STEP parsing failed
- **500 Internal Server Error**: Unexpected error

All errors include:
- `error`: Error type
- `message`: Human-readable message
- `details`: Additional context (filename, values, etc.)

## 📊 STEP Parser Features

The STEP parser (using OpenCascade) provides:

1. **Shape Detection**
   - Compounds, Solids, Shells, Faces, Wires, Edges, Vertices
   - Count by type

2. **Geometric Analysis**
   - Bounding box (min/max XYZ coordinates)
   - Volume calculation (for solids)
   - Surface area calculation
   - Dimensions (length, width, height)

3. **Metadata Extraction**
   - Number of shapes
   - File size
   - Shape type distribution

## 🔧 Development

### Live Code Editing
The `./app` directory is bind-mounted, so code changes are reflected immediately with uvicorn's `--reload` flag.

### Adding Dependencies
```bash
# Add to requirements.txt, then rebuild
docker-compose down
docker-compose up --build
```

### Logs
```bash
# View logs
docker-compose logs -f api

# Structured JSON logs
docker-compose logs api | jq
```

## 🛣️ Roadmap

### Stage 3: DXF/DWG Generation
- [ ] Profile extraction from STEP geometry
- [ ] DXF file generation with ezdxf
- [ ] Layer management
- [ ] Dimension annotations

### Stage 4: NC File Generation
- [ ] DSTV format support
- [ ] Tool path generation
- [ ] Cutting parameters

### Stage 5: Bill of Materials
- [ ] Component extraction
- [ ] Material identification
- [ ] Quantity calculation
- [ ] Excel/CSV export

### Stage 6: Frontend
- [ ] Next.js frontend
- [ ] File upload interface
- [ ] 3D preview
- [ ] Download management

## 📚 Tech Stack

- **Backend**: FastAPI 0.109.0
- **Python**: 3.11
- **CAD Libraries**: 
  - OCP 7.7.2.1 (OpenCascade)
  - CadQuery 2.4.0
  - ezdxf 1.1.3
- **Testing**: pytest 7.4.4, pytest-asyncio
- **Logging**: structlog 24.1.0
- **Container**: Docker, Docker Compose

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Add tests
4. Run test suite
5. Submit PR

## 📄 License

[Your License]

## 🆘 Troubleshooting

### Issue: OpenCascade import errors
**Solution**: Rebuild container to ensure system dependencies are installed
```bash
docker-compose down
docker-compose up --build
```

### Issue: File validation fails
**Solution**: Check that file starts with `ISO-10303-21;` header

### Issue: Cannot connect to API
**Solution**: Ensure port 8000 is not in use
```bash
lsof -i :8000
docker-compose down
docker-compose up
```

## 📞 Support

- GitHub Issues: [Your repo]/issues
- Documentation: http://localhost:8000/docs
