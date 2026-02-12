# CAD Automation - Stage 2

## Project Overview

FastAPI backend for processing STEP (ISO 10303) CAD files. Extracts assembly structure, part names, solid geometry, and metadata from uploaded STEP files. Frontend web UI for upload and interactive assembly tree inspection.

The long-term goal is STEP to DXF/DWG/NC conversion with BOM generation. Stage 2 focuses on upload, validation, parsing, and assembly tree analysis.

## Architecture

### Runtime Environment

- Runs in Docker via `docker-compose.yml`
- Base image: `condaforge/miniforge3` (Debian-based)
- Python 3.11 pinned (required by OCP 7.7.2)
- OCP (OpenCascade Python bindings) and CadQuery installed via conda (not pip-installable on Linux)
- Remaining Python deps installed via pip from `requirements.txt`
- Single uvicorn worker with `--reload` for dev
- Container name: `cad-automation-api`, exposed on port 8000
- Volumes: `./app`, `./tests`, `./frontend`, `./uploads`, `./temp`, `./outputs` are bind-mounted

### Key Directories

```
app/
  main.py              # FastAPI app, lifespan, exception handlers, router wiring
  config.py            # Pydantic settings (upload limits, dirs, API metadata)
  exceptions.py        # Exception hierarchy (CADAutomationException base)
  validators/          # FileValidator: extension, size, STEP content checks
  parsers/
    step_parser.py     # STEPControl_Reader based parser (basic geometry extraction)
    assembly_analyzer.py  # STEPCAFControl_Reader / XCAF based (named assembly trees)
  routers/
    upload.py          # POST /api/v1/upload/, GET /api/v1/validate/{f}, POST /api/v1/parse/{f}
    analysis.py        # GET /api/v1/analysis/files, GET /api/v1/analysis/assembly/{f}
    frontend.py        # GET /ui (serves index.html)
frontend/
  templates/index.html # Single-page app shell
  static/
    css/main.css, components.css
    js/app.js, api.js, upload.js, analysis.js, components.js, utils.js
    img/favicon.svg
tests/
  test_upload.py       # 8 tests: upload, validation, root, health endpoints
  test_validators.py   # 12 tests: FileValidator unit tests
```

### Two STEP Parsers (Important)

There are two distinct STEP parsing approaches in the codebase:

1. **`STEPParser`** (`step_parser.py`) - Uses `STEPControl_Reader`. Extracts raw geometry: all solids with volume, surface area, bounding box. Slow on large files (6+ minutes for 97MB/34K solids). Used by upload's `parse_geometry=true` option and `/api/v1/parse/{f}`.

2. **`AssemblyAnalyzer`** (`assembly_analyzer.py`) - Uses `STEPCAFControl_Reader` with XCAF framework. Preserves product structure, part names, and assembly hierarchy. Much faster (~27s for same 97MB file). Used by `/api/v1/analysis/assembly/{f}`.

The upload page sends `parse_geometry=false` to avoid blocking. Deep analysis is done via the Analysis tab using `AssemblyAnalyzer`.

## OCP API Gotcha

OCP Python bindings use a `_s` suffix convention for static/class methods on XCAF tools:

```python
# CORRECT:
XCAFDoc_ShapeTool.GetComponents_s(label, components)
XCAFDoc_ShapeTool.IsReference_s(label)
XCAFDoc_ShapeTool.GetReferredShape_s(label, ref_label)
XCAFDoc_ShapeTool.GetShape_s(label)

# WRONG (AttributeError at runtime):
shape_tool.GetComponents(label, components)
shape_tool.IsReference(label)
```

The `shape_tool` instance returned by `XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())` has these as `_s` methods, not plain methods. This applies to all XCAFDoc tool classes.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/ui` | Frontend web UI |
| GET | `/docs` | Swagger/OpenAPI docs |
| POST | `/api/v1/upload/` | Upload STEP file (params: `parse_geometry=bool`) |
| GET | `/api/v1/validate/{filename}` | Validate existing file |
| POST | `/api/v1/parse/{filename}` | Full geometry parse (slow for large files) |
| GET | `/api/v1/analysis/files` | List uploaded files (with display_name, uploaded_at) |
| GET | `/api/v1/analysis/assembly/{filename}` | XCAF assembly tree with named nodes |

## Assembly Tree Node Schema

Each node in the assembly tree returned by `/analysis/assembly/{f}`:

```json
{
  "id": "0:1:1:1",           // XCAF label entry (colon-separated)
  "name": "Full Top Stringer", // Reference element name (definition/part name)
  "instance_ref": "Full Top Stringer:1", // Instance reference (may differ for multiple instances)
  "node_type": "assembly",    // assembly | part_single_solid | part_multi_solid | part_no_solid
  "solid_count": 0,
  "children": [...]
}
```

The `name` field shows the definition name from the referred shape. The `instance_ref` preserves the instance-level name (often has `:N` suffix for multiple instances of the same part).

## Frontend

Single-page app with two tabs:
- **Upload**: Drag-drop STEP files, client-side validation, server validation, results with "go to Analysis" prompt
- **Analysis**: File selector (sorted by upload date, shows original filename), assembly tree viewer with expand/collapse, node type badges, action buttons (CNC/Bought-out/Explode), elapsed timer during analysis

Uses Pico CSS (CDN) for base styles. Custom CSS in `components.css`. JS modules (ES6 imports) in `static/js/`.

## Uploaded Files

Files are stored in `/app/uploads/` (bind-mounted to `./uploads/`) as `<8-hex-chars>_<original_filename>`. The hex prefix is stripped for display in the UI.

## Testing

```bash
# Run all tests inside the container:
docker exec cad-automation-api python -m pytest tests/ -v

# 20 tests total (8 upload + 12 validator), all passing
```

Tests use `FastAPI.TestClient` and run inside the container (need OCP/CadQuery).

## Git Status

- Branch: `stage2` (tracks `origin/stage2`)
- Base commit: `a0b3379 Stage 2: FastAPI backend with conda-based OCC/CadQuery Docker setup`
- **Uncommitted changes** (all working, tested):
  - Modified: `app/main.py`, `app/parsers/step_parser.py`, `app/routers/__init__.py`, `docker-compose.yml`
  - New: `app/parsers/assembly_analyzer.py`, `app/routers/analysis.py`, `app/routers/frontend.py`, `frontend/` (full UI)
  - Misc: `Dockerfile.old`, `docker-compose.yml.old` (backups)

## What's Working

- File upload with validation (extension, size, STEP header content)
- XCAF assembly tree extraction with named nodes, instance refs, solid counts
- Frontend upload page (drag-drop, validation feedback, directs to Analysis)
- Frontend analysis page (file selector with date/name, tree viewer with expand/collapse, action buttons, elapsed timer)
- All 20 tests passing

## What's Not Yet Built

- DXF/DWG output generation (ezdxf is installed but not wired up)
- NC code generation
- BOM generation
- Part classification persistence (buttons work in UI but state is in-memory only)
- Background task queue for long-running operations (currently synchronous)
- The `/api/v1/parse/{filename}` endpoint is very slow for large assemblies (uses STEPParser with full geometry extraction) - consider deprecating in favour of the XCAF-based analysis endpoint

## Running the Project

```bash
# Start:
docker compose up -d

# Rebuild after Dockerfile changes:
docker compose up -d --build

# View logs:
docker logs -f cad-automation-api

# Run tests:
docker exec cad-automation-api python -m pytest tests/ -v

# Access:
# UI: http://localhost:8000/ui
# API docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```
