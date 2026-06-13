# CAD Automation - Stage 2

## Project Overview

FastAPI backend for processing STEP (ISO 10303) CAD files. Extracts assembly structure, part names, solid geometry, and metadata from uploaded STEP files. Frontend web UI for upload, interactive assembly tree inspection, and 3D STL preview.

The long-term goal is STEP to DXF/DWG/NC conversion with BOM generation. Stage 2 focuses on upload, validation, parsing, assembly tree analysis, STL preview generation, part classification (CNC/Bought-out), and persisted project state.

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
  config.py            # Pydantic settings (upload limits, dirs, API metadata, STL params)
  exceptions.py        # Exception hierarchy (CADAutomationException base)
  validators/          # FileValidator: extension, size, STEP content checks
  parsers/
    step_parser.py     # STEPControl_Reader based parser (basic geometry extraction)
    assembly_analyzer.py  # STEPCAFControl_Reader / XCAF based (named assembly trees)
  services/
    __init__.py        # Package init
    task_manager.py    # Background task manager (ThreadPoolExecutor + asyncio.Task)
    stl_generator.py   # XCAF-based STL generation for assembly items
  workers/
    analyze_step.py    # Standalone subprocess worker for XCAF assembly analysis
    generate_stl.py    # Standalone subprocess worker for STL mesh generation
  routers/
    upload.py          # POST /api/v1/upload/, GET /api/v1/validate/{f}, POST /api/v1/parse/{f}
    analysis.py        # Assembly tree, analysis cache, project state endpoints
    stl.py             # STL generation trigger, status poll, file listing
    frontend.py        # GET /ui (serves index.html)
frontend/
  templates/index.html # Single-page app shell (includes Three.js import map)
  static/
    css/main.css, components.css
    js/app.js, api.js, upload.js, analysis.js, stl-viewer.js, components.js, utils.js
    img/favicon.svg
tests/
  test_upload.py       # 8 tests: upload, validation, root, health endpoints
  test_validators.py   # 12 tests: FileValidator unit tests
outputs/
  analysis/            # JSON sidecar files: analysis cache + project state per STEP file
  stl/                 # Generated STL meshes, organised by run_id prefix
```

### Critical Architecture Decision: Asyncio Subprocesses

**All OCC/OCP operations run in subprocess workers, not in the FastAPI process.**

OCC (OpenCascade) holds the Python GIL during XCAF analysis and STL mesh generation. If these run in a `ThreadPoolExecutor` via `asyncio.run_in_executor()`, the GIL prevents the event loop from executing any Python — including `anyio.to_thread.run_sync` that Starlette uses for static file serving. This causes Three.js STL fetch requests to hang indefinitely, locking the viewer in "loading" state.

**Solution**: Both analysis and STL generation run as standalone child processes via `asyncio.create_subprocess_exec()`. The parent `await proc.communicate()` yields to the event loop while waiting, keeping uvicorn fully responsive throughout.

```
FastAPI event loop (parent)
  ├── asyncio.create_subprocess_exec() → analyze_step.py
  │     runs XCAF analysis in thread with 64MB stack, prints JSON to stdout
  ├── asyncio.create_subprocess_exec() → generate_stl.py
  │     runs STL mesh generation in thread with 64MB stack, prints JSON to stdout
  └── ... all other HTTP requests served normally, no GIL interference
```

### TaskManager (`app/services/task_manager.py`)

Two submission modes:

- **`submit()`** — sync callable in `ThreadPoolExecutor`. Kept for any remaining sync uses.
- **`submit_async()`** — async coroutine scheduled as `asyncio.create_task()`. Used for all OCC work (analysis + STL). The coroutine must be truly non-blocking (i.e., use `asyncio.create_subprocess_exec`, not `subprocess.run`).

Both modes track status (`pending → running → completed/failed`), progress, and results in an in-memory dict. Tasks are identified by 12-char hex IDs.

### Worker Pattern (`app/workers/`)

Both worker scripts follow the same pattern:

1. `faulthandler.enable()` at the very top — dumps C-level traceback on SIGSEGV/SIGABRT
2. Redirect structlog and stdlib logging to stderr **before** any `app.*` imports
3. Add project root to `sys.path` so `app` package is importable
4. Run OCC work in a `threading.Thread` with `threading.stack_size(64 * 1024 * 1024)` (64 MB) — prevents stack overflow during deep OCC recursion on complex assemblies
5. Print JSON to stdout on success, print error to stderr and `sys.exit(1)` on failure

The parent process (`_run_stl_subprocess` / `run_analysis_async` closures) scans stdout for the first line starting with `{` or `[` and parses that as JSON, ignoring any diagnostic output.

### Two STEP Parsers (Important)

There are two distinct STEP parsing approaches in the codebase:

1. **`STEPParser`** (`step_parser.py`) — Uses `STEPControl_Reader`. Extracts raw geometry: all solids with volume, surface area, bounding box. Very slow on large files (6+ minutes for 97MB/34K solids). Used by upload's `parse_geometry=true` option and `/api/v1/parse/{f}`.

2. **`AssemblyAnalyzer`** (`assembly_analyzer.py`) — Uses `STEPCAFControl_Reader` with XCAF framework. Preserves product structure, part names, and assembly hierarchy. Much faster (~27s for same 97MB file). Used by `/api/v1/analysis/assembly/{f}` via `analyze_step.py` worker.

The upload page sends `parse_geometry=false` to avoid blocking. Deep analysis is done via the Analysis tab.

### Analysis Caching + Project State (`outputs/analysis/`)

Each analysed STEP file gets a JSON sidecar at `outputs/analysis/<hex_prefix>_<stem>.json`:

```json
{
  "analysis": {
    "analyzed_at": "2025-02-13T10:30:00Z",
    "assembly_tree": [...],
    "summary": { "total_assemblies": 5, "total_parts": 34, "total_solids": 34 }
  },
  "project_state": {
    "classifications": { "0:1:1:2": "postprocess", "0:1:1:3": "bought-out" },
    "exploded_nodes": ["0:1:1:1"],
    "stl_map": { "0:1:1:2": "/outputs/stl/c79fd101/Part.stl" },
    "solid_children": { "0:1:2": [{ "name": "Part - Solid 1", "nodeId": "0:1:2:s0" }] }
  }
}
```

- `analysis` section — written by backend after first XCAF parse. Subsequent requests to `/analysis/assembly/{filename}` return this immediately (no re-parse).
- `project_state` section — written by frontend via `PUT /analysis/project-state/{filename}` as the user classifies nodes and explodes assemblies. Restored automatically when the user opens the same file again.
- The JSON lives in `outputs/` which is bind-mounted, so it survives container restarts.

### STL Generation Pipeline

```
POST /stl/generate/{filename}           → triggers "all" mode  (root-level items)
POST /stl/generate-children/{filename}  → triggers "children" mode (assembly explode)
POST /stl/generate-solids/{filename}    → triggers "solids" mode (multi-solid part explode)
```

All three endpoints are **idempotent** — return the existing task if one is already running or completed for the same `(filename, task_type)` pair.

`generate_stl.py` worker:
- Mode `all`: `STLGenerator.generate_all()` — meshes all root assembly children
- Mode `children`: `STLGenerator.generate_children(node_id)` — meshes children of a specific node
- Mode `solids`: `STLGenerator.generate_solids(node_id)` — meshes individual solids of a multi-solid part

STL files stored at `outputs/stl/<8-hex-prefix>/PartName.stl`. Served via `StaticFiles` mount at `/outputs/stl/`.

### BOM / Classification Tables

The frontend `analysis.js` `_updateClassificationTables()` deduplicates by `ref_id`:

- Multiple instances of the same part (same XCAF reference/prototype) share a `ref_id`
- Classifying one instance classifies all instances (same `ref_id`)
- The BOM table shows one row per unique `ref_id` with a Qty column showing instance count
- Deduplicated in a `Map<refId, {name, usedIn, qty}>` built from `this.classifications`

## OCP API Gotchas

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

**StlAPI_Writer**: Uses property assignment, not setter methods:
```python
writer = StlAPI_Writer()
writer.ASCIIMode = False          # CORRECT (property)
# writer.SetASCIIMode(False)      # WRONG (AttributeError)
```

## OCC Chirality Detection

**`AssemblyAnalyzer._is_mirrored(label)`** detects mirrored instances using BRep-level location:

```python
shape = XCAFDoc_ShapeTool.GetShape_s(label)   # safe — avoids XCAFDoc_Location attribute
loc = shape.Location()
return loc.IsNegative()                        # True if transformation determinant is -1
```

**Why not `XCAFDoc_Location`**: Accessing `XCAFDoc_Location` XCAF attributes (via `FindAttribute`, `GetID_s`, `Get`) causes **SIGSEGV** in OCC 7.7.x for certain XCAF labels in complex STEP files. The BRep path avoids this entirely.

**STL meshes for mirrored parts**: The STL generator currently meshes the prototype shape (via the referred/prototype label), not the instance shape. Mirrored instances therefore show the non-mirrored mesh in the 3D viewer — visually wrong for asymmetric parts but functionally harmless for the classification workflow. Fixing this would require applying the instance's mirror transform to the mesh before writing the STL.

## Upcoming Work

- **Postprocess item consolidation worker**: Background worker to build a cross-assembly flat parts list — all unique `ref_id` parts with total instance counts across the whole tree, regardless of which parent assembly they live in. Currently the BOM deduplicates visible classified nodes by ref_id, but doesn't aggregate across unexploded assemblies.
- **Mirrored-part STL meshes**: STL generator always meshes the prototype shape; mirrored instances show the non-mirrored mesh. Fix: apply the instance's mirror transform to the mesh before writing the STL.
- **DXF/DWG output** (ezdxf installed, not wired)
- **NC code generation**
- **Full BOM export** (CSV/Excel)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/ui` | Frontend web UI |
| GET | `/docs` | Swagger/OpenAPI docs |
| POST | `/api/v1/upload/` | Upload STEP file (`parse_geometry=bool`) |
| GET | `/api/v1/validate/{filename}` | Validate existing file |
| POST | `/api/v1/parse/{filename}` | Full geometry parse (slow — avoid for large files) |
| GET | `/api/v1/analysis/files` | List uploaded STEP files (display_name, uploaded_at) |
| GET | `/api/v1/analysis/assembly/{filename}` | Assembly tree — cache hit: immediate; cache miss: returns `{analysis_task_id, status: "pending"}` |
| GET | `/api/v1/analysis/status/{task_id}` | Poll background analysis task; returns full tree on completion |
| PUT | `/api/v1/analysis/project-state/{filename}` | Save classifications/exploded_nodes/stl_map/solid_children |
| POST | `/api/v1/stl/generate/{filename}` | Trigger root-level STL generation (idempotent) |
| POST | `/api/v1/stl/generate-children/{filename}?parent_id=` | Trigger STL for assembly children (idempotent) |
| POST | `/api/v1/stl/generate-solids/{filename}?node_id=` | Trigger STL for individual solids (idempotent) |
| GET | `/api/v1/stl/status/{task_id}` | Poll STL generation task |
| GET | `/api/v1/stl/files/{filename}` | List generated STL files for an upload |

STL files served as static files at `/outputs/stl/{run_id}/{file}.stl`.
Analysis JSON served (read-only) via bind-mount; not exposed as a separate endpoint.

## Assembly Tree Node Schema

Each node returned by `/analysis/assembly/{f}` or `/analysis/status/{task_id}`:

```json
{
  "id": "0:1:1:1",
  "name": "Full Top Stringer",
  "instance_ref": "Full Top Stringer:1",
  "node_type": "assembly",
  "solid_count": 0,
  "ref_id": "0:1:1:1",
  "chiral_key": "0:1:1:1:N",
  "children": [...]
}
```

- `id` — XCAF label entry (instance label, unique per placement)
- `name` — definition name from the referred shape (shared across all instances of the same part)
- `node_type` — `assembly` | `part_single_solid` | `part_multi_solid` | `part_no_solid`
- `ref_id` — XCAF label of the shape definition (shared by all instances of the same prototype)
- `chiral_key` — `"{ref_id}:N"` normally, `"{ref_id}:M"` for mirrored instances (currently always N, see chirality note above)

## Frontend

Single-page app with two tabs:
- **Upload**: Drag-drop STEP files, client-side validation, server validation, results
- **Analysis**: File selector → assembly tree + 3D preview side-by-side layout

Key JS modules:
- `api.js` — thin wrapper around `fetch`, all endpoint calls go through here
- `analysis.js` — `AnalysisPage` class; manages tree, classifications, STL polling, viewer, project state save/restore
- `stl-viewer.js` — `STLViewer` class; Three.js scene/camera/controls/lights/loader, ResizeObserver, dispose

**Three.js**: Import map in `index.html` (three@0.164.1 from CDN). `STLViewer.loadSTL(url)` uses a load-generation counter (`_loadGen`) to cancel stale loads and always settles the Promise (resolve or reject — never hangs).

**Project state auto-save**: `analysis.js` debounces saves (1s) via `_debouncedSave()`. Called after every classification, explode, and STL result. On next analysis of the same file the `project_state` is restored from the JSON sidecar automatically.

**Pico CSS note**: Pico heavily styles `<table>` and can intercept click events. STL file list uses `<div>` elements for reliable click handling.

## AR Weld-Ratification (markers + constellation)

A `PRODUCTION` UI area (Calibrate + Capture tabs) for aligning CAD to a real photo of a
fabrication, as the basis for weld overlay / robotic guidance. Marker-based (markerless was
investigated and deferred — see below).

**Pipeline:** camera **calibration** (ChArUco, `app/routers/calibration.py`, profiles in
`outputs/calibration/<name>.json`) → **AR alignment** (`app/routers/ar.py` + `app/services/pose.py`,
`constellation.py`; CAD edges via `app/workers/extract_edges.py`):

- `GET  /ar/geometry/{f}` — CAD edges + corner anchors + OBB (world coords; cached in sidecar `ar_geometry`)
- `GET  /ar/markers.pdf` — printable AprilTag sheet (size/count params)
- `POST /ar/detect-markers/{f}` — detect AprilTags + per-marker pose
- `POST /ar/solve/{f}` — manual-correspondence pose (clicked points → solvePnP **SQPNP**)
- `POST /ar/register-marker/{f}` — store T(marker→model) for all visible tags from one solve (`ar_markers`)
- `POST /ar/auto-solve/{f}` — detect registered tags, **fuse all** → pose → project edges (no clicking)
- `POST /ar/calibrate-constellation/{f}` — link many tags across photos into one frame (`ar_constellation`)
- `POST /ar/anchor-constellation/{f}` — tie constellation to CAD via one solved photo → writes `ar_markers`
- `POST /ar/capture/{f}` — persist photo + provenance record to `outputs/captures/<run_id>/<id>/`

**Invariants / gotchas:**
- Pose maths is `app/services/pose.py` (solvePnP uses **SQPNP**, not ITERATIVE — ITERATIVE's DLT needs ≥6
  non-coplanar pts and threw on 5). Marker pose uses `SOLVEPNP_IPPE_SQUARE`.
- **Resolution gate:** every pose-critical endpoint calls `_require_resolution_match` — a photo whose
  resolution ≠ the calibration profile's `image_size` is rejected (wrong intrinsics = silently wrong pose).
- One calibration profile per camera/lens/zoom; markers same physical size; ArUco detector params tuned
  aggressive (`_detector_params`) to catch small/oblique tags.
- Markerless (single-image edge/chamfer/VP-axis) is **NOT viable on bare sections** (deferred, bd `ase`);
  the viable markerless route is multi-view/depth (future). Full log: vault `AR-weld-ratification-tool.md`.

## Files Stored

- Uploads: `/uploads/<8-hex>_<original_name>.step` (bind-mounted to `./uploads/`)
- Analysis cache: `/outputs/analysis/<8-hex>_<stem>.json` (bind-mounted to `./outputs/`)
- STL meshes: `/outputs/stl/<8-hex>/PartName.stl` (bind-mounted to `./outputs/`)
- Calibration profiles: `/outputs/calibration/<name>.json`; AR captures: `/outputs/captures/<run_id>/<id>/`

## Testing

```bash
# Run all tests inside the container:
docker exec cad-automation-api python -m pytest tests/ -v

# 20 tests total (8 upload + 12 validator), all passing
```

Tests use `FastAPI.TestClient` and run inside the container (need OCP/CadQuery).

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
# UI:       http://localhost:8000/ui
# API docs: http://localhost:8000/docs
# Health:   http://localhost:8000/health
```

## What's Working

- File upload with validation (extension, size, STEP header content)
- XCAF assembly tree extraction with named nodes, instance refs, solid counts, ref_id, chiral_key
- Analysis result caching — re-opening same file is instant (no re-parse)
- Project state persistence — classifications, exploded nodes, STL map restored on next visit
- Background STL generation for root-level items (auto-triggered by frontend after analysis)
- On-demand STL generation for assembly children (Explode button) and individual solids
- STL generation progress polling with real-time status updates
- 3D STL preview with Three.js (OrbitControls, auto-centering, responsive resize, proper dispose)
- Part classification (Postprocess / Bought-out) with BOM tables deduplicated by ref_id + Qty
- Asyncio subprocess architecture — event loop stays responsive throughout OCC work
- All 20 tests passing

## Portal Tree Export

`GET /api/v1/analysis/portal-tree/{filename}` returns a slim payload for the platform portal viewer:

- `runId`, `projectName` (from `cnc_project_number`), `summary`
- `assembly_tree` — with solid children embedded on `part_multi_solid` nodes and local placements replaced by world-space `placement` (accumulated 4x4 transform), rounded (rotation 3dp, translation 1dp)
- `stl_map`, `classifications`
- Excludes: `consolidation`, `cnc_analysis`, `solid_children` (already merged into tree)

Solid children are embedded into the sidecar's `assembly_tree` at save time (`_embed_solid_children` in `_save_project_state`). This means the cad-automation frontend also sees them on re-open — `_createSolidChildrenDOM` is idempotent so it replaces them during restore, but there is a brief render before restore where solid children appear without action buttons.

**Known issue — solid children in sidecar tree**: Embedding children at save time couples the portal export format with the cad-automation frontend's data. A cleaner approach would be to keep the sidecar tree clean (children: []) and only merge during portal-tree export. This avoids the brief no-action-buttons flash on the frontend and keeps the sidecar format as a pure parse cache + separate project_state. Low priority since it's cosmetic only.

## What's Not Yet Built

- **Mirrored-part STL meshes** — `is_mirrored` is now correctly detected; STL generator still meshes the prototype (non-mirrored) shape; mirrored parts show wrong-handed mesh in viewer
- **Postprocess consolidation worker** — cross-assembly flat parts list, identify truly unique parts by ref_id across all assembly levels
- **Portal ancillary data** — world placement, material, shape type, dimensions etc. should move to a separate ancillary JSON fetched on-demand by the portal service, reducing the main portal-tree payload (~4.2 MB → ~2.7 MB for the test file)
- DXF/DWG output generation (ezdxf is installed but not wired up)
- NC code generation
- BOM export (CSV/Excel)
- The `/api/v1/parse/{filename}` endpoint is very slow for large assemblies — consider deprecating in favour of the XCAF-based analysis endpoint


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
