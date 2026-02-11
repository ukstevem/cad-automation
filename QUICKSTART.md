# Quick Start Guide - Testing Stage 2

Get up and running with Stage 2 in 5 minutes.

## 🚀 Fast Track Setup

### 1. Start the API (30 seconds)
```bash
cd cad-automation-stage2
docker-compose up --build -d
```

Wait for the build (first time takes ~3-5 minutes due to OpenCascade compilation).

### 2. Check Health (5 seconds)
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "directories": {
    "upload": true,
    "temp": true,
    "output": true
  }
}
```

### 3. Run Example Tests (1 minute)
```bash
# Install requests if not already available
pip install requests

# Run the examples
python examples.py
```

This will:
- Create a test STEP file
- Upload it to the API
- Validate it
- Parse the geometry
- Test error cases

## 🧪 Manual Testing

### Test 1: Upload a Valid STEP File
```bash
# Create a test file
cat > test.step << 'EOF'
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Test'),'2;1');
FILE_NAME('test.step','2024-01-01',('Author'),('Org'),'','','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
ENDSEC;
END-ISO-10303-21;
EOF

# Upload
curl -X POST "http://localhost:8000/api/v1/upload/" \
  -F "file=@test.step"
```

### Test 2: Upload with Geometry Parsing
```bash
curl -X POST "http://localhost:8000/api/v1/upload/?parse_geometry=true" \
  -F "file=@test.step"
```

### Test 3: Test Error - Invalid Extension
```bash
echo "fake content" > test.stl
curl -X POST "http://localhost:8000/api/v1/upload/" \
  -F "file=@test.stl"
```

Expected: **415 Unsupported Media Type**

### Test 4: Test Error - Invalid STEP Content
```bash
echo "NOT A STEP FILE" > fake.step
curl -X POST "http://localhost:8000/api/v1/upload/" \
  -F "file=@fake.step"
```

Expected: **415 Unsupported Media Type** with message about STEP format

### Test 5: Validate Existing File
```bash
# First upload a file and get the saved filename
RESPONSE=$(curl -X POST "http://localhost:8000/api/v1/upload/" \
  -F "file=@test.step" 2>/dev/null)
FILENAME=$(echo $RESPONSE | jq -r '.file_info.saved_filename')

# Then validate it
curl "http://localhost:8000/api/v1/validate/$FILENAME"
```

### Test 6: Parse Existing File
```bash
curl -X POST "http://localhost:8000/api/v1/parse/$FILENAME"
```

## 📊 Using the Interactive Docs

The easiest way to test all endpoints:

1. Open browser: http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Fill in parameters
5. Click "Execute"
6. See the response

### Try These Workflows:

**Workflow 1: Basic Upload**
1. Go to `POST /api/v1/upload/`
2. Upload `test.step` file
3. Set `parse_geometry` to `false`
4. Execute

**Workflow 2: Upload + Parse**
1. Go to `POST /api/v1/upload/`
2. Upload `test.step` file
3. Set `parse_geometry` to `true`
4. Execute
5. Review geometry data in response

**Workflow 3: Validate Existing**
1. Copy `saved_filename` from previous upload
2. Go to `GET /api/v1/validate/{filename}`
3. Paste filename
4. Execute

## 🧹 Running Tests

### All Tests
```bash
docker-compose exec api pytest
```

### Verbose Output
```bash
docker-compose exec api pytest -v
```

### Specific Test File
```bash
docker-compose exec api pytest tests/test_validators.py -v
```

### With Coverage
```bash
docker-compose exec api pytest --cov=app --cov-report=term-missing
```

## 📝 View Logs

### Follow logs in real-time
```bash
docker-compose logs -f api
```

### View structured JSON logs
```bash
docker-compose logs api | jq
```

### Filter for errors only
```bash
docker-compose logs api | grep -i error
```

## 🔍 Debugging

### Check if container is running
```bash
docker-compose ps
```

### Get shell access
```bash
docker-compose exec api bash
```

### Inside container:
```bash
# Check Python version
python --version

# Test imports
python -c "import cadquery; print('CadQuery OK')"
python -c "from OCP.STEPControl import STEPControl_Reader; print('OCP OK')"

# List uploaded files
ls -lh /uploads/

# Run a single test
pytest tests/test_validators.py::TestFileValidator::test_validate_file_size_valid -v
```

### Restart containers
```bash
docker-compose restart
```

### Rebuild from scratch
```bash
docker-compose down
docker-compose up --build
```

## ✅ Success Indicators

You're ready to proceed if:
- [x] Health check returns 200 OK
- [x] Can upload valid STEP files
- [x] Invalid files are rejected properly
- [x] Tests pass (`pytest`)
- [x] Interactive docs work (`/docs`)
- [x] Logs show structured JSON

## 🎯 Next: Real STEP Files

Once basic tests pass, try with real STEP files:

```bash
# Upload a real CAD model
curl -X POST "http://localhost:8000/api/v1/upload/?parse_geometry=true" \
  -F "file=@/path/to/your/model.step" \
  | jq
```

Check the response for:
- `num_shapes`: Number of geometry entities
- `bounding_box`: XYZ dimensions
- `total_volume`: Calculated volume
- `total_surface_area`: Calculated area

## 🐛 Common Issues & Fixes

**Issue: Connection refused**
```bash
# Check if running
docker-compose ps

# Start if not running
docker-compose up -d
```

**Issue: Import errors (OCP/CadQuery)**
```bash
# Rebuild with dependencies
docker-compose down
docker-compose up --build
```

**Issue: Permission errors on /uploads**
```bash
# Check permissions
docker-compose exec api ls -ld /uploads

# Fix if needed
docker-compose exec api chmod 777 /uploads
```

**Issue: Tests fail**
```bash
# Ensure pytest is installed
docker-compose exec api pip install pytest pytest-asyncio httpx

# Run with verbose output
docker-compose exec api pytest -v
```

## 📚 Helpful Commands

```bash
# View API endpoints
curl http://localhost:8000/ | jq

# Format JSON responses
curl http://localhost:8000/health | jq

# Save response to file
curl -X POST "http://localhost:8000/api/v1/upload/" \
  -F "file=@test.step" \
  -o response.json

# Watch logs with timestamps
docker-compose logs -f --timestamps api

# Check disk usage
docker system df

# Clean up old images
docker system prune
```

## 🎓 Learning the API

1. **Start simple**: Upload a basic STEP file
2. **Add parsing**: Enable `parse_geometry=true`
3. **Explore metadata**: Check the geometry data returned
4. **Test errors**: Try invalid files to see error handling
5. **Read the docs**: http://localhost:8000/docs

## ⏱️ Time Estimates

- Initial setup: 5 minutes
- Running examples: 1 minute
- Manual testing: 5-10 minutes
- Exploring docs: 10-15 minutes
- Real STEP files: 5+ minutes

**Total: ~30 minutes to full competency**

---

Ready for Stage 3? See the roadmap in README.md!
