#!/usr/bin/env bash
# ---------------------------------------------------------------
# Convert a 3D DWG file to a multibody STEP file.
#
# Usage:  ./dwg_to_step.sh <input.dwg> [output.step]
#
# Requires:
#   - ODA File Converter (installed at default location)
#   - Docker container 'cad-automation-api' running
#
# If output.step is omitted, defaults to same name as input.
# ---------------------------------------------------------------
set -euo pipefail

ODA="/c/Program Files/ODA/ODAFileConverter 27.1.0/ODAFileConverter.exe"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input.dwg> [output.step]"
    exit 1
fi

INPUT="$1"
BASENAME="$(basename "$INPUT" .dwg)"
BASENAME="$(basename "$BASENAME" .DWG)"
OUTPUT="${2:-$(dirname "$INPUT")/${BASENAME}.step}"

# Temporary folder for DXF conversion
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== Step 1: DWG -> DXF (ODA File Converter) ==="
INPUT_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
INPUT_FILE="$(basename "$INPUT")"

# ODA needs input and output directories; copy file to temp to avoid clobbering
cp "$INPUT" "$TMPDIR/$INPUT_FILE"

# ODA args: input_dir output_dir version format recurse audit
# ACAD2018 DXF 0=no-recurse 1=audit
"$ODA" "$TMPDIR" "$TMPDIR" "ACAD2018" "DXF" "0" "1"

DXF_FILE="$TMPDIR/${BASENAME}.dxf"
if [[ ! -f "$DXF_FILE" ]]; then
    # Try uppercase extension
    DXF_FILE="$TMPDIR/${BASENAME}.DXF"
fi
if [[ ! -f "$DXF_FILE" ]]; then
    echo "ERROR: DXF file not generated. Check ODA output above."
    ls -la "$TMPDIR"
    exit 1
fi

echo "DXF created: $DXF_FILE"

echo ""
echo "=== Step 2: DXF -> STEP (Docker) ==="

# Ensure dxf directory exists in container
docker exec cad-automation-api mkdir -p //app/dxf

# Copy DXF and converter script into container
docker cp "$DXF_FILE" "cad-automation-api:/app/dxf/${BASENAME}.dxf"
docker cp "$(dirname "$0")/dxf_to_step.py" "cad-automation-api:/app/dxf/dxf_to_step.py"

# Run conversion
docker exec cad-automation-api python3 //app/dxf/dxf_to_step.py \
    "//app/dxf/${BASENAME}.dxf" \
    "//app/dxf/${BASENAME}.step"

# Copy result back to host
docker cp "cad-automation-api:/app/dxf/${BASENAME}.step" "$OUTPUT"

echo ""
echo "=== Done ==="
echo "STEP file: $OUTPUT"
