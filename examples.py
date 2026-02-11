#!/usr/bin/env python3
"""
Example API usage scripts for CAD Automation System
"""
import requests
import json
from pathlib import Path


API_BASE = "http://localhost:8000/api/v1"


def upload_step_file(file_path: str, parse_geometry: bool = False):
    """
    Upload a STEP file to the API
    
    Args:
        file_path: Path to STEP file
        parse_geometry: Whether to parse geometry
    """
    url = f"{API_BASE}/upload/"
    
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'application/step')}
        params = {'parse_geometry': parse_geometry}
        
        response = requests.post(url, files=files, params=params)
    
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    return response.json()


def validate_file(filename: str):
    """
    Validate an existing file
    
    Args:
        filename: Name of file in uploads directory
    """
    url = f"{API_BASE}/validate/{filename}"
    response = requests.get(url)
    
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    return response.json()


def parse_file(filename: str):
    """
    Parse an existing STEP file
    
    Args:
        filename: Name of file in uploads directory
    """
    url = f"{API_BASE}/parse/{filename}"
    response = requests.post(url)
    
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    return response.json()


def health_check():
    """Check API health"""
    url = "http://localhost:8000/health"
    response = requests.get(url)
    
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    return response.json()


def create_test_step_file(output_path: str = "test_model.step"):
    """
    Create a minimal valid STEP file for testing
    
    Args:
        output_path: Where to save the file
    """
    step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Test STEP file'),'2;1');
FILE_NAME('test_model.step','2024-01-01T00:00:00',('Author'),('Organization'),'PreProcessor','CAD System','Authorization');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
ENDSEC;
END-ISO-10303-21;
"""
    
    with open(output_path, 'w') as f:
        f.write(step_content)
    
    print(f"Created test STEP file: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    
    # Check if API is running
    try:
        print("=== Checking API Health ===")
        health_check()
        print()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Is it running on http://localhost:8000?")
        sys.exit(1)
    
    # Create test file
    print("=== Creating Test STEP File ===")
    test_file = create_test_step_file()
    print()
    
    # Upload without parsing
    print("=== Upload Test (No Parsing) ===")
    result = upload_step_file(test_file, parse_geometry=False)
    print()
    
    if result.get("success"):
        saved_filename = result["file_info"]["saved_filename"]
        
        # Validate existing file
        print("=== Validate Existing File ===")
        validate_file(saved_filename)
        print()
        
        # Parse existing file
        print("=== Parse Existing File ===")
        parse_file(saved_filename)
        print()
    
    # Upload with parsing
    print("=== Upload Test (With Parsing) ===")
    upload_step_file(test_file, parse_geometry=True)
    print()
    
    # Test error cases
    print("=== Error Test: Invalid Extension ===")
    try:
        # Create invalid file
        with open("test.stl", 'w') as f:
            f.write("Invalid content")
        upload_step_file("test.stl")
    except Exception as e:
        print(f"Expected error: {e}")
    print()
    
    print("=== Error Test: Invalid STEP Content ===")
    try:
        # Create file with wrong content
        with open("invalid.step", 'w') as f:
            f.write("This is not a STEP file")
        upload_step_file("invalid.step")
    except Exception as e:
        print(f"Expected error: {e}")
    print()
    
    print("=== All tests completed ===")
