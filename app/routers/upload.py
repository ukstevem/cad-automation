from fastapi import APIRouter, UploadFile, File
import os

router = APIRouter(prefix="/upload", tags=["Upload"])

UPLOAD_DIR = "/app/uploads"

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"message": "Upload successful", "file": file.filename, "path": file_path}

