from fastapi import FastAPI
from app.routers import upload

app = FastAPI(title="CAD Automation API")

app.include_router(upload.router)
