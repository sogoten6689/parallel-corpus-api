from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal
from models import RowWord
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# from routers import api
from routers import rowword_api, analysis_api, analysis_update_api

app = FastAPI(
    title="Parallel Corpus API",
    description="API for Parallel Corpus application",
    version="1.0.0"
)

# CORS configuration
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency inject DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Parallel Corpus API is running"}

# Test database connection
@app.get("/test-db")
async def test_db():
    try:
        db = SessionLocal()
        count = db.query(RowWord).count()
        db.close()
        return {"status": "success", "count": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# app.include_router(api.router)
app.include_router(rowword_api.router, prefix="/api")
app.include_router(analysis_api.router, prefix="/api")
app.include_router(analysis_update_api.router, prefix="/api/analysis")
