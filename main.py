from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
# from routers import api
from fastapi.middleware.cors import CORSMiddleware
from routers import rowword_api
from init_db import create_database_if_not_exists
create_database_if_not_exists()

app = FastAPI()

# 👇 Add this block to allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # Replace with your frontend URL
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

# app.include_router(api.router)
app.include_router(rowword_api.router)
