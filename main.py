from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
# from routers import api
from routers import rowword_api

app = FastAPI()

# Dependency inject DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# app.include_router(api.router)
app.include_router(rowword_api.router)
