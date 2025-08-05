import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
# DATABASE_URL = "postgresql://corpus_user:Xw4AUWglNyuzfZ489j2ehs0uNL1H0poN@dpg-d1noveer433s738haveg-a/corpus_db"
# DATABASE_URL = "postgresql://corpus_user:corpus_pw@localhost:5431/corpus_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

#  inject DB session vào routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()