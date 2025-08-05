# 📚 Parallel Corpus API

A simple FastAPI project using PostgreSQL, SQLAlchemy ORM, Alembic migrations, and dotenv configuration.

---

## 📦 Requirements

- Python >= 3.8
- PostgreSQL
- pip or poetry

---

## 🚀 Setup Guide

### 1. Clone the repo
```bash
git clone https://github.com/sogoten6689/parallel-corpus-api.git
cd parallel-corpus-api
````

### 2. create .env 
```bash
 cp .env.sample .env 
````

```bash
POSTGRES_DB=corpus_db
POSTGRES_USER=corpus_user
POSTGRES_PASSWORD=corpus_pw
POSTGRES_HOST=localhost # db or localhost. 
POSTGRES_PORT=5432
````
### 2.a - upgrade db
```bash
alembic revision --autogenerate -m "create sentence and point tables"
alembic upgrade head
````

### 3. Install dependencies


```bash
pip install -r requirements.txt
````
### 4. Start the API

```bash
uvicorn main:app --reload
````