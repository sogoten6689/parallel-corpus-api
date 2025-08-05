# init_db.py
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database_if_not_exists():
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "db")
    port = os.environ.get("POSTGRES_PORT", "5432")
    dbname = os.environ["POSTGRES_DB"]

    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    print("🔧 Connecting with DB URL:", db_url)

    # db_url = os.environ["DATABASE_URL"]

    # Extract connection params to connect to default `postgres` DB
    import re
    match = re.match(r"postgresql://(.*?):(.*?)@(.+?):(\d+)/(.*)", db_url)
    if not match:
        raise ValueError("Invalid DATABASE_URL format")

    user, password, host, port, dbname = match.groups()

    # Connect to default database
    con = psycopg2.connect(
        dbname="postgres",
        user=user,
        password=password,
        host=host,
        port=port
    )
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = con.cursor()

    # Check and create database
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}';")
    exists = cur.fetchone()
    if not exists:
        print(f"Database '{dbname}' not found. Creating...")
        cur.execute(f"CREATE DATABASE {dbname};")
    else:
        print(f"Database '{dbname}' already exists.")

    cur.close()
    con.close()
