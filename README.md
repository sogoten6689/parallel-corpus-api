# Parallel Corpus API (Backend)

FastAPI backend cho ứng dụng Parallel Corpus.

## Cài đặt và chạy

### Sử dụng Docker (Khuyến nghị)

1. **Clone repository:**
```bash
git clone <your-repo-url>
cd backend
```

2. **Tạo file .env:**
```bash
cp env.example .env
# Chỉnh sửa các giá trị trong .env theo môi trường của bạn
```

3. **Chạy với Docker Compose:**
```bash
docker-compose up -d
```

4. **Chạy migrations:**
```bash
docker-compose exec api alembic upgrade head
```

### Chạy trực tiếp (Development)

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Cài đặt PostgreSQL và tạo database**

3. **Tạo file .env và cấu hình DATABASE_URL**

4. **Chạy migrations:**
```bash
alembic upgrade head
```

5. **Chạy server:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- **Health check:** `GET /health`
- **API docs:** `GET /docs`
- **Row words:** `GET /api/rowwords`

## Deploy lên Production

### Sử dụng Docker

1. **Build image:**
```bash
docker build -t parallel-corpus-api .
```

2. **Chạy container:**
```bash
docker run -d \
  --name parallel-corpus-api \
  -p 8000:8000 \
  -e DATABASE_URL=your-production-db-url \
  -e ALLOWED_ORIGINS=https://your-frontend-domain.com \
  parallel-corpus-api
```

### Sử dụng Docker Compose (Production)

1. **Tạo docker-compose.prod.yml:**
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=your-production-db-url
      - ALLOWED_ORIGINS=https://your-frontend-domain.com
    restart: unless-stopped
```

2. **Chạy:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Environment Variables

- `DATABASE_URL`: URL kết nối database
- `API_HOST`: Host cho API server (default: 0.0.0.0)
- `API_PORT`: Port cho API server (default: 8000)
- `DEBUG`: Chế độ debug (default: False)
- `ALLOWED_ORIGINS`: CORS origins được phép
- `SECRET_KEY`: Secret key cho JWT (nếu sử dụng)

## Database Migrations

```bash
# Tạo migration mới
alembic revision --autogenerate -m "description"

# Chạy migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```
