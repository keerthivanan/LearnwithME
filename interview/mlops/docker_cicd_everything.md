# Docker + CI/CD — Everything

---

## 1. DOCKERFILE — ML API

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create non-root user (security)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "fastapi_everything:app", "--host", "0.0.0.0", "--port", "8000"]
```

```text
# requirements.txt
fastapi==0.104.0
uvicorn==0.24.0
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
joblib==1.3.0
pydantic==2.4.0
```

---

## 2. DOCKER COMMANDS

```bash
# Build image
docker build -t ml-api:latest .
docker build -t ml-api:v1.2 -f Dockerfile.prod .

# Run container
docker run -d \
  --name ml-api \
  -p 8000:8000 \
  -e MODEL_PATH=/app/models/model.pkl \
  -v $(pwd)/models:/app/models \
  ml-api:latest

# Check logs
docker logs ml-api
docker logs ml-api -f          # follow (live)
docker logs ml-api --tail 100  # last 100 lines

# Enter container
docker exec -it ml-api bash

# Check running containers
docker ps
docker ps -a   # all including stopped

# Stop / remove
docker stop ml-api
docker rm ml-api

# Images
docker images
docker rmi ml-api:latest
docker image prune   # clean unused images

# Stats (CPU, memory)
docker stats ml-api
```

---

## 3. DOCKER COMPOSE — Multi-Container

```yaml
# docker-compose.yml
version: "3.9"

services:

  api:
    build: .
    image: ml-api:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.pkl
      - DB_URL=postgresql://user:pass@db:5432/mldb
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped
    networks:
      - ml-network

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mldb
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mldb"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ml-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - ml-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
    networks:
      - ml-network

volumes:
  postgres_data:

networks:
  ml-network:
    driver: bridge
```

```bash
# Docker Compose commands
docker compose up -d          # start all services
docker compose down           # stop all
docker compose down -v        # stop + delete volumes
docker compose logs api -f    # follow API logs
docker compose ps             # service status
docker compose exec api bash  # enter API container
docker compose build          # rebuild images
docker compose pull           # pull latest images
```

---

## 4. GITHUB ACTIONS — CI/CD Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ml-api

jobs:

  # ── Job 1: Test ────────────────────────────
  test:
    name: Run Tests
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ -v \
            --cov=src \
            --cov-report=xml \
            --cov-report=term-missing

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml

  # ── Job 2: Lint ────────────────────────────
  lint:
    name: Code Quality
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install linters
        run: pip install ruff black mypy

      - name: Run ruff (linting)
        run: ruff check .

      - name: Run black (formatting)
        run: black --check .

      - name: Run mypy (type check)
        run: mypy src/ --ignore-missing-imports

  # ── Job 3: Train Model ─────────────────────
  train:
    name: Train ML Model
    runs-on: ubuntu-latest
    needs: [test, lint]
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python train.py --output models/model.pkl

      - name: Evaluate model
        run: |
          python evaluate.py --model models/model.pkl
          # Fails if accuracy < threshold

      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: models/model.pkl
          retention-days: 30

  # ── Job 4: Build Docker ────────────────────
  build:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    needs: [train]

    steps:
      - uses: actions/checkout@v4

      - name: Download model
        uses: actions/download-artifact@v3
        with:
          name: trained-model
          path: models/

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Job 5: Deploy ──────────────────────────
  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build]
    environment: production    # requires manual approval!

    steps:
      - name: Deploy to server
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /opt/ml-api
            docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            docker compose down
            docker compose up -d
            docker system prune -f
            echo "Deployed successfully!"

      - name: Notify Slack
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {"text": "✅ ML API deployed to production! Commit: ${{ github.sha }}"}
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 5. GITHUB ACTIONS — Scheduled Retraining

```yaml
# .github/workflows/retrain.yml
name: Weekly Model Retraining

on:
  schedule:
    - cron: "0 2 * * 0"   # Every Sunday at 2 AM UTC
  workflow_dispatch:        # Manual trigger button

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install deps
        run: pip install -r requirements.txt

      - name: Pull latest data
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: aws s3 cp s3://my-bucket/data/latest.csv data/train.csv

      - name: Retrain model
        run: python retrain.py

      - name: Run tests
        run: pytest tests/model_tests.py -v

      - name: Push model to S3
        run: aws s3 cp models/model.pkl s3://my-bucket/models/model_$(date +%Y%m%d).pkl

      - name: Update production model
        run: aws s3 cp models/model.pkl s3://my-bucket/models/model_latest.pkl
```

---

## 6. NGINX CONFIG (Reverse Proxy)

```nginx
# nginx.conf
upstream ml_api {
    server api:8000;
}

server {
    listen 80;
    server_name myapp.com;

    # Redirect to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name myapp.com;

    ssl_certificate     /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass         http://ml_api;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    $http_upgrade;
        proxy_set_header   Connection keep-alive;
        proxy_set_header   Host       $host;
        proxy_set_header   X-Real-IP  $remote_addr;
        proxy_cache_bypass $http_upgrade;

        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }

    # Increase body size for file uploads
    client_max_body_size 100M;
}
```

---

## 7. QUICK REFERENCE

```bash
# Full deploy from scratch
git clone https://github.com/you/ml-api
cd ml-api
docker compose up -d

# Update deployment
git pull
docker compose build api
docker compose up -d api

# Rollback
docker compose stop api
docker tag ml-api:previous ml-api:latest
docker compose up -d api

# View all logs
docker compose logs -f

# Scale up (3 API instances)
docker compose up -d --scale api=3
```
