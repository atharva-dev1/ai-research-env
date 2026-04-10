# ── Stage 1: Build React frontend ──────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
# Build outputs to /build/backend/static (per vite.config.js outDir: '../backend/static')
RUN npm run build

# ── Stage 2: Python backend ────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY inference.py openenv.yaml ./
# Copy built frontend static files from Stage 1
COPY --from=frontend-builder /build/backend/static ./backend/static
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1
CMD ["python", "-m", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860"]
