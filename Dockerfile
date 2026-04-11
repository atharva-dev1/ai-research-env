# ── Stage 1: Build React frontend ──────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# Build outputs to /build/backend/static (per vite.config.js outDir: '../backend/static')
RUN npm run build

# ── Stage 2: Python backend ────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="atharvashaah01@gmail.com"
LABEL description="AI-Research-Env — Scientific Discovery Environment for AI Agents"

WORKDIR /app

# Critical: set PYTHONPATH so all imports resolve from /app
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default inference configuration (override with -e at docker run time)
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV API_KEY=""
ENV HF_TOKEN=""
ENV OPENAI_API_KEY=""

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY inference.py openenv.yaml ./

# Copy built frontend static files from Stage 1
COPY --from=frontend-builder /build/backend/static ./backend/static

# Validate imports and basic functionality at build time
RUN python3 -c "\
from backend.server import app; \
from backend.env.research_env import Action, ResearchEnv; \
env = ResearchEnv(); \
r = env.reset(); \
assert r is not None; \
print('Build-time validation passed')"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860", \
    "--workers", "1", "--timeout-keep-alive", "30"]
