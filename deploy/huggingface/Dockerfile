# Local AI Lab – public online demo (Hugging Face Spaces, Docker SDK).
# Build context: the repository root (deploy/huggingface/build_space.py copies this file there).

FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production LOG_JSON=true \
    DATA_DIR=/home/user/data \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 EMBEDDING_MODEL=sim-embed DEFAULT_MODEL=sim-small:1b \
    OLLAMA_CLOUD_ENABLED=false NETWORK_GUARD_ENABLED=true OFFLINE_ONLY_DEFAULT=true \
    MAX_UPLOAD_MB=5 MAX_PDF_PAGES=60 MAX_OUTPUT_TOKENS=512 \
    ARENA_MAX_UPLOAD_MB=10 ARENA_MAX_ROWS=20000 ARENA_MAX_COLUMNS=100 ARENA_MAX_IMAGES=2000 \
    ARENA_SANDBOX=true ARENA_SANDBOX_MEMORY_MB=1536 ARENA_SANDBOX_TIMEOUT_S=300 \
    SCHEDULER_ENABLED=false MLFLOW_ENABLED=true \
    DEMO_RESET_HOURS=6 DEMO_WRITES_PER_MINUTE=30

RUN useradd --create-home --uid 1000 user
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY --chown=user backend/app backend/app
COPY --chown=user backend/datasets backend/datasets
COPY --chown=user backend/devtools backend/devtools
COPY --chown=user samples samples
COPY --chown=user deploy/huggingface deploy/huggingface
COPY --from=ui --chown=user /ui/dist frontend/dist
RUN python deploy/huggingface/add_banner.py frontend/dist/index.html "$DEMO_RESET_HOURS" \
 && chmod 0555 deploy/huggingface/start.sh \
 && chown -R root:root backend/app backend/datasets backend/devtools deploy frontend/dist \
 && chmod -R a-w /app

USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=3)" || exit 1
CMD ["/app/deploy/huggingface/start.sh"]
