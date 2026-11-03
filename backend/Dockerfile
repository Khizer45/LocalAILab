# Local AI Lab – backend (FastAPI)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/app/data

WORKDIR /app
RUN useradd --create-home --uid 1000 lab

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY datasets ./datasets
COPY devtools ./devtools
RUN mkdir -p /app/data && chown -R lab:lab /app/data

USER lab
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

# Inside the container the API must listen on all interfaces; docker-compose
# only publishes the frontend, and only on 127.0.0.1 of the host.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
