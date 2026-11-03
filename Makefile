# Linux / macOS shortcuts (Windows users: see scripts\*.bat)
PY := backend/.venv/bin/python

.PHONY: setup backend frontend demo test smoke docker docker-demo models

setup:
	python3 -m venv backend/.venv
	$(PY) -m pip install -r backend/requirements-dev.txt
	cd frontend && npm install && npm run build
	[ -f .env ] || cp .env.example .env
	$(PY) scripts/check_system.py

models:
	ollama pull llama3.2:1b && ollama pull qwen2.5:1.5b && ollama pull nomic-embed-text

backend:
	cd backend && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

frontend:
	cd frontend && npm run dev

demo:
	cd backend && (.venv/bin/python -m devtools.fake_ollama --port 11500 --speed 0.5 &) && \
	OLLAMA_BASE_URL=http://127.0.0.1:11500 EMBEDDING_MODEL=sim-embed DEFAULT_MODEL=sim-small:1b DATA_DIR=$$PWD/data-demo \
	.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

test:
	cd backend && .venv/bin/ruff check . && .venv/bin/pytest -q
	cd frontend && npm run build

smoke:
	$(PY) scripts/smoke_test.py --model $${MODEL:-llama3.2:1b}

docker:
	docker compose up -d --build

docker-demo:
	docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d --build
