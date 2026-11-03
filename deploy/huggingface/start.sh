#!/usr/bin/env bash
# Public demo: Ollama simulator + Local AI Lab. The data folder is wiped and the demo re-seeded every
# DEMO_RESET_HOURS hours, so nothing a visitor uploads stays for long.
set -u
cd /app/backend
RESET_HOURS="${DEMO_RESET_HOURS:-6}"

python -m devtools.fake_ollama --host 127.0.0.1 --port 11434 --speed 0.5 &

while true; do
  rm -rf "${DATA_DIR:?}" && mkdir -p "$DATA_DIR/workspace"
  (sleep 5; python /app/deploy/huggingface/seed_demo.py "http://127.0.0.1:7860") &
  timeout --signal=TERM --kill-after=30s "${RESET_HOURS}h" \
    python -m uvicorn public_demo:app --app-dir /app/deploy/huggingface --host 0.0.0.0 --port 7860 \
      --no-server-header --timeout-keep-alive 30 --limit-concurrency 200
  echo "[demo] periodic reset" >&2
  sleep 1
done
