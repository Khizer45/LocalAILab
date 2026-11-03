"""Inspect this computer and suggest Ollama models that fit.

    backend\\.venv\\Scripts\\python scripts\\check_system.py      (Windows)
    backend/.venv/bin/python scripts/check_system.py            (Linux/macOS)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.system_monitor import recommendations, snapshot  # noqa: E402

snap = snapshot(Path(__file__).resolve().parents[1])
rec = recommendations(snap)
print(f"OS:        {snap['os']} ({snap['machine']}), Python {snap['python']}")
print(f"CPU:       {snap['cpu_count_physical']} cores / {snap['cpu_count_logical']} threads")
print(f"RAM:       {snap['ram_total_mb'] / 1024:.1f} GB total, {snap['ram_available_mb'] / 1024:.1f} GB available")
print(f"Disk:      {snap['disk_free_gb']} GB free of {snap['disk_total_gb']} GB")
gpus = ", ".join(f"{g['name']} ({g['memory_total_mb'] / 1024:.0f} GB)" for g in snap["gpus"]) or "none detected (CPU only)"
print(f"GPU:       {gpus}")
print()
print(f"Suggested chat models: {', '.join(rec['chat_models'])}")
print(f"Embedding model:       {rec['embedding_model']}")
print(rec["note"])
print("\nDownload commands:")
for c in rec["commands"]:
    print("  " + c)
