"""Assemble the folder that is uploaded to the Hugging Face Space.

    python deploy/huggingface/build_space.py [output_dir]      (default: ../local-ai-lab-space)

The Space gets only what the demo needs: backend code, built-in datasets, the simulator, samples, the frontend
sources (built inside Docker) and this folder, plus the Space README and Dockerfile at its root.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parent / "local-ai-lab-space"
HERE = Path(__file__).resolve().parent

KEEP = ["backend/requirements.txt", "backend/app", "backend/datasets", "backend/devtools", "samples",
        "frontend", "deploy/huggingface"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules", "dist", "test-results", "playwright-report",
                              "*.tsbuildinfo", ".hypothesis", ".pytest_cache", "data", "data-demo", "*.joblib")

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
for rel in KEEP:
    src, dst = ROOT / rel, OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=SKIP)
    else:
        shutil.copy2(src, dst)
shutil.copy2(HERE / "Dockerfile", OUT / "Dockerfile")
shutil.copy2(HERE / "README.md", OUT / "README.md")
(OUT / ".dockerignore").write_text("**/node_modules\n**/__pycache__\n**/dist\n.git\n", encoding="utf-8")
(OUT / ".gitattributes").write_text("*.pdf filter=lfs diff=lfs merge=lfs -text\n*.zip filter=lfs diff=lfs merge=lfs -text\n"
                                    "*.onnx filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n"
                                    "*.svg filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
files = [p for p in OUT.rglob("*") if p.is_file()]
print(f"Space folder: {OUT}  ({len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB)")
