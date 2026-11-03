"""Download the pretrained image feature extractors used by the Model Arena.

Runs outside the app, so Offline-Only mode can stay on. Every file is verified
with SHA-256 before it is used.

    backend\\.venv\\Scripts\\python scripts\\download_backbones.py              (both)
    backend\\.venv\\Scripts\\python scripts\\download_backbones.py mobilenetv2  (one)

Files are written to backend/data/arena/backbones (or $DATA_DIR/arena/backbones).
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("LOCAL_AI_LAB_NO_APP", "1")

from app.arena.images import BACKBONES  # noqa: E402


def target_dir() -> Path:
    data = os.environ.get("DATA_DIR")
    base = Path(data) if data else ROOT / "backend" / "data"
    return base / "arena" / "backbones"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(bid: str) -> None:
    b = BACKBONES[bid]
    dest = target_dir() / b["file"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256(dest) == b["sha256"]:
        print(f"{b['name']}: already installed ({dest})")
        return
    tmp = dest.with_suffix(".part")
    print(f"{b['name']}: downloading {b['size_mb']} MB from {b['url']}")
    with urllib.request.urlopen(b["url"], timeout=60) as r, tmp.open("wb") as f:  # noqa: S310 - fixed https URL
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1024 * 256):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%}", end="", flush=True)
    print()
    if sha256(tmp) != b["sha256"]:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{b['name']}: checksum mismatch – the file was deleted. Try again later.")
    tmp.replace(dest)
    print(f"{b['name']}: OK -> {dest}")


def main() -> None:
    wanted = sys.argv[1:] or list(BACKBONES)
    unknown = [w for w in wanted if w not in BACKBONES]
    if unknown:
        raise SystemExit(f"Unknown backbone(s): {', '.join(unknown)}. Choose from: {', '.join(BACKBONES)}")
    for bid in wanted:
        download(bid)
    print("Done. Restart is not needed – refresh the Model Arena page.")


if __name__ == "__main__":
    main()
