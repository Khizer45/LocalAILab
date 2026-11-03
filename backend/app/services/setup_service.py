"""First-run checks, model catalogue with memory fit, and model download / removal through Ollama.

Memory fit is an *estimate*: model file size (the weights Ollama loads) + KV cache for the chosen context length
(2 × layers × KV heads × head size × 2 bytes per token, read from ``/api/show`` when Ollama reports the
architecture) + ~0.4 GB runtime overhead. Real usage depends on the runtime version, quantisation of the cache,
GPU offload and other programs, so the verdicts are "fits", "tight" and "too big", not guarantees.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx

from app.core.errors import AppError, OfflineModeViolation, ValidationFailed
from app.services.system_monitor import snapshot

GB = 1024**3
OVERHEAD_GB = 0.4
MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,100}(:[A-Za-z0-9._-]{1,60})?$")


def load_catalog(datasets_dir: Path) -> dict:
    return json.loads((datasets_dir / "model_catalog.json").read_text(encoding="utf-8"))


def kv_cache_gb(show: dict | None, num_ctx: int) -> float | None:
    info = (show or {}).get("model_info") or {}
    arch = info.get("general.architecture")
    if not arch:
        return None
    layers = info.get(f"{arch}.block_count")
    emb = info.get(f"{arch}.embedding_length")
    heads = info.get(f"{arch}.attention.head_count")
    kv_heads = info.get(f"{arch}.attention.head_count_kv") or heads
    key_len = info.get(f"{arch}.attention.key_length")
    if not (layers and heads and (emb or key_len)):
        return None
    head_dim = key_len or emb / heads
    return round(2 * layers * kv_heads * head_dim * 2 * num_ctx / GB, 2)


def fit(size_bytes: float | None, kv_gb: float | None, num_ctx: int, snap: dict, params_b: float | None = None) -> dict:
    """Estimate memory need and compare it with this computer."""
    if size_bytes is None and params_b is not None:
        size_bytes = params_b * 0.6 * GB  # ≈ 4.5-bit quantisation
    if size_bytes is None:
        return {"verdict": "unknown"}
    kv = kv_gb if kv_gb is not None else round(num_ctx / 4096 * 0.25 * max(1.0, (params_b or size_bytes / 0.6 / GB) / 3), 2)
    need = round(size_bytes / GB + kv + OVERHEAD_GB, 2)
    ram_total = snap["ram_total_mb"] / 1024
    ram_free = snap["ram_available_mb"] / 1024
    vram = max((g["memory_total_mb"] for g in snap.get("gpus", [])), default=0) / 1024
    if vram and need <= vram * 0.9:
        verdict, where = "fits", "GPU"
    elif need <= ram_free * 0.9:
        verdict, where = "fits", "RAM"
    elif need <= ram_total * 0.8:
        verdict, where = "tight", "RAM – close other programs first"
    else:
        verdict, where = "too_big", "not enough memory"
    return {"verdict": verdict, "where": where, "estimated_gb": need, "kv_cache_gb": kv, "num_ctx": num_ctx,
            "kv_source": "model architecture" if kv_gb is not None else "rule of thumb"}


class SetupService:
    def __init__(self, settings, registry, settings_service, engine):
        self.settings = settings
        self.registry = registry
        self.settings_service = settings_service
        self.engine = engine

    # ------------------------------------------------------------ checks
    def check(self) -> dict:
        from app.core.backup import list_backups
        from app.db.migrate import current_revision, head_revision

        s = self.settings
        snap = snapshot(s.data_dir)
        items: list[dict] = []

        def add(id_, label, status, detail, fix=None):
            items.append({"id": id_, "label": label, "status": status, "detail": detail, "fix": fix})

        add("python", "Python", "ok" if sys.version_info >= (3, 11) else "fail", sys.version.split()[0],
            None if sys.version_info >= (3, 11) else "Install Python 3.11 or newer.")
        version = None
        try:
            version = self.registry.local.version()
            add("ollama", "Ollama runtime", "ok", f"version {version} at {self.registry.local.base_url}")
        except AppError as exc:
            add("ollama", "Ollama runtime", "fail", exc.message, exc.hint)
        models, _ = self.registry.list_models(force=True) if version else ([], {})
        chat = [m for m in models if m.get("provider") == "local" and "embedding" not in (m.get("capabilities") or [])
                and "completion" in (m.get("capabilities") or ["completion"])]
        emb = [m for m in models if "embedding" in (m.get("capabilities") or []) or "embed" in m["id"]]
        add("chat_model", "A chat model is installed", "ok" if chat else ("fail" if version else "skip"),
            f"{len(chat)} installed" if chat else "none",
            None if chat else "Pick one under Setup → Models (or run: ollama pull qwen3:1.7b).")
        add("embedding_model", "Embedding model for document search", "ok" if emb else "warn",
            ", ".join(m["id"] for m in emb[:3]) or "none – a built-in fallback embedder is used (lower quality)",
            None if emb else "ollama pull nomic-embed-text")
        free = snap["disk_free_gb"]
        add("disk", "Free disk space", "ok" if free >= 10 else ("warn" if free >= 3 else "fail"), f"{free} GB free",
            None if free >= 10 else "Models need 1–6 GB each; free some space.")
        ram = round(snap["ram_total_mb"] / 1024, 1)
        add("ram", "Memory", "ok" if ram >= 8 else "warn", f"{ram} GB RAM, {round(snap['ram_available_mb'] / 1024, 1)} GB "
            "available" + (f"; GPU: {', '.join(g['name'] for g in snap['gpus'])}" if snap["gpus"] else "; no NVIDIA GPU detected"),
            None if ram >= 8 else "Use models of 1–2B parameters and a 2048-token context.")
        rs = self.settings_service.get()
        add("offline", "Offline-Only mode", "ok" if rs.offline_only else "warn",
            "on – cloud models and outbound requests from the app are blocked" if rs.offline_only else
            "off – cloud features may be used", None if rs.offline_only else "Turn it on in Settings when you work "
            "with confidential files.")
        local_bind = s.host in ("127.0.0.1", "localhost", "::1")
        add("access", "Network exposure", "ok" if local_bind or s.auth_enabled else "fail",
            f"bound to {s.host}, authentication {'on' if s.auth_enabled else 'off'}",
            None if local_bind or s.auth_enabled else "Set AUTH_ENABLED=true before exposing the app to a network, "
                                                       "and put HTTPS in front of it.")
        try:
            cur, head = current_revision(self.engine), head_revision()
            add("database", "Database schema", "ok" if cur == head else "fail", f"revision {cur} (latest {head})")
        except Exception as exc:  # noqa: BLE001
            add("database", "Database schema", "fail", type(exc).__name__)
        backups = [b for b in list_backups(s.backups_dir) if b["name"].endswith(".zip")]
        add("backup", "Backups", "ok" if backups else "warn",
            f"last backup {backups[0]['created_at']}" if backups else "no backup yet",
            None if backups else "Create one under Settings → Backup & restore.")
        dist = s.frontend_dist / "index.html"
        add("frontend", "Web interface built", "ok" if dist.exists() else "warn",
            "served by the backend" if dist.exists() else "not built (development mode uses the Vite server)")
        worst = "fail" if any(i["status"] == "fail" for i in items) else (
            "warn" if any(i["status"] == "warn" for i in items) else "ok")
        return {"status": worst, "checks": items, "system": snap}

    # ------------------------------------------------------------ catalogue
    def catalog(self, num_ctx: int = 4096) -> dict:
        cat = load_catalog(self.settings.datasets_dir)
        snap = snapshot(self.settings.data_dir)
        installed: dict[str, dict] = {}
        try:
            for t in self.registry.local.tags():
                installed[t.get("name") or t.get("model")] = t
        except AppError:
            pass
        rows = []
        for m in cat["models"]:
            tag = installed.get(m["name"]) or installed.get(m["name"] + ":latest")
            show = None
            if tag:
                try:
                    show = self.registry.local.show(m["name"])
                except AppError:
                    show = None
            size = tag.get("size") if tag else m["download_gb"] * GB
            rows.append({**m, "installed": bool(tag),
                         "fit": fit(size, kv_cache_gb(show, num_ctx), num_ctx, snap, m.get("params_b"))})
        others = []
        for name, t in installed.items():
            if any(r["name"] == name or r["name"] + ":latest" == name for r in rows):
                continue
            try:
                show = self.registry.local.show(name)
            except AppError:
                show = None
            others.append({"name": name, "installed": True, "size_gb": round((t.get("size") or 0) / GB, 2),
                           "license": (show or {}).get("license", "")[:200] or "see the model page",
                           "fit": fit(t.get("size"), kv_cache_gb(show, num_ctx), num_ctx, snap)})
        return {"note": cat["note"], "catalog": rows, "other_installed": others, "num_ctx": num_ctx,
                "system": {"ram_total_gb": round(snap["ram_total_mb"] / 1024, 1),
                           "ram_available_gb": round(snap["ram_available_mb"] / 1024, 1),
                           "gpus": snap["gpus"], "disk_free_gb": snap["disk_free_gb"]}}

    # ------------------------------------------------------------ pull / delete
    def _require_online(self) -> None:
        if self.settings_service.get().offline_only:
            raise OfflineModeViolation(
                "Offline-Only mode is on, and downloading a model needs the internet.",
                hint="Switch Offline-Only off in Settings for the download (Ollama downloads the files, not this "
                     "app), then switch it back on – or run `ollama pull <name>` yourself.")

    def pull(self, name: str) -> Iterator[dict]:
        if not MODEL_NAME.match(name or ""):
            raise ValidationFailed("Invalid model name.")
        self._require_online()
        client = self.registry.local
        try:
            with client._http.stream("POST", "/api/pull", json={"model": name, "stream": True},
                                     timeout=httpx.Timeout(None, connect=client.connect_timeout_s)) as r:
                if r.status_code >= 400:
                    r.read()
                    raise client._error_from_body(r.status_code, r.text, name)
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if "error" in obj:
                        yield {"type": "error", "message": str(obj["error"])[:300]}
                        return
                    total, done = obj.get("total"), obj.get("completed")
                    yield {"type": "progress", "status": obj.get("status"), "total": total, "completed": done,
                           "percent": round(done / total * 100, 1) if total and done else None}
        except AppError as exc:
            yield {"type": "error", "message": exc.message, "hint": exc.hint}
            return
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": client._translate(exc).message}
            return
        self.registry.invalidate()
        yield {"type": "done", "model": name}

    def delete(self, name: str) -> None:
        if not MODEL_NAME.match(name or ""):
            raise ValidationFailed("Invalid model name.")
        client = self.registry.local
        try:
            r = client._http.request("DELETE", "/api/delete", json={"model": name}, timeout=60)
        except Exception as exc:  # noqa: BLE001
            raise client._translate(exc) from exc
        if r.status_code >= 400:
            raise client._error_from_body(r.status_code, r.text, name)
        self.registry.invalidate()


def disk_free_gb(path: Path) -> float:
    return round(shutil.disk_usage(path).free / GB, 1)
