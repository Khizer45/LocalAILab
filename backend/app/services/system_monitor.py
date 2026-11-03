"""System resource inspection, GPU detection, model recommendations and a
background sampler used by the benchmark runner."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import psutil

MB = 1024 * 1024
GB = 1024 * MB


def detect_gpus() -> list[dict]:
    """Best-effort NVIDIA GPU detection via nvidia-smi (fixed argv, no shell)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(  # noqa: S603 - fixed arguments, no user input
            [exe, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append({"name": parts[0], "memory_total_mb": float(parts[1]), "memory_used_mb": float(parts[2])})
            except ValueError:
                continue
    return gpus


def runtime_processes() -> list[psutil.Process]:
    procs = []
    for p in psutil.process_iter(["name"]):
        name = (p.info.get("name") or "").lower()
        if name.startswith("ollama") or name in ("llama-server", "ollama_llama_server"):
            procs.append(p)
    return procs


def runtime_rss_mb() -> float | None:
    """Resident memory of local Ollama processes (None if not visible, e.g. Docker)."""
    total, seen = 0, False
    for p in runtime_processes():
        try:
            total += p.memory_info().rss
            seen = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return round(total / MB, 1) if seen else None


def snapshot(data_dir: Path) -> dict:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(data_dir if data_dir.exists() else Path.cwd())
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_total_mb": round(vm.total / MB),
        "ram_used_mb": round((vm.total - vm.available) / MB),
        "ram_available_mb": round(vm.available / MB),
        "ram_percent": vm.percent,
        "disk_total_gb": round(disk.total / GB, 1),
        "disk_free_gb": round(disk.free / GB, 1),
        "gpus": detect_gpus(),
        "runtime_rss_mb": runtime_rss_mb(),
        "pid": os.getpid(),
    }


def recommendations(snap: dict) -> dict:
    """Suggest model sizes for the detected hardware (CPU-friendly first)."""
    ram_gb = snap["ram_total_mb"] / 1024
    vram_gb = max((g["memory_total_mb"] for g in snap["gpus"]), default=0) / 1024
    budget = max(ram_gb * 0.5, vram_gb * 0.85)
    tiers = [  # licences: see datasets/model_catalog.json (qwen2.5:3b is research-only, so it is not suggested)
        (1.5, ["qwen3:0.6b", "gemma3:1b"], "Tiny models – fast on any CPU, limited quality."),
        (3.0, ["qwen3:1.7b", "llama3.2:1b", "granite3.3:2b"], "Small models – good starting point for 8 GB RAM laptops."),
        (5.0, ["qwen3:4b", "phi4-mini", "llama3.2:3b"], "3–4B models – noticeably better answers, slower on CPU."),
        (9.0, ["qwen3:8b", "qwen2.5:7b", "gemma3:4b"], "7–8B models – need ~16 GB RAM or a 8 GB GPU."),
    ]
    chat, note = tiers[0][1], tiers[0][2]
    for need, models, text in tiers:
        if budget >= need:
            chat, note = models, text
    if snap["disk_free_gb"] < 10:
        note += " Disk space is low (<10 GB free): download one model at a time."
    return {
        "memory_budget_gb": round(budget, 1),
        "chat_models": chat,
        "embedding_model": "nomic-embed-text",
        "note": note + " Check each model's licence before commercial use (Setup → Models shows it).",
        "commands": [f"ollama pull {chat[0]}", "ollama pull nomic-embed-text"],
    }


class ResourceSampler:
    """Samples system CPU/RAM (and Ollama RSS when visible) in a background thread."""

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu: list[float] = []
        self.ram_used_mb: list[float] = []
        self.rss_mb: list[float] = []
        self._procs: list[psutil.Process] = []

    def __enter__(self) -> ResourceSampler:
        self._procs = runtime_processes()
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True, name="resource-sampler")
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()

    def _sample(self) -> None:
        vm = psutil.virtual_memory()
        self.ram_used_mb.append((vm.total - vm.available) / MB)
        self.cpu.append(psutil.cpu_percent(interval=None))
        rss, seen = 0, False
        for p in self._procs:
            try:
                rss += p.memory_info().rss
                seen = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if seen:
            self.rss_mb.append(rss / MB)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def summary(self) -> dict:
        return {
            "peak_ram_used_mb": round(max(self.ram_used_mb), 1) if self.ram_used_mb else None,
            "avg_cpu_pct": round(sum(self.cpu) / len(self.cpu), 1) if self.cpu else None,
            "runtime_rss_mb": round(max(self.rss_mb), 1) if self.rss_mb else None,
        }
