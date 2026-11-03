"""Application configuration.

All configuration comes from environment variables (or a `.env` file in the
backend folder). Values that a user may change at runtime from the Settings
screen (offline mode, allowed models, limits, ...) are only *defaults* here;
the live values are stored in the database and managed by
`app.services.settings_service.SettingsService`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(REPO_DIR / ".env"), str(BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- general -----------------------------------------------------------
    app_name: str = "Local AI Lab"
    app_env: str = Field(default="development", description="development | production | test")
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = True

    # --- storage -----------------------------------------------------------
    data_dir: Path = BACKEND_DIR / "data"
    datasets_dir: Path = BACKEND_DIR / "datasets"
    database_url: str | None = None  # default: sqlite file inside data_dir
    frontend_dist: Path = REPO_DIR / "frontend" / "dist"

    # --- local model runtime ----------------------------------------------
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_connect_timeout_s: float = 5.0

    # --- optional Ollama Cloud (only usable when Offline-Only mode is OFF) --
    # The API key is read from the backend environment and is never sent to
    # the browser. Cloud models are always blocked in Offline-Only mode.
    ollama_cloud_enabled: bool = False
    ollama_cloud_base_url: str = "https://ollama.com"
    ollama_api_key: str | None = None

    # --- privacy -----------------------------------------------------------
    offline_only_default: bool = True
    # Hosts that remain reachable while Offline-Only mode is on. Loopback is
    # always allowed. Add the Ollama container name when using Docker.
    offline_allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "::1"]
    )
    # Process-wide socket guard (Python audit hook). Blocks every outbound TCP
    # connection to a non-allowed address while Offline-Only mode is on.
    network_guard_enabled: bool = True

    # --- auth --------------------------------------------------------------
    auth_enabled: bool = False
    # Without authentication only clients from these networks may use the API (remote-access guard).
    # Docker Compose adds its private bridge networks (its ports are published on 127.0.0.1 only).
    unauthenticated_networks: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["127.0.0.0/8", "::1/128"])
    # Without authentication, requests must address the API by one of these host names (blocks DNS-rebinding
    # attacks from malicious websites) and state-changing requests from other web origins are refused (CSRF).
    local_host_names: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "::1"])
    allow_registration: bool = True
    token_ttl_hours: int = 24

    # --- defaults for runtime-editable settings ---------------------------
    default_model: str = ""
    embedding_model: str = "nomic-embed-text"
    max_upload_mb: int = 20
    max_pdf_pages: int = 500
    # pypdf (BSD-3-Clause, default) | pymupdf (optional, AGPL-3.0 – see docs/LEGAL_AND_LICENSING.md)
    pdf_parser: str = "pypdf"
    max_context_tokens: int = 8192
    max_output_tokens: int = 1024
    request_timeout_s: int = 180
    benchmark_timeout_s: int = 120
    retrieval_top_k: int = 4
    chunk_size: int = 900
    chunk_overlap: int = 150

    # --- processing --------------------------------------------------------
    doc_processing_mode: str = Field(default="background", description="background | inline")

    # --- ML / MLOps --------------------------------------------------------
    ml_baseline_enabled: bool = True
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str | None = None  # default: sqlite db inside data_dir

    # --- Model Arena (bring your own dataset / model) ----------------------
    arena_enabled: bool = True
    arena_max_upload_mb: int = 300  # CSV, image ZIP or model file
    arena_max_rows: int = 200_000
    arena_max_columns: int = 500
    arena_max_images: int = 20_000
    # uploaded models are inspected / run in a separate, limited process (see app/arena/sandbox.py)
    arena_sandbox: bool = True
    arena_sandbox_memory_mb: int = 4096
    arena_sandbox_timeout_s: int = 1800

    # --- automation ----------------------------------------------------------
    scheduler_enabled: bool = True
    work_agent_enabled: bool = True
    # The Work Agent can only read/write files inside this folder (default: data/workspace)
    work_agent_workspace: Path | None = None

    # --- Agent Arena -------------------------------------------------------
    agents_enabled: bool = True
    agent_max_steps: int = 6
    agent_task_timeout_s: int = 180

    # --- CORS --------------------------------------------------------------
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    @field_validator("offline_allowed_hosts", "cors_origins", "unauthenticated_networks", "local_host_names",
                     mode="before")
    @classmethod
    def _split_csv(cls, value):  # allow "a,b,c" in env files
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                import json

                return json.loads(value)
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    # --- derived paths -----------------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def ml_dir(self) -> Path:
        return self.data_dir / "ml"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def arena_dir(self) -> Path:
        return self.data_dir / "arena"

    @property
    def backups_dir(self) -> Path:
        return self.data_dir / "backups"

    @property
    def workspace_dir(self) -> Path:
        return self.work_agent_workspace or (self.data_dir / "workspace")

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'local_ai_lab.db').as_posix()}"

    @property
    def mlflow_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return f"sqlite:///{(self.data_dir / 'mlflow.db').as_posix()}"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.logs_dir, self.ml_dir, self.reports_dir, self.arena_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
