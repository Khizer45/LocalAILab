"""Optional MLflow experiment tracking (local storage only).

MLflow is imported lazily so the app starts quickly and still works when
MLflow is not installed. Tracking data is written to a local SQLite database
and a local artifact folder inside ``data/`` – nothing is sent to a server.
MLflow's own usage telemetry is disabled via environment variables, and the
offline network guard blocks any remaining outbound connection.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

log = logging.getLogger("app.tracking")

for _k, _v in {"MLFLOW_DISABLE_TELEMETRY": "true", "DO_NOT_TRACK": "true"}.items():
    os.environ.setdefault(_k, _v)


class Tracker:
    def __init__(self, enabled: bool, tracking_uri: str, artifact_dir: Path):
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.artifact_dir = artifact_dir
        self._mlflow = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def _client(self):
        if not self.enabled:
            return None
        if self._mlflow is None and self._error is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                self._mlflow = mlflow
            except Exception as exc:  # noqa: BLE001 - optional dependency
                self._error = f"MLflow unavailable: {type(exc).__name__}"
                log.warning("mlflow disabled", extra={"reason": self._error})
        return self._mlflow

    def status(self) -> dict:
        installed = True
        try:
            import importlib.util

            installed = importlib.util.find_spec("mlflow") is not None
        except Exception:  # noqa: BLE001
            installed = False
        return {"enabled": self.enabled, "installed": installed, "tracking_uri": self.tracking_uri,
                "artifact_dir": str(self.artifact_dir), "error": self._error,
                "ui_command": f"mlflow ui --backend-store-uri {self.tracking_uri} --port 5000"}

    def _experiment(self, mlflow, name: str) -> str:
        exp = mlflow.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        return mlflow.create_experiment(name, artifact_location=(self.artifact_dir / name).resolve().as_uri())

    def log_run(self, experiment: str, run_name: str, params: dict, metrics: dict, artifacts: dict[str, object],
                tags: dict | None = None) -> str | None:
        """Log one run. `artifacts` maps file name -> JSON-serialisable object. Never raises."""
        mlflow = self._client()
        if mlflow is None:
            return None
        try:
            with self._lock:
                exp_id = self._experiment(mlflow, experiment)
                with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
                    mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
                    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
                    if clean:
                        mlflow.log_metrics(clean)
                    if tags:
                        mlflow.set_tags({k: str(v) for k, v in tags.items()})
                    with tempfile.TemporaryDirectory() as tmp:
                        for name, obj in artifacts.items():
                            p = Path(tmp) / name
                            p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
                            mlflow.log_artifact(str(p))
                    return run.info.run_id
        except Exception as exc:  # noqa: BLE001
            self._error = f"MLflow logging failed: {type(exc).__name__}: {str(exc)[:200]}"
            log.warning("mlflow logging failed", extra={"error_type": type(exc).__name__})
            return None
