"""Local AI Lab backend package."""

import os as _os

__version__ = "1.2.0"

# Third-party telemetry opt-outs. They must be set before any ML / web library is imported, and every entry
# point (server, CLI, scripts, tests, Alembic) imports this package first.
#
# Verified finding (v1.2): ONNX Runtime 1.31 on Linux sends usage telemetry to mobile.events.data.microsoft.com
# from long-running processes unless ORT_DISABLE_TELEMETRY is set before it is imported;
# onnxruntime.disable_telemetry_events() alone did not stop it. Native code is not covered by the Python
# socket guard, so these switches matter.
_FORCED = {"ORT_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
_DEFAULTS = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    "MLFLOW_DISABLE_TELEMETRY": "true", "ANONYMIZED_TELEMETRY": "False",
    "OTEL_SDK_DISABLED": "true", "OTEL_TRACES_EXPORTER": "none", "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none", "FASTAPI_OTEL_AUTO_CONFIGURE": "false",
}
for _k, _v in _FORCED.items():
    _os.environ[_k] = _v
for _k, _v in _DEFAULTS.items():
    _os.environ.setdefault(_k, _v)
TELEMETRY_OPT_OUTS = {**_DEFAULTS, **_FORCED}
