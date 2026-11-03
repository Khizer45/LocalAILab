"""Application exceptions mapped to readable JSON error responses."""

from __future__ import annotations


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code

    def to_dict(self) -> dict:
        data = {"code": self.code, "message": self.message}
        if self.hint:
            data["hint"] = self.hint
        return {"error": data}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"


class OfflineModeViolation(AppError):
    """Raised when something tries to leave the machine while Offline-Only mode is on."""

    status_code = 403
    code = "offline_mode_blocked"


class RuntimeUnavailable(AppError):
    status_code = 503
    code = "runtime_unavailable"


class ModelNotAvailable(AppError):
    status_code = 404
    code = "model_not_available"


class InferenceTimeout(AppError):
    status_code = 504
    code = "inference_timeout"


class InsufficientResources(AppError):
    status_code = 507
    code = "insufficient_resources"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
