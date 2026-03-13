"""Canonical error codes matching SDK_FACADE_CONTRACT.md."""

from __future__ import annotations

from octomil._generated.error_code import (
    ERROR_CLASSIFICATION,
    ErrorCategory,
    ErrorClassification,
    RetryClass,
    SuggestedAction,
)
from octomil._generated.error_code import (
    ErrorCode as OctomilErrorCode,
)

__all__ = [
    "OctomilErrorCode",
    "ErrorCategory",
    "ErrorClassification",
    "RetryClass",
    "SuggestedAction",
    "ERROR_CLASSIFICATION",
    "OctomilError",
]


_HTTP_STATUS_MAP: dict[int, OctomilErrorCode] = {
    401: OctomilErrorCode.INVALID_API_KEY,
    403: OctomilErrorCode.FORBIDDEN,
    404: OctomilErrorCode.MODEL_NOT_FOUND,
    408: OctomilErrorCode.REQUEST_TIMEOUT,
    429: OctomilErrorCode.RATE_LIMITED,
    500: OctomilErrorCode.SERVER_ERROR,
    502: OctomilErrorCode.SERVER_ERROR,
    503: OctomilErrorCode.SERVER_ERROR,
}


class OctomilError(Exception):
    """Unified exception for all Octomil SDK errors."""

    def __init__(
        self,
        code: OctomilErrorCode,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.error_message = message
        self.cause = cause

    @property
    def _classification(self) -> ErrorClassification:
        return ERROR_CLASSIFICATION[self.code]

    @property
    def category(self) -> ErrorCategory:
        return self._classification.category

    @property
    def retry_class(self) -> RetryClass:
        return self._classification.retry_class

    @property
    def fallback_eligible(self) -> bool:
        return self._classification.fallback_eligible

    @property
    def suggested_action(self) -> SuggestedAction:
        return self._classification.suggested_action

    @property
    def retryable(self) -> bool:
        return self.retry_class != RetryClass.NEVER

    @classmethod
    def from_http_status(cls, status: int, message: str | None = None) -> OctomilError:
        code = _HTTP_STATUS_MAP.get(status, OctomilErrorCode.UNKNOWN)
        msg = message or f"HTTP {status}"
        return cls(code=code, message=msg)

    def __repr__(self) -> str:
        return f"OctomilError(code={self.code.value}, retryable={self.retryable}, message={self.error_message!r})"
