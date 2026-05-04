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
    400: OctomilErrorCode.INVALID_INPUT,
    401: OctomilErrorCode.AUTHENTICATION_FAILED,
    403: OctomilErrorCode.FORBIDDEN,
    404: OctomilErrorCode.MODEL_NOT_FOUND,
    # Cutover follow-up #70: round-trip parity with the
    # serve-layer status map. 413 = CONTEXT_TOO_LARGE,
    # 422 = UNSUPPORTED_MODALITY, 429 = RATE_LIMITED,
    # 499 = CANCELLED (nginx convention; widely understood
    # by API clients).
    413: OctomilErrorCode.CONTEXT_TOO_LARGE,
    422: OctomilErrorCode.UNSUPPORTED_MODALITY,
    429: OctomilErrorCode.RATE_LIMITED,
    499: OctomilErrorCode.CANCELLED,
    500: OctomilErrorCode.SERVER_ERROR,
    502: OctomilErrorCode.SERVER_ERROR,
    503: OctomilErrorCode.SERVER_ERROR,
    504: OctomilErrorCode.REQUEST_TIMEOUT,
    507: OctomilErrorCode.INSUFFICIENT_STORAGE,
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
