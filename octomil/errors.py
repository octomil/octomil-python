"""Canonical error codes matching SDK_FACADE_CONTRACT.md."""

from __future__ import annotations

from enum import Enum

from octomil._generated.error_code import ErrorCode as _ContractErrorCode


class OctomilErrorCode(str, Enum):
    """All 19 canonical error codes from the SDK Facade Contract."""

    # Transport
    NETWORK_UNAVAILABLE = "network_unavailable"
    REQUEST_TIMEOUT = "request_timeout"
    SERVER_ERROR = "server_error"

    # Auth
    INVALID_API_KEY = "invalid_api_key"
    AUTHENTICATION_FAILED = "authentication_failed"
    FORBIDDEN = "forbidden"

    # Model lifecycle
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_DISABLED = "model_disabled"

    # Download
    DOWNLOAD_FAILED = "download_failed"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    INSUFFICIENT_STORAGE = "insufficient_storage"

    # Runtime
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    MODEL_LOAD_FAILED = "model_load_failed"
    INFERENCE_FAILED = "inference_failed"
    INSUFFICIENT_MEMORY = "insufficient_memory"

    # Policy
    RATE_LIMITED = "rate_limited"

    # Validation
    INVALID_INPUT = "invalid_input"

    # Control
    CANCELLED = "cancelled"

    # Catch-all
    UNKNOWN = "unknown"

    @property
    def retryable(self) -> bool:
        return self in _RETRYABLE_CODES

    @classmethod
    def from_http_status(cls, status: int) -> OctomilErrorCode:
        return _HTTP_STATUS_MAP.get(status, cls.UNKNOWN)


_RETRYABLE_CODES = frozenset(
    {
        OctomilErrorCode.NETWORK_UNAVAILABLE,
        OctomilErrorCode.REQUEST_TIMEOUT,
        OctomilErrorCode.SERVER_ERROR,
        OctomilErrorCode.DOWNLOAD_FAILED,
        OctomilErrorCode.CHECKSUM_MISMATCH,
        OctomilErrorCode.INFERENCE_FAILED,
        OctomilErrorCode.RATE_LIMITED,
    }
)

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
    def retryable(self) -> bool:
        return self.code.retryable

    @classmethod
    def from_http_status(cls, status: int, message: str | None = None) -> OctomilError:
        code = OctomilErrorCode.from_http_status(status)
        msg = message or f"HTTP {status}"
        return cls(code=code, message=msg)

    def __repr__(self) -> str:
        return f"OctomilError(code={self.code.value}, retryable={self.retryable}, message={self.error_message!r})"


# ---------------------------------------------------------------------------
# Contract parity assertion
# ---------------------------------------------------------------------------
# Verify at import time that every value in the contract-generated ErrorCode
# enum has a corresponding member in the SDK's OctomilErrorCode.  This catches
# drift between the contract repo and the SDK.

_sdk_values = {m.value for m in OctomilErrorCode}
_contract_values = {m.value for m in _ContractErrorCode}
_missing = _contract_values - _sdk_values
if _missing:
    raise ImportError(
        f"OctomilErrorCode is missing contract error codes: {sorted(_missing)}. "
        "Update the enum to match octomil-contracts."
    )
