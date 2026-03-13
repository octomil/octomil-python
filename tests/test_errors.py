"""Tests for octomil.errors — canonical error codes and unified exception."""

from __future__ import annotations

import pytest

from octomil.errors import OctomilError, OctomilErrorCode

# ---------------------------------------------------------------------------
# All 19 canonical codes
# ---------------------------------------------------------------------------

ALL_CODES = [
    "invalid_api_key",
    "authentication_failed",
    "forbidden",
    "device_not_registered",
    "token_expired",
    "device_revoked",
    "network_unavailable",
    "request_timeout",
    "server_error",
    "rate_limited",
    "invalid_input",
    "unsupported_modality",
    "context_too_large",
    "model_not_found",
    "model_disabled",
    "version_not_found",
    "download_failed",
    "checksum_mismatch",
    "insufficient_storage",
    "insufficient_memory",
    "runtime_unavailable",
    "accelerator_unavailable",
    "model_load_failed",
    "inference_failed",
    "stream_interrupted",
    "policy_denied",
    "cloud_fallback_disallowed",
    "max_tool_rounds_exceeded",
    "control_sync_failed",
    "assignment_not_found",
    "cancelled",
    "app_backgrounded",
    "training_failed",
    "training_not_supported",
    "weight_upload_failed",
    "unknown",
]


class TestOctomilErrorCodeEnum:
    def test_has_exactly_36_members(self) -> None:
        assert len(OctomilErrorCode) == 36

    @pytest.mark.parametrize("value", ALL_CODES)
    def test_all_canonical_codes_exist(self, value: str) -> None:
        code = OctomilErrorCode(value)
        assert code.value == value

    def test_is_str_enum(self) -> None:
        """Each member should be usable as a plain string."""
        assert isinstance(OctomilErrorCode.UNKNOWN, str)
        assert OctomilErrorCode.UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# Retryable property
# ---------------------------------------------------------------------------

RETRYABLE_CODES = {
    OctomilErrorCode.NETWORK_UNAVAILABLE,
    OctomilErrorCode.REQUEST_TIMEOUT,
    OctomilErrorCode.SERVER_ERROR,
    OctomilErrorCode.RATE_LIMITED,
    OctomilErrorCode.DOWNLOAD_FAILED,
    OctomilErrorCode.CHECKSUM_MISMATCH,
    OctomilErrorCode.MODEL_LOAD_FAILED,
    OctomilErrorCode.INFERENCE_FAILED,
    OctomilErrorCode.STREAM_INTERRUPTED,
    OctomilErrorCode.CONTROL_SYNC_FAILED,
    OctomilErrorCode.APP_BACKGROUNDED,
    OctomilErrorCode.TRAINING_FAILED,
    OctomilErrorCode.WEIGHT_UPLOAD_FAILED,
}

NON_RETRYABLE_CODES = set(OctomilErrorCode) - RETRYABLE_CODES


class TestRetryableProperty:
    @pytest.mark.parametrize("code", sorted(RETRYABLE_CODES, key=lambda c: c.value))
    def test_retryable_codes(self, code: OctomilErrorCode) -> None:
        err = OctomilError(code=code, message="test")
        assert err.retryable is True

    @pytest.mark.parametrize("code", sorted(NON_RETRYABLE_CODES, key=lambda c: c.value))
    def test_non_retryable_codes(self, code: OctomilErrorCode) -> None:
        err = OctomilError(code=code, message="test")
        assert err.retryable is False


# ---------------------------------------------------------------------------
# from_http_status
# ---------------------------------------------------------------------------


class TestFromHttpStatus:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (400, OctomilErrorCode.INVALID_INPUT),
            (401, OctomilErrorCode.AUTHENTICATION_FAILED),
            (403, OctomilErrorCode.FORBIDDEN),
            (404, OctomilErrorCode.MODEL_NOT_FOUND),
            (429, OctomilErrorCode.RATE_LIMITED),
            (500, OctomilErrorCode.SERVER_ERROR),
            (502, OctomilErrorCode.SERVER_ERROR),
            (503, OctomilErrorCode.SERVER_ERROR),
        ],
    )
    def test_mapped_status_codes(self, status: int, expected: OctomilErrorCode) -> None:
        err = OctomilError.from_http_status(status)
        assert err.code is expected

    @pytest.mark.parametrize("status", [200, 201, 204, 301, 408, 418, 504])
    def test_unmapped_status_returns_unknown(self, status: int) -> None:
        err = OctomilError.from_http_status(status)
        assert err.code is OctomilErrorCode.UNKNOWN


# ---------------------------------------------------------------------------
# OctomilError construction and properties
# ---------------------------------------------------------------------------


class TestOctomilError:
    def test_is_exception_subclass(self) -> None:
        assert issubclass(OctomilError, Exception)

    def test_basic_construction(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.INVALID_API_KEY,
            message="bad key",
        )
        assert err.code is OctomilErrorCode.INVALID_API_KEY
        assert err.error_message == "bad key"
        assert str(err) == "bad key"
        assert err.cause is None

    def test_construction_with_cause(self) -> None:
        cause = ConnectionError("socket closed")
        err = OctomilError(
            code=OctomilErrorCode.NETWORK_UNAVAILABLE,
            message="connection lost",
            cause=cause,
        )
        assert err.cause is cause

    def test_retryable_delegates_to_code(self) -> None:
        retryable_err = OctomilError(
            code=OctomilErrorCode.SERVER_ERROR,
            message="500",
        )
        assert retryable_err.retryable is True

        non_retryable_err = OctomilError(
            code=OctomilErrorCode.FORBIDDEN,
            message="403",
        )
        assert non_retryable_err.retryable is False

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            raise OctomilError(
                code=OctomilErrorCode.MODEL_NOT_FOUND,
                message="no such model",
            )
        assert exc_info.value.code is OctomilErrorCode.MODEL_NOT_FOUND


# ---------------------------------------------------------------------------
# OctomilError.from_http_status factory
# ---------------------------------------------------------------------------


class TestOctomilErrorFromHttpStatus:
    def test_with_explicit_message(self) -> None:
        err = OctomilError.from_http_status(401, "Invalid token")
        assert err.code is OctomilErrorCode.AUTHENTICATION_FAILED
        assert err.error_message == "Invalid token"

    def test_without_message_uses_default(self) -> None:
        err = OctomilError.from_http_status(503)
        assert err.code is OctomilErrorCode.SERVER_ERROR
        assert err.error_message == "HTTP 503"

    def test_unknown_status_code(self) -> None:
        err = OctomilError.from_http_status(418)
        assert err.code is OctomilErrorCode.UNKNOWN
        assert err.error_message == "HTTP 418"


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestOctomilErrorRepr:
    def test_repr_format(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.RATE_LIMITED,
            message="slow down",
        )
        r = repr(err)
        assert "code=rate_limited" in r
        assert "retryable=True" in r
        assert "message='slow down'" in r

    def test_repr_non_retryable(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.FORBIDDEN,
            message="nope",
        )
        r = repr(err)
        assert "retryable=False" in r
