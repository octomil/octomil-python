"""Conformance tests: SDK error handling matches contract fixtures.

Validates that OctomilError / OctomilErrorCode correctly deserialise
error payloads defined in octomil-contracts/fixtures/errors/.
"""

from __future__ import annotations

import pytest

from octomil.errors import OctomilError, OctomilErrorCode

# ---------------------------------------------------------------------------
# Fixtures embedded inline from octomil-contracts/fixtures/errors/
# ---------------------------------------------------------------------------

MODEL_NOT_FOUND_FIXTURE = {
    "code": "model_not_found",
    "message": "Model 'nonexistent-7b' not found in registry.",
    "retryable": False,
}

RATE_LIMITED_FIXTURE = {
    "code": "rate_limited",
    "message": "Too many requests. Retry after 30 seconds.",
    "retryable": True,
}

INFERENCE_FAILED_FIXTURE = {
    "code": "inference_failed",
    "message": "CoreML prediction failed: input tensor shape mismatch.",
    "retryable": True,
}

UNKNOWN_ERROR_FALLBACK_INPUT = {
    "code": "some_future_error_code",
    "message": "Something the SDK has never seen before.",
    "retryable": False,
}

UNKNOWN_ERROR_FALLBACK_EXPECTED = {
    "code": "unknown",
    "message": "Something the SDK has never seen before.",
    "retryable": False,
}


def _error_from_fixture(fixture: dict[str, object]) -> OctomilError:
    """Simulate SDK deserialisation of a contract error payload."""
    code_str = str(fixture["code"])
    try:
        code = OctomilErrorCode(code_str)
    except ValueError:
        code = OctomilErrorCode.UNKNOWN
    return OctomilError(code=code, message=str(fixture["message"]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestErrorFixtureDeserialization:
    """OctomilError must correctly represent each contract fixture."""

    @pytest.mark.parametrize(
        "fixture",
        [MODEL_NOT_FOUND_FIXTURE, RATE_LIMITED_FIXTURE, INFERENCE_FAILED_FIXTURE],
        ids=["model_not_found", "rate_limited", "inference_failed"],
    )
    def test_known_error_codes(self, fixture: dict[str, object]) -> None:
        err = _error_from_fixture(fixture)
        assert err.code.value == fixture["code"]
        assert err.error_message == fixture["message"]
        assert err.retryable is fixture["retryable"]

    def test_unknown_error_fallback(self) -> None:
        """Unrecognised codes MUST map to UNKNOWN (per contract spec)."""
        err = _error_from_fixture(UNKNOWN_ERROR_FALLBACK_INPUT)
        assert err.code == OctomilErrorCode.UNKNOWN
        assert err.code.value == UNKNOWN_ERROR_FALLBACK_EXPECTED["code"]
        # Original message must be preserved
        assert err.error_message == UNKNOWN_ERROR_FALLBACK_INPUT["message"]
        assert err.retryable is UNKNOWN_ERROR_FALLBACK_EXPECTED["retryable"]

    def test_retryable_property(self) -> None:
        """Retryable flag must match contract for all fixture codes."""
        for fixture in [MODEL_NOT_FOUND_FIXTURE, RATE_LIMITED_FIXTURE, INFERENCE_FAILED_FIXTURE]:
            err = _error_from_fixture(fixture)
            assert err.retryable is fixture["retryable"], (
                f"retryable mismatch for {fixture['code']}: SDK={err.retryable}, contract={fixture['retryable']}"
            )


class TestErrorSchema:
    """OctomilError must expose the three fields required by schemas/core/error.json."""

    REQUIRED_FIELDS = ["code", "message", "retryable"]

    def test_error_has_required_fields(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.SERVER_ERROR,
            message="internal error",
        )
        # code (as string value)
        assert hasattr(err, "code")
        assert isinstance(err.code.value, str)
        # message
        assert hasattr(err, "error_message")
        assert isinstance(err.error_message, str)
        # retryable
        assert hasattr(err, "retryable")
        assert isinstance(err.retryable, bool)
