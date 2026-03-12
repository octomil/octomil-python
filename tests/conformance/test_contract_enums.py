"""Conformance tests: SDK enums match octomil-contracts definitions.

Each test loads the canonical enum values from the contract-generated code
(octomil._generated) and asserts that the SDK's own types cover every
contract-defined value.
"""

from __future__ import annotations

import pytest

from octomil._generated.compatibility_level import CompatibilityLevel
from octomil._generated.device_class import DeviceClass
from octomil._generated.error_code import ErrorCode as ContractErrorCode
from octomil._generated.finish_reason import FinishReason
from octomil._generated.model_status import ModelStatus
from octomil.errors import OctomilErrorCode

# ---------------------------------------------------------------------------
# ErrorCode parity
# ---------------------------------------------------------------------------


class TestErrorCodeParity:
    """OctomilErrorCode must cover every value in the contract ErrorCode enum."""

    def test_all_contract_error_codes_present(self) -> None:
        sdk_values = {m.value for m in OctomilErrorCode}
        contract_values = {m.value for m in ContractErrorCode}
        missing = contract_values - sdk_values
        assert not missing, f"SDK missing contract error codes: {sorted(missing)}"

    def test_member_names_match(self) -> None:
        """Enum *names* (UPPER_CASE) should also match 1:1."""
        sdk_names = {m.name for m in OctomilErrorCode}
        contract_names = {m.name for m in ContractErrorCode}
        missing = contract_names - sdk_names
        assert not missing, f"SDK missing contract error code names: {sorted(missing)}"

    def test_error_code_count(self) -> None:
        """SDK must have at least as many codes as the contract."""
        assert len(OctomilErrorCode) >= len(ContractErrorCode)

    @pytest.mark.parametrize(
        "code_value",
        [m.value for m in ContractErrorCode],
        ids=[m.name for m in ContractErrorCode],
    )
    def test_each_code_roundtrips(self, code_value: str) -> None:
        """Every contract code string must parse back into OctomilErrorCode."""
        sdk_code = OctomilErrorCode(code_value)
        assert sdk_code.value == code_value


# ---------------------------------------------------------------------------
# ModelStatus parity
# ---------------------------------------------------------------------------


class TestModelStatusParity:
    """Contract ModelStatus values must all exist as valid string constants."""

    CONTRACT_STATUSES = [m.value for m in ModelStatus]

    def test_expected_statuses(self) -> None:
        assert set(self.CONTRACT_STATUSES) == {
            "not_cached",
            "downloading",
            "ready",
            "error",
        }

    @pytest.mark.parametrize("status", CONTRACT_STATUSES)
    def test_status_is_string(self, status: str) -> None:
        assert isinstance(status, str)
        assert len(status) > 0


# ---------------------------------------------------------------------------
# DeviceClass parity
# ---------------------------------------------------------------------------


class TestDeviceClassParity:
    """Contract DeviceClass values must all exist."""

    CONTRACT_VALUES = [m.value for m in DeviceClass]

    def test_expected_device_classes(self) -> None:
        assert set(self.CONTRACT_VALUES) == {
            "flagship",
            "high",
            "mid",
            "low",
        }


# ---------------------------------------------------------------------------
# FinishReason parity
# ---------------------------------------------------------------------------


class TestFinishReasonParity:
    """Contract FinishReason values must all exist."""

    CONTRACT_VALUES = [m.value for m in FinishReason]

    def test_expected_finish_reasons(self) -> None:
        assert set(self.CONTRACT_VALUES) == {
            "stop",
            "tool_calls",
            "length",
            "content_filter",
        }


# ---------------------------------------------------------------------------
# CompatibilityLevel parity
# ---------------------------------------------------------------------------


class TestCompatibilityLevelParity:
    """Contract CompatibilityLevel values must all exist."""

    CONTRACT_VALUES = [m.value for m in CompatibilityLevel]

    def test_expected_levels(self) -> None:
        assert set(self.CONTRACT_VALUES) == {
            "stable",
            "beta",
            "experimental",
            "compatibility",
        }
