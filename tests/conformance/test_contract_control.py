"""Conformance tests: ControlSyncResult matches contract schema.

Validates that the SDK's ControlSyncResult dataclass has the required
fields defined in octomil-contracts/schemas/core/control_sync_result.json.
"""

from __future__ import annotations

import dataclasses

import pytest

from octomil.control import ControlSyncResult

# ---------------------------------------------------------------------------
# Schema (inline from octomil-contracts/schemas/core/control_sync_result.json)
# ---------------------------------------------------------------------------

CONTROL_SYNC_SCHEMA = {
    "required": ["updated", "configVersion", "assignmentsChanged", "rolloutsChanged", "fetchedAt"],
    "properties": {
        "updated": {"type": "boolean"},
        "configVersion": {"type": "string"},
        "assignmentsChanged": {"type": "boolean"},
        "rolloutsChanged": {"type": "boolean"},
        "fetchedAt": {"type": "string"},
    },
}

# JSON camelCase -> Python snake_case mapping
_FIELD_MAP: dict[str, str] = {
    "updated": "updated",
    "configVersion": "config_version",
    "assignmentsChanged": "assignments_changed",
    "rolloutsChanged": "rollouts_changed",
    "fetchedAt": "fetched_at",
}


# ---------------------------------------------------------------------------
# Fixtures (inline from octomil-contracts/fixtures/control/)
# ---------------------------------------------------------------------------

REFRESH_NO_CHANGE = {
    "updated": False,
    "configVersion": "v42",
    "assignmentsChanged": False,
    "rolloutsChanged": False,
    "fetchedAt": "2026-03-12T16:00:00Z",
}

REFRESH_ASSIGNMENT_CHANGED = {
    "updated": True,
    "configVersion": "v43",
    "assignmentsChanged": True,
    "rolloutsChanged": False,
    "fetchedAt": "2026-03-12T16:05:00Z",
}


def _fixture_to_dataclass(fixture: dict[str, object]) -> ControlSyncResult:
    """Map a JSON fixture (camelCase) to the SDK dataclass (snake_case)."""
    return ControlSyncResult(
        updated=bool(fixture["updated"]),
        config_version=str(fixture["configVersion"]),
        assignments_changed=bool(fixture["assignmentsChanged"]),
        rollouts_changed=bool(fixture["rolloutsChanged"]),
        fetched_at=str(fixture["fetchedAt"]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestControlSyncResultSchema:
    """ControlSyncResult must expose all fields from the contract schema."""

    def test_has_all_required_fields(self) -> None:
        dc_fields = {f.name for f in dataclasses.fields(ControlSyncResult)}
        for json_key, python_key in _FIELD_MAP.items():
            assert python_key in dc_fields, (
                f"ControlSyncResult missing field '{python_key}' (contract key: '{json_key}')"
            )

    def test_field_count(self) -> None:
        """SDK must have at least as many fields as the contract requires."""
        dc_fields = dataclasses.fields(ControlSyncResult)
        assert len(dc_fields) >= len(CONTROL_SYNC_SCHEMA["required"])


class TestControlSyncResultFixtures:
    """ControlSyncResult must correctly represent contract fixtures."""

    @pytest.mark.parametrize(
        "fixture",
        [REFRESH_NO_CHANGE, REFRESH_ASSIGNMENT_CHANGED],
        ids=["no_change", "assignment_changed"],
    )
    def test_fixture_roundtrip(self, fixture: dict[str, object]) -> None:
        result = _fixture_to_dataclass(fixture)
        assert result.updated is bool(fixture["updated"])
        assert result.config_version == fixture["configVersion"]
        assert result.assignments_changed is bool(fixture["assignmentsChanged"])
        assert result.rollouts_changed is bool(fixture["rolloutsChanged"])
        assert result.fetched_at == fixture["fetchedAt"]
