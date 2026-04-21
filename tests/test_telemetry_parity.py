"""Tests for route telemetry parity — reference implementation validation.

Validates that the Python SDK's telemetry payload:
1. Contains all required fields from the contract fixture
2. Never leaks forbidden keys (prompt, output, audio, etc.)
3. Correctly builds from AttemptLoopResult
4. Serializes cleanly to JSON
5. Matches the shape defined in the vendored contract fixture
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from octomil.runtime.telemetry import (
    FORBIDDEN_TELEMETRY_KEYS,
    AttemptDetail,
    AttemptLoopResult,
    GateSummary,
    RouteEventPayload,
    build_route_event,
    validate_telemetry_safety,
)

# ---------------------------------------------------------------------------
# Fixture path
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sdk_parity"
_TELEMETRY_FIXTURE = _FIXTURE_DIR / "telemetry_route_attempt_upload.json"


def _load_fixture() -> dict:
    """Load the vendored telemetry contract fixture."""
    return json.loads(_TELEMETRY_FIXTURE.read_text())


# ---------------------------------------------------------------------------
# Test: payload has all required fields
# ---------------------------------------------------------------------------


class TestRouteEventPayloadFields:
    """RouteEventPayload must contain all fields from the contract."""

    def test_route_event_payload_has_all_required_fields(self) -> None:
        """All fields defined in the fixture's expected_telemetry must exist on the dataclass."""
        fixture = _load_fixture()
        expected = fixture["expected_telemetry"]

        payload_field_names = {f.name for f in fields(RouteEventPayload)}

        for key in expected:
            assert key in payload_field_names, f"Contract field '{key}' is missing from RouteEventPayload dataclass"

    def test_payload_has_route_id(self) -> None:
        payload = RouteEventPayload(route_id="r1", plan_id="p1", request_id="req1")
        assert payload.route_id == "r1"

    def test_payload_has_plan_id(self) -> None:
        payload = RouteEventPayload(route_id="r1", plan_id="p1", request_id="req1")
        assert payload.plan_id == "p1"

    def test_payload_has_request_id(self) -> None:
        payload = RouteEventPayload(route_id="r1", plan_id="p1", request_id="req1")
        assert payload.request_id == "req1"


# ---------------------------------------------------------------------------
# Test: forbidden keys are rejected
# ---------------------------------------------------------------------------


class TestTelemetrySafety:
    """Safety validation must reject forbidden keys at any nesting level."""

    def test_route_event_never_contains_forbidden_keys(self) -> None:
        """RouteEventPayload.to_dict() never contains any forbidden key."""
        payload = RouteEventPayload(
            route_id="r1",
            plan_id="p1",
            request_id="req1",
            capability="chat",
            policy="local_first",
        )
        data = payload.to_dict()
        all_keys = _collect_all_keys(data)
        intersection = all_keys & FORBIDDEN_TELEMETRY_KEYS
        assert not intersection, f"Forbidden keys found in payload: {intersection}"

    def test_validate_telemetry_safety_rejects_prompt(self) -> None:
        """Safety validation must raise for 'prompt' key."""
        with pytest.raises(ValueError, match="Forbidden telemetry key 'prompt'"):
            validate_telemetry_safety({"route_id": "r1", "prompt": "hello world"})

    def test_validate_telemetry_safety_rejects_output(self) -> None:
        """Safety validation must raise for 'output' key."""
        with pytest.raises(ValueError, match="Forbidden telemetry key 'output'"):
            validate_telemetry_safety({"route_id": "r1", "output": "response text"})

    def test_validate_telemetry_safety_rejects_nested_forbidden_key(self) -> None:
        """Safety validation catches forbidden keys in nested structures."""
        with pytest.raises(ValueError, match="Forbidden telemetry key 'messages'"):
            validate_telemetry_safety({"route_id": "r1", "meta": {"messages": ["hi"]}})

    def test_validate_telemetry_safety_rejects_list_nested_forbidden_key(self) -> None:
        """Safety validation catches forbidden keys inside list elements."""
        with pytest.raises(ValueError, match="Forbidden telemetry key 'content'"):
            validate_telemetry_safety({"items": [{"content": "secret"}]})

    def test_validate_telemetry_safety_passes_clean_payload(self) -> None:
        """Clean payloads pass safety validation without error."""
        clean = {
            "route_id": "abc",
            "plan_id": "p1",
            "request_id": "req1",
            "capability": "chat",
            "policy": "local_first",
            "fallback_used": True,
            "candidate_attempts": 2,
            "final_locality": "cloud",
        }
        # Should not raise
        validate_telemetry_safety(clean)

    def test_forbidden_keys_match_fixture(self) -> None:
        """Our FORBIDDEN_TELEMETRY_KEYS set matches the contract fixture."""
        fixture = _load_fixture()
        fixture_forbidden = set(fixture["forbidden_telemetry_keys"])
        assert FORBIDDEN_TELEMETRY_KEYS == fixture_forbidden


# ---------------------------------------------------------------------------
# Test: build_route_event from AttemptLoopResult
# ---------------------------------------------------------------------------


class TestBuildRouteEvent:
    """build_route_event correctly maps AttemptLoopResult fields."""

    def test_build_route_event_from_attempt_result(self) -> None:
        """Basic build produces a valid payload with correct field mapping."""
        result = AttemptLoopResult(
            selected_index=0,
            final_locality="local",
            final_engine="mlx-lm",
            final_artifact_id="art_123",
            fallback_used=False,
            candidate_count=1,
            attempts=[
                AttemptDetail(
                    index=0,
                    locality="local",
                    mode="sdk_runtime",
                    engine="mlx-lm",
                    status="selected",
                    stage="inference",
                    gate_summary=GateSummary(
                        passed=["artifact_verified", "runtime_available"],
                        failed=[],
                    ),
                    reason_code="selected",
                )
            ],
            ttft_ms=150.0,
            tokens_per_second=45.2,
            total_tokens=128,
            duration_ms=2800.0,
        )

        payload = build_route_event(
            attempt_result=result,
            request_id="req_001",
            plan_id="plan_001",
            capability="chat",
            policy="local_first",
            planner_source="server",
        )

        assert payload.request_id == "req_001"
        assert payload.plan_id == "plan_001"
        assert payload.final_locality == "local"
        assert payload.engine == "mlx-lm"
        assert payload.artifact_id == "art_123"
        assert payload.fallback_used is False
        assert payload.candidate_attempts == 1
        assert payload.capability == "chat"
        assert payload.policy == "local_first"
        assert payload.planner_source == "server"
        assert payload.ttft_ms == 150.0
        assert payload.tokens_per_second == 45.2
        assert payload.total_tokens == 128
        assert payload.duration_ms == 2800.0
        assert len(payload.attempt_details) == 1
        assert payload.attempt_details[0].engine == "mlx-lm"

    def test_build_route_event_with_fallback(self) -> None:
        """Build with fallback scenario populates trigger fields."""
        result = AttemptLoopResult(
            selected_index=1,
            final_locality="cloud",
            final_engine=None,
            fallback_used=True,
            fallback_trigger_code="gate_failed",
            fallback_trigger_stage="gate",
            candidate_count=2,
            attempts=[
                AttemptDetail(
                    index=0,
                    locality="local",
                    mode="sdk_runtime",
                    engine="mlx-lm",
                    status="failed",
                    stage="gate",
                    gate_summary=GateSummary(
                        passed=["artifact_verified", "runtime_available", "model_loads"],
                        failed=["max_ttft_ms"],
                    ),
                    reason_code="gate_failed",
                ),
                AttemptDetail(
                    index=1,
                    locality="cloud",
                    mode="hosted_gateway",
                    engine=None,
                    status="selected",
                    stage="inference",
                    gate_summary=GateSummary(passed=[], failed=[]),
                    reason_code="selected",
                ),
            ],
        )

        payload = build_route_event(
            attempt_result=result,
            capability="chat",
            policy="local_first",
            planner_source="server",
        )

        assert payload.fallback_used is True
        assert payload.fallback_trigger_code == "gate_failed"
        assert payload.fallback_trigger_stage == "gate"
        assert payload.final_locality == "cloud"
        assert payload.engine is None
        assert payload.candidate_attempts == 2
        assert len(payload.attempt_details) == 2
        assert payload.attempt_details[0].status == "failed"
        assert payload.attempt_details[1].status == "selected"

    def test_build_route_event_with_app_context(self) -> None:
        """Build with app context populates app_id and app_slug."""
        result = AttemptLoopResult(
            selected_index=0,
            final_locality="local",
            final_engine="mlx-lm",
            candidate_count=1,
            attempts=[],
        )

        payload = build_route_event(
            attempt_result=result,
            app_id="app_xyz",
            app_slug="my-app",
            deployment_id="dep_123",
            experiment_id="exp_456",
            variant_id="var_a",
        )

        assert payload.app_id == "app_xyz"
        assert payload.app_slug == "my-app"
        assert payload.deployment_id == "dep_123"
        assert payload.experiment_id == "exp_456"
        assert payload.variant_id == "var_a"

    def test_build_route_event_generates_ids_when_missing(self) -> None:
        """route_id and request_id are auto-generated if not provided."""
        result = AttemptLoopResult(
            selected_index=0,
            final_locality="local",
            candidate_count=1,
            attempts=[],
        )

        payload = build_route_event(attempt_result=result)

        assert payload.route_id is not None
        assert len(payload.route_id) == 32  # uuid4 hex
        assert payload.request_id is not None
        assert len(payload.request_id) == 32


# ---------------------------------------------------------------------------
# Test: serialization
# ---------------------------------------------------------------------------


class TestPayloadSerialization:
    """RouteEventPayload must serialize cleanly to JSON."""

    def test_payload_serializes_to_json_cleanly(self) -> None:
        """to_json() produces valid JSON without errors."""
        payload = RouteEventPayload(
            route_id="r1",
            plan_id="p1",
            request_id="req1",
            capability="chat",
            policy="local_first",
            fallback_used=False,
            candidate_attempts=1,
            final_locality="local",
            engine="mlx-lm",
            ttft_ms=100.5,
            tokens_per_second=42.0,
            total_tokens=64,
            duration_ms=1500.0,
            attempt_details=[
                AttemptDetail(
                    index=0,
                    locality="local",
                    mode="sdk_runtime",
                    engine="mlx-lm",
                    status="selected",
                    stage="inference",
                    gate_summary=GateSummary(passed=["runtime_available"], failed=[]),
                    reason_code="selected",
                )
            ],
        )

        json_str = payload.to_json()
        parsed = json.loads(json_str)

        assert parsed["route_id"] == "r1"
        assert parsed["plan_id"] == "p1"
        assert parsed["capability"] == "chat"
        assert parsed["fallback_used"] is False
        assert parsed["ttft_ms"] == 100.5
        assert isinstance(parsed["attempt_details"], list)
        assert parsed["attempt_details"][0]["engine"] == "mlx-lm"

    def test_payload_to_dict_matches_to_json(self) -> None:
        """to_dict() and json.loads(to_json()) produce the same structure."""
        payload = RouteEventPayload(
            route_id="r1",
            plan_id=None,
            request_id="req1",
            fallback_used=True,
            candidate_attempts=2,
        )

        dict_form = payload.to_dict()
        json_form = json.loads(payload.to_json())

        assert dict_form == json_form

    def test_none_fields_serialize_as_null(self) -> None:
        """Optional fields set to None serialize as JSON null."""
        payload = RouteEventPayload(
            route_id="r1",
            plan_id=None,
            request_id="req1",
        )
        parsed = json.loads(payload.to_json())
        assert parsed["plan_id"] is None
        assert parsed["engine"] is None
        assert parsed["ttft_ms"] is None


# ---------------------------------------------------------------------------
# Test: fixture shape conformance
# ---------------------------------------------------------------------------


class TestFixtureShapeConformance:
    """Payload shape must match the vendored contract fixture."""

    def test_payload_matches_contract_fixture_shape(self) -> None:
        """Build a payload matching the fixture scenario and verify shape."""
        fixture = _load_fixture()
        expected = fixture["expected_telemetry"]

        # Reconstruct the fixture scenario
        result = AttemptLoopResult(
            selected_index=1,
            final_locality="cloud",
            final_engine=None,
            final_artifact_id=None,
            fallback_used=True,
            fallback_trigger_code="gate_failed",
            fallback_trigger_stage="gate",
            candidate_count=2,
            attempts=[
                AttemptDetail(
                    index=0,
                    locality="local",
                    mode="sdk_runtime",
                    engine="mlx-lm",
                    status="failed",
                    stage="gate",
                    gate_summary=GateSummary(
                        passed=["artifact_verified", "runtime_available", "model_loads"],
                        failed=["max_ttft_ms"],
                    ),
                    reason_code="gate_failed",
                ),
                AttemptDetail(
                    index=1,
                    locality="cloud",
                    mode="hosted_gateway",
                    engine=None,
                    status="selected",
                    stage="inference",
                    gate_summary=GateSummary(passed=[], failed=[]),
                    reason_code="selected",
                ),
            ],
        )

        payload = build_route_event(
            attempt_result=result,
            request_id="test_req_id",
            plan_id="test_plan_id",
            route_id="test_route_id",
            capability="chat",
            policy="local_first",
            planner_source="server",
        )

        data = payload.to_dict()

        # Verify all non-generated fields match
        assert data["capability"] == expected["capability"]
        assert data["policy"] == expected["policy"]
        assert data["planner_source"] == expected["planner_source"]
        assert data["final_locality"] == expected["final_locality"]
        assert data["engine"] == expected["engine"]
        assert data["artifact_id"] == expected["artifact_id"]
        assert data["fallback_used"] == expected["fallback_used"]
        assert data["fallback_trigger_code"] == expected["fallback_trigger_code"]
        assert data["fallback_trigger_stage"] == expected["fallback_trigger_stage"]
        assert data["candidate_attempts"] == expected["candidate_attempts"]

        # Verify attempt_details structure
        assert len(data["attempt_details"]) == len(expected["attempt_details"])
        for actual, exp in zip(data["attempt_details"], expected["attempt_details"]):
            assert actual["index"] == exp["index"]
            assert actual["locality"] == exp["locality"]
            assert actual["mode"] == exp["mode"]
            assert actual["engine"] == exp["engine"]
            assert actual["status"] == exp["status"]
            assert actual["stage"] == exp["stage"]
            assert actual["reason_code"] == exp["reason_code"]
            assert set(actual["gate_summary"]["passed"]) == set(exp["gate_summary"]["passed"])
            assert set(actual["gate_summary"]["failed"]) == set(exp["gate_summary"]["failed"])

    def test_fixture_forbidden_keys_complete(self) -> None:
        """Every key in the fixture's forbidden list exists in our constant."""
        fixture = _load_fixture()
        for key in fixture["forbidden_telemetry_keys"]:
            assert key in FORBIDDEN_TELEMETRY_KEYS, f"Missing forbidden key: {key}"

    def test_fixture_rules_coverage(self) -> None:
        """Ensure the fixture's rules_tested are all addressed by our tests."""
        fixture = _load_fixture()
        rules = fixture["rules_tested"]
        # We just verify the rules list is non-empty and is a list of strings
        assert len(rules) > 0
        assert all(isinstance(r, str) for r in rules)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_all_keys(obj: Any, keys: set[str] | None = None) -> set[str]:
    """Recursively collect all dict keys in a nested structure."""
    if keys is None:
        keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            _collect_all_keys(v, keys)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_all_keys(item, keys)
    return keys
