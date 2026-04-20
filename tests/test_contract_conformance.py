"""Contract conformance tests for the Python SDK.

Loads vendored contract fixtures from octomil-contracts and validates that
the SDK can decode planner responses, route metadata, and telemetry correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sdk_parity"

# Load all fixtures
FIXTURE_FILES = sorted(FIXTURES_DIR.glob("*.json"))


@pytest.fixture(params=FIXTURE_FILES, ids=[f.stem for f in FIXTURE_FILES])
def fixture_data(request: pytest.FixtureRequest) -> dict[str, Any]:
    return json.loads(request.param.read_text())  # type: ignore[no-any-return]


def _require_section(fixture_data: dict[str, Any], key: str) -> dict[str, Any]:
    """Get a required fixture section or skip the test."""
    section = fixture_data.get(key)
    if section is None:
        pytest.skip(f"no {key}")
    assert isinstance(section, dict)
    return section


class TestPlannerResponseDecoding:
    """SDK can decode all planner response fixtures."""

    def test_can_parse_candidates(self, fixture_data: dict[str, Any]) -> None:
        resp = fixture_data["planner_response"]
        candidates = resp["candidates"]
        assert len(candidates) > 0
        for c in candidates:
            assert "locality" in c
            assert c["locality"] in ("local", "cloud")
            assert "priority" in c
            assert isinstance(c["priority"], int)
            # If gates present, verify structure
            for gate in c.get("gates", []):
                assert "code" in gate
                assert "required" in gate
                assert isinstance(gate["required"], bool)
                assert "source" in gate

    def test_fallback_allowed_field_present(self, fixture_data: dict[str, Any]) -> None:
        resp = fixture_data["planner_response"]
        assert "fallback_allowed" in resp
        assert isinstance(resp["fallback_allowed"], bool)

    def test_model_field_present(self, fixture_data: dict[str, Any]) -> None:
        resp = fixture_data["planner_response"]
        assert "model" in resp
        assert isinstance(resp["model"], str)

    def test_candidate_engine_nullable(self, fixture_data: dict[str, Any]) -> None:
        """Engine field can be null (for cloud candidates)."""
        resp = fixture_data["planner_response"]
        for c in resp["candidates"]:
            # engine must be present as key (even if null)
            assert "engine" in c
            if c["locality"] == "local":
                assert c["engine"] is not None

    def test_candidate_artifact_when_local(self, fixture_data: dict[str, Any]) -> None:
        """Local candidates should have artifact info."""
        resp = fixture_data["planner_response"]
        for c in resp["candidates"]:
            if c["locality"] == "local":
                assert "artifact" in c
                art = c["artifact"]
                assert "artifact_id" in art
                assert "digest" in art


class TestRouteMetadataDecoding:
    """SDK can decode all route metadata fixtures."""

    def test_can_parse_status_and_execution(self, fixture_data: dict[str, Any]) -> None:
        meta = _require_section(fixture_data, "expected_route_metadata")
        assert meta["status"] in ("selected", "unavailable", "failed")
        if meta["status"] == "selected":
            assert meta["execution"] is not None
            exec_info = meta["execution"]
            assert "locality" in exec_info
            assert "mode" in exec_info
        elif meta["status"] in ("unavailable", "failed"):
            assert meta["execution"] is None

    def test_can_parse_attempts(self, fixture_data: dict[str, Any]) -> None:
        meta = _require_section(fixture_data, "expected_route_metadata")
        attempts = meta.get("attempts", [])
        assert len(attempts) > 0
        for attempt in attempts:
            assert "index" in attempt
            assert isinstance(attempt["index"], int)
            assert "locality" in attempt
            assert attempt["locality"] in ("local", "cloud")
            assert "status" in attempt
            assert attempt["status"] in ("selected", "failed", "skipped")
            assert "stage" in attempt
            assert attempt["stage"] in (
                "policy",
                "prepare",
                "download",
                "verify",
                "load",
                "benchmark",
                "gate",
                "inference",
            )
            assert "reason" in attempt

    def test_can_parse_fallback(self, fixture_data: dict[str, Any]) -> None:
        meta = _require_section(fixture_data, "expected_route_metadata")
        fb = meta["fallback"]
        assert "used" in fb
        assert isinstance(fb["used"], bool)
        if fb["used"]:
            assert fb["trigger"] is not None
            trigger = fb["trigger"]
            assert "code" in trigger
            assert "stage" in trigger
            assert "message" in trigger
        else:
            assert fb["trigger"] is None

    def test_execution_mode_matches_locality(self, fixture_data: dict[str, Any]) -> None:
        """Execution mode should be sdk_runtime for local, hosted_gateway for cloud."""
        meta = _require_section(fixture_data, "expected_route_metadata")
        if meta["status"] != "selected":
            pytest.skip("no execution to check")
        exec_info = meta["execution"]
        assert isinstance(exec_info, dict)
        if exec_info["locality"] == "local":
            assert exec_info["mode"] == "sdk_runtime"
        elif exec_info["locality"] == "cloud":
            assert exec_info["mode"] == "hosted_gateway"


class TestTelemetrySafety:
    """SDK must not emit unsafe telemetry fields."""

    FORBIDDEN = frozenset(
        {
            "prompt",
            "input",
            "output",
            "audio",
            "file_path",
            "content",
            "messages",
            "system_prompt",
        }
    )

    def test_no_forbidden_keys_in_telemetry(self, fixture_data: dict[str, Any]) -> None:
        telem = fixture_data.get("expected_telemetry", {})
        all_keys: set[str] = set()
        _collect_keys(telem, all_keys)
        violations = all_keys & self.FORBIDDEN
        assert not violations, f"Forbidden telemetry keys found: {violations}"

    def test_no_forbidden_keys_in_route_metadata(self, fixture_data: dict[str, Any]) -> None:
        meta = _require_section(fixture_data, "expected_route_metadata")
        all_keys: set[str] = set()
        _collect_keys(meta, all_keys)
        violations = all_keys & self.FORBIDDEN
        assert not violations, f"Forbidden route metadata keys found: {violations}"

    def test_telemetry_has_required_fields(self, fixture_data: dict[str, Any]) -> None:
        """Telemetry must contain at minimum: route_id, policy, model_id."""
        telem = fixture_data.get("expected_telemetry", {})
        assert "route_id" in telem
        assert "policy" in telem
        assert "model_id" in telem


class TestPolicyResult:
    """SDK policy resolution produces correct allow/deny decisions."""

    def test_policy_result_structure(self, fixture_data: dict[str, Any]) -> None:
        policy = _require_section(fixture_data, "expected_policy_result")
        assert "cloud_allowed" in policy
        assert "fallback_allowed" in policy
        assert "private" in policy
        assert isinstance(policy["cloud_allowed"], bool)
        assert isinstance(policy["fallback_allowed"], bool)
        assert isinstance(policy["private"], bool)

    def test_local_only_implies_no_cloud(self, fixture_data: dict[str, Any]) -> None:
        """If routing_policy is local_only, cloud must not be allowed."""
        request = fixture_data["request"]
        policy = _require_section(fixture_data, "expected_policy_result")
        if request["routing_policy"] == "local_only":
            assert policy["cloud_allowed"] is False
            assert policy["private"] is True

    def test_cloud_only_implies_cloud_allowed(self, fixture_data: dict[str, Any]) -> None:
        """If routing_policy is cloud_only, cloud must be allowed."""
        request = fixture_data["request"]
        policy = _require_section(fixture_data, "expected_policy_result")
        if request["routing_policy"] == "cloud_only":
            assert policy["cloud_allowed"] is True


class TestPlatformLimitations:
    """Python SDK platform rules."""

    def test_python_supports_local_runtime(self) -> None:
        """Python SDK supports local runtime via CandidateAttemptRunner."""
        from octomil.runtime.routing.attempt_runner import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=True)
        assert runner is not None

    def test_python_supports_all_gate_codes(self) -> None:
        """GateStatus enum has required values."""
        from octomil.runtime.routing.attempt_runner import GateStatus

        assert GateStatus.PASSED.value == "passed"
        assert GateStatus.FAILED.value == "failed"
        assert GateStatus.UNKNOWN.value == "unknown"
        assert GateStatus.NOT_REQUIRED.value == "not_required"

    def test_python_supports_all_attempt_stages(self) -> None:
        """AttemptStage enum covers all contract stages."""
        from octomil.runtime.routing.attempt_runner import AttemptStage

        expected_stages = {
            "policy",
            "prepare",
            "download",
            "verify",
            "load",
            "benchmark",
            "gate",
            "inference",
        }
        actual_stages = {s.value for s in AttemptStage}
        assert expected_stages == actual_stages

    def test_python_supports_all_attempt_statuses(self) -> None:
        """AttemptStatus enum covers all contract statuses."""
        from octomil.runtime.routing.attempt_runner import AttemptStatus

        expected_statuses = {"skipped", "failed", "selected"}
        actual_statuses = {s.value for s in AttemptStatus}
        assert expected_statuses == actual_statuses

    def test_python_streaming_fallback_semantics(self) -> None:
        """Streaming runner blocks fallback after first token."""
        from octomil.runtime.routing.attempt_runner import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=True, streaming=True)
        # Before first token: fallback allowed
        assert runner.should_fallback_after_inference_error(first_token_emitted=False) is True
        # After first token: fallback blocked
        assert runner.should_fallback_after_inference_error(first_token_emitted=True) is False

    def test_python_non_streaming_always_allows_fallback(self) -> None:
        """Non-streaming runner always allows fallback when configured."""
        from octomil.runtime.routing.attempt_runner import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=True, streaming=False)
        assert runner.should_fallback_after_inference_error(first_token_emitted=False) is True
        assert runner.should_fallback_after_inference_error(first_token_emitted=True) is True

    def test_python_fallback_disabled_blocks_all(self) -> None:
        """When fallback_allowed=False, no fallback regardless of state."""
        from octomil.runtime.routing.attempt_runner import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=False, streaming=True)
        assert runner.should_fallback_after_inference_error(first_token_emitted=False) is False


class TestFixtureIntegrity:
    """Validate that the fixture set itself is complete and well-formed."""

    def test_fixture_count(self) -> None:
        """All 10 expected fixtures are present."""
        assert len(FIXTURE_FILES) == 10

    def test_all_fixtures_valid_json(self) -> None:
        """Every fixture file is valid JSON."""
        for f in FIXTURE_FILES:
            data = json.loads(f.read_text())
            assert isinstance(data, dict)

    def test_all_fixtures_have_required_sections(self) -> None:
        """Every fixture has the required top-level keys."""
        required_keys = {
            "description",
            "request",
            "planner_response",
            "expected_route_metadata",
            "expected_telemetry",
            "expected_policy_result",
            "rules_tested",
        }
        for f in FIXTURE_FILES:
            data = json.loads(f.read_text())
            missing = required_keys - set(data.keys())
            assert not missing, f"{f.name} missing keys: {missing}"


def _collect_keys(obj: object, keys: set[str]) -> None:
    """Recursively collect all dictionary keys from a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            _collect_keys(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            _collect_keys(item, keys)
