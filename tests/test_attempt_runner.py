"""Tests for CandidateAttemptRunner — per-request candidate attempt loop."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from octomil.runtime.routing.attempt_runner import (
    ArtifactChecker,
    AttemptStage,
    AttemptStatus,
    CandidateAttemptRunner,
    GateEvaluator,
    GateResult,
    GateStatus,
    RuntimeChecker,
)

# ---------------------------------------------------------------------------
# Test helpers — checker stubs
# ---------------------------------------------------------------------------


class _AlwaysAvailableRuntime(RuntimeChecker):
    def check(self, *, engine: str | None, locality: str) -> tuple[bool, str | None]:
        return True, None


class _NeverAvailableRuntime(RuntimeChecker):
    """Fails for a specific engine, passes for all others."""

    def __init__(self, *, failing_engine: str | None = None) -> None:
        self._failing_engine = failing_engine

    def check(self, *, engine: str | None, locality: str) -> tuple[bool, str | None]:
        if self._failing_engine is None or engine == self._failing_engine:
            return False, "not_installed"
        return True, None


class _AlwaysOkArtifact(ArtifactChecker):
    def check(self, artifact_plan: dict[str, Any]) -> tuple[bool, str, str | None]:
        return True, "hit", None


class _FailingArtifact(ArtifactChecker):
    def check(self, artifact_plan: dict[str, Any]) -> tuple[bool, str, str | None]:
        return False, "unavailable", "digest_mismatch"


class _PassAllGates(GateEvaluator):
    def evaluate(self, gate: dict[str, Any], *, engine: str | None, locality: str) -> GateResult:
        return GateResult(code=gate.get("code", "unknown"), status=GateStatus.PASSED)


class _FailGate(GateEvaluator):
    """Fails gates whose code is in the failing_codes set."""

    def __init__(self, failing_codes: set[str]) -> None:
        self._failing_codes = failing_codes

    def evaluate(self, gate: dict[str, Any], *, engine: str | None, locality: str) -> GateResult:
        code = gate.get("code", "unknown")
        if code in self._failing_codes:
            return GateResult(
                code=code,
                status=GateStatus.FAILED,
                observed_number=gate.get("threshold_number", 0) - 1 if gate.get("threshold_number") else None,
                threshold_number=gate.get("threshold_number"),
                reason_code="below_threshold",
            )
        return GateResult(code=code, status=GateStatus.PASSED)


# ---------------------------------------------------------------------------
# Test candidates fixtures
# ---------------------------------------------------------------------------


def _local_candidate(engine: str = "mlx-lm", gates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {
        "locality": "local",
        "engine": engine,
        "artifact": {"artifact_id": "art-001", "digest": "sha256:abc123"},
    }
    if gates is not None:
        c["gates"] = gates
    return c


def _cloud_candidate(gates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {"locality": "cloud"}
    if gates is not None:
        c["gates"] = gates
    return c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleLocalCandidateAllGatesPass:
    def test_selects_first_candidate(self) -> None:
        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            [_local_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert result.selected_attempt is not None
        assert result.selected_attempt.index == 0
        assert result.selected_attempt.locality == "local"
        assert result.selected_attempt.mode == "sdk_runtime"
        assert result.selected_attempt.status == AttemptStatus.SELECTED
        assert result.selected_attempt.stage == AttemptStage.INFERENCE
        assert len(result.attempts) == 1
        assert not result.fallback_used
        assert result.fallback_trigger is None


class TestLocalFailsRuntimeFallsBackToCloud:
    def test_fallback_to_cloud(self) -> None:
        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            [_local_candidate(engine="mlx-lm"), _cloud_candidate()],
            runtime_checker=_NeverAvailableRuntime(failing_engine="mlx-lm"),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert len(result.attempts) == 2

        # First attempt failed at prepare stage
        first = result.attempts[0]
        assert first.index == 0
        assert first.status == AttemptStatus.FAILED
        assert first.stage == AttemptStage.PREPARE
        assert first.locality == "local"

        # Second attempt selected (cloud)
        second = result.attempts[1]
        assert second.index == 1
        assert second.status == AttemptStatus.SELECTED
        assert second.locality == "cloud"
        assert second.mode == "hosted_gateway"

        assert result.fallback_used
        assert result.fallback_trigger is not None
        assert result.fallback_trigger.code == "runtime_unavailable"
        assert result.from_attempt == 0
        assert result.to_attempt == 1


class TestGateFailureTriggersAndFallback:
    def test_gate_failure_falls_back(self) -> None:
        gates = [
            {"code": "min_tokens_per_second", "required": True, "threshold_number": 30, "source": "server"},
        ]
        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            [_local_candidate(gates=gates), _cloud_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_FailGate({"min_tokens_per_second"}),
        )

        assert result.succeeded
        assert len(result.attempts) == 2

        # First attempt failed at gate stage
        first = result.attempts[0]
        assert first.status == AttemptStatus.FAILED
        assert first.stage == AttemptStage.GATE
        assert first.reason_code == "gate_failed"

        # Verify gate_results contain the failed gate
        gate_codes = [g.code for g in first.gate_results]
        assert "runtime_available" in gate_codes
        assert "artifact_verified" in gate_codes
        assert "min_tokens_per_second" in gate_codes
        failed_gate = next(g for g in first.gate_results if g.code == "min_tokens_per_second")
        assert failed_gate.status == GateStatus.FAILED
        assert failed_gate.threshold_number == 30

        # Second attempt selected
        assert result.attempts[1].status == AttemptStatus.SELECTED
        assert result.fallback_used
        assert result.fallback_trigger is not None
        assert result.fallback_trigger.stage == "gate"


class TestPrivateNoFallbackFails:
    def test_no_fallback_when_disabled(self) -> None:
        runner = CandidateAttemptRunner(fallback_allowed=False)
        result = runner.run(
            [_local_candidate(engine="mlx-lm"), _cloud_candidate()],
            runtime_checker=_NeverAvailableRuntime(failing_engine="mlx-lm"),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert not result.succeeded
        assert result.selected_attempt is None
        assert len(result.attempts) == 1
        assert result.attempts[0].status == AttemptStatus.FAILED
        assert not result.fallback_used
        assert result.fallback_trigger is None


class TestArtifactVerificationFailure:
    def test_artifact_fail_triggers_fallback(self) -> None:
        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            [_local_candidate(), _cloud_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_FailingArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert len(result.attempts) == 2

        first = result.attempts[0]
        assert first.status == AttemptStatus.FAILED
        assert first.stage == AttemptStage.VERIFY
        assert first.reason_code == "artifact_verification_failed"
        assert first.artifact is not None
        assert first.artifact.cache_status == "unavailable"

        second = result.attempts[1]
        assert second.status == AttemptStatus.SELECTED
        assert second.locality == "cloud"

        assert result.fallback_used
        assert result.fallback_trigger is not None
        assert result.fallback_trigger.code == "artifact_verification_failed"
        assert result.fallback_trigger.stage == "verify"

    def test_artifact_fail_no_fallback(self) -> None:
        runner = CandidateAttemptRunner(fallback_allowed=False)
        result = runner.run(
            [_local_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_FailingArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert not result.succeeded
        assert len(result.attempts) == 1
        assert result.attempts[0].stage == AttemptStage.VERIFY


class TestStreamingFlagPreserved:
    def test_streaming_flag_stored(self) -> None:
        runner = CandidateAttemptRunner(streaming=True)
        assert runner.streaming is True

    def test_streaming_default_false(self) -> None:
        runner = CandidateAttemptRunner()
        assert runner.streaming is False

    def test_streaming_disallows_fallback_after_first_token(self) -> None:
        runner = CandidateAttemptRunner(fallback_allowed=True, streaming=True)
        assert runner.should_fallback_after_inference_error(first_token_emitted=False)
        assert not runner.should_fallback_after_inference_error(first_token_emitted=True)

    def test_non_streaming_allows_inference_fallback(self) -> None:
        runner = CandidateAttemptRunner(fallback_allowed=True, streaming=False)
        assert runner.should_fallback_after_inference_error(first_token_emitted=True)


class TestInferenceExecutionLoop:
    @pytest.mark.asyncio
    async def test_inference_error_before_output_falls_back(self) -> None:
        runner = CandidateAttemptRunner(fallback_allowed=True)

        async def execute(candidate: dict[str, Any]) -> str:
            if candidate.get("locality") == "local":
                raise RuntimeError("local model load failed")
            return "cloud response"

        result = await runner.run_with_inference(
            [_local_candidate(), _cloud_candidate()],
            execute_candidate=execute,
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert result.value == "cloud response"
        assert result.fallback_used
        assert result.fallback_trigger is not None
        assert result.fallback_trigger.code == "inference_error"
        assert result.attempts[0].status == AttemptStatus.FAILED
        assert result.attempts[0].stage == AttemptStage.INFERENCE
        assert result.attempts[1].status == AttemptStatus.SELECTED

    @pytest.mark.asyncio
    async def test_streaming_error_after_first_token_does_not_fallback(self) -> None:
        class StreamCrashed(RuntimeError):
            first_token_emitted = True

        runner = CandidateAttemptRunner(fallback_allowed=True, streaming=True)

        async def execute(candidate: dict[str, Any]) -> str:
            if candidate.get("locality") == "local":
                raise StreamCrashed("crashed after first token")
            return "cloud response"

        result = await runner.run_with_inference(
            [_local_candidate(), _cloud_candidate()],
            execute_candidate=execute,
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert not result.succeeded
        assert result.error is not None
        assert len(result.attempts) == 1
        assert result.attempts[0].reason_code == "inference_error_after_first_token"
        assert not result.fallback_used


class TestAttemptIndicesSequential:
    def test_indices_are_sequential(self) -> None:
        """Attempt indices match the candidate list position, even across failures."""

        class _FailFirstTwoEngines(RuntimeChecker):
            def check(self, *, engine: str | None, locality: str) -> tuple[bool, str | None]:
                if engine in ("engine-a", "engine-b"):
                    return False, "not_available"
                return True, None

        candidates = [
            _local_candidate(engine="engine-a"),
            _local_candidate(engine="engine-b"),
            _cloud_candidate(),
        ]
        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            candidates,
            runtime_checker=_FailFirstTwoEngines(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert len(result.attempts) == 3
        for i, attempt in enumerate(result.attempts):
            assert attempt.index == i, f"Attempt {i} has index {attempt.index}"


class TestToDictMatchesContractShape:
    """Validate that to_dict output structurally matches the route_attempt schema."""

    def test_route_attempt_to_dict_keys(self) -> None:
        runner = CandidateAttemptRunner()
        result = runner.run(
            [_local_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
        )

        attempt_dict = result.attempts[0].to_dict()

        # Required keys per route_attempt.schema.json
        assert "index" in attempt_dict
        assert "locality" in attempt_dict
        assert "mode" in attempt_dict
        assert "status" in attempt_dict
        assert "stage" in attempt_dict
        assert "reason" in attempt_dict
        assert "code" in attempt_dict["reason"]
        assert "message" in attempt_dict["reason"]

        # Optional keys
        assert "engine" in attempt_dict
        assert "artifact" in attempt_dict
        assert "gate_results" in attempt_dict

        # Enum values match contract
        assert attempt_dict["status"] in ("skipped", "failed", "selected")
        assert attempt_dict["stage"] in (
            "policy",
            "prepare",
            "download",
            "verify",
            "load",
            "benchmark",
            "gate",
            "inference",
        )
        assert attempt_dict["locality"] in ("local", "cloud")
        assert attempt_dict["mode"] in ("sdk_runtime", "hosted_gateway", "external_endpoint")

    def test_gate_result_to_dict_keys(self) -> None:
        runner = CandidateAttemptRunner()
        gates = [{"code": "min_tokens_per_second", "required": True, "threshold_number": 10, "source": "server"}]
        result = runner.run(
            [_local_candidate(gates=gates)],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        gate_dicts = result.attempts[0].to_dict()["gate_results"]
        for gd in gate_dicts:
            assert "code" in gd
            assert "status" in gd
            assert gd["status"] in ("passed", "failed", "unknown", "not_required")

    def test_route_metadata_fields_shape(self) -> None:
        runner = CandidateAttemptRunner()
        result = runner.run(
            [_local_candidate(engine="mlx-lm"), _cloud_candidate()],
            runtime_checker=_NeverAvailableRuntime(failing_engine="mlx-lm"),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        meta = result.to_route_metadata_fields()
        assert "attempts" in meta
        assert "fallback" in meta
        assert isinstance(meta["attempts"], list)
        assert "used" in meta["fallback"]
        assert "from_attempt" in meta["fallback"]
        assert "to_attempt" in meta["fallback"]
        assert "trigger" in meta["fallback"]

    def test_validate_against_schema_if_jsonschema_available(self) -> None:
        """If jsonschema is available, validate to_dict output against the contract schema."""
        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")

        schema_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "octomil-contracts",
            "schemas",
            "runtime_planner",
            "route_attempt.schema.json",
        )
        if not os.path.exists(schema_path):
            # Try the workspace-level contracts path
            schema_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "octomil-contracts",
                "schemas",
                "runtime_planner",
                "route_attempt.schema.json",
            )
        if not os.path.exists(schema_path):
            pytest.skip("route_attempt.schema.json not found")

        with open(schema_path) as f:
            schema = json.load(f)

        runner = CandidateAttemptRunner()
        result = runner.run(
            [_local_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        attempt_dict = result.attempts[0].to_dict()
        jsonschema.validate(instance=attempt_dict, schema=schema)


class TestFallbackNotUsedWhenNoTrigger:
    """Edge case: all candidates pass, no fallback trigger should be emitted."""

    def test_no_fallback_fields_when_first_wins(self) -> None:
        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            [_local_candidate(), _cloud_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert not result.fallback_used
        assert result.fallback_trigger is None
        assert result.from_attempt is None
        assert result.to_attempt is None


class TestEmptyCandidateList:
    def test_empty_candidates(self) -> None:
        runner = CandidateAttemptRunner()
        result = runner.run([])
        assert not result.succeeded
        assert result.selected_attempt is None
        assert len(result.attempts) == 0
        assert not result.fallback_used


class TestModeDerivation:
    def test_local_candidate_mode(self) -> None:
        assert CandidateAttemptRunner._mode_for_candidate({"locality": "local"}) == "sdk_runtime"

    def test_cloud_candidate_mode(self) -> None:
        assert CandidateAttemptRunner._mode_for_candidate({"locality": "cloud"}) == "hosted_gateway"

    def test_default_locality_is_local(self) -> None:
        assert CandidateAttemptRunner._mode_for_candidate({}) == "sdk_runtime"
