"""Tests for output_quality gate phases in the candidate attempt runner."""

from __future__ import annotations

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
    OutputQualityGateEvaluator,
    RuntimeChecker,
    classify_gate,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _AlwaysAvailableRuntime(RuntimeChecker):
    def check(self, *, engine: str | None, locality: str) -> tuple[bool, str | None]:
        return True, None


class _AlwaysOkArtifact(ArtifactChecker):
    def check(self, artifact_plan: dict[str, Any]) -> tuple[bool, str, str | None]:
        return True, "hit", None


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


class _FailOutputQualityEvaluator(OutputQualityGateEvaluator):
    """Fails output quality gates whose code is in the failing_codes set."""

    def __init__(self, failing_codes: set[str]) -> None:
        self._failing_codes = failing_codes

    def evaluate(self, gate: dict[str, Any], response: Any) -> GateResult:
        code = gate.get("code", "unknown")
        if code in self._failing_codes:
            return GateResult(
                code=code,
                status=GateStatus.FAILED,
                reason_code="quality_check_failed",
            )
        return GateResult(code=code, status=GateStatus.PASSED)


class _PassOutputQualityEvaluator(OutputQualityGateEvaluator):
    def evaluate(self, gate: dict[str, Any], response: Any) -> GateResult:
        return GateResult(code=gate.get("code", "unknown"), status=GateStatus.PASSED)


# ---------------------------------------------------------------------------
# Candidate fixtures
# ---------------------------------------------------------------------------


def _local_candidate(
    engine: str = "mlx-lm",
    gates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
# Tests: readiness hard failure
# ---------------------------------------------------------------------------


class TestReadinessHardFailureFallsBack:
    def test_runtime_unavailable_falls_back(self) -> None:
        class _FailLocal(RuntimeChecker):
            def check(self, *, engine: str | None, locality: str) -> tuple[bool, str | None]:
                if locality == "local":
                    return False, "not_installed"
                return True, None

        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            [_local_candidate(), _cloud_candidate()],
            runtime_checker=_FailLocal(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert result.fallback_used
        assert result.attempts[0].status == AttemptStatus.FAILED
        assert result.attempts[0].stage == AttemptStage.PREPARE
        assert result.attempts[1].status == AttemptStatus.SELECTED


# ---------------------------------------------------------------------------
# Tests: performance gates
# ---------------------------------------------------------------------------


class TestPerformanceHardFailureFallsBack:
    def test_performance_required_gate_failure_falls_back(self) -> None:
        gates = [
            {"code": "min_tokens_per_second", "required": True, "threshold_number": 30},
        ]
        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            [_local_candidate(gates=gates), _cloud_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_FailGate({"min_tokens_per_second"}),
        )

        assert result.succeeded
        assert result.fallback_used
        assert result.attempts[0].status == AttemptStatus.FAILED
        assert result.attempts[0].stage == AttemptStage.GATE
        assert result.attempts[1].status == AttemptStatus.SELECTED
        assert result.attempts[1].locality == "cloud"


class TestPerformanceAdvisoryDoesNotFallback:
    def test_performance_advisory_gate_failure_selects_local(self) -> None:
        gates = [
            {"code": "min_tokens_per_second", "required": False, "threshold_number": 30},
        ]
        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            [_local_candidate(gates=gates), _cloud_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_FailGate({"min_tokens_per_second"}),
        )

        assert result.succeeded
        assert not result.fallback_used
        # Local still selected since gate was advisory
        assert result.selected_attempt is not None
        assert result.selected_attempt.locality == "local"


# ---------------------------------------------------------------------------
# Tests: output quality gates
# ---------------------------------------------------------------------------


class TestOutputQualityFailureBeforeReturnFallsBack:
    @pytest.mark.asyncio
    async def test_oq_gate_failure_triggers_fallback(self) -> None:
        gates = [
            {"code": "schema_valid", "required": True},
        ]

        runner = CandidateAttemptRunner(
            fallback_allowed=True,
            output_quality_evaluator=_FailOutputQualityEvaluator({"schema_valid"}),
        )

        async def execute(candidate: dict[str, Any]) -> str:
            if candidate.get("locality") == "local":
                return "local response"
            return "cloud response"

        result = await runner.run_with_inference(
            [_local_candidate(gates=gates), _cloud_candidate()],
            execute_candidate=execute,
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert result.value == "cloud response"
        assert result.fallback_used
        assert result.fallback_trigger is not None
        assert result.fallback_trigger.stage == "output_quality"
        assert result.fallback_trigger.gate_class == "output_quality"
        assert result.fallback_trigger.evaluation_phase == "post_inference"
        assert result.attempts[0].status == AttemptStatus.FAILED
        assert result.attempts[0].stage == AttemptStage.OUTPUT_QUALITY


class TestOutputQualityFailureAfterFirstTokenNoFallback:
    @pytest.mark.asyncio
    async def test_oq_gate_failure_after_stream_does_not_fallback(self) -> None:
        """When output_quality gate fails but first token was emitted, record failure, do NOT fallback."""
        gates = [
            {"code": "schema_valid", "required": True},
        ]
        runner = CandidateAttemptRunner(
            fallback_allowed=True,
            streaming=True,
            output_quality_evaluator=_FailOutputQualityEvaluator({"schema_valid"}),
        )
        # Simulate post-inference quality check with first_token_emitted=True
        # by directly calling _evaluate_output_quality_gates
        from octomil.runtime.routing.attempt_runner import RouteAttempt

        ready_attempt = RouteAttempt(
            index=0,
            locality="local",
            mode="sdk_runtime",
            engine="mlx-lm",
            status=AttemptStatus.SELECTED,
            stage=AttemptStage.INFERENCE,
            gate_results=[],
            reason_code="selected",
            reason_message="all gates passed",
        )

        quality_failure = runner._evaluate_output_quality_gates(
            _local_candidate(gates=gates),
            "streamed response text",
            ready_attempt,
            0,
            first_token_emitted=True,
        )
        # Should return None because first token was emitted → no fallback
        assert quality_failure is None
        # But the gate result should be recorded on the attempt
        assert len(ready_attempt.gate_results) == 1
        assert ready_attempt.gate_results[0].code == "schema_valid"
        assert ready_attempt.gate_results[0].status == GateStatus.FAILED


# ---------------------------------------------------------------------------
# Tests: private / local_only never cloud-fallback
# ---------------------------------------------------------------------------


class TestPrivateNoFallbackDespiteQualityFail:
    @pytest.mark.asyncio
    async def test_private_mode_no_cloud_fallback_on_quality_failure(self) -> None:
        gates = [
            {"code": "schema_valid", "required": True},
        ]
        runner = CandidateAttemptRunner(
            fallback_allowed=False,
            output_quality_evaluator=_FailOutputQualityEvaluator({"schema_valid"}),
        )

        async def execute(candidate: dict[str, Any]) -> str:
            return "local response"

        result = await runner.run_with_inference(
            [_local_candidate(gates=gates), _cloud_candidate()],
            execute_candidate=execute,
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert not result.succeeded
        assert len(result.attempts) == 1
        assert result.attempts[0].status == AttemptStatus.FAILED
        assert result.attempts[0].stage == AttemptStage.OUTPUT_QUALITY
        assert not result.fallback_used


# ---------------------------------------------------------------------------
# Tests: unknown gate handling
# ---------------------------------------------------------------------------


class TestUnknownRequiredGateFailsClosed:
    @pytest.mark.asyncio
    async def test_unknown_required_gate_fails_closed(self) -> None:
        """Required gate with no evaluator → fail closed."""
        gates = [
            {"code": "schema_valid", "required": True},
        ]
        runner = CandidateAttemptRunner(
            fallback_allowed=True,
            output_quality_evaluator=None,  # no evaluator
        )

        async def execute(candidate: dict[str, Any]) -> str:
            if candidate.get("locality") == "local":
                return "local response"
            return "cloud response"

        result = await runner.run_with_inference(
            [_local_candidate(gates=gates), _cloud_candidate()],
            execute_candidate=execute,
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert result.value == "cloud response"
        assert result.fallback_used
        # First attempt failed at output_quality stage
        assert result.attempts[0].status == AttemptStatus.FAILED
        assert result.attempts[0].stage == AttemptStage.OUTPUT_QUALITY


class TestUnknownAdvisoryGateRecordsUnknown:
    @pytest.mark.asyncio
    async def test_unknown_advisory_gate_records_unknown(self) -> None:
        """Advisory gate with no evaluator → status=unknown, continue."""
        gates = [
            {"code": "evaluator_score_min", "required": False},
        ]
        runner = CandidateAttemptRunner(
            fallback_allowed=True,
            output_quality_evaluator=None,
        )

        async def execute(candidate: dict[str, Any]) -> str:
            return "local response"

        result = await runner.run_with_inference(
            [_local_candidate(gates=gates)],
            execute_candidate=execute,
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert result.value == "local response"
        assert result.selected_attempt is not None
        # Gate result should be recorded as "unknown"
        oq_results = [g for g in result.selected_attempt.gate_results if g.gate_class == "output_quality"]
        assert len(oq_results) == 1
        assert oq_results[0].status == GateStatus.UNKNOWN
        assert oq_results[0].reason_code == "no_evaluator"


# ---------------------------------------------------------------------------
# Tests: route metadata includes gate_class and evaluation_phase
# ---------------------------------------------------------------------------


class TestRouteMetadataIncludesGateClassification:
    def test_gate_results_include_classification_in_to_dict(self) -> None:
        gates = [
            {"code": "min_tokens_per_second", "required": True, "threshold_number": 10},
        ]
        runner = CandidateAttemptRunner()
        result = runner.run(
            [_local_candidate(gates=gates)],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        attempt_dict = result.attempts[0].to_dict()
        gate_dicts = attempt_dict["gate_results"]

        # runtime_available gate should have classification
        runtime_gate = next(g for g in gate_dicts if g["code"] == "runtime_available")
        assert runtime_gate["gate_class"] == "readiness"
        assert runtime_gate["evaluation_phase"] == "pre_inference"

        # artifact_verified gate should have classification
        artifact_gate = next(g for g in gate_dicts if g["code"] == "artifact_verified")
        assert artifact_gate["gate_class"] == "readiness"
        assert artifact_gate["evaluation_phase"] == "pre_inference"

        # min_tokens_per_second gate should have classification
        perf_gate = next(g for g in gate_dicts if g["code"] == "min_tokens_per_second")
        assert perf_gate["gate_class"] == "performance"
        assert perf_gate["evaluation_phase"] == "pre_inference"


# ---------------------------------------------------------------------------
# Tests: FallbackTrigger includes new fields
# ---------------------------------------------------------------------------


class TestFallbackTriggerIncludesNewFields:
    def test_fallback_trigger_includes_gate_fields(self) -> None:
        gates = [
            {"code": "min_tokens_per_second", "required": True, "threshold_number": 30},
        ]
        runner = CandidateAttemptRunner(fallback_allowed=True)
        result = runner.run(
            [_local_candidate(gates=gates), _cloud_candidate()],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_FailGate({"min_tokens_per_second"}),
        )

        assert result.fallback_trigger is not None
        trigger = result.fallback_trigger
        assert trigger.gate_code == "min_tokens_per_second"
        assert trigger.gate_class == "performance"
        assert trigger.evaluation_phase == "pre_inference"
        assert trigger.candidate_index == 0

        # to_dict should include the new fields
        trigger_dict = trigger.to_dict()
        assert trigger_dict["gate_code"] == "min_tokens_per_second"
        assert trigger_dict["gate_class"] == "performance"
        assert trigger_dict["evaluation_phase"] == "pre_inference"
        assert trigger_dict["candidate_index"] == 0

    def test_fallback_trigger_to_dict_omits_none_fields(self) -> None:
        """Backward compat: FallbackTrigger.to_dict() omits None fields."""
        from octomil.runtime.routing.attempt_runner import FallbackTrigger

        trigger = FallbackTrigger(code="gate_failed", stage="gate", message="test")
        d = trigger.to_dict()
        assert "gate_code" not in d
        assert "gate_class" not in d
        assert "evaluation_phase" not in d
        assert "candidate_index" not in d
        assert "output_visible_before_failure" not in d


# ---------------------------------------------------------------------------
# Tests: classify_gate
# ---------------------------------------------------------------------------


class TestClassifyGate:
    def test_known_readiness_gate(self) -> None:
        g_cls, g_phase, blocking = classify_gate("runtime_available")
        assert g_cls == "readiness"
        assert g_phase == "pre_inference"
        assert blocking is True

    def test_known_performance_gate(self) -> None:
        g_cls, g_phase, blocking = classify_gate("min_tokens_per_second")
        assert g_cls == "performance"
        assert g_phase == "pre_inference"
        assert blocking is False

    def test_known_output_quality_gate(self) -> None:
        g_cls, g_phase, blocking = classify_gate("schema_valid")
        assert g_cls == "output_quality"
        assert g_phase == "post_inference"
        assert blocking is True

    def test_unknown_gate_defaults_to_readiness(self) -> None:
        g_cls, g_phase, blocking = classify_gate("totally_unknown_gate")
        assert g_cls == "readiness"
        assert g_phase == "pre_inference"
        assert blocking is True


# ---------------------------------------------------------------------------
# Tests: output_quality gates are skipped in pre-inference gate loop
# ---------------------------------------------------------------------------


class TestOutputQualityGatesSkippedInPreInference:
    def test_oq_gates_not_evaluated_in_run(self) -> None:
        """Output quality gates should be skipped during the pre-inference gate loop."""
        gates = [
            {"code": "min_tokens_per_second", "required": True, "threshold_number": 10},
            {"code": "schema_valid", "required": True},
        ]
        runner = CandidateAttemptRunner()
        result = runner.run(
            [_local_candidate(gates=gates)],
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert result.selected_attempt is not None
        # Gate results should NOT include schema_valid (it's output_quality, deferred)
        gate_codes = [g.code for g in result.selected_attempt.gate_results]
        assert "schema_valid" not in gate_codes
        assert "runtime_available" in gate_codes
        assert "artifact_verified" in gate_codes
        assert "min_tokens_per_second" in gate_codes


# ---------------------------------------------------------------------------
# Tests: output_quality gate with evaluator that passes
# ---------------------------------------------------------------------------


class TestOutputQualityPassingGate:
    @pytest.mark.asyncio
    async def test_oq_gate_passes_no_fallback(self) -> None:
        gates = [
            {"code": "schema_valid", "required": True},
        ]
        runner = CandidateAttemptRunner(
            fallback_allowed=True,
            output_quality_evaluator=_PassOutputQualityEvaluator(),
        )

        async def execute(candidate: dict[str, Any]) -> str:
            return "local response"

        result = await runner.run_with_inference(
            [_local_candidate(gates=gates), _cloud_candidate()],
            execute_candidate=execute,
            runtime_checker=_AlwaysAvailableRuntime(),
            artifact_checker=_AlwaysOkArtifact(),
            gate_evaluator=_PassAllGates(),
        )

        assert result.succeeded
        assert result.selected_attempt is not None
        assert result.value == "local response"
        assert not result.fallback_used
        # Gate result should be recorded on the selected attempt
        oq_results = [g for g in result.selected_attempt.gate_results if g.gate_class == "output_quality"]
        assert len(oq_results) == 1
        assert oq_results[0].status == GateStatus.PASSED
