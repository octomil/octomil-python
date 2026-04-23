"""Per-request candidate attempt runner.

Evaluates candidates in priority order through staged gates.
Produces RouteAttempt records for structured route metadata.

Contract schemas:
- candidate_gate.schema.json — 18 gate codes (12 existing + 6 output_quality)
- route_attempt.schema.json — attempt record with index, locality, mode, engine, artifact, status, stage, gate_results[], reason
- route_metadata.schema.json — extended route metadata with attempts array and fallback trigger
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AttemptStage(str, Enum):
    """Stage at which an attempt resolved (succeeded or failed)."""

    POLICY = "policy"
    PREPARE = "prepare"
    DOWNLOAD = "download"
    VERIFY = "verify"
    LOAD = "load"
    BENCHMARK = "benchmark"
    GATE = "gate"
    INFERENCE = "inference"
    OUTPUT_QUALITY = "output_quality"


class AttemptStatus(str, Enum):
    """Outcome of a single attempt."""

    SKIPPED = "skipped"
    FAILED = "failed"
    SELECTED = "selected"


class GateStatus(str, Enum):
    """Outcome of evaluating a single gate."""

    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    NOT_REQUIRED = "not_required"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Result of evaluating one gate against a candidate."""

    code: str
    status: GateStatus
    observed_number: float | None = None
    threshold_number: float | None = None
    reason_code: str | None = None
    gate_class: str | None = None
    evaluation_phase: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "status": self.status.value}
        if self.observed_number is not None:
            d["observed_number"] = self.observed_number
        if self.threshold_number is not None:
            d["threshold_number"] = self.threshold_number
        if self.reason_code is not None:
            d["reason_code"] = self.reason_code
        if self.gate_class is not None:
            d["gate_class"] = self.gate_class
        if self.evaluation_phase is not None:
            d["evaluation_phase"] = self.evaluation_phase
        return d


@dataclass
class AttemptArtifact:
    """Artifact state at time of an attempt."""

    id: str | None = None
    digest: str | None = None
    cache_status: str = "not_applicable"
    managed_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "digest": self.digest}
        d["cache"] = {"status": self.cache_status, "managed_by": self.managed_by}
        return d


@dataclass
class RouteAttempt:
    """A single attempt record in the per-request candidate loop.

    Matches the route_attempt.schema.json contract.
    """

    index: int
    locality: str
    mode: str
    status: AttemptStatus
    stage: AttemptStage
    reason_code: str
    reason_message: str
    engine: str | None = None
    artifact: AttemptArtifact | None = None
    gate_results: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "locality": self.locality,
            "mode": self.mode,
            "engine": self.engine,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "status": self.status.value,
            "stage": self.stage.value,
            "gate_results": [g.to_dict() for g in self.gate_results],
            "reason": {"code": self.reason_code, "message": self.reason_message},
        }


@dataclass
class FallbackTrigger:
    """Describes what triggered the fallback from one candidate to another."""

    code: str
    stage: str
    message: str
    gate_code: str | None = None
    gate_class: str | None = None
    evaluation_phase: str | None = None
    candidate_index: int | None = None
    output_visible_before_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "stage": self.stage, "message": self.message}
        if self.gate_code is not None:
            d["gate_code"] = self.gate_code
        if self.gate_class is not None:
            d["gate_class"] = self.gate_class
        if self.evaluation_phase is not None:
            d["evaluation_phase"] = self.evaluation_phase
        if self.candidate_index is not None:
            d["candidate_index"] = self.candidate_index
        if self.output_visible_before_failure:
            d["output_visible_before_failure"] = True
        return d


@dataclass
class AttemptLoopResult:
    """Result of running the candidate attempt loop."""

    selected_attempt: RouteAttempt | None
    attempts: list[RouteAttempt]
    fallback_used: bool = False
    fallback_trigger: FallbackTrigger | None = None
    from_attempt: int | None = None
    to_attempt: int | None = None
    value: Any | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.selected_attempt is not None

    def to_route_metadata_fields(self) -> dict[str, Any]:
        """Returns the attempts/fallback portion for route_metadata."""
        return {
            "attempts": [a.to_dict() for a in self.attempts],
            "fallback": {
                "used": self.fallback_used,
                "from_attempt": self.from_attempt,
                "to_attempt": self.to_attempt,
                "trigger": self.fallback_trigger.to_dict() if self.fallback_trigger else None,
            },
        }


# ---------------------------------------------------------------------------
# Checker protocols
# ---------------------------------------------------------------------------


class RuntimeChecker:
    """Protocol for checking runtime/engine availability."""

    def check(self, *, engine: str | None, locality: str) -> tuple[bool, str | None]:
        """Returns (available, reason_code_if_not)."""
        raise NotImplementedError


class ArtifactChecker:
    """Protocol for checking artifact cache and verification."""

    def check(self, artifact_plan: dict[str, Any]) -> tuple[bool, str, str | None]:
        """Returns (ok, cache_status, reason_code_if_not)."""
        raise NotImplementedError


class GateEvaluator:
    """Protocol for evaluating per-request gates."""

    def evaluate(self, gate: dict[str, Any], *, engine: str | None, locality: str) -> GateResult:
        """Evaluate a single gate and return the result."""
        raise NotImplementedError


class OutputQualityGateEvaluator:
    """Protocol for post-inference output quality evaluation."""

    def evaluate(self, gate: dict[str, Any], response: Any) -> GateResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Gate classification
# ---------------------------------------------------------------------------

# code -> (gate_class, evaluation_phase, blocking_default)
GATE_CLASSIFICATION: dict[str, tuple[str, str, bool]] = {
    "artifact_verified": ("readiness", "pre_inference", True),
    "runtime_available": ("readiness", "pre_inference", True),
    "model_loads": ("readiness", "pre_inference", True),
    "context_fits": ("readiness", "pre_inference", True),
    "modality_supported": ("readiness", "pre_inference", True),
    "tool_support": ("readiness", "pre_inference", True),
    "min_tokens_per_second": ("performance", "pre_inference", False),
    "max_ttft_ms": ("performance", "during_inference", False),
    "max_error_rate": ("performance", "pre_inference", False),
    "min_free_memory_bytes": ("performance", "pre_inference", True),
    "min_free_storage_bytes": ("performance", "pre_inference", True),
    "benchmark_fresh": ("performance", "pre_inference", False),
    "schema_valid": ("output_quality", "post_inference", True),
    "tool_call_valid": ("output_quality", "post_inference", True),
    "safety_passed": ("output_quality", "post_inference", True),
    "evaluator_score_min": ("output_quality", "post_inference", False),
    "json_parseable": ("output_quality", "post_inference", True),
    "max_refusal_rate": ("output_quality", "post_inference", False),
}


def classify_gate(code: str) -> tuple[str, str, bool]:
    """Returns (gate_class, evaluation_phase, blocking_default) for a gate code."""
    return GATE_CLASSIFICATION.get(code, ("readiness", "pre_inference", True))


class _NoOpRuntimeChecker(RuntimeChecker):
    def check(self, *, engine: str | None, locality: str) -> tuple[bool, str | None]:
        return True, None


class _NoOpArtifactChecker(ArtifactChecker):
    def check(self, artifact_plan: dict[str, Any]) -> tuple[bool, str, str | None]:
        return True, "hit", None


class _NoOpGateEvaluator(GateEvaluator):
    def evaluate(self, gate: dict[str, Any], *, engine: str | None, locality: str) -> GateResult:
        return GateResult(code=gate.get("code", "unknown"), status=GateStatus.PASSED)


# ---------------------------------------------------------------------------
# CandidateAttemptRunner
# ---------------------------------------------------------------------------


class CandidateAttemptRunner:
    """Evaluates candidates in priority order through the attempt loop.

    For each candidate:
    1. Check policy constraints (locality allowed?)
    2. Prepare: check runtime availability
    3. Download/verify artifact if needed
    4. Load model
    5. Run benchmark if required
    6. Evaluate gates
    7. Attempt inference

    If a candidate fails at any stage and fallback is allowed,
    move to the next candidate.
    """

    def __init__(
        self,
        *,
        fallback_allowed: bool = True,
        streaming: bool = False,
        output_quality_evaluator: OutputQualityGateEvaluator | None = None,
    ) -> None:
        self._fallback_allowed = fallback_allowed
        self._streaming = streaming
        self._output_quality_evaluator = output_quality_evaluator
        self._attempts: list[RouteAttempt] = []

    @property
    def streaming(self) -> bool:
        """Whether the caller intends streaming inference."""
        return self._streaming

    def should_fallback_after_inference_error(self, *, first_token_emitted: bool) -> bool:
        """Return whether an inference error may fall back to another candidate.

        Streaming requests may only fall back before the first token is emitted.
        After the user has seen output, switching routes would splice two model
        responses into one stream, so the error must surface instead.
        """
        if self._streaming and first_token_emitted:
            return False
        return self._fallback_allowed

    def run(
        self,
        candidates: list[dict[str, Any]],
        *,
        runtime_checker: RuntimeChecker | None = None,
        artifact_checker: ArtifactChecker | None = None,
        gate_evaluator: GateEvaluator | None = None,
    ) -> AttemptLoopResult:
        """Run readiness stages over candidates without invoking inference.

        This method is useful for tests and diagnostics. User-facing execution
        paths should use ``run_with_inference`` so inference failures and
        streaming first-token semantics participate in fallback decisions.

        Args:
            candidates: Ordered list of candidate dicts from the plan response.
            runtime_checker: Checks if a runtime/engine is available.
            artifact_checker: Checks artifact cache and verification.
            gate_evaluator: Evaluates per-request gates.

        Returns:
            AttemptLoopResult with the selected attempt or failure info.
        """
        self._attempts = []
        _runtime = runtime_checker or _NoOpRuntimeChecker()
        _artifact = artifact_checker or _NoOpArtifactChecker()
        _gates = gate_evaluator or _NoOpGateEvaluator()

        selected: RouteAttempt | None = None
        fallback_trigger: FallbackTrigger | None = None
        from_attempt: int | None = None
        to_attempt: int | None = None

        for idx, candidate in enumerate(candidates):
            locality = candidate.get("locality", "local")
            mode = self._mode_for_candidate(candidate)
            engine = candidate.get("engine")

            # ------------------------------------------------------------------
            # Stage: prepare — check runtime/engine availability
            # ------------------------------------------------------------------
            runtime_ok, runtime_reason = _runtime.check(engine=engine, locality=locality)
            gate_results: list[GateResult] = []

            if not runtime_ok:
                rt_cls, rt_phase, _ = classify_gate("runtime_available")
                gate_results.append(
                    GateResult(
                        code="runtime_available",
                        status=GateStatus.FAILED,
                        reason_code=runtime_reason,
                        gate_class=rt_cls,
                        evaluation_phase=rt_phase,
                    )
                )
                attempt = RouteAttempt(
                    index=idx,
                    locality=locality,
                    mode=mode,
                    engine=engine,
                    status=AttemptStatus.FAILED,
                    stage=AttemptStage.PREPARE,
                    gate_results=gate_results,
                    reason_code="runtime_unavailable",
                    reason_message=f"{engine or 'runtime'} not available: {runtime_reason}",
                )
                self._attempts.append(attempt)
                if self._fallback_allowed and idx < len(candidates) - 1:
                    if fallback_trigger is None:
                        fallback_trigger = FallbackTrigger(
                            code="runtime_unavailable",
                            stage="prepare",
                            message=attempt.reason_message,
                            gate_code="runtime_available",
                            gate_class=rt_cls,
                            evaluation_phase=rt_phase,
                            candidate_index=idx,
                        )
                        from_attempt = idx
                    continue
                break

            rt_cls, rt_phase, _ = classify_gate("runtime_available")
            gate_results.append(
                GateResult(
                    code="runtime_available",
                    status=GateStatus.PASSED,
                    gate_class=rt_cls,
                    evaluation_phase=rt_phase,
                )
            )

            # ------------------------------------------------------------------
            # Stage: verify artifact (if local with artifact)
            # ------------------------------------------------------------------
            artifact_info: AttemptArtifact | None = None
            artifact_plan = candidate.get("artifact")
            if artifact_plan and locality == "local":
                art_ok, art_status, art_reason = _artifact.check(artifact_plan)
                artifact_info = AttemptArtifact(
                    id=artifact_plan.get("artifact_id"),
                    digest=artifact_plan.get("digest"),
                    cache_status=art_status,
                    managed_by="octomil",
                )
                art_cls, art_phase, _ = classify_gate("artifact_verified")
                if art_ok:
                    gate_results.append(
                        GateResult(
                            code="artifact_verified",
                            status=GateStatus.PASSED,
                            gate_class=art_cls,
                            evaluation_phase=art_phase,
                        )
                    )
                else:
                    gate_results.append(
                        GateResult(
                            code="artifact_verified",
                            status=GateStatus.FAILED,
                            reason_code=art_reason,
                            gate_class=art_cls,
                            evaluation_phase=art_phase,
                        )
                    )
                    attempt = RouteAttempt(
                        index=idx,
                        locality=locality,
                        mode=mode,
                        engine=engine,
                        artifact=artifact_info,
                        status=AttemptStatus.FAILED,
                        stage=AttemptStage.VERIFY,
                        gate_results=gate_results,
                        reason_code="artifact_verification_failed",
                        reason_message=f"artifact verification failed: {art_reason}",
                    )
                    self._attempts.append(attempt)
                    if self._fallback_allowed and idx < len(candidates) - 1:
                        if fallback_trigger is None:
                            fallback_trigger = FallbackTrigger(
                                code="artifact_verification_failed",
                                stage="verify",
                                message=attempt.reason_message,
                                gate_code="artifact_verified",
                                gate_class=art_cls,
                                evaluation_phase=art_phase,
                                candidate_index=idx,
                            )
                            from_attempt = idx
                        continue
                    break

            # ------------------------------------------------------------------
            # Stage: gate — evaluate per-request gates (skip output_quality; those run post-inference)
            # ------------------------------------------------------------------
            gates = candidate.get("gates", [])
            gate_failed = False
            for gate in gates:
                code = gate.get("code", "")
                if code in ("runtime_available", "artifact_verified"):
                    continue  # already evaluated above
                g_cls, g_phase, _ = classify_gate(code)
                if g_cls == "output_quality":
                    continue  # deferred to post-inference
                required = gate.get("required", True)
                result = _gates.evaluate(gate, engine=engine, locality=locality)
                result.gate_class = g_cls
                result.evaluation_phase = g_phase
                gate_results.append(result)
                if result.status == GateStatus.FAILED and required:
                    gate_failed = True
                    attempt = RouteAttempt(
                        index=idx,
                        locality=locality,
                        mode=mode,
                        engine=engine,
                        artifact=artifact_info,
                        status=AttemptStatus.FAILED,
                        stage=AttemptStage.GATE,
                        gate_results=gate_results,
                        reason_code="gate_failed",
                        reason_message=f"{code} gate failed",
                    )
                    self._attempts.append(attempt)
                    if self._fallback_allowed and idx < len(candidates) - 1:
                        if fallback_trigger is None:
                            fallback_trigger = FallbackTrigger(
                                code="gate_failed",
                                stage="gate",
                                message=f"{code} gate failed",
                                gate_code=code,
                                gate_class=g_cls,
                                evaluation_phase=g_phase,
                                candidate_index=idx,
                            )
                            from_attempt = idx
                        continue
                    break

            if gate_failed:
                if not self._fallback_allowed or idx >= len(candidates) - 1:
                    break
                continue

            # ------------------------------------------------------------------
            # Stage: inference — candidate selected
            # ------------------------------------------------------------------
            attempt = RouteAttempt(
                index=idx,
                locality=locality,
                mode=mode,
                engine=engine,
                artifact=artifact_info,
                status=AttemptStatus.SELECTED,
                stage=AttemptStage.INFERENCE,
                gate_results=gate_results,
                reason_code="selected",
                reason_message="all gates passed, candidate selected",
            )
            self._attempts.append(attempt)
            selected = attempt
            if fallback_trigger is not None:
                to_attempt = idx
            break

        return AttemptLoopResult(
            selected_attempt=selected,
            attempts=self._attempts,
            fallback_used=fallback_trigger is not None and selected is not None,
            fallback_trigger=fallback_trigger if (fallback_trigger is not None and selected is not None) else None,
            from_attempt=from_attempt if (fallback_trigger is not None and selected is not None) else None,
            to_attempt=to_attempt,
        )

    @staticmethod
    def _mode_for_candidate(candidate: dict[str, Any]) -> str:
        """Derive execution mode from candidate locality."""
        locality = candidate.get("locality", "local")
        if locality == "cloud":
            return "hosted_gateway"
        return "sdk_runtime"

    def _evaluate_output_quality_gates(
        self,
        candidate: dict[str, Any],
        response: Any,
        ready_attempt: RouteAttempt,
        idx: int,
        *,
        first_token_emitted: bool = False,
    ) -> RouteAttempt | None:
        """Evaluate post-inference output_quality gates for a candidate.

        Returns a failed RouteAttempt if a required gate fails and fallback
        is possible (output not yet visible). Returns None if all gates pass
        or failures are advisory/non-blocking.
        """
        gates = candidate.get("gates", [])
        oq_gates = [g for g in gates if classify_gate(g.get("code", ""))[0] == "output_quality"]
        if not oq_gates:
            return None

        evaluator = self._output_quality_evaluator
        advisory_failures: list[GateResult] = []

        for gate in oq_gates:
            code = gate.get("code", "")
            required = gate.get("required", True)
            g_cls, g_phase, blocking_default = classify_gate(code)

            if evaluator is None:
                # No evaluator: unknown gate handling
                if required:
                    # Unknown required gate → fail closed
                    result = GateResult(
                        code=code,
                        status=GateStatus.FAILED,
                        reason_code="no_evaluator",
                        gate_class=g_cls,
                        evaluation_phase=g_phase,
                    )
                    ready_attempt.gate_results.append(result)

                    if first_token_emitted:
                        # Output already visible — record failure, no fallback
                        advisory_failures.append(result)
                        continue

                    return RouteAttempt(
                        index=idx,
                        locality=ready_attempt.locality,
                        mode=ready_attempt.mode,
                        engine=ready_attempt.engine,
                        artifact=ready_attempt.artifact,
                        status=AttemptStatus.FAILED,
                        stage=AttemptStage.OUTPUT_QUALITY,
                        gate_results=ready_attempt.gate_results,
                        reason_code=f"quality_gate_{code}",
                        reason_message=f"output quality gate {code} failed (no evaluator)",
                    )
                else:
                    # Unknown advisory gate → record "unknown", continue
                    result = GateResult(
                        code=code,
                        status=GateStatus.UNKNOWN,
                        reason_code="no_evaluator",
                        gate_class=g_cls,
                        evaluation_phase=g_phase,
                    )
                    ready_attempt.gate_results.append(result)
                    continue

            result = evaluator.evaluate(gate, response)
            result.gate_class = g_cls
            result.evaluation_phase = g_phase
            ready_attempt.gate_results.append(result)

            if result.status == GateStatus.FAILED:
                if not required:
                    # Advisory failure → record, continue
                    advisory_failures.append(result)
                    continue

                if first_token_emitted:
                    # Output already visible — record failure, no fallback
                    advisory_failures.append(result)
                    continue

                # Required failure before output visible → trigger fallback
                return RouteAttempt(
                    index=idx,
                    locality=ready_attempt.locality,
                    mode=ready_attempt.mode,
                    engine=ready_attempt.engine,
                    artifact=ready_attempt.artifact,
                    status=AttemptStatus.FAILED,
                    stage=AttemptStage.OUTPUT_QUALITY,
                    gate_results=ready_attempt.gate_results,
                    reason_code=f"quality_gate_{code}",
                    reason_message=f"output quality gate {code} failed",
                )

        return None

    async def run_with_inference(
        self,
        candidates: list[dict[str, Any]],
        *,
        execute_candidate: Callable[[dict[str, Any]], Any],
        runtime_checker: RuntimeChecker | None = None,
        artifact_checker: ArtifactChecker | None = None,
        gate_evaluator: GateEvaluator | None = None,
    ) -> AttemptLoopResult:
        """Run readiness gates and actual inference for each candidate.

        The executor is called only after prepare/verify/gate checks pass. If
        inference fails and fallback is allowed, the runner records an inference
        failure and advances to the next candidate. For streaming requests,
        executor exceptions may set ``first_token_emitted=True`` to prevent
        fallback after output has reached the caller.
        """
        self._attempts = []
        fallback_trigger: FallbackTrigger | None = None
        from_attempt: int | None = None
        to_attempt: int | None = None
        last_error: Exception | None = None

        for idx, candidate in enumerate(candidates):
            readiness_runner = CandidateAttemptRunner(
                fallback_allowed=False,
                streaming=self._streaming,
            )
            readiness = readiness_runner.run(
                [candidate],
                runtime_checker=runtime_checker,
                artifact_checker=artifact_checker,
                gate_evaluator=gate_evaluator,
            )

            ready_attempt = readiness.attempts[0] if readiness.attempts else None
            if ready_attempt is not None:
                ready_attempt.index = idx

            if not readiness.succeeded:
                if ready_attempt is not None:
                    self._attempts.append(ready_attempt)
                    if fallback_trigger is None:
                        fallback_trigger = FallbackTrigger(
                            code=ready_attempt.reason_code,
                            stage=ready_attempt.stage.value,
                            message=ready_attempt.reason_message,
                        )
                        from_attempt = idx
                if self._fallback_allowed and idx < len(candidates) - 1:
                    continue
                return AttemptLoopResult(
                    selected_attempt=None,
                    attempts=self._attempts,
                    error=last_error,
                )

            assert ready_attempt is not None
            try:
                maybe_value = execute_candidate(candidate)
                value = await maybe_value if inspect.isawaitable(maybe_value) else maybe_value
            except Exception as exc:
                last_error = exc
                first_token_emitted = bool(getattr(exc, "first_token_emitted", False))
                reason_code = (
                    "inference_error_after_first_token"
                    if first_token_emitted
                    else "inference_error_before_first_token"
                    if self._streaming
                    else "inference_error"
                )
                failed_attempt = RouteAttempt(
                    index=idx,
                    locality=ready_attempt.locality,
                    mode=ready_attempt.mode,
                    engine=ready_attempt.engine,
                    artifact=ready_attempt.artifact,
                    status=AttemptStatus.FAILED,
                    stage=AttemptStage.INFERENCE,
                    gate_results=ready_attempt.gate_results,
                    reason_code=reason_code,
                    reason_message=str(exc) or reason_code,
                )
                self._attempts.append(failed_attempt)

                if (
                    self.should_fallback_after_inference_error(first_token_emitted=first_token_emitted)
                    and idx < len(candidates) - 1
                ):
                    if fallback_trigger is None:
                        fallback_trigger = FallbackTrigger(
                            code=reason_code,
                            stage=AttemptStage.INFERENCE.value,
                            message=failed_attempt.reason_message,
                            candidate_index=idx,
                        )
                        from_attempt = idx
                    continue

                return AttemptLoopResult(
                    selected_attempt=None,
                    attempts=self._attempts,
                    error=exc,
                )

            # ------------------------------------------------------------------
            # Post-inference: evaluate output_quality gates
            # ------------------------------------------------------------------
            quality_failure = self._evaluate_output_quality_gates(
                candidate,
                value,
                ready_attempt,
                idx,
                first_token_emitted=False,  # non-streaming: output not visible yet
            )
            if quality_failure is not None:
                # Required output_quality gate failed before output visible → fallback
                self._attempts.append(quality_failure)
                if self._fallback_allowed and idx < len(candidates) - 1:
                    if fallback_trigger is None:
                        fallback_trigger = FallbackTrigger(
                            code=quality_failure.reason_code,
                            stage=AttemptStage.OUTPUT_QUALITY.value,
                            message=quality_failure.reason_message,
                            gate_code=quality_failure.reason_code.replace("quality_gate_", ""),
                            gate_class="output_quality",
                            evaluation_phase="post_inference",
                            candidate_index=idx,
                        )
                        from_attempt = idx
                    continue
                return AttemptLoopResult(
                    selected_attempt=None,
                    attempts=self._attempts,
                    error=None,
                )

            self._attempts.append(ready_attempt)
            ready_attempt.reason_message = "inference succeeded"
            if fallback_trigger is not None:
                to_attempt = idx
            return AttemptLoopResult(
                selected_attempt=ready_attempt,
                attempts=self._attempts,
                fallback_used=fallback_trigger is not None,
                fallback_trigger=fallback_trigger,
                from_attempt=from_attempt if fallback_trigger is not None else None,
                to_attempt=to_attempt,
                value=value,
            )

        return AttemptLoopResult(
            selected_attempt=None,
            attempts=self._attempts,
            error=last_error,
        )
