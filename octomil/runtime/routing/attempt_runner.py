"""Per-request candidate attempt runner.

Evaluates candidates in priority order through staged gates.
Produces RouteAttempt records for structured route metadata.

Contract schemas:
- candidate_gate.schema.json — 12 gate codes
- route_attempt.schema.json — attempt record with index, locality, mode, engine, artifact, status, stage, gate_results[], reason
- route_metadata.schema.json — extended route metadata with attempts array and fallback trigger
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "status": self.status.value}
        if self.observed_number is not None:
            d["observed_number"] = self.observed_number
        if self.threshold_number is not None:
            d["threshold_number"] = self.threshold_number
        if self.reason_code is not None:
            d["reason_code"] = self.reason_code
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

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "stage": self.stage, "message": self.message}


@dataclass
class AttemptLoopResult:
    """Result of running the candidate attempt loop."""

    selected_attempt: RouteAttempt | None
    attempts: list[RouteAttempt]
    fallback_used: bool = False
    fallback_trigger: FallbackTrigger | None = None
    from_attempt: int | None = None
    to_attempt: int | None = None

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

    The runner does NOT perform actual inference — it evaluates readiness.
    The caller invokes inference after the runner selects a candidate.
    """

    def __init__(
        self,
        *,
        fallback_allowed: bool = True,
        streaming: bool = False,
    ) -> None:
        self._fallback_allowed = fallback_allowed
        self._streaming = streaming
        self._attempts: list[RouteAttempt] = []

    @property
    def streaming(self) -> bool:
        """Whether the caller intends streaming inference."""
        return self._streaming

    def run(
        self,
        candidates: list[dict[str, Any]],
        *,
        runtime_checker: RuntimeChecker | None = None,
        artifact_checker: ArtifactChecker | None = None,
        gate_evaluator: GateEvaluator | None = None,
    ) -> AttemptLoopResult:
        """Run the attempt loop over candidates.

        Args:
            candidates: Ordered list of candidate dicts from the plan response.
            runtime_checker: Checks if a runtime/engine is available.
            artifact_checker: Checks artifact cache and verification.
            gate_evaluator: Evaluates per-request gates.

        Returns:
            AttemptLoopResult with the selected attempt or failure info.
        """
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
                gate_results.append(
                    GateResult(
                        code="runtime_available",
                        status=GateStatus.FAILED,
                        reason_code=runtime_reason,
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
                        )
                        from_attempt = idx
                    continue
                break

            gate_results.append(GateResult(code="runtime_available", status=GateStatus.PASSED))

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
                if art_ok:
                    gate_results.append(GateResult(code="artifact_verified", status=GateStatus.PASSED))
                else:
                    gate_results.append(
                        GateResult(
                            code="artifact_verified",
                            status=GateStatus.FAILED,
                            reason_code=art_reason,
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
                            )
                            from_attempt = idx
                        continue
                    break

            # ------------------------------------------------------------------
            # Stage: gate — evaluate per-request gates
            # ------------------------------------------------------------------
            gates = candidate.get("gates", [])
            gate_failed = False
            for gate in gates:
                code = gate.get("code", "")
                if code in ("runtime_available", "artifact_verified"):
                    continue  # already evaluated above
                required = gate.get("required", True)
                result = _gates.evaluate(gate, engine=engine, locality=locality)
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
