"""Routing layer: candidate attempt runner and route event telemetry."""

from octomil.runtime.routing.attempt_runner import (
    ArtifactChecker,
    AttemptArtifact,
    AttemptLoopResult,
    AttemptStage,
    AttemptStatus,
    CandidateAttemptRunner,
    FallbackTrigger,
    GateEvaluator,
    GateResult,
    GateStatus,
    RouteAttempt,
    RuntimeChecker,
)
from octomil.runtime.routing.model_ref import (
    CANONICAL_MODEL_REF_KINDS,
    ParsedModelRef,
    parse_model_ref,
)
from octomil.runtime.routing.route_event import (
    FORBIDDEN_TELEMETRY_KEYS,
    CandidateAttemptSummary,
    RouteEvent,
    build_route_event,
    emit_route_event,
    strip_forbidden_keys,
)

__all__ = [
    "ArtifactChecker",
    "AttemptArtifact",
    "AttemptLoopResult",
    "AttemptStage",
    "AttemptStatus",
    "CandidateAttemptRunner",
    "FallbackTrigger",
    "FORBIDDEN_TELEMETRY_KEYS",
    "CANONICAL_MODEL_REF_KINDS",
    "GateEvaluator",
    "GateResult",
    "GateStatus",
    "CandidateAttemptSummary",
    "RouteAttempt",
    "RouteEvent",
    "RuntimeChecker",
    "ParsedModelRef",
    "build_route_event",
    "emit_route_event",
    "parse_model_ref",
    "strip_forbidden_keys",
]
