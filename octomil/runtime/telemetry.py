"""Route telemetry payload — reference implementation for SDK parity.

Defines the structured telemetry payload emitted after each inference request,
the builder that constructs it from an AttemptLoopResult, and the safety
validation that prevents prompt/output/PII leakage.

This module is the REFERENCE IMPLEMENTATION. Other SDKs (Node, iOS, Android,
Browser) must match this shape exactly.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Safety: forbidden keys that must NEVER appear in telemetry payloads
# ---------------------------------------------------------------------------

FORBIDDEN_TELEMETRY_KEYS = frozenset(
    {
        "prompt",
        "input",
        "output",
        "completion",
        "audio",
        "audio_bytes",
        "file_path",
        "text",
        "content",
        "messages",
        "system_prompt",
        "documents",
        "image",
        "image_url",
        "embedding",
        "embeddings",
    }
)


def validate_telemetry_safety(payload: dict[str, Any]) -> None:
    """Raises ValueError if payload contains forbidden keys.

    Recursively checks nested dicts and list elements to ensure no
    prompt/output/audio/PII data leaks into telemetry.
    """
    _check_keys_recursive(payload, path="")


def _check_keys_recursive(obj: Any, path: str) -> None:
    """Recursively validate that no forbidden keys exist in the structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in FORBIDDEN_TELEMETRY_KEYS:
                raise ValueError(
                    f"Forbidden telemetry key '{key}' found at path '{path}.{key}'. "
                    f"Telemetry payloads must NEVER contain prompt, input, output, "
                    f"audio, file paths, or PII data."
                )
            _check_keys_recursive(value, path=f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _check_keys_recursive(item, path=f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Attempt detail — per-candidate structured telemetry
# ---------------------------------------------------------------------------


@dataclass
class GateSummary:
    """Summary of gate evaluation for a candidate attempt."""

    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


@dataclass
class AttemptDetail:
    """Structured detail for a single candidate attempt in the route loop."""

    index: int
    locality: str  # "local" | "cloud"
    mode: str  # "sdk_runtime" | "hosted_gateway"
    engine: str | None = None
    status: str = "failed"  # "failed" | "selected" | "skipped"
    stage: str = "gate"  # "gate" | "inference" | "selection"
    gate_summary: GateSummary = field(default_factory=GateSummary)
    reason_code: str = ""


# ---------------------------------------------------------------------------
# RouteEventPayload — the reference telemetry shape
# ---------------------------------------------------------------------------


@dataclass
class RouteEventPayload:
    """Telemetry payload emitted after each inference request.

    SAFETY: This payload must NEVER contain:
    - prompt / input / output / completion text
    - audio bytes
    - file paths
    - raw documents
    - PII
    """

    route_id: str
    plan_id: str | None
    request_id: str
    app_id: str | None = None
    app_slug: str | None = None
    capability: str | None = None
    deployment_id: str | None = None
    experiment_id: str | None = None
    variant_id: str | None = None
    policy: str | None = None
    planner_source: str | None = None  # "server" | "cache" | "offline"
    candidate_attempts: int = 0
    fallback_used: bool = False
    fallback_trigger_code: str | None = None
    fallback_trigger_stage: str | None = None
    final_locality: str | None = None  # "local" | "cloud"
    engine: str | None = None
    artifact_id: str | None = None
    # Safe benchmark metrics (no prompt/output data)
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    total_tokens: int | None = None
    duration_ms: float | None = None
    # Structured attempt details
    attempt_details: list[AttemptDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict, validating safety before returning."""
        data = asdict(self)
        validate_telemetry_safety(data)
        return data

    def to_json(self) -> str:
        """Serialize to JSON string, validating safety before returning."""
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# AttemptLoopResult — input type for the builder
# ---------------------------------------------------------------------------


@dataclass
class AttemptLoopResult:
    """Result from the candidate attempt loop (input to telemetry builder).

    This captures the outcome of iterating through planner candidates,
    evaluating gates, and selecting a final runtime for inference.
    """

    selected_index: int | None = None
    final_locality: str | None = None  # "local" | "cloud"
    final_engine: str | None = None
    final_artifact_id: str | None = None
    fallback_used: bool = False
    fallback_trigger_code: str | None = None
    fallback_trigger_stage: str | None = None
    candidate_count: int = 0
    attempts: list[AttemptDetail] = field(default_factory=list)
    # Performance metrics from the selected candidate
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    total_tokens: int | None = None
    duration_ms: float | None = None


# ---------------------------------------------------------------------------
# Builder — constructs RouteEventPayload from AttemptLoopResult + context
# ---------------------------------------------------------------------------


def build_route_event(
    *,
    attempt_result: AttemptLoopResult,
    request_id: str | None = None,
    plan_id: str | None = None,
    route_id: str | None = None,
    app_id: str | None = None,
    app_slug: str | None = None,
    capability: str | None = None,
    deployment_id: str | None = None,
    experiment_id: str | None = None,
    variant_id: str | None = None,
    policy: str | None = None,
    planner_source: str | None = None,
) -> RouteEventPayload:
    """Build a RouteEventPayload from an AttemptLoopResult and context.

    Generates route_id and request_id if not provided. All fields are
    validated for safety before the payload is returned.

    Parameters
    ----------
    attempt_result:
        The result from the candidate attempt loop.
    request_id:
        Unique identifier for this inference request. Generated if None.
    plan_id:
        Identifier of the plan that produced the candidates. May be None
        for offline/local-only resolution.
    route_id:
        Correlation ID for this route event. Generated if None.
    app_id:
        Application ID if the request is scoped to a specific app.
    app_slug:
        Application slug if the request is scoped to a specific app.
    capability:
        The capability requested (e.g. "chat", "embeddings", "audio").
    deployment_id:
        Deployment ID if the request is part of a managed deployment.
    experiment_id:
        Experiment ID if this request is part of an A/B experiment.
    variant_id:
        Variant ID within an experiment.
    policy:
        The routing policy used (e.g. "local_first", "cloud_first").
    planner_source:
        Where the plan came from ("server", "cache", "offline").

    Returns
    -------
    RouteEventPayload
        The structured telemetry payload, validated for safety.
    """
    payload = RouteEventPayload(
        route_id=route_id or uuid.uuid4().hex,
        plan_id=plan_id,
        request_id=request_id or uuid.uuid4().hex,
        app_id=app_id,
        app_slug=app_slug,
        capability=capability,
        deployment_id=deployment_id,
        experiment_id=experiment_id,
        variant_id=variant_id,
        policy=policy,
        planner_source=planner_source,
        candidate_attempts=attempt_result.candidate_count,
        fallback_used=attempt_result.fallback_used,
        fallback_trigger_code=attempt_result.fallback_trigger_code,
        fallback_trigger_stage=attempt_result.fallback_trigger_stage,
        final_locality=attempt_result.final_locality,
        engine=attempt_result.final_engine,
        artifact_id=attempt_result.final_artifact_id,
        ttft_ms=attempt_result.ttft_ms,
        tokens_per_second=attempt_result.tokens_per_second,
        total_tokens=attempt_result.total_tokens,
        duration_ms=attempt_result.duration_ms,
        attempt_details=list(attempt_result.attempts),
    )

    # Safety validation before returning
    validate_telemetry_safety(payload.to_dict())

    return payload
