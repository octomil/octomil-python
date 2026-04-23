"""Canonical route event emission for SDK telemetry correlation.

Emits a privacy-safe route event after every routing decision completes.
The event carries correlation IDs (route_id, request_id), context identifiers
(app, deployment, experiment), locality outcome, fallback metadata, and
candidate attempt summaries — but NEVER user content.

Public API:
- ``RouteEvent`` dataclass — the canonical event shape
- ``CandidateAttemptSummary`` dataclass — per-candidate outcome
- ``FORBIDDEN_TELEMETRY_KEYS`` — keys that must never appear in route events
- ``strip_forbidden_keys()`` — recursively remove forbidden keys from a dict
- ``emit_route_event()`` — emit a route event via a TelemetryReporter
- ``build_route_event()`` — construct a RouteEvent without emitting
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Privacy: forbidden telemetry keys
# ---------------------------------------------------------------------------

FORBIDDEN_TELEMETRY_KEYS: frozenset[str] = frozenset(
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


def strip_forbidden_keys(data: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Recursively strip forbidden keys from a dictionary.

    Returns a tuple of (cleaned_data, set_of_stripped_key_names).
    """
    stripped: set[str] = set()

    def _scrub(node: Any) -> Any:
        if isinstance(node, dict):
            clean: dict[str, Any] = {}
            for key, value in node.items():
                if key in FORBIDDEN_TELEMETRY_KEYS:
                    stripped.add(key)
                    continue
                clean[key] = _scrub(value)
            return clean
        if isinstance(node, list):
            return [_scrub(item) for item in node]
        return node

    return _scrub(data), stripped


# ---------------------------------------------------------------------------
# Candidate attempt summary
# ---------------------------------------------------------------------------


@dataclass
class CandidateAttemptSummary:
    """Summary of a single candidate attempt within a route decision."""

    locality: str  # "on_device" | "cloud"
    engine: str = ""
    model: str = ""
    success: bool = False
    failure_reason: str = ""
    duration_ms: float = 0.0
    ttft_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Route event
# ---------------------------------------------------------------------------


@dataclass
class RouteEvent:
    """Canonical route event emitted after a routing decision completes.

    All fields are privacy-safe — no user content is included.
    """

    # Correlation
    route_id: str = field(default_factory=lambda: f"route_{uuid.uuid4().hex[:16]}")
    request_id: str = ""
    plan_id: Optional[str] = None

    # Context identifiers
    app_id: Optional[str] = None
    app_slug: Optional[str] = None
    deployment_id: Optional[str] = None
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None

    # Policy / planning
    capability: Optional[str] = None
    policy: Optional[str] = None
    planner_source: Optional[str] = None  # canonical: "server" | "cache" | "offline"

    # Model ref metadata
    model_ref: Optional[str] = None
    model_ref_kind: Optional[str] = None  # model|app|capability|deployment|experiment|alias|default|unknown

    # Outcome
    selected_locality: str = ""  # "local" | "cloud"
    final_locality: str = ""  # backward-compat alias for selected_locality
    final_mode: str = ""  # "sdk_runtime" | "hosted_gateway" | "external_endpoint"
    fallback_used: bool = False
    fallback_trigger_code: Optional[str] = None
    fallback_trigger_stage: Optional[str] = None  # "prepare" | "verify" | "gate" | "inference"

    # Candidate attempts
    candidate_attempts: int = 0
    attempt_details: list[CandidateAttemptSummary] = field(default_factory=list)

    # Performance (final successful candidate)
    engine: Optional[str] = None
    artifact_id: Optional[str] = None
    cache_status: Optional[str] = None  # "hit" | "miss" | "not_applicable"
    ttft_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    total_tokens: Optional[int] = None
    duration_ms: Optional[float] = None

    # Output quality evaluation
    quality_evaluator_name: Optional[str] = None
    quality_score: Optional[float] = None
    quality_reason_code: Optional[str] = None
    advisory_failures: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for telemetry upload.

        Applies privacy stripping as a safety net.
        """
        raw = asdict(self)
        # Ensure no forbidden keys leaked in
        clean, stripped = strip_forbidden_keys(raw)
        if stripped:
            logger.warning(
                "Route event contained forbidden keys (stripped): %s",
                ", ".join(sorted(stripped)),
            )
        return clean


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_route_event(
    *,
    request_id: str = "",
    plan_id: Optional[str] = None,
    app_id: Optional[str] = None,
    app_slug: Optional[str] = None,
    deployment_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    variant_id: Optional[str] = None,
    capability: Optional[str] = None,
    policy: Optional[str] = None,
    planner_source: Optional[str] = None,
    model_ref: Optional[str] = None,
    model_ref_kind: Optional[str] = None,
    selected_locality: str = "",
    final_locality: Optional[str] = None,
    final_mode: str = "",
    fallback_used: bool = False,
    fallback_trigger_code: Optional[str] = None,
    fallback_trigger_stage: Optional[str] = None,
    candidate_attempts: Optional[list[CandidateAttemptSummary]] = None,
    engine: Optional[str] = None,
    artifact_id: Optional[str] = None,
    cache_status: Optional[str] = None,
    ttft_ms: Optional[float] = None,
    tokens_per_second: Optional[float] = None,
    total_tokens: Optional[int] = None,
    duration_ms: Optional[float] = None,
    quality_evaluator_name: Optional[str] = None,
    quality_score: Optional[float] = None,
    quality_reason_code: Optional[str] = None,
    advisory_failures: Optional[list[dict[str, Any]]] = None,
) -> RouteEvent:
    """Construct a RouteEvent with a fresh route_id."""
    from octomil.runtime.planner.schemas import normalize_planner_source

    attempt_details = candidate_attempts or []
    locality = selected_locality or final_locality or ""
    # Normalize planner_source to canonical enum: server | cache | offline
    if planner_source is not None:
        planner_source = normalize_planner_source(planner_source)
    return RouteEvent(
        route_id=f"route_{uuid.uuid4().hex[:16]}",
        request_id=request_id or f"req_{uuid.uuid4().hex[:12]}",
        plan_id=plan_id,
        app_id=app_id,
        app_slug=app_slug,
        deployment_id=deployment_id,
        experiment_id=experiment_id,
        variant_id=variant_id,
        capability=capability,
        policy=policy,
        planner_source=planner_source,
        model_ref=model_ref,
        model_ref_kind=model_ref_kind,
        selected_locality=locality,
        final_locality=locality,
        final_mode=final_mode,
        fallback_used=fallback_used,
        fallback_trigger_code=fallback_trigger_code,
        fallback_trigger_stage=fallback_trigger_stage,
        candidate_attempts=len(attempt_details),
        attempt_details=attempt_details,
        engine=engine,
        artifact_id=artifact_id,
        cache_status=cache_status,
        ttft_ms=ttft_ms,
        tokens_per_second=tokens_per_second,
        total_tokens=total_tokens,
        duration_ms=duration_ms,
        quality_evaluator_name=quality_evaluator_name,
        quality_score=quality_score,
        quality_reason_code=quality_reason_code,
        advisory_failures=advisory_failures,
    )


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


def emit_route_event(
    event: RouteEvent,
    reporter: Any,
) -> None:
    """Emit a route event via the given TelemetryReporter.

    This is a best-effort operation — failures are logged but never raised.

    Parameters
    ----------
    event:
        The RouteEvent to emit.
    reporter:
        A TelemetryReporter instance (or any object with an ``_enqueue`` method).
    """
    if reporter is None:
        return

    try:
        payload = event.to_dict()
        # Use the reporter's internal _enqueue to send as a structured event
        reporter._enqueue(
            name="route.decision",
            attributes=_flatten_for_telemetry(payload),
        )
        logger.debug(
            "Route event emitted: route_id=%s request_id=%s locality=%s fallback=%s",
            event.route_id,
            event.request_id,
            event.final_locality,
            event.fallback_used,
        )
    except Exception:
        logger.debug("Failed to emit route event", exc_info=True)


def _flatten_for_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested structures for OTLP attribute format.

    Attempt details are serialized as a JSON-encoded list summary rather
    than nested OTLP attributes.
    """
    import json

    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "attempt_details" and isinstance(value, list):
            flat["route.attempt_details"] = json.dumps(value)
        elif value is None:
            continue
        elif isinstance(value, (str, int, float, bool)):
            flat[f"route.{key}"] = value
        else:
            flat[f"route.{key}"] = str(value)
    return flat
