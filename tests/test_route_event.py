from __future__ import annotations

from octomil.runtime.routing.model_ref import parse_model_ref
from octomil.runtime.routing.route_event import (
    FORBIDDEN_TELEMETRY_KEYS,
    CandidateAttemptSummary,
    build_route_event,
    emit_route_event,
    strip_forbidden_keys,
)
from octomil.runtime.telemetry import (
    FORBIDDEN_TELEMETRY_KEYS as TELEMETRY_FORBIDDEN_KEYS,
)


class _Reporter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _enqueue(self, *, name: str, attributes: dict[str, object]) -> None:
        self.calls.append((name, attributes))


def test_build_route_event_uses_canonical_fields() -> None:
    event = build_route_event(
        request_id="req_123",
        capability="responses",
        final_locality="cloud",
        fallback_used=True,
        fallback_trigger_code="runtime_unavailable",
        fallback_trigger_stage="prepare",
        candidate_attempts=[
            CandidateAttemptSummary(
                locality="local",
                engine="mlx",
                model="gemma3-1b",
                success=False,
                failure_reason="runtime missing",
            )
        ],
    )

    payload = event.to_dict()
    assert payload["request_id"] == "req_123"
    assert payload["final_locality"] == "cloud"
    assert payload["selected_locality"] == "cloud"
    assert payload["candidate_attempts"] == 1
    assert payload["attempt_details"][0]["failure_reason"] == "runtime missing"


def test_build_route_event_persists_ttft_and_throughput() -> None:
    """Latency telemetry surfaces from the inference adapter through
    ``RuntimeResponse.ttft_ms`` / ``tokens_per_second`` into the
    persisted route event. The dashboard's ``Avg TTFT`` /
    ``Avg throughput`` cards aggregate over these — pre-fix every
    event posted them as ``None`` and the cards rendered em-dashes.

    Regression test: builds a ``RouteEvent`` with explicit ttft +
    tps values and asserts they survive both the dataclass and the
    ``to_dict()`` serialization."""
    event = build_route_event(
        request_id="req_ttft",
        capability="responses",
        final_locality="local",
        engine="mlx-lm",
        ttft_ms=42.5,
        tokens_per_second=128.3,
    )

    assert event.ttft_ms == 42.5
    assert event.tokens_per_second == 128.3

    payload = event.to_dict()
    assert payload["ttft_ms"] == 42.5
    assert payload["tokens_per_second"] == 128.3


def test_build_route_event_omits_ttft_when_unknown() -> None:
    """When the backend can't measure latency (cloud routes that
    don't surface per-token timings, or local engines whose
    ``InferenceMetrics`` returned 0.0), the SDK passes ``None`` and
    the event keeps the field unset. Pre-fix this was the only
    behavior; post-fix we want it to remain valid for partial
    coverage during rollout."""
    event = build_route_event(
        request_id="req_no_ttft",
        capability="responses",
        final_locality="local",
        engine="mlx-lm",
    )

    assert event.ttft_ms is None
    assert event.tokens_per_second is None


def test_emit_route_event_uses_route_decision_and_json_attempt_details() -> None:
    reporter = _Reporter()
    event = build_route_event(
        request_id="req_abc",
        capability="responses",
        selected_locality="local",
        candidate_attempts=[
            CandidateAttemptSummary(locality="local", success=True),
        ],
    )

    emit_route_event(event, reporter)

    assert len(reporter.calls) == 1
    name, attributes = reporter.calls[0]
    assert name == "route.decision"
    assert "route.id" in attributes
    assert "route.route_id" not in attributes
    assert attributes["route.final_locality"] == "local"
    assert attributes["route.candidate_attempts"] == 1
    assert "route.attempt_details" in attributes


def test_parse_model_ref_uses_canonical_kinds() -> None:
    cases = {
        "gemma3-1b": "model",
        "@app/translator/chat": "app",
        "@capability/embeddings": "capability",
        "deploy_abc123": "deployment",
        "exp_v1/variant_a": "experiment",
        "alias:prod-chat": "alias",
        "": "default",
        "@bad/ref": "unknown",
        "https://example.com/model.gguf": "unknown",
    }

    for model, expected_kind in cases.items():
        assert parse_model_ref(model).kind == expected_kind

    assert parse_model_ref("deploy_abc123").deployment_id == "deploy_abc123"


def test_route_event_forbidden_keys_match_telemetry_contract() -> None:
    """The route-event sanitizer must enforce the same forbidden set as
    octomil.runtime.telemetry. Otherwise a payload routed through
    emit_route_event() can carry secrets that the central contract
    would have stripped."""
    assert FORBIDDEN_TELEMETRY_KEYS == TELEMETRY_FORBIDDEN_KEYS


def test_strip_forbidden_keys_drops_auth_case_insensitively() -> None:
    """Authorization (capital A), API_KEY (uppercase) — both must be
    stripped. The earlier case-sensitive impl let these through, which
    was the privacy gap PR 7 missed in route_event.py."""
    payload = {
        "Authorization": "Bearer sk-secret",
        "API_KEY": "sk-secret",
        "metadata": {
            "Token": "abc",
            "password": "hunter2",
            "engine": "llamacpp",
        },
        "engine": "mlx",
    }
    cleaned, stripped = strip_forbidden_keys(payload)
    assert "Authorization" not in cleaned
    assert "API_KEY" not in cleaned
    assert "Token" not in cleaned["metadata"]
    assert "password" not in cleaned["metadata"]
    assert cleaned["engine"] == "mlx"
    assert cleaned["metadata"]["engine"] == "llamacpp"
    # Stripped names preserve original case for logging.
    assert "Authorization" in stripped
    assert "API_KEY" in stripped


def test_strip_forbidden_keys_recurses_through_lists() -> None:
    payload = {
        "candidates": [
            {"engine": "mlx", "Authorization": "Bearer leak"},
            {"engine": "llamacpp", "secret": "..."},
        ]
    }
    cleaned, _ = strip_forbidden_keys(payload)
    for cand in cleaned["candidates"]:
        assert "Authorization" not in cand
        assert "secret" not in cand
        assert "engine" in cand
