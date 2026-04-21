from __future__ import annotations

from octomil.runtime.routing.route_event import (
    CandidateAttemptSummary,
    build_route_event,
    emit_route_event,
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
    assert payload["candidate_attempts"] == 1
    assert "selected_locality" not in payload
    assert payload["attempt_details"][0]["failure_reason"] == "runtime missing"


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
    assert attributes["route.final_locality"] == "local"
    assert attributes["route.candidate_attempts"] == 1
    assert "route.attempt_details" in attributes
