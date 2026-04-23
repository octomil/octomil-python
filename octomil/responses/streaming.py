"""Streaming adaptation — stream event generation and first-token fallback.

Handles the mechanics of converting runtime stream chunks into
ResponseStreamEvents and assembling the final DoneEvent.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from octomil.execution.route_metadata_mapper import (
    _route_metadata_from_selection,
)
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.router import (
    LOCALITY_ON_DEVICE,
    RouterModelRuntime,
)
from octomil.runtime.core.types import (
    RuntimeRequest,
    RuntimeUsage,
)
from octomil.runtime.routing.attempt_runner import (
    AttemptLoopResult,
    AttemptStage,
    AttemptStatus,
    CandidateAttemptRunner,
    FallbackTrigger,
    RouteAttempt,
)

from .response_builder import ToolCallBuffer, generate_id
from .route_attachment import _locality_for_candidate
from .types import (
    DoneEvent,
    OutputItem,
    Response,
    ResponseRequest,
    ResponseStreamEvent,
    ResponseToolCall,
    ResponseUsage,
    TextDeltaEvent,
    TextOutput,
    ToolCallDeltaEvent,
    ToolCallOutput,
)

# ---------------------------------------------------------------------------
# Direct streaming (no attempt runner)
# ---------------------------------------------------------------------------


async def stream_direct(
    request: ResponseRequest,
    runtime: Any,
    runtime_request: RuntimeRequest,
    *,
    routing_policy: Optional[RoutingPolicy],
    locality: str,
    route: Any,
) -> AsyncIterator[ResponseStreamEvent]:
    """Stream a response directly from a runtime (no planner-driven fallback)."""
    response_id = generate_id()
    text_parts: list[str] = []
    tool_call_buffers: dict[int, ToolCallBuffer] = {}
    last_usage: Optional[RuntimeUsage] = None

    if isinstance(runtime, RouterModelRuntime) and routing_policy is not None:
        stream_iter = runtime.stream(runtime_request, policy=routing_policy)
    else:
        stream_iter = runtime.stream(runtime_request)
    async for chunk in stream_iter:
        if chunk.text is not None:
            text_parts.append(chunk.text)
            yield TextDeltaEvent(delta=chunk.text)
        if chunk.tool_call_delta is not None:
            delta = chunk.tool_call_delta
            buffer = tool_call_buffers.setdefault(delta.index, ToolCallBuffer())
            if delta.id is not None:
                buffer.id = delta.id
            if delta.name is not None:
                buffer.name = delta.name
            if delta.arguments_delta is not None:
                buffer.arguments += delta.arguments_delta
            yield ToolCallDeltaEvent(
                index=delta.index,
                id=delta.id,
                name=delta.name,
                arguments_delta=delta.arguments_delta,
            )
        if chunk.usage is not None:
            last_usage = chunk.usage

    output = _assemble_output(text_parts, tool_call_buffers)
    finish_reason = "tool_calls" if tool_call_buffers else "stop"
    usage = _map_usage(last_usage)

    yield DoneEvent(
        response=Response(
            id=response_id,
            model=request.model,
            output=output,
            finish_reason=finish_reason,
            usage=usage,
            locality=locality,
            route=route,
        )
    )


# ---------------------------------------------------------------------------
# Streaming with attempt runner (first-token fallback)
# ---------------------------------------------------------------------------


async def stream_with_attempt_runner(
    request: ResponseRequest,
    model_id: str,
    runtime_request: RuntimeRequest,
    candidates: list[dict[str, Any]],
    runner: CandidateAttemptRunner,
    selection: Any,
    *,
    resolve_runtime_for_candidate: Any,
    telemetry: Optional[object],
) -> AsyncIterator[ResponseStreamEvent]:
    """Stream with CandidateAttemptRunner fallback semantics.

    - If streaming fails BEFORE first token: fall back to next candidate
    - If streaming fails AFTER first token: raise (do not fall back)
    """
    from .dispatch import _runtime_model_for_selection

    response_id = generate_id()
    text_parts: list[str] = []
    tool_call_buffers: dict[int, ToolCallBuffer] = {}
    last_usage: Optional[RuntimeUsage] = None
    selected_locality = LOCALITY_ON_DEVICE
    is_fallback = False
    last_error: Optional[Exception] = None
    runtime_model_id = _runtime_model_for_selection(selection, model_id)
    stream_attempts: list[RouteAttempt] = []
    selected_attempt: RouteAttempt | None = None
    fallback_trigger: FallbackTrigger | None = None
    from_attempt: int | None = None

    for idx, candidate in enumerate(candidates):
        first_token_emitted = False
        readiness = CandidateAttemptRunner(
            fallback_allowed=False,
            streaming=True,
        ).run([candidate])
        ready_attempt = readiness.attempts[0] if readiness.attempts else None
        if ready_attempt is not None:
            ready_attempt.index = idx
        if not readiness.succeeded:
            if ready_attempt is not None:
                stream_attempts.append(ready_attempt)
                if fallback_trigger is None:
                    fallback_trigger = FallbackTrigger(
                        code=ready_attempt.reason_code,
                        stage=ready_attempt.stage.value,
                        message=ready_attempt.reason_message,
                    )
                    from_attempt = idx
            if runner.should_fallback_after_inference_error(first_token_emitted=False) and idx < len(candidates) - 1:
                continue
            raise RuntimeError(ready_attempt.reason_message if ready_attempt else "No runtime available")

        try:
            runtime = resolve_runtime_for_candidate(runtime_model_id, candidate)
            policy = RoutingPolicy.cloud_only() if candidate.get("locality") == "cloud" else RoutingPolicy.local_only()
            if isinstance(runtime, RouterModelRuntime):
                stream_iter = runtime.stream(runtime_request, policy=policy)
            else:
                stream_iter = runtime.stream(runtime_request)

            async for chunk in stream_iter:
                if chunk.text is not None:
                    first_token_emitted = True
                    text_parts.append(chunk.text)
                    yield TextDeltaEvent(delta=chunk.text)
                if chunk.tool_call_delta is not None:
                    first_token_emitted = True
                    delta = chunk.tool_call_delta
                    buffer = tool_call_buffers.setdefault(delta.index, ToolCallBuffer())
                    if delta.id is not None:
                        buffer.id = delta.id
                    if delta.name is not None:
                        buffer.name = delta.name
                    if delta.arguments_delta is not None:
                        buffer.arguments += delta.arguments_delta
                    yield ToolCallDeltaEvent(
                        index=delta.index,
                        id=delta.id,
                        name=delta.name,
                        arguments_delta=delta.arguments_delta,
                    )
                if chunk.usage is not None:
                    last_usage = chunk.usage

            # Post-inference: evaluate output_quality gates
            assert ready_attempt is not None
            quality_failure = runner._evaluate_output_quality_gates(
                candidate,
                "".join(text_parts),  # assembled text as response
                ready_attempt,
                idx,
                first_token_emitted=first_token_emitted,
            )
            if quality_failure is not None and not first_token_emitted:
                # Buffered streaming: output not visible, can fallback
                stream_attempts.append(quality_failure)
                if (
                    runner.should_fallback_after_inference_error(first_token_emitted=False)
                    and idx < len(candidates) - 1
                ):
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
                    text_parts.clear()
                    tool_call_buffers.clear()
                    last_usage = None
                    continue
                raise RuntimeError(quality_failure.reason_message)

            selected_locality = _locality_for_candidate(candidate)
            is_fallback = idx > 0
            selected_attempt = ready_attempt
            if selected_attempt is not None:
                selected_attempt.reason_message = "stream completed"
                stream_attempts.append(selected_attempt)
            break

        except Exception as exc:
            last_error = exc
            reason_code = (
                "inference_error_after_first_token" if first_token_emitted else "inference_error_before_first_token"
            )
            failed_attempt = RouteAttempt(
                index=idx,
                locality=candidate.get("locality", "local"),
                mode=CandidateAttemptRunner._mode_for_candidate(candidate),
                engine=candidate.get("engine"),
                artifact=ready_attempt.artifact if ready_attempt is not None else None,
                status=AttemptStatus.FAILED,
                stage=AttemptStage.INFERENCE,
                gate_results=ready_attempt.gate_results if ready_attempt is not None else [],
                reason_code=reason_code,
                reason_message=str(exc) or reason_code,
            )
            stream_attempts.append(failed_attempt)
            if (
                runner.should_fallback_after_inference_error(first_token_emitted=first_token_emitted)
                and idx < len(candidates) - 1
            ):
                if fallback_trigger is None:
                    fallback_trigger = FallbackTrigger(
                        code=reason_code,
                        stage=AttemptStage.INFERENCE.value,
                        message=failed_attempt.reason_message,
                    )
                    from_attempt = idx
                text_parts.clear()
                tool_call_buffers.clear()
                last_usage = None
                continue
            raise
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError("No runtime available")

    route = _route_metadata_from_selection(
        selection,
        selected_locality,
        is_fallback,
        model_name=model_id,
        capability="chat",
        attempt_loop=AttemptLoopResult(
            selected_attempt=selected_attempt,
            attempts=stream_attempts,
            fallback_used=is_fallback,
            fallback_trigger=fallback_trigger if is_fallback else None,
            from_attempt=from_attempt if is_fallback else None,
            to_attempt=selected_attempt.index if is_fallback and selected_attempt is not None else None,
        ),
    )

    if is_fallback and telemetry is not None:
        try:
            telemetry.report_fallback_cloud(  # type: ignore[attr-defined]
                model_id=model_id,
                fallback_reason="local_unavailable",
            )
        except Exception:
            pass

    output = _assemble_output(text_parts, tool_call_buffers)
    finish_reason = "tool_calls" if tool_call_buffers else "stop"
    usage = _map_usage(last_usage)

    yield DoneEvent(
        response=Response(
            id=response_id,
            model=request.model,
            output=output,
            finish_reason=finish_reason,
            usage=usage,
            locality=selected_locality,
            route=route,
        )
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _assemble_output(
    text_parts: list[str],
    tool_call_buffers: dict[int, ToolCallBuffer],
) -> list[OutputItem]:
    """Assemble final output items from accumulated stream parts."""
    output: list[OutputItem] = []
    full_text = "".join(text_parts)
    if full_text:
        output.append(TextOutput(text=full_text))
    for idx in sorted(tool_call_buffers):
        buf = tool_call_buffers[idx]
        output.append(
            ToolCallOutput(
                tool_call=ResponseToolCall(
                    id=buf.id or generate_id(),
                    name=buf.name or "",
                    arguments=buf.arguments,
                )
            )
        )
    return output


def _map_usage(
    last_usage: Optional[RuntimeUsage],
) -> Optional[ResponseUsage]:
    """Map RuntimeUsage to ResponseUsage."""
    if last_usage is None:
        return None
    return ResponseUsage(
        prompt_tokens=last_usage.prompt_tokens,
        completion_tokens=last_usage.completion_tokens,
        total_tokens=last_usage.total_tokens,
    )
