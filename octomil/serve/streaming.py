"""SSE streaming response generators."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Optional

from .types import GenerationRequest, StreamableState

__all__ = ["_stream_response", "_queued_stream_response"]


async def _stream_response(
    state: StreamableState,
    request: GenerationRequest,
    req_id: str,
    session_id: str = "",
    is_reasoning: bool = False,
) -> AsyncIterator[str]:
    """Yield SSE chunks in OpenAI streaming format."""
    assert state.backend is not None

    from .thinking import ThinkingStreamParser

    _reporter = state.reporter
    model_version = "latest"
    chunk_index = 0
    stream_start = time.monotonic()
    first_chunk_time: Optional[float] = None
    prev_chunk_time = stream_start
    failed = False
    parser = ThinkingStreamParser() if is_reasoning else None

    try:
        async for chunk in state.backend.generate_stream(request):
            now = time.monotonic()
            chunk_latency_ms = (now - prev_chunk_time) * 1000
            prev_chunk_time = now

            if first_chunk_time is None and chunk.text:
                first_chunk_time = now

            # Report each chunk (best-effort)
            if _reporter is not None and chunk.text:
                try:
                    ttfc = (
                        (first_chunk_time - stream_start) * 1000
                        if first_chunk_time is not None and chunk_index == 0
                        else None
                    )
                    _reporter.report_inference_chunk(
                        session_id=session_id,
                        model_id=request.model,
                        version=model_version,
                        chunk_index=chunk_index,
                        ttfc_ms=ttfc,
                        chunk_latency_ms=chunk_latency_ms,
                    )
                except Exception:
                    pass
                chunk_index += 1

            if parser and chunk.text:
                for field, text in parser.feed(chunk.text):
                    delta = {field: text}
                    data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                if chunk.finish_reason:
                    # Flush parser and emit finish
                    for field, text in parser.flush():
                        delta = {field: text}
                        data = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": chunk.finish_reason}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            else:
                data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.text} if chunk.text else {},
                            "finish_reason": chunk.finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
    except Exception:
        failed = True
        if _reporter is not None:
            try:
                _reporter.report_inference_failed(
                    session_id=session_id,
                    model_id=request.model,
                    version=model_version,
                )
            except Exception:
                pass
        raise

    # Flush any remaining buffered text from the parser
    if parser:
        for field, text in parser.flush():
            delta = {field: text}
            data = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"

    # Report inference_completed (best-effort)
    if _reporter is not None and not failed:
        try:
            total_duration_ms = (time.monotonic() - stream_start) * 1000
            ttfc_ms = (first_chunk_time - stream_start) * 1000 if first_chunk_time is not None else 0.0
            throughput = chunk_index / (total_duration_ms / 1000) if total_duration_ms > 0 else 0.0
            _reporter.report_inference_completed(
                session_id=session_id,
                model_id=request.model,
                version=model_version,
                total_chunks=chunk_index,
                total_duration_ms=total_duration_ms,
                ttfc_ms=ttfc_ms,
                throughput=throughput,
                attention_backend=(state.backend.attention_backend if state.backend is not None else None),
            )
        except Exception:
            pass


async def _queued_stream_response(
    state: Any,
    request: GenerationRequest,
    req_id: str,
    session_id: str,
    queue: Any,
    is_reasoning: bool = False,
) -> AsyncIterator[str]:
    """Yield SSE chunks after waiting in the request queue.

    Submits a streaming request to the queue.  Once the request reaches
    the front, chunks are forwarded as SSE events.  Queue errors (full,
    timeout) are surfaced as SSE error events so the client gets useful
    feedback even on a streaming connection.
    """
    from ..batch import QueueFullError, QueueTimeoutError
    from .thinking import ThinkingStreamParser

    assert state.backend is not None

    _reporter = state.reporter
    model_version = "latest"
    stream_start = time.monotonic()
    first_chunk_time: Optional[float] = None
    prev_chunk_time = stream_start
    parser = ThinkingStreamParser() if is_reasoning else None

    try:
        chunk_iter = queue.submit_generate_stream(request, state.backend.generate_stream)
        # Build SSE events from the chunk iterator (same format as _stream_response)
        chunk_index = 0
        async for chunk in chunk_iter:
            now = time.monotonic()
            chunk_latency_ms = (now - prev_chunk_time) * 1000
            prev_chunk_time = now

            if first_chunk_time is None and chunk.text:
                first_chunk_time = now

            # Report each chunk (best-effort)
            if _reporter is not None and chunk.text:
                try:
                    ttfc = (
                        (first_chunk_time - stream_start) * 1000
                        if first_chunk_time is not None and chunk_index == 0
                        else None
                    )
                    _reporter.report_inference_chunk(
                        session_id=session_id,
                        model_id=request.model,
                        version=model_version,
                        chunk_index=chunk_index,
                        ttfc_ms=ttfc,
                        chunk_latency_ms=chunk_latency_ms,
                    )
                except Exception:
                    pass

            if parser and chunk.text:
                for field, text in parser.feed(chunk.text):
                    delta = {field: text}
                    data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                if chunk.finish_reason:
                    for field, text in parser.flush():
                        delta = {field: text}
                        data = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": chunk.finish_reason}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            else:
                data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.text} if chunk.text else {},
                            "finish_reason": chunk.finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
            chunk_index += 1

        # Flush parser at end of stream
        if parser:
            for field, text in parser.flush():
                delta = {field: text}
                data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

        # Report inference_completed (best-effort)
        if _reporter is not None:
            try:
                total_duration_ms = (time.monotonic() - stream_start) * 1000
                ttfc_ms = (first_chunk_time - stream_start) * 1000 if first_chunk_time is not None else 0.0
                throughput = chunk_index / (total_duration_ms / 1000) if total_duration_ms > 0 else 0.0
                _reporter.report_inference_completed(
                    session_id=session_id,
                    model_id=request.model,
                    version=model_version,
                    total_chunks=chunk_index,
                    total_duration_ms=total_duration_ms,
                    ttfc_ms=ttfc_ms,
                    throughput=throughput,
                )
            except Exception:
                pass
    except QueueFullError:
        error_data = {
            "error": {
                "message": "Server busy -- request queue full. Try again later.",
                "type": "server_error",
                "code": 503,
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    except QueueTimeoutError:
        error_data = {
            "error": {
                "message": "Request timed out waiting in queue.",
                "type": "server_error",
                "code": 504,
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
