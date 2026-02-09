"""
Streaming inference client with per-chunk timing metrics.

Provides ``StreamingInferenceClient`` which wraps modality-specific backends
(mlx-lm for text, diffusers for images, etc.) and yields
``InferenceChunk`` objects with timing instrumentation.  On stream completion
the aggregated ``StreamingInferenceResult`` is automatically reported to the
server via ``POST /api/v1/inference/events``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Optional

if TYPE_CHECKING:
    from .api_client import _ApiClient


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class InferenceChunk:
    """A single chunk emitted during streaming inference."""

    index: int
    data: bytes
    modality: Modality
    timestamp: float  # epoch seconds
    latency_ms: float  # ms since previous chunk (or session start)


@dataclass
class StreamingInferenceResult:
    """Aggregated metrics for a completed streaming inference session."""

    session_id: str
    modality: Modality
    ttfc_ms: float
    avg_chunk_latency_ms: float
    total_chunks: int
    total_duration_ms: float
    throughput: float  # chunks per second

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "modality": self.modality.value,
            "ttfc_ms": self.ttfc_ms,
            "avg_chunk_latency_ms": self.avg_chunk_latency_ms,
            "total_chunks": self.total_chunks,
            "total_duration_ms": self.total_duration_ms,
            "throughput": self.throughput,
        }


class StreamingInferenceClient:
    """
    Client for streaming inference with automatic timing instrumentation.

    Usage::

        client = StreamingInferenceClient(api, device_id="...")
        for chunk in client.generate("my-model", prompt="Hello", modality=Modality.TEXT):
            print(chunk.data.decode(), end="", flush=True)
    """

    def __init__(
        self,
        api: _ApiClient,
        device_id: str,
        org_id: str = "default",
    ):
        self.api = api
        self.device_id = device_id
        self.org_id = org_id
        self.last_result: Optional[StreamingInferenceResult] = None

    # ------------------------------------------------------------------
    # Session helpers (shared by sync and async paths)
    # ------------------------------------------------------------------

    def _init_session(
        self, model_id: str, version: str, modality: Modality, resolved_input: Any,
    ) -> dict[str, Any]:
        """Create session tracking state and report ``generation_started``."""
        state: dict[str, Any] = {
            "session_id": uuid.uuid4().hex,
            "session_start": time.monotonic(),
            "first_chunk_time": None,
            "previous_time": time.monotonic(),
            "latencies": [],
            "chunk_count": 0,
            "resolved_input": resolved_input,
        }
        self._report_event(
            model_id=model_id,
            version=version,
            modality=modality,
            session_id=state["session_id"],
            event_type="generation_started",
        )
        return state

    def _make_chunk(self, raw_data: Any, modality: Modality, state: dict[str, Any]) -> InferenceChunk:
        """Process a raw backend datum into an :class:`InferenceChunk`, updating *state* in-place."""
        now = time.monotonic()
        if state["first_chunk_time"] is None:
            state["first_chunk_time"] = now

        latency_ms = (now - state["previous_time"]) * 1000
        state["previous_time"] = now
        state["latencies"].append(latency_ms)
        state["chunk_count"] += 1

        if isinstance(raw_data, bytes):
            chunk_data = raw_data
        elif isinstance(raw_data, str):
            chunk_data = raw_data.encode("utf-8")
        else:
            chunk_data = bytes(raw_data)

        return InferenceChunk(
            index=state["chunk_count"] - 1,
            data=chunk_data,
            modality=modality,
            timestamp=time.time(),
            latency_ms=latency_ms,
        )

    def _report_failure(self, model_id: str, version: str, modality: Modality, session_id: str) -> None:
        """Report ``generation_failed`` event."""
        self._report_event(
            model_id=model_id,
            version=version,
            modality=modality,
            session_id=session_id,
            event_type="generation_failed",
        )

    def _finalize_session(self, model_id: str, version: str, modality: Modality, state: dict[str, Any]) -> None:
        """Compute result metrics, store ``last_result``, and report ``generation_completed``."""
        session_end = time.monotonic()
        total_duration_ms = (session_end - state["session_start"]) * 1000
        ttfc_ms = ((state["first_chunk_time"] or session_end) - state["session_start"]) * 1000
        latencies = state["latencies"]
        chunk_count = state["chunk_count"]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        throughput = chunk_count / (total_duration_ms / 1000) if total_duration_ms > 0 else 0.0

        result = StreamingInferenceResult(
            session_id=state["session_id"],
            modality=modality,
            ttfc_ms=ttfc_ms,
            avg_chunk_latency_ms=avg_latency,
            total_chunks=chunk_count,
            total_duration_ms=total_duration_ms,
            throughput=throughput,
        )
        self.last_result = result

        self._report_event(
            model_id=model_id,
            version=version,
            modality=modality,
            session_id=state["session_id"],
            event_type="generation_completed",
            metrics={
                "ttfc_ms": ttfc_ms,
                "total_chunks": chunk_count,
                "total_duration_ms": total_duration_ms,
                "throughput": throughput,
            },
        )

    # ------------------------------------------------------------------
    # Synchronous streaming
    # ------------------------------------------------------------------

    def generate(
        self,
        model_id: str,
        input: Any = None,
        *,
        prompt: Optional[str] = None,
        modality: Modality = Modality.TEXT,
        version: str = "latest",
        max_tokens: int = 512,
    ) -> Iterator[InferenceChunk]:
        """
        Stream inference chunks with automatic timing.

        Keyword arguments are forwarded to the modality-specific backend.
        After the iterator is exhausted, ``self.last_result`` contains the
        aggregated metrics and the result has been reported to the server.
        """
        resolved_input = prompt if prompt is not None else input
        state = self._init_session(model_id, version, modality, resolved_input)

        try:
            for raw_data in self._backend_generate(
                modality=modality, resolved_input=resolved_input, max_tokens=max_tokens,
            ):
                yield self._make_chunk(raw_data, modality, state)
        except Exception:
            self._report_failure(model_id, version, modality, state["session_id"])
            raise

        self._finalize_session(model_id, version, modality, state)

    # ------------------------------------------------------------------
    # Async streaming
    # ------------------------------------------------------------------

    async def generate_async(
        self,
        model_id: str,
        input: Any = None,
        *,
        prompt: Optional[str] = None,
        modality: Modality = Modality.TEXT,
        version: str = "latest",
        max_tokens: int = 512,
    ) -> AsyncIterator[InferenceChunk]:
        """Async version of :meth:`generate`."""
        resolved_input = prompt if prompt is not None else input
        state = self._init_session(model_id, version, modality, resolved_input)

        try:
            async for raw_data in self._backend_generate_async(
                modality=modality, resolved_input=resolved_input, max_tokens=max_tokens,
            ):
                yield self._make_chunk(raw_data, modality, state)
        except Exception:
            self._report_failure(model_id, version, modality, state["session_id"])
            raise

        self._finalize_session(model_id, version, modality, state)

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _backend_generate(
        self, modality: Modality, resolved_input: Any, max_tokens: int,
    ) -> Iterator[bytes]:
        """Dispatch to modality-specific backend (sync)."""
        if modality == Modality.TEXT:
            yield from self._generate_text(resolved_input, max_tokens)
        elif modality == Modality.IMAGE:
            yield from self._generate_image(resolved_input)
        elif modality == Modality.AUDIO:
            yield from self._generate_audio(resolved_input)
        elif modality == Modality.VIDEO:
            yield from self._generate_video(resolved_input)

    async def _backend_generate_async(
        self, modality: Modality, resolved_input: Any, max_tokens: int,
    ) -> AsyncIterator[bytes]:
        """Dispatch to modality-specific backend (async)."""
        import asyncio

        # Wrap sync generator for now; backends can override with native async
        for chunk in self._backend_generate(modality, resolved_input, max_tokens):
            yield chunk
            await asyncio.sleep(0)  # yield to event loop

    # ------------------------------------------------------------------
    # Text generation (mlx-lm when available)
    # ------------------------------------------------------------------

    def _generate_text(self, prompt: Any, max_tokens: int) -> Iterator[bytes]:
        prompt_str = str(prompt) if prompt is not None else ""

        try:
            import mlx_lm  # type: ignore

            model, tokenizer = mlx_lm.load("mlx-community/phi-3-mini")
            for token in mlx_lm.stream_generate(model, tokenizer, prompt=prompt_str, max_tokens=max_tokens):
                yield token.encode("utf-8")
            return
        except ImportError:
            pass

        # Placeholder when mlx-lm is not available
        response = f"Generated response for: {prompt_str[:30]}..."
        for word in response.split():
            yield (word + " ").encode("utf-8")

    # ------------------------------------------------------------------
    # Image generation (diffusers when available)
    # ------------------------------------------------------------------

    def _generate_image(self, _input: Any) -> Iterator[bytes]:
        # Placeholder — each yield represents one denoising step
        for step in range(20):
            yield bytes([step] * 64)

    # ------------------------------------------------------------------
    # Audio generation
    # ------------------------------------------------------------------

    def _generate_audio(self, _input: Any) -> Iterator[bytes]:
        for _ in range(80):
            yield bytes(1024 * 2)

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    def _generate_video(self, _input: Any) -> Iterator[bytes]:
        for frame in range(30):
            yield bytes([frame % 256] * 1024)

    # ------------------------------------------------------------------
    # Server event reporting
    # ------------------------------------------------------------------

    def _report_event(
        self,
        model_id: str,
        version: str,
        modality: Modality,
        session_id: str,
        event_type: str,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            payload: dict[str, Any] = {
                "device_id": self.device_id,
                "model_id": model_id,
                "version": version,
                "modality": modality.value,
                "session_id": session_id,
                "event_type": event_type,
                "timestamp_ms": int(time.time() * 1000),
                "org_id": self.org_id,
            }
            if metrics:
                payload["metrics"] = metrics
            self.api.post("/inference/events", payload)
        except (OSError, ValueError):
            pass  # best-effort reporting
