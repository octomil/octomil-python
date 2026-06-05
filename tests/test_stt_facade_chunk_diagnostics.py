"""WU-1 — public facade surfacing of STT chunk controls + diagnostics.

Pure-Python (no dylib / cffi): every test mocks the backend dict or the
``ModelRuntime`` / kernel seam. Covers the three plumbed layers:

1. ``audio.transcriptions.create()`` via ``runtime.run(RuntimeRequest)``:
   chunk kwargs reach the request as ``stt_options``; ``RuntimeResponse``
   STT carriers project up onto the public ``TranscriptionResult``.
2. ``FacadeTranscriptions.create()`` via the kernel: chunk kwargs reach
   ``transcribe_audio``; ``ExecutionResult.segments`` + ``raw`` project up.
3. ``kernel._local_transcribe`` -> ``backend.transcribe(...)``: chunk
   kwargs reach a native-shaped backend; a legacy backend (no chunk
   kwargs in its signature) is called WITHOUT them and yields
   ``chunk_diagnostics=None`` without error.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import pytest

from octomil.audio import FacadeTranscriptions
from octomil.audio.transcriptions import AudioTranscriptions
from octomil.audio.types import ChunkDiagnostics, TranscriptionResult
from octomil.execution.kernel import ExecutionResult
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
)

# ---------------------------------------------------------------------------
# Fixtures: a backend dict in the native serve-adapter shape.
# ---------------------------------------------------------------------------


def _native_backend_dict(*, with_diagnostics: bool) -> dict[str, Any]:
    segments = [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "hello",
            "start_ms": 0,
            "end_ms": 1500,
            "avg_logprob": -0.21,
            "no_speech_prob": 0.02,
        },
        {
            "start": 1.5,
            "end": 3.0,
            "text": "world",
            "start_ms": 1500,
            "end_ms": 3000,
            "avg_logprob": -0.33,
            "no_speech_prob": 0.05,
        },
    ]
    payload: dict[str, Any] = {
        "text": "hello world",
        "segments": segments,
        "duration_ms": 3000,
        "chunk_diagnostics": None,
    }
    if with_diagnostics:
        payload["chunk_diagnostics"] = {
            "window_ms": 15000,
            "overlap_ms": 0,
            "step_ms": 15000,
            "window_count": 2,
            "windows": [
                {"index": 0, "start_ms": 0, "end_ms": 15000, "off_ms": 0, "is_last": False},
                {"index": 1, "start_ms": 15000, "end_ms": 28000, "off_ms": 15000, "is_last": True},
            ],
            "segment_owner_window": [0, 0],
            "audio_duration_ms": 28000,
            "final_segment_end_ms": 3000,
            "tail_gap_ms": 25000,
        }
    return payload


# ---------------------------------------------------------------------------
# Layer 1 — AudioTranscriptions.create() via ModelRuntime.run()
# ---------------------------------------------------------------------------


class _RecordingRuntime(ModelRuntime):
    """Captures the request and returns a fixed STT-shaped response."""

    def __init__(self, response: RuntimeResponse) -> None:
        self._response = response
        self.last_request: Optional[RuntimeRequest] = None

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        self.last_request = request
        return self._response

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        yield RuntimeChunk(text=self._response.text)


@pytest.mark.asyncio
async def test_create_chunk_kwargs_reach_request_as_stt_options() -> None:
    runtime = _RecordingRuntime(RuntimeResponse(text="hi"))
    transcriptions = AudioTranscriptions(runtime_resolver=lambda ref: runtime)

    await transcriptions.create(audio=b"x", chunk_window_ms=15000, chunk_overlap_ms=2000)

    assert runtime.last_request is not None
    opts = runtime.last_request.stt_options
    assert opts is not None
    assert opts.chunk_window_ms == 15000
    assert opts.chunk_overlap_ms == 2000


@pytest.mark.asyncio
async def test_create_default_has_no_stt_options() -> None:
    """No chunk kwargs -> stt_options stays None (byte-identical to today)."""
    runtime = _RecordingRuntime(RuntimeResponse(text="hi"))
    transcriptions = AudioTranscriptions(runtime_resolver=lambda ref: runtime)

    result = await transcriptions.create(audio=b"x")

    assert runtime.last_request is not None
    assert runtime.last_request.stt_options is None
    assert result.text == "hi"
    assert result.segments == []
    assert result.duration_ms == 0
    assert result.chunk_diagnostics is None


@pytest.mark.asyncio
async def test_create_projects_response_carriers_up() -> None:
    diag = ChunkDiagnostics(window_ms=15000, overlap_ms=0, step_ms=15000, window_count=2, tail_gap_ms=25000)
    from octomil.audio.types import TranscriptionSegment

    response = RuntimeResponse(
        text="hello world",
        stt_segments=[TranscriptionSegment(text="hello", start_ms=0, end_ms=1500, avg_logprob=-0.21)],
        stt_duration_ms=3000,
        stt_chunk_diagnostics=diag,
    )
    transcriptions = AudioTranscriptions(runtime_resolver=lambda ref: _RecordingRuntime(response))

    result = await transcriptions.create(audio=b"x", chunk_window_ms=15000)

    assert result.duration_ms == 3000
    assert result.chunk_diagnostics is diag
    assert result.chunk_diagnostics.tail_gap_ms == 25000
    assert result.segments[0].avg_logprob == pytest.approx(-0.21)


@pytest.mark.asyncio
async def test_create_ignores_bad_diagnostics_shape() -> None:
    """A runtime that returns a non-ChunkDiagnostics carrier must not
    poison the public result — it is treated as no diagnostics."""
    response = RuntimeResponse(text="hi", stt_chunk_diagnostics={"not": "a dataclass"})
    transcriptions = AudioTranscriptions(runtime_resolver=lambda ref: _RecordingRuntime(response))

    result = await transcriptions.create(audio=b"x", chunk_window_ms=15000)

    assert result.chunk_diagnostics is None


# ---------------------------------------------------------------------------
# Layer 2 — FacadeTranscriptions.create() via the kernel ExecutionResult
# ---------------------------------------------------------------------------


class _FakeKernel:
    def __init__(self, result: ExecutionResult) -> None:
        self._result = result
        self.captured: dict[str, Any] = {}

    async def transcribe_audio(self, audio_data, **kwargs):  # type: ignore[no-untyped-def]
        self.captured = {"audio_data": audio_data, **kwargs}
        return self._result


@pytest.mark.asyncio
async def test_facade_forwards_chunk_kwargs_to_kernel() -> None:
    kernel = _FakeKernel(ExecutionResult(output_text="hello world"))
    facade = FacadeTranscriptions(kernel)

    await facade.create(audio=b"x", chunk_window_ms=15000, chunk_overlap_ms=1000)

    assert kernel.captured["chunk_window_ms"] == 15000
    assert kernel.captured["chunk_overlap_ms"] == 1000


@pytest.mark.asyncio
async def test_facade_projects_native_diagnostics_up() -> None:
    payload = _native_backend_dict(with_diagnostics=True)
    result = ExecutionResult(output_text=payload["text"], segments=payload["segments"], raw=payload)
    facade = FacadeTranscriptions(_FakeKernel(result))

    out = await facade.create(audio=b"x", chunk_window_ms=15000, language="en")

    assert isinstance(out, TranscriptionResult)
    assert out.duration_ms == 3000
    assert out.chunk_diagnostics is not None
    assert out.chunk_diagnostics.window_count == 2
    assert out.chunk_diagnostics.tail_gap_ms == 25000
    assert out.chunk_diagnostics.segment_owner_window == [0, 0]
    assert len(out.chunk_diagnostics.windows) == 2
    assert out.chunk_diagnostics.windows[1].is_last is True
    assert out.segments[0].avg_logprob == pytest.approx(-0.21)
    assert out.segments[0].start_ms == 0
    assert out.segments[1].no_speech_prob == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_facade_legacy_result_yields_no_diagnostics() -> None:
    """Legacy / cloud ExecutionResult (seconds-only segments, no
    chunk_diagnostics in raw) projects with chunk_diagnostics=None and
    logprob defaults, without error."""
    legacy_segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    result = ExecutionResult(
        output_text="hi", segments=legacy_segments, raw={"text": "hi", "segments": legacy_segments}
    )
    facade = FacadeTranscriptions(_FakeKernel(result))

    out = await facade.create(audio=b"x")

    assert out.chunk_diagnostics is None
    assert out.duration_ms == 0
    assert out.segments[0].start_ms == 0
    assert out.segments[0].end_ms == 1000
    assert out.segments[0].avg_logprob == 0.0


@pytest.mark.asyncio
async def test_facade_default_no_segments_no_diagnostics() -> None:
    result = ExecutionResult(output_text="plain")
    facade = FacadeTranscriptions(_FakeKernel(result))

    out = await facade.create(audio=b"x")

    assert out.text == "plain"
    assert out.segments == []
    assert out.chunk_diagnostics is None


# ---------------------------------------------------------------------------
# Layer 3 — kernel._local_transcribe -> backend.transcribe(...)
# ---------------------------------------------------------------------------


class _NativeShapeBackend:
    """Backend whose transcribe ACCEPTS chunk kwargs (native shape)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        audio_path: str,
        *,
        chunk_window_ms: int | None = None,
        chunk_overlap_ms: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"chunk_window_ms": chunk_window_ms, "chunk_overlap_ms": chunk_overlap_ms})
        return self._payload


class _LegacyShapeBackend:
    """Backend whose transcribe takes path only (legacy shape)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls = 0

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        self.calls += 1
        return self._payload


def _kernel_with_backend(backend: Any):
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel()
    kernel._resolve_local_transcription_backend = lambda *a, **k: backend  # type: ignore[method-assign]
    return kernel


@pytest.mark.asyncio
async def test_local_transcribe_passes_chunk_kwargs_to_native_backend() -> None:
    backend = _NativeShapeBackend(_native_backend_dict(with_diagnostics=True))
    kernel = _kernel_with_backend(backend)

    result = await kernel._local_transcribe(
        b"\x00" * 8, "whisper-tiny", "en", chunk_window_ms=15000, chunk_overlap_ms=2000
    )

    assert backend.calls == [{"chunk_window_ms": 15000, "chunk_overlap_ms": 2000}]
    # ExecutionResult carries the segments + raw diagnostics through.
    assert result.segments is not None
    assert result.raw is not None
    assert result.raw["chunk_diagnostics"]["window_count"] == 2


@pytest.mark.asyncio
async def test_local_transcribe_default_calls_native_without_chunk_kwargs() -> None:
    backend = _NativeShapeBackend(_native_backend_dict(with_diagnostics=False))
    kernel = _kernel_with_backend(backend)

    await kernel._local_transcribe(b"\x00" * 8, "whisper-tiny", "en")

    # Backend is called exactly once; with NO chunk kwargs forwarded the
    # native shape defaults both to None (default path unchanged).
    assert backend.calls == [{"chunk_window_ms": None, "chunk_overlap_ms": None}]


@pytest.mark.asyncio
async def test_local_transcribe_legacy_backend_drops_chunk_kwargs() -> None:
    """A legacy backend (transcribe(path) only) must not receive chunk
    kwargs even when requested — it runs full-buffer, no diagnostics, no
    TypeError."""
    backend = _LegacyShapeBackend({"text": "hi", "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]})
    kernel = _kernel_with_backend(backend)

    result = await kernel._local_transcribe(b"\x00" * 8, "whisper-tiny", "en", chunk_window_ms=15000)

    assert backend.calls == 1
    # Legacy dict omits chunk_diagnostics -> facade projection yields None.
    from octomil.audio import _execution_result_to_transcription

    public = _execution_result_to_transcription(result, "en")
    assert public.chunk_diagnostics is None
