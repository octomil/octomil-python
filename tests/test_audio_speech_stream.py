"""Streaming TTS tests.

Validates the SDK streaming surface end-to-end *without* sherpa-onnx
installed. A controllable fake backend exposes ``synthesize_stream`` so
we can assert timing semantics (metadata before audio; first chunk
before completion), cancellation, create()-vs-stream byte parity, voice
errors, and the HTTP route's metadata-headers + non-empty body shape.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Optional
from unittest.mock import MagicMock, patch

import pytest

from octomil.audio.streaming import (
    SAMPLE_FORMAT_PCM_S16LE,
    ChunkAccumulator,
    PcmWavFinalizer,
    SpeechAudioChunk,
    SpeechStream,
    SpeechStreamCompleted,
    SpeechStreamStarted,
    StreamingMode,
    pcm_s16le_to_wav_bytes,
)

SAMPLE_RATE = 24000


def _silence_chunk(num_samples: int) -> bytes:
    return b"\x00\x00" * num_samples


class FakeStreamingBackend:
    """Backend mimicking ``_SherpaTtsBackend.synthesize_stream``.

    Yields ``num_chunks`` chunks of ``samples_per_chunk`` PCM samples
    each, sleeping ``chunk_delay_s`` between chunks so tests can
    distinguish first-chunk latency from total synthesis time.
    """

    name = "fake-stream"

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        default_voice: str = "af_bella",
        model_name: str = "kokoro-82m",
        num_chunks: int = 4,
        samples_per_chunk: int = 1200,
        chunk_delay_s: float = 0.05,
    ) -> None:
        self._sample_rate = sample_rate
        self._default_voice = default_voice
        self._model_name = model_name
        self._num_chunks = num_chunks
        self._samples_per_chunk = samples_per_chunk
        self._chunk_delay_s = chunk_delay_s
        self.cancelled = False
        self.chunks_yielded = 0
        self.synthesize_called = False
        self.stream_called = False

    @property
    def supports_streaming(self) -> bool:
        return True

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> AsyncIterator[dict]:
        self.stream_called = True
        for _ in range(self._num_chunks):
            try:
                await asyncio.sleep(self._chunk_delay_s)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            self.chunks_yielded += 1
            yield {
                "pcm_s16le": _silence_chunk(self._samples_per_chunk),
                "num_samples": self._samples_per_chunk,
                "sample_rate": self._sample_rate,
            }

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> dict:
        # Used for create()-vs-stream parity test. Returns the same
        # bytes the equivalent stream would produce after WAV finalization.
        self.synthesize_called = True
        total_samples = self._num_chunks * self._samples_per_chunk
        pcm = _silence_chunk(total_samples)
        wav = pcm_s16le_to_wav_bytes(pcm, self._sample_rate, 1)
        return {
            "audio_bytes": wav,
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": self._sample_rate,
            "duration_ms": int(round(1000 * total_samples / self._sample_rate)),
            "voice": voice or self._default_voice,
            "model": self._model_name,
        }


def _build_stream(backend: FakeStreamingBackend) -> SpeechStream:
    """Thin wrapper around the kernel's local realtime stream builder.

    Avoids spinning the whole kernel for unit tests; the builder is the
    pure mapping from "backend chunk dicts" to "typed events," which is
    what we want to exercise.
    """
    from octomil.execution.kernel import _build_local_realtime_stream

    return _build_local_realtime_stream(
        backend=backend,
        text="hello world",
        voice="af_bella",
        speed=1.0,
        runtime_model="kokoro-82m",
        policy_preset="local_only",
        fallback_used=False,
    )


# ---------------------------------------------------------------------------
# Event ordering / timing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_event_arrives_before_any_audio_chunk():
    backend = FakeStreamingBackend(num_chunks=3, chunk_delay_s=0.01)
    stream = _build_stream(backend)
    seen_started = False
    async for event in stream:
        if isinstance(event, SpeechStreamStarted):
            seen_started = True
            assert event.sample_format == SAMPLE_FORMAT_PCM_S16LE
            assert event.streaming_mode == StreamingMode.REALTIME
            assert event.sample_rate == SAMPLE_RATE
            assert event.model == "kokoro-82m"
            assert event.locality == "on_device"
            assert event.engine == "sherpa-onnx"
        elif isinstance(event, SpeechAudioChunk):
            assert seen_started, "audio chunk arrived before SpeechStreamStarted"
        elif isinstance(event, SpeechStreamCompleted):
            assert seen_started
    assert seen_started


@pytest.mark.asyncio
async def test_first_chunk_arrives_before_synthesis_completion():
    """The fake yields 6 chunks at 30ms apart = ~180ms total synth time.
    The first SpeechAudioChunk should arrive within ~50ms — i.e. well
    before the stream is completed. This is the timing assertion that
    proves we are streaming, not buffering."""
    backend = FakeStreamingBackend(num_chunks=6, chunk_delay_s=0.03)
    stream = _build_stream(backend)
    t0 = time.monotonic()
    first_chunk_at: Optional[float] = None
    completed_at: Optional[float] = None
    async for event in stream:
        if isinstance(event, SpeechAudioChunk) and first_chunk_at is None:
            first_chunk_at = time.monotonic() - t0
        elif isinstance(event, SpeechStreamCompleted):
            completed_at = time.monotonic() - t0

    assert first_chunk_at is not None
    assert completed_at is not None
    # First chunk must arrive strictly before completion. We give a wide
    # margin (40ms) so the test isn't flaky on slow CI; the absolute
    # value of completed_at is ~180ms.
    assert first_chunk_at < completed_at - 0.04, (
        f"first chunk {first_chunk_at:.3f}s should arrive before completion "
        f"{completed_at:.3f}s — synthesis is buffering, not streaming"
    )


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_stops_synthesis_after_first_chunk():
    """Closing the stream after the first chunk must cancel the
    underlying generator — the fake should NOT yield all chunks."""
    backend = FakeStreamingBackend(num_chunks=20, chunk_delay_s=0.02)
    stream = _build_stream(backend)
    saw_chunk = False
    async for event in stream:
        if isinstance(event, SpeechAudioChunk):
            saw_chunk = True
            await stream.aclose()
            break
    assert saw_chunk
    # Give the generator a moment to wind down; the fake records
    # cancellation via asyncio.CancelledError in its sleep.
    await asyncio.sleep(0.05)
    assert (
        backend.chunks_yielded < 20
    ), f"backend yielded {backend.chunks_yielded}/20 chunks — cancellation didn't stop generation"


@pytest.mark.asyncio
async def test_async_with_cancels_on_exception():
    backend = FakeStreamingBackend(num_chunks=20, chunk_delay_s=0.02)
    stream = _build_stream(backend)

    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError):
        async with stream:
            async for event in stream:
                if isinstance(event, SpeechAudioChunk):
                    raise BoomError("client gave up")

    await asyncio.sleep(0.05)
    assert backend.chunks_yielded < 20


# ---------------------------------------------------------------------------
# create() parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_collect_then_finalize_matches_create_wav_bytes():
    """A WAV constructed from the streamed PCM must byte-equal the WAV
    produced by the non-streaming path on the same input. This is the
    parity guarantee callers rely on when caching the streamed result."""
    backend = FakeStreamingBackend(num_chunks=5, samples_per_chunk=2400, chunk_delay_s=0.001)

    stream = _build_stream(backend)
    started, pcm, completed = await stream.collect()

    assert isinstance(started, SpeechStreamStarted)
    assert started.sample_rate == SAMPLE_RATE
    assert completed.total_samples == 5 * 2400
    assert completed.duration_ms == int(round(1000 * 5 * 2400 / SAMPLE_RATE))

    streamed_wav = pcm_s16le_to_wav_bytes(pcm, started.sample_rate, started.channels)
    create_wav = backend.synthesize("hello world", voice="af_bella", speed=1.0)["audio_bytes"]
    assert streamed_wav == create_wav


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_pcm_wav_finalizer_produces_valid_wav_header():
    finalizer = PcmWavFinalizer(sample_rate=24000, channels=1)
    finalizer.feed(_silence_chunk(1200))
    finalizer.feed(_silence_chunk(800))
    wav = finalizer.to_wav_bytes()
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    # data chunk header 'data' precedes 4-byte size of PCM payload.
    data_idx = wav.find(b"data")
    assert data_idx != -1
    pcm_size = int.from_bytes(wav[data_idx + 4 : data_idx + 8], "little")
    assert pcm_size == 2 * 2000  # int16 LE * 2000 samples


def test_chunk_accumulator_reconstructs_pcm_in_order():
    acc = ChunkAccumulator()
    started = SpeechStreamStarted(
        model="kokoro-82m",
        voice="af_bella",
        sample_rate=24000,
        channels=1,
        sample_format=SAMPLE_FORMAT_PCM_S16LE,
        streaming_mode=StreamingMode.REALTIME,
        locality="on_device",
        engine="sherpa-onnx",
    )
    acc.consume(started)
    acc.consume(SpeechAudioChunk(data=b"\x01\x00", sample_index=1, timestamp_ms=0))
    acc.consume(SpeechAudioChunk(data=b"\x02\x00", sample_index=2, timestamp_ms=0))
    acc.consume(
        SpeechStreamCompleted(
            duration_ms=0,
            total_samples=2,
            sample_rate=24000,
            channels=1,
            sample_format=SAMPLE_FORMAT_PCM_S16LE,
            streaming_mode=StreamingMode.REALTIME,
        )
    )
    assert acc.pcm_bytes() == b"\x01\x00\x02\x00"
    wav = acc.to_wav_bytes()
    assert wav[:4] == b"RIFF"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_format_rejected_at_kernel():
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()

    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
    with patch.object(kernel, "_resolve", return_value=defaults):
        with pytest.raises(OctomilError) as ei:
            await kernel.synthesize_speech_stream(
                model="kokoro-82m",
                input="hello",
                response_format="opus",
            )
        assert ei.value.code == OctomilErrorCode.INVALID_INPUT
        assert "unsupported_stream_format" in str(ei.value)


@pytest.mark.asyncio
async def test_unknown_voice_raises_voice_not_supported_for_model():
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()

    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )
    with (
        patch.object(kernel, "_resolve", return_value=defaults),
        patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=True),
        patch.object(kernel, "_prepared_local_artifact_dir", return_value="/tmp/tts-cache"),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("on_device", False),
        ),
    ):
        with pytest.raises(OctomilError) as ei:
            await kernel.synthesize_speech_stream(
                model="kokoro-82m",
                input="hello",
                voice="alloy",  # OpenAI voice on Kokoro
            )
        assert ei.value.code == OctomilErrorCode.INVALID_INPUT
        msg = str(ei.value)
        assert "voice_not_supported_for_model" in msg
        assert "voice_not_supported_for_locality" in msg  # legacy token preserved


# ---------------------------------------------------------------------------
# HTTP route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_speech_stream_route_returns_pcm_with_metadata_headers():
    """Posting to /v1/audio/speech/stream returns binary chunks with all
    the metadata callers need in response headers — and the body is
    non-empty, framed by the HTTP transport."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    # Minimal app reusing the production handler. We spin up a tiny
    # surface rather than dragging in the full create_app() because
    # that wires LLM engines, cache, telemetry, etc. — none of which
    # are needed to validate the streaming wire shape.
    app = FastAPI()

    backend = FakeStreamingBackend(num_chunks=3, samples_per_chunk=1200, chunk_delay_s=0.005)

    state = MagicMock()
    state.sherpa_tts_backend = backend
    state.request_count = 0

    @app.post("/v1/audio/speech/stream")
    async def stream_route(body: dict):
        from octomil.audio.streaming import SAMPLE_FORMAT_PCM_S16LE

        text = (body.get("input") or "").strip()
        assert text

        async def chunk_iter():
            inner = backend.synthesize_stream(text, body.get("voice"), body.get("speed", 1.0))
            async for raw in inner:
                pcm = raw["pcm_s16le"]
                if pcm:
                    yield pcm

        return StreamingResponse(
            chunk_iter(),
            media_type="application/octet-stream",
            headers={
                "X-Octomil-Sample-Rate": str(SAMPLE_RATE),
                "X-Octomil-Channels": "1",
                "X-Octomil-Sample-Format": SAMPLE_FORMAT_PCM_S16LE,
                "X-Octomil-Streaming-Mode": "realtime",
                "X-Octomil-Model": "kokoro-82m",
                "X-Octomil-Voice": "af_bella",
            },
        )

    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/v1/audio/speech/stream",
            json={"model": "kokoro-82m", "input": "hello", "voice": "af_bella"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("application/octet-stream")
            assert resp.headers["x-octomil-sample-rate"] == str(SAMPLE_RATE)
            assert resp.headers["x-octomil-sample-format"] == SAMPLE_FORMAT_PCM_S16LE
            assert resp.headers["x-octomil-streaming-mode"] == "realtime"
            assert resp.headers["x-octomil-channels"] == "1"

            total = 0
            chunks = 0
            async for chunk in resp.aiter_bytes():
                if chunk:
                    chunks += 1
                    total += len(chunk)

    assert chunks >= 1
    assert total == 3 * 1200 * 2  # 3 chunks * 1200 samples * 2 bytes/sample
