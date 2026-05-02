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
    TtsStreamingCapability,
    TtsStreamingGranularity,
    TtsStreamingMode,
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

    def load_model(self, model_name: str) -> None:
        # No-op; the production lifespan calls load_model after
        # create_backend, so the fake mirrors that contract.
        self._model_name = model_name

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


async def _start_lifespan(app) -> None:
    """Fire a FastAPI app's lifespan startup the way uvicorn would.

    httpx.ASGITransport doesn't trigger lifespan events, so the
    production-route tests have to do it themselves before the first
    request — otherwise ``state.sherpa_tts_backend`` is never wired
    and every call returns 503.
    """
    cm = app.router.lifespan_context(app)
    await cm.__aenter__()


def _stub_kernel_warmup_for_tts(monkeypatch_or_path):
    """Patch ``ExecutionKernel.prepare`` so it returns a fake
    PrepareOutcome with ``artifact_dir = monkeypatch_or_path``.

    The lifespan deliberately calls ``prepare`` (not ``warmup``) to
    avoid the double-load P2: ``warmup`` also constructs and caches a
    backend, but the lifespan needs to own the canonical backend on
    ``state.sherpa_tts_backend``. This stub mirrors the production
    path so tests don't depend on a real PrepareManager / planner /
    network or a cached artifact on the dev box.
    """
    fake_prepare_outcome = MagicMock()
    fake_prepare_outcome.artifact_dir = str(monkeypatch_or_path)
    return patch(
        "octomil.execution.kernel.ExecutionKernel.prepare",
        return_value=fake_prepare_outcome,
    )


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
        sdk_t0=time.monotonic(),
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
            # The fake backend doesn't implement streaming_capability so
            # _build_local_realtime_stream falls back to final_only.
            assert event.streaming_capability.mode == TtsStreamingMode.FINAL_CHUNK
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
    cap = TtsStreamingCapability.sentence(verified=True)
    started = SpeechStreamStarted(
        model="kokoro-82m",
        voice="af_bella",
        sample_rate=24000,
        channels=1,
        sample_format=SAMPLE_FORMAT_PCM_S16LE,
        streaming_capability=cap,
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
            streaming_capability=cap,
            setup_ms=0.0,
            engine_first_chunk_ms=None,
            e2e_first_chunk_ms=None,
            total_latency_ms=0.0,
            observed_chunks=2,
            capability_verified=True,
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
async def test_http_speech_stream_unit_route_shape():
    """Unit-level wire shape check.

    Builds a minimal FastAPI route that mimics the production response
    shape so the wire contract (binary chunks + metadata headers) is
    pinned independently of create_app() wiring. Pair this with
    test_production_http_speech_stream_route_returns_pcm_with_metadata_headers
    below — that test calls the *real* handler so regressions in
    octomil/serve/app.py break the build.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

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
                "X-Octomil-Streaming-Capability-Mode": "final_chunk",
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
            assert resp.headers["x-octomil-streaming-capability-mode"] == "final_chunk"
            assert resp.headers["x-octomil-channels"] == "1"

            total = 0
            chunks = 0
            async for chunk in resp.aiter_bytes():
                if chunk:
                    chunks += 1
                    total += len(chunk)

    assert chunks >= 1
    assert total == 3 * 1200 * 2  # 3 chunks * 1200 samples * 2 bytes/sample


# ---------------------------------------------------------------------------
# Production HTTP route — exercises octomil.serve.app.create_app
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_production_http_speech_stream_route_returns_pcm_with_metadata_headers(tmp_path):
    """End-to-end through the real ``create_app`` factory.

    Patches the sherpa engine factory to return a FakeStreamingBackend
    so we don't need a Kokoro artifact on disk, but every other piece
    of the server stack (FastAPI app, lifespan, state wiring, route
    handler) is the production code in ``octomil/serve/app.py``.
    Regressions in route registration / header naming / error
    envelope / state plumbing fail this test.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    backend = FakeStreamingBackend(num_chunks=3, samples_per_chunk=1200, chunk_delay_s=0.005)

    fake_engine = MagicMock()
    fake_engine.create_backend.return_value = backend

    from httpx import ASGITransport, AsyncClient

    with (
        patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", return_value=fake_engine),
        patch("octomil.serve.app._is_sherpa_tts_model", return_value=True),
        _stub_kernel_warmup_for_tts(tmp_path),
    ):
        from octomil.serve.app import create_app

        app = create_app("kokoro-82m")
        # httpx ASGITransport doesn't auto-fire lifespan; do it manually
        # so create_app's startup wires state.sherpa_tts_backend.
        await _start_lifespan(app)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with client.stream(
                "POST",
                "/v1/audio/speech/stream",
                json={"model": "kokoro-82m", "input": "hello", "voice": "af_bella"},
            ) as resp:
                assert resp.status_code == 200, await resp.aread()
                assert resp.headers["content-type"].startswith("application/octet-stream")
                # All metadata required by the spec must be in headers,
                # not trailers — clients/proxies routinely drop trailers.
                assert resp.headers["x-octomil-sample-rate"] == str(SAMPLE_RATE)
                assert resp.headers["x-octomil-sample-format"] == SAMPLE_FORMAT_PCM_S16LE
                # Single-sentence input "hello" — backend honestly
                # advertises final_chunk, not the legacy "realtime" lie.
                assert resp.headers["x-octomil-streaming-capability-mode"] == "final_chunk"
                assert resp.headers["x-octomil-channels"] == "1"
                assert resp.headers["x-octomil-model"] == "kokoro-82m"
                assert resp.headers["x-octomil-voice"] == "af_bella"

                total = 0
                chunks = 0
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        chunks += 1
                        total += len(chunk)

    assert chunks >= 1
    assert total == 3 * 1200 * 2  # 3 chunks * 1200 samples * 2 bytes/sample
    assert backend.stream_called is True


@pytest.mark.asyncio
async def test_production_http_speech_stream_route_rejects_unsupported_format(tmp_path):
    """Production route returns a 4xx error for unsupported formats."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    backend = FakeStreamingBackend(num_chunks=2, chunk_delay_s=0.001)
    fake_engine = MagicMock()
    fake_engine.create_backend.return_value = backend

    from httpx import ASGITransport, AsyncClient

    with (
        patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", return_value=fake_engine),
        patch("octomil.serve.app._is_sherpa_tts_model", return_value=True),
        _stub_kernel_warmup_for_tts(tmp_path),
    ):
        from octomil.serve.app import create_app

        app = create_app("kokoro-82m")
        # httpx ASGITransport doesn't auto-fire lifespan; do it manually
        # so create_app's startup wires state.sherpa_tts_backend.
        await _start_lifespan(app)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/audio/speech/stream",
                json={"model": "kokoro-82m", "input": "hello", "response_format": "opus"},
            )
            assert 400 <= resp.status_code < 500
            text = resp.text
            assert "unsupported_stream_format" in text or "server_streaming_format" in text


# ---------------------------------------------------------------------------
# Artifact-specific voice manifest — pins the P1 reviewer concern that
# the streaming branch must not reintroduce the global-catalog aliasing
# bug. The kernel must validate the voice against the static recipe's
# voice_manifest, not a hardcoded tuple.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_voice_validation_uses_static_recipe_manifest():
    """Streaming voice validation must agree with the recipe manifest.

    Pre-fix the streaming branch reused the global 28-name Kokoro tuple
    and silently mapped unknown voices to ``sid=0`` (the bm_george /
    am_echo aliasing bug). This test pins that the streaming kernel
    path delegates to ``_validate_local_voice``, which reads the static
    recipe's ``voice_manifest`` instead of any hardcoded list.
    """
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    # Sanity: the recipe under test actually carries the active
    # Kokoro multi-lang voice catalog. If this regresses, the rest
    # of the test is meaningless.
    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    manifest = {n.lower() for n in recipe.materialization.voice_manifest}
    assert manifest, "kokoro-82m static recipe must declare voice_manifest"
    assert "af_bella" in manifest, "af_bella must be in the recipe's catalog"
    # Pick a name guaranteed not to be in any Kokoro manifest so the
    # rejection is unambiguously about catalog enforcement and not a
    # voice that happens to be in some future expansion.
    legacy_only_voice = "octomil_test_unknown_voice"
    assert (
        legacy_only_voice not in manifest
    ), f"test premise: {legacy_only_voice} must be absent from manifest (found: {sorted(manifest)})"

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
                voice=legacy_only_voice,
            )
        assert ei.value.code == OctomilErrorCode.INVALID_INPUT
        msg = str(ei.value)
        assert "voice_not_supported_for_model" in msg
        assert legacy_only_voice in msg
        # Supported list must be the manifest, so af_bella appears.
        assert "af_bella" in msg


# ---------------------------------------------------------------------------
# P1 follow-ups: catalog-less default voice + planner-artifact preflight bypass
# ---------------------------------------------------------------------------


def test_piper_default_voice_does_not_raise_on_create_or_stream(tmp_path):
    """Piper has no voices.txt and an empty fallback catalog.

    Pre-fix the backend collapsed ``self._default_voice`` (e.g.
    ``'amy'``) into the same parameter as the explicit voice, so a
    caller passing ``voice=None`` still hit
    ``voice_not_supported_for_model``. The fix distinguishes
    explicit-voice from default-label: empty/None must map to sid=0
    silently, only an explicit unknown voice raises.
    """
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend("piper-en-amy", model_dir=str(tmp_path))
    # Explicit None -> sid=0, no error.
    sid, _ = backend.validate_voice(None)
    assert sid == 0
    # Empty string also routes to sid=0 (no explicit name).
    assert backend._voice_to_sid("", explicit=False) == 0
    # Explicit unknown voice on a catalog-less model is rejected with
    # the structured error, not silently aliased to 0.
    from octomil.errors import OctomilError, OctomilErrorCode

    with pytest.raises(OctomilError) as ei:
        backend.validate_voice("amy")  # explicit, even though it matches the default label
    assert ei.value.code == OctomilErrorCode.INVALID_INPUT
    assert "voice_not_supported_for_model" in str(ei.value)


def test_kokoro_default_voice_label_resolves_via_manifest(tmp_path):
    """sid=0's reported label should be the manifest's first entry, not
    the engine's static ``_default_voice`` string.

    Pre-fix Kokoro reported ``af_bella`` for ``voice=None`` even when
    the prepared manifest's sid=0 was actually ``af``. validate_voice
    now reads manifest[0] for the default-label so reports are
    accurate. Catalog-less models (Piper) keep falling back to
    ``_default_voice`` since there's no authoritative source.
    """
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    sidecar = tmp_path / "voices.txt"
    # First entry intentionally != backend's _default_voice ('af_bella').
    sidecar.write_text("af\naf_bella\nam_michael\n")

    backend = _SherpaTtsBackend("kokoro-82m", model_dir=str(tmp_path))
    sid, label = backend.validate_voice(None)
    assert sid == 0
    assert label == "af", (
        f"sid=0 should resolve to manifest[0] ('af'), got {label!r} "
        f"— this is the Kokoro default-label drift the reviewer flagged"
    )

    # Explicit name still wins.
    sid, label = backend.validate_voice("af_bella")
    assert sid == 1
    assert label == "af_bella"


def test_piper_default_voice_label_falls_back_to_default_string(tmp_path):
    """Catalog-less Piper has no manifest; default label keeps using
    ``_default_voice`` rather than failing or returning empty."""
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend("piper-en-amy", model_dir=str(tmp_path))
    sid, label = backend.validate_voice(None)
    assert sid == 0
    assert label == "amy"


def test_validate_voice_raises_synchronously_for_unsupported_explicit_voice(tmp_path):
    """The whole point of validate_voice is to raise BEFORE
    SpeechStreamStarted / 200 OK is committed."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    sidecar = tmp_path / "voices.txt"
    sidecar.write_text("af\naf_bella\n")

    backend = _SherpaTtsBackend("kokoro-82m", model_dir=str(tmp_path))
    with pytest.raises(OctomilError) as ei:
        backend.validate_voice("alloy")
    assert ei.value.code == OctomilErrorCode.INVALID_INPUT


@pytest.mark.asyncio
async def test_stream_does_not_emit_started_for_unsupported_voice(tmp_path):
    """Reviewer P1: validate_voice runs synchronously in
    _build_local_realtime_stream, so an unsupported explicit voice
    raises during synthesize_speech_stream's await — the caller never
    receives a SpeechStream object that would emit a successful
    SpeechStreamStarted event."""
    from octomil.errors import OctomilError

    class _UnsupportedVoiceBackend:
        """Minimal backend that raises in validate_voice."""

        _sample_rate = SAMPLE_RATE
        _default_voice = "af_bella"
        _model_name = "kokoro-82m"
        supports_streaming = True
        stream_started = False

        def validate_voice(self, voice):
            raise OctomilError(
                code=__import__("octomil.errors", fromlist=["OctomilErrorCode"]).OctomilErrorCode.INVALID_INPUT,
                message="voice_not_supported_for_model: alloy",
            )

        async def synthesize_stream(self, text, voice=None, speed=1.0):
            self.stream_started = True
            yield {"pcm_s16le": b"\x00\x00", "num_samples": 1, "sample_rate": SAMPLE_RATE}

    from octomil.execution.kernel import _build_local_realtime_stream

    backend = _UnsupportedVoiceBackend()
    with pytest.raises(OctomilError):
        # Construction itself must raise — no SpeechStream returned.
        _build_local_realtime_stream(
            backend=backend,
            text="hello",
            voice="alloy",
            speed=1.0,
            runtime_model="kokoro-82m",
            policy_preset="local_only",
            fallback_used=False,
            sdk_t0=time.monotonic(),
        )
    # The producer was never even constructed, so synthesize_stream
    # was not consumed.
    assert backend.stream_started is False


@pytest.mark.asyncio
async def test_tts_server_lifespan_passes_prepared_model_dir_to_sherpa_backend(tmp_path):
    """The TTS lifespan must inject the prepared ``model_dir`` into
    ``SherpaTtsEngine.create_backend`` — otherwise
    ``_SherpaTtsBackend._resolve_model_dir`` raises and ``octomil
    serve kokoro-82m`` fails before any route is reachable.

    The earlier production-route tests patched the engine factory to
    return a fake, masking this regression. This test patches at the
    *engine class* level so we can capture the kwargs the production
    lifespan uses, AND patches ``kernel.warmup`` so the test doesn't
    require a real PrepareManager / planner.
    """
    pytest.importorskip("fastapi")

    captured_kwargs: dict = {}

    class _RecordingBackend:
        supports_streaming = True
        _sample_rate = SAMPLE_RATE
        _default_voice = "af"
        _model_name = "kokoro-82m"

        def load_model(self, model_name: str) -> None:
            self._model_name = model_name

    class _RecordingEngine:
        def create_backend(self, model_name, **kwargs):
            captured_kwargs.update(kwargs)
            return _RecordingBackend()

    with (
        patch(
            "octomil.runtime.engines.sherpa.SherpaTtsEngine",
            return_value=_RecordingEngine(),
        ),
        patch("octomil.serve.app._is_sherpa_tts_model", return_value=True),
        _stub_kernel_warmup_for_tts(tmp_path),
    ):
        from octomil.serve.app import create_app

        app = create_app("kokoro-82m")
        await _start_lifespan(app)

    # The lifespan MUST pass model_dir=<artifact_dir> so the sherpa
    # backend can resolve the prepared layout. Pre-fix the lifespan
    # called create_backend(model_name) with no kwargs and immediately
    # raised in load_model.
    assert (
        "model_dir" in captured_kwargs
    ), f"TTS lifespan must inject prepared model_dir from kernel.prepare() — captured kwargs: {captured_kwargs}"
    assert captured_kwargs["model_dir"] == str(tmp_path)


@pytest.mark.asyncio
async def test_tts_server_lifespan_uses_prepare_not_warmup(tmp_path):
    """P2: lifespan must call ``kernel.prepare()`` (artifact-only),
    not ``kernel.warmup()`` (which also loads + caches a second
    backend in ``kernel._warmed_backends``). Calling warmup would
    double startup time and resident memory because the lifespan
    already owns the canonical backend on
    ``state.sherpa_tts_backend``.
    """
    pytest.importorskip("fastapi")

    class _RecordingBackend:
        supports_streaming = True
        _sample_rate = SAMPLE_RATE
        _default_voice = "af"
        _model_name = "kokoro-82m"

        def load_model(self, model_name: str) -> None:
            self._model_name = model_name

    class _RecordingEngine:
        def create_backend(self, model_name, **kwargs):
            return _RecordingBackend()

    fake_prepare_outcome = MagicMock()
    fake_prepare_outcome.artifact_dir = str(tmp_path)

    with (
        patch(
            "octomil.runtime.engines.sherpa.SherpaTtsEngine",
            return_value=_RecordingEngine(),
        ),
        patch("octomil.serve.app._is_sherpa_tts_model", return_value=True),
        patch(
            "octomil.execution.kernel.ExecutionKernel.prepare",
            return_value=fake_prepare_outcome,
        ) as prepare_mock,
        patch(
            "octomil.execution.kernel.ExecutionKernel.warmup",
            side_effect=AssertionError("lifespan called warmup() — should call prepare() to avoid double load"),
        ) as warmup_mock,
    ):
        from octomil.serve.app import create_app

        app = create_app("kokoro-82m")
        await _start_lifespan(app)

    assert prepare_mock.called, "lifespan must call kernel.prepare() to materialize the artifact"
    assert not warmup_mock.called, (
        "lifespan must NOT call kernel.warmup() — that double-loads the model and "
        "doubles resident memory while the kernel-cached backend sits unused"
    )
    call_kwargs = prepare_mock.call_args.kwargs
    assert call_kwargs.get("model") == "kokoro-82m"
    assert call_kwargs.get("capability") == "tts"


@pytest.mark.asyncio
async def test_http_route_returns_4xx_for_unsupported_voice_before_streaming(tmp_path):
    """The production /v1/audio/speech/stream route must return a 4xx
    JSON envelope for unsupported voices, not 200 + binary body that
    later EOFs with an exception. Binary streaming clients have no
    way to recover a structured error after status/headers are
    committed, so pre-validation is mandatory."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from octomil.errors import OctomilError, OctomilErrorCode

    class _ValidatingBackend(FakeStreamingBackend):
        def validate_voice(self, voice):
            if (voice or "").strip().lower() not in {"af_bella", "am_michael"}:
                raise OctomilError(
                    code=OctomilErrorCode.INVALID_INPUT,
                    message=f"voice_not_supported_for_model: {voice!r} not in catalog",
                )
            return 0, voice or "af_bella"

    backend = _ValidatingBackend(num_chunks=1, samples_per_chunk=10, chunk_delay_s=0.001)
    fake_engine = MagicMock()
    fake_engine.create_backend.return_value = backend

    from httpx import ASGITransport, AsyncClient

    with (
        patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", return_value=fake_engine),
        patch("octomil.serve.app._is_sherpa_tts_model", return_value=True),
        _stub_kernel_warmup_for_tts(tmp_path),
    ):
        from octomil.serve.app import create_app

        app = create_app("kokoro-82m")
        await _start_lifespan(app)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/audio/speech/stream",
                json={"model": "kokoro-82m", "input": "hello", "voice": "alloy"},
            )
            # 4xx before any streaming body is sent — clients see a real
            # JSON error envelope, not a truncated octet-stream.
            assert 400 <= resp.status_code < 500, resp.status_code
            assert resp.headers["content-type"].startswith("application/json"), (
                f"unsupported voice must return JSON envelope, got {resp.headers['content-type']!r} — "
                f"streaming clients cannot recover from a 200 octet-stream that EOFs mid-body"
            )
            assert "voice_not_supported_for_model" in resp.text

    # Critically: the backend's stream method was never invoked.
    assert backend.stream_called is False


def test_kokoro_explicit_unknown_voice_still_raises(tmp_path):
    """Pinned to make sure the explicit-vs-default refactor didn't
    regress the catalog enforcement. Kokoro has a sidecar; an
    explicit name not in it must raise."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    sidecar = tmp_path / "voices.txt"
    sidecar.write_text("af_bella\nam_michael\n")

    backend = _SherpaTtsBackend("kokoro-82m", model_dir=str(tmp_path))
    sid, _ = backend.validate_voice(None)  # default still works
    assert sid == 0
    sid, _ = backend.validate_voice("af_bella")  # name in sidecar
    assert sid == 0
    sid, _ = backend.validate_voice("am_michael")
    assert sid == 1
    with pytest.raises(OctomilError) as ei:
        backend.validate_voice("alloy")
    assert ei.value.code == OctomilErrorCode.INVALID_INPUT


def _make_planner_candidate(*, artifact_id: str, digest: str):
    """Build a minimal planner candidate object the kernel's gating
    logic can introspect.

    The real candidate type carries far more metadata; the kernel
    helpers we exercise (``_candidate_has_meaningful_identity``,
    ``_candidate_matches_static_recipe``) only read
    ``candidate.artifact.artifact_id`` / ``.digest``.
    """

    class _Artifact:
        def __init__(self, artifact_id: str, digest: str) -> None:
            self.artifact_id = artifact_id
            self.digest = digest
            self.model_id = artifact_id

    class _Candidate:
        def __init__(self, artifact_id: str, digest: str) -> None:
            self.artifact = _Artifact(artifact_id, digest)

    return _Candidate(artifact_id, digest)


@pytest.mark.asyncio
async def test_private_kokoro_artifact_with_private_voice_in_voices_txt_is_accepted(tmp_path):
    """Reviewer P1#2 reproducer: a planner-selected *private* Kokoro
    artifact whose own ``voices.txt`` contains a private voice must
    be accepted by both ``create()`` and ``stream()`` even though
    that voice is NOT in the public static recipe's manifest.

    Pre-fix: the kernel's pre-prepare voice preflight read the static
    recipe and rejected the private voice before
    ``_prepare_local_tts_artifact`` could materialize the planner's
    artifact. The fix gates the preflight on candidate identity so
    private artifacts validate against their own sidecar.
    """
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    # Pick a voice that's guaranteed absent from any public manifest
    # — the test's gating-bypass property only matters when the
    # private artifact's voice is NOT in the public recipe.
    private_voice = "octomil_test_private_voice"

    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None and recipe.materialization.voice_manifest
    public_manifest = {n.lower() for n in recipe.materialization.voice_manifest}
    assert (
        private_voice not in public_manifest
    ), f"test premise broken: {private_voice} is in the public recipe manifest"

    # The planner-prepared artifact dir contains a voices.txt that
    # *does* include the private voice. The fake backend reads the
    # dir the kernel hands it, so this is what the production code
    # would see post-PrepareManager.
    voices_txt = tmp_path / "voices.txt"
    voices_txt.write_text(f"af_bella\naf_sarah\n{private_voice}\n")

    # Synthetic planner candidate that is *meaningful* (digest set)
    # and explicitly does NOT match the static recipe (different
    # artifact_id, different digest). This is the gating signal the
    # preflight is supposed to honor.
    private_candidate = _make_planner_candidate(
        artifact_id="private-kokoro-v2",
        digest="sha256:" + "f" * 64,
    )

    backend = FakeStreamingBackend(num_chunks=2, samples_per_chunk=600, chunk_delay_s=0.001)
    backend._model_name = "kokoro-82m"

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    kernel._config_set = MagicMock()

    defaults = ResolvedExecutionDefaults(
        model="kokoro-82m",
        policy_preset="local_only",
        inline_policy=None,
        cloud_profile=None,
    )

    # Wire the kernel so the local route lands on our backend without
    # touching disk for the artifact bytes themselves.
    common_patches = [
        patch.object(kernel, "_resolve", return_value=defaults),
        patch(
            "octomil.execution.kernel._resolve_planner_selection",
            return_value=MagicMock(),
        ),
        patch(
            "octomil.execution.kernel._local_sdk_runtime_candidate",
            return_value=private_candidate,
        ),
        patch("octomil.execution.kernel._resolve_routing_policy"),
        patch.object(kernel, "_sherpa_tts_runtime_loadable", return_value=True),
        # No prepared static-recipe cache; we want the planner branch
        # to drive prepare and write the private voices.txt.
        patch.object(kernel, "_prepared_cache_may_short_circuit", return_value=False),
        patch.object(kernel, "_local_candidate_is_unpreparable", return_value=False),
        patch.object(kernel, "_can_prepare_local_tts", return_value=True),
        patch.object(kernel, "_prepare_local_tts_artifact", return_value=str(tmp_path)),
        patch.object(kernel, "_resolve_local_tts_backend", return_value=backend),
        patch(
            "octomil.execution.kernel._select_locality_for_capability",
            return_value=("on_device", False),
        ),
    ]

    # ---- stream() must accept the private voice for the private candidate. ----
    with contextlib_ExitStack() as st:
        for p in common_patches:
            st.enter_context(p)
        stream = await kernel.synthesize_speech_stream(
            model="kokoro-82m",
            input="hello",
            voice=private_voice,
        )
        started, pcm, completed = await stream.collect()
        assert started.voice == private_voice
        assert completed.total_samples == 2 * 600
        assert pcm  # non-empty PCM body

    # ---- create() (non-streaming path) must also accept the private voice. ----
    with contextlib_ExitStack() as st:
        for p in common_patches:
            st.enter_context(p)
        response = await kernel.synthesize_speech(
            model="kokoro-82m",
            input="hello",
            voice=private_voice,
        )
        assert response.audio_bytes
        assert response.route.locality == "on_device"
        # The fake backend reports the requested voice.
        assert response.voice == private_voice


# Lightweight import — keeps the patch-stack assembly above readable
# without adding unittest.mock as a top-level import in this file.
from contextlib import ExitStack as contextlib_ExitStack  # noqa: E402

# ---------------------------------------------------------------------------
# Truthfulness contract — capability advertisement vs. observed cadence
# ---------------------------------------------------------------------------


class CapabilityAdvertisingBackend(FakeStreamingBackend):
    """FakeStreamingBackend that exposes ``streaming_capability(text)``.

    Lets tests pin the advertised mode explicitly so we can drive
    every branch of ``_verify_capability``: advertised better than
    delivered (downgrade), advertised matches delivered (verified
    True), and advertised final_chunk + single chunk (also verified
    True because that *is* the truth).
    """

    def __init__(
        self,
        *,
        advertised: TtsStreamingCapability,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._advertised = advertised

    def streaming_capability(self, text: str) -> TtsStreamingCapability:
        return self._advertised


@pytest.mark.asyncio
async def test_advertised_final_chunk_with_single_chunk_is_verified_true():
    backend = CapabilityAdvertisingBackend(
        advertised=TtsStreamingCapability.final_only(verified=False),
        num_chunks=1,
        chunk_delay_s=0.001,
    )
    stream = _build_stream(backend)
    started, _pcm, completed = await stream.collect()
    assert started.streaming_capability.mode == TtsStreamingMode.FINAL_CHUNK
    assert completed.observed_chunks == 1
    assert completed.streaming_capability.mode == TtsStreamingMode.FINAL_CHUNK
    assert completed.capability_verified is True


@pytest.mark.asyncio
async def test_advertised_sentence_chunk_with_single_chunk_downgrades_to_final_unverified():
    """The bug that motivated this PR: engine advertises sentence_chunk
    but only delivers one chunk. The completion event must downgrade to
    final_chunk with capability_verified=False so callers stop trusting
    the advertised label for this (model, input) shape."""
    backend = CapabilityAdvertisingBackend(
        advertised=TtsStreamingCapability.sentence(verified=False),
        num_chunks=1,
        chunk_delay_s=0.001,
    )
    stream = _build_stream(backend)
    started, _pcm, completed = await stream.collect()
    # Started carries the advertised (unverified) cadence.
    assert started.streaming_capability.mode == TtsStreamingMode.SENTENCE_CHUNK
    assert started.streaming_capability.verified is False
    # Completed carries the observed truth — downgraded.
    assert completed.observed_chunks == 1
    assert completed.streaming_capability.mode == TtsStreamingMode.FINAL_CHUNK
    assert completed.streaming_capability.verified is False
    assert completed.capability_verified is False


@pytest.mark.asyncio
async def test_advertised_sentence_chunk_with_multiple_chunks_is_verified_true():
    backend = CapabilityAdvertisingBackend(
        advertised=TtsStreamingCapability.sentence(verified=False),
        num_chunks=3,
        chunk_delay_s=0.001,
    )
    stream = _build_stream(backend)
    started, _pcm, completed = await stream.collect()
    assert started.streaming_capability.mode == TtsStreamingMode.SENTENCE_CHUNK
    assert completed.observed_chunks == 3
    assert completed.streaming_capability.mode == TtsStreamingMode.SENTENCE_CHUNK
    assert completed.streaming_capability.verified is True
    assert completed.capability_verified is True


@pytest.mark.asyncio
async def test_advertised_progressive_with_multiple_chunks_is_verified_true():
    backend = CapabilityAdvertisingBackend(
        advertised=TtsStreamingCapability.progressive(verified=False),
        num_chunks=4,
        chunk_delay_s=0.001,
    )
    stream = _build_stream(backend)
    started, _pcm, completed = await stream.collect()
    assert started.streaming_capability.mode == TtsStreamingMode.PROGRESSIVE
    assert started.streaming_capability.granularity == TtsStreamingGranularity.FRAME
    assert completed.streaming_capability.mode == TtsStreamingMode.PROGRESSIVE
    assert completed.capability_verified is True


# ---------------------------------------------------------------------------
# Honest metrics — setup_ms / engine_first_chunk_ms / e2e_first_chunk_ms /
# total_latency_ms must measure the boundaries their names claim.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_metrics_measure_their_named_boundaries():
    """All four metrics are advertised by name — verify they actually
    measure those boundaries:

      * setup_ms <= e2e_first_chunk_ms <= total_latency_ms
      * engine_first_chunk_ms <= e2e_first_chunk_ms (engine TTFB is a
        subset of customer-visible TTFB by construction)
      * total_latency_ms is large enough to span all the chunks.
    """
    chunk_delay = 0.03
    num_chunks = 4
    backend = FakeStreamingBackend(num_chunks=num_chunks, chunk_delay_s=chunk_delay)
    stream = _build_stream(backend)
    _started, _pcm, completed = await stream.collect()

    # Every metric must be a non-negative finite float.
    assert completed.setup_ms >= 0.0
    assert completed.total_latency_ms >= 0.0
    assert completed.engine_first_chunk_ms is not None
    assert completed.e2e_first_chunk_ms is not None
    assert completed.engine_first_chunk_ms >= 0.0
    assert completed.e2e_first_chunk_ms >= 0.0

    # Setup happens BEFORE first chunk arrives at the consumer.
    assert completed.setup_ms <= completed.e2e_first_chunk_ms + 1.0  # slack for clock noise

    # First-chunk arrival is BEFORE total completion.
    assert completed.e2e_first_chunk_ms <= completed.total_latency_ms

    # Engine TTFB is a subset of e2e TTFB (e2e includes setup).
    assert completed.engine_first_chunk_ms <= completed.e2e_first_chunk_ms + 1.0

    # Total wall-time must be at least the synthesis budget — fake
    # backend yields num_chunks chunks at chunk_delay each, so total
    # synthesis is ~num_chunks * chunk_delay seconds. We expect total
    # latency to be at least 80% of that (CI clock noise floor).
    expected_synth_ms = num_chunks * chunk_delay * 1000.0
    assert completed.total_latency_ms >= expected_synth_ms * 0.8


@pytest.mark.asyncio
async def test_kernel_honors_sdk_t0_argument_for_setup_ms():
    """The kernel-level stream builder measures setup_ms from
    whatever ``sdk_t0`` it's handed — the boundary policy is the
    *caller's* responsibility (``FacadeSpeech.stream`` captures it
    on first iteration). Verify the kernel honors the value.
    """
    from octomil.execution.kernel import _build_local_realtime_stream

    backend = FakeStreamingBackend(num_chunks=2, chunk_delay_s=0.001)
    sdk_t0 = time.monotonic() - 0.10  # pretend SDK work started 100ms ago

    stream = _build_local_realtime_stream(
        backend=backend,
        text="hello world",
        voice="af_bella",
        speed=1.0,
        runtime_model="kokoro-82m",
        policy_preset="local_only",
        fallback_used=False,
        sdk_t0=sdk_t0,
    )
    _started, _pcm, completed = await stream.collect()

    # The synthetic 100ms gap must be visible in setup_ms and bound
    # e2e_first_chunk_ms / total_latency_ms from below.
    assert completed.setup_ms >= 90.0  # slack
    assert completed.e2e_first_chunk_ms is not None
    assert completed.e2e_first_chunk_ms >= 90.0
    assert completed.total_latency_ms >= 90.0


@pytest.mark.asyncio
async def test_idle_time_before_iteration_does_not_count_against_setup_ms():
    """Caller idle time between ``client.audio.speech.stream(...)`` and
    the first ``async for`` MUST NOT be charged to setup_ms / e2e /
    total. The boundary is "iteration begins," not "object constructed."

    Regression for: a caller that creates the stream, awaits something
    else for 200ms, then iterates would otherwise see 200ms+ of fake
    setup_ms attributed to SDK work.
    """
    from octomil.audio.speech import FacadeSpeech

    backend = FakeStreamingBackend(num_chunks=2, chunk_delay_s=0.001)

    # A fake kernel whose ``synthesize_speech_stream`` returns the
    # kernel-level builder against the fake backend. We aren't going
    # through routing; we just need a real ``sdk_t0`` round-trip.
    class _FakeKernel:
        async def synthesize_speech_stream(self, **kwargs):
            from octomil.execution.kernel import _build_local_realtime_stream

            return _build_local_realtime_stream(
                backend=backend,
                text=kwargs["input"],
                voice=kwargs["voice"],
                speed=kwargs["speed"],
                runtime_model=kwargs["model"],
                policy_preset="local_only",
                fallback_used=False,
                sdk_t0=kwargs["sdk_t0"],
            )

    facade = FacadeSpeech(_FakeKernel())

    stream = facade.stream(model="kokoro-82m", input="hello", voice="af_bella")
    # Caller-side idle time — sleep BEFORE iterating. Under the bug
    # this 150ms would show up in setup_ms / total_latency_ms.
    await asyncio.sleep(0.15)

    _started, _pcm, completed = await stream.collect()

    # Generous upper bound: the actual SDK work is microseconds (no
    # routing, just the fake builder), so total wall-clock should be
    # well under 50ms even on slow CI. The 150ms idle MUST NOT be in
    # the metrics.
    assert completed.setup_ms < 50.0, (
        f"setup_ms={completed.setup_ms:.1f}ms — caller idle time leaked into "
        f"the metric. sdk_t0 must be captured on first iteration, not at "
        f"stream() construction."
    )
    assert completed.total_latency_ms < 50.0
    assert completed.e2e_first_chunk_ms is not None
    assert completed.e2e_first_chunk_ms < 50.0


# ---------------------------------------------------------------------------
# _backend_can_stream — replaces the legacy supports_streaming bool
# ---------------------------------------------------------------------------


def test_backend_can_stream_detects_synthesize_stream_method():
    """A backend is a streaming backend iff it implements
    ``synthesize_stream``. The legacy ``supports_streaming`` attribute
    is not consulted any more; this is the contract every test that
    builds a fake backend depends on."""
    from octomil.execution.kernel import _backend_can_stream

    streaming = FakeStreamingBackend(num_chunks=1)
    assert _backend_can_stream(streaming) is True

    class NonStreamingBackend:
        # Has the legacy bool but NOT the method — must NOT be treated as streaming.
        supports_streaming = True

        def synthesize(self, text, voice=None, speed=1.0):
            return {"audio_bytes": b"", "sample_rate": 24000}

    assert _backend_can_stream(NonStreamingBackend()) is False


# ---------------------------------------------------------------------------
# Sherpa engine — _count_sentences + streaming_capability
# Pure-function unit tests (no sherpa-onnx import required).
# ---------------------------------------------------------------------------


def test_count_sentences_single_sentence_returns_one():
    from octomil.runtime.engines.sherpa.engine import _count_sentences

    # Single sentence — no terminator inside the body.
    assert _count_sentences("Hello there.") == 1
    assert _count_sentences("hello") == 1
    # Trailing terminator with no following text is still one sentence.
    assert _count_sentences("Wait!") == 1


def test_count_sentences_multi_sentence_matches_kokoro_chunking():
    """The counter must agree with what sherpa-onnx's
    ``max_num_sentences=1`` would actually do — split on terminator
    + whitespace + non-whitespace, which is the same regex sherpa
    uses internally for sentence boundaries."""
    from octomil.runtime.engines.sherpa.engine import _count_sentences

    assert _count_sentences("Hello there. How are you?") == 2
    assert _count_sentences("First. Second. Third.") == 3
    # Mixed terminators (punctuation + question + exclamation).
    assert _count_sentences("Welcome! Are you ready? Let's go.") == 3
    # CJK terminators participate when followed by whitespace + char,
    # mirroring sherpa-onnx's split-on-boundary behaviour.
    assert _count_sentences("你好。 今天很好。 再见。") >= 2


def test_sherpa_streaming_capability_advertises_sentence_chunk_only_for_multi_sentence():
    """The engine's per-input capability advertisement must be
    truthful: single-sentence input gets ``final_chunk`` because
    Kokoro will only invoke the callback once. The bug this prevents:
    advertising sentence_chunk for a single-sentence prompt and then
    delivering one chunk → fake realtime."""
    # Light-weight construction: skip __init__ so we don't load sherpa.
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend.__new__(_SherpaTtsBackend)

    cap_single = backend.streaming_capability("Hello there.")
    assert cap_single.mode == TtsStreamingMode.FINAL_CHUNK
    assert cap_single.granularity == TtsStreamingGranularity.UTTERANCE
    assert cap_single.verified is False

    cap_multi = backend.streaming_capability("Hello there. How are you?")
    assert cap_multi.mode == TtsStreamingMode.SENTENCE_CHUNK
    assert cap_multi.granularity == TtsStreamingGranularity.SENTENCE
    assert cap_multi.verified is False


@pytest.mark.asyncio
async def test_pocket_missing_reference_raises_before_started_is_emitted():
    """P1 regression — pre-fix, ``backend.synthesize_stream()`` was
    an async generator, so Pocket's missing-reference validation
    inside the generator body would only run when the consumer
    started iterating. ``_build_local_realtime_stream`` would
    yield ``SpeechStreamStarted`` first, so an invalid Pocket
    speaker profile looked successfully started before failing.

    Post-fix: backends expose ``validate_speaker_profile`` and the
    builder calls it synchronously BEFORE constructing the
    producer. A missing-reference profile raises during
    ``_build_local_realtime_stream`` itself; the consumer never
    receives a SpeechStream object that would emit Started."""
    from octomil.errors import OctomilError, OctomilErrorCode

    started_yielded = False

    class _PocketStubBackend:
        """Minimal Pocket-shaped backend whose
        ``validate_speaker_profile`` raises for missing
        reference_audio (mirrors the real engine)."""

        _sample_rate = SAMPLE_RATE
        _default_voice = ""
        _model_name = "pocket-tts-int8"
        supports_streaming = True

        def validate_speaker_profile(self, speaker_profile):
            if speaker_profile is None or not getattr(speaker_profile, "reference_audio", None):
                raise OctomilError(
                    code=OctomilErrorCode.INVALID_INPUT,
                    message=(
                        "speaker_profile_missing_reference: PocketTTS requires "
                        "a planner-supplied speaker profile with reference_audio."
                    ),
                )

        def validate_voice(self, voice):
            return 0, ""

        async def synthesize_stream(self, text, voice=None, speed=1.0, **kw):
            nonlocal started_yielded
            started_yielded = True
            yield {"pcm_s16le": b"\x00\x00", "num_samples": 1, "sample_rate": SAMPLE_RATE}

    from octomil.execution.kernel import _build_local_realtime_stream

    backend = _PocketStubBackend()
    # No speaker profile passed → Pocket validation must raise
    # synchronously, before any SpeechStream is constructed.
    with pytest.raises(OctomilError) as ei:
        _build_local_realtime_stream(
            backend=backend,
            text="hello",
            voice=None,
            resolved_speaker=None,
            speed=1.0,
            runtime_model="pocket-tts-int8",
            policy_preset="local_only",
            fallback_used=False,
            sdk_t0=time.monotonic(),
        )
    assert "speaker_profile_missing_reference" in str(ei.value)
    # The async generator's body never ran.
    assert started_yielded is False


# ---------------------------------------------------------------------------
# Hard-cutover invariants — the v4.13 ``StreamingMode`` alias and
# the ``streaming_mode`` / ``latency_ms`` / ``first_chunk_ms``
# compat properties have been removed. These tests pin the new
# canonical API and assert the old names DO NOT resolve, so any
# attempt to re-add a compat shim fails the build.
# ---------------------------------------------------------------------------


def test_streaming_mode_alias_no_longer_exported():
    """v4.13's ``StreamingMode`` alias was removed in the hard
    cutover. ``TtsStreamingMode`` is the canonical name; any
    consumer importing the old name must migrate."""
    import octomil.audio.streaming as streaming

    assert hasattr(streaming, "TtsStreamingMode")
    assert not hasattr(streaming, "StreamingMode"), (
        "StreamingMode alias is gone post-cutover; consumers must " "import TtsStreamingMode."
    )


def test_started_and_completed_no_longer_expose_v4_13_attribute_names():
    """``SpeechStreamStarted.streaming_mode`` and
    ``SpeechStreamCompleted.{streaming_mode,latency_ms,first_chunk_ms}``
    are gone. Use ``streaming_capability.mode`` /
    ``total_latency_ms`` / ``e2e_first_chunk_ms``."""
    from octomil.audio.streaming import (
        SpeechStreamCompleted,
        SpeechStreamStarted,
        TtsStreamingCapability,
        TtsStreamingMode,
    )

    started = SpeechStreamStarted(
        model="kokoro-82m",
        voice="af_bella",
        sample_rate=24000,
        channels=1,
        sample_format=SAMPLE_FORMAT_PCM_S16LE,
        streaming_capability=TtsStreamingCapability.sentence(verified=True),
        locality="on_device",
    )
    assert not hasattr(started, "streaming_mode"), "streaming_mode property removed; use streaming_capability.mode"
    assert started.streaming_capability.mode is TtsStreamingMode.SENTENCE_CHUNK

    completed = SpeechStreamCompleted(
        duration_ms=1000,
        total_samples=24000,
        sample_rate=24000,
        channels=1,
        sample_format=SAMPLE_FORMAT_PCM_S16LE,
        streaming_capability=TtsStreamingCapability.final_only(verified=True),
        setup_ms=12.0,
        engine_first_chunk_ms=5.0,
        e2e_first_chunk_ms=18.0,
        total_latency_ms=950.0,
        observed_chunks=1,
        capability_verified=True,
    )
    assert not hasattr(completed, "streaming_mode")
    assert not hasattr(completed, "latency_ms")
    assert not hasattr(completed, "first_chunk_ms")
    # Canonical fields are reachable under their new names.
    assert completed.streaming_capability.mode is TtsStreamingMode.FINAL_CHUNK
    assert completed.total_latency_ms == 950.0
    assert completed.e2e_first_chunk_ms == 18.0
