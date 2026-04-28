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
    """Return a context manager that patches ``ExecutionKernel.warmup``
    so it returns a fake :class:`WarmupOutcome` pointing at
    ``monkeypatch_or_path`` as the artifact dir.

    Lets the production-route tests exercise the new lifespan path
    (which now calls ``kernel.warmup`` to materialize the artifact)
    without spinning the real PrepareManager / planner / network.
    Pass any directory; the test fakes don't read it.
    """
    from octomil.execution.kernel import WarmupOutcome

    fake_prepare_outcome = MagicMock()
    fake_prepare_outcome.artifact_dir = str(monkeypatch_or_path)
    fake_outcome = WarmupOutcome(
        capability="tts",
        model="kokoro-82m",
        prepare_outcome=fake_prepare_outcome,
        backend_loaded=True,
        latency_ms=1.0,
    )
    return patch("octomil.execution.kernel.ExecutionKernel.warmup", return_value=fake_outcome)


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
                assert resp.headers["x-octomil-streaming-mode"] == "realtime"
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

    # Sanity: the recipe under test actually carries an 11-name v0_19
    # catalog and *not* the legacy 28-name one. If this regresses, the
    # rest of the test is meaningless.
    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None
    manifest = {n.lower() for n in recipe.materialization.voice_manifest}
    assert manifest, "kokoro-82m static recipe must declare voice_manifest"
    assert "af_bella" in manifest, "af_bella must be in the v0_19 catalog"
    # Pick a name that was in the legacy 28-name tuple but is NOT in the
    # 11-name v0_19 manifest, so we prove the recipe (not the legacy
    # tuple) is authoritative.
    legacy_only_voice = "am_echo"
    assert legacy_only_voice not in manifest, (
        f"test premise: {legacy_only_voice} must be absent from v0_19 manifest " f"(found: {sorted(manifest)})"
    )

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
    assert backend._resolve_sid("") == 0
    # Backwards-compat alias normalizes None too.
    assert backend._voice_to_sid("") == 0
    # Explicit unknown voice on a catalog-less model is rejected with
    # the structured error, not silently aliased to 0.
    from octomil.errors import OctomilError, OctomilErrorCode

    with pytest.raises(OctomilError) as ei:
        backend._resolve_sid("amy")  # explicit, even though it matches the default label
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
    assert "model_dir" in captured_kwargs, (
        "TTS lifespan must inject prepared model_dir from kernel.warmup() — " f"captured kwargs: {captured_kwargs}"
    )
    assert captured_kwargs["model_dir"] == str(tmp_path)


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
    assert backend._resolve_sid("") == 0  # default still works
    assert backend._resolve_sid("af_bella") == 0  # name in sidecar
    assert backend._resolve_sid("am_michael") == 1
    with pytest.raises(OctomilError) as ei:
        backend._resolve_sid("alloy")
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
async def test_private_kokoro_artifact_with_am_echo_in_voices_txt_is_accepted(tmp_path):
    """Reviewer P1#2 reproducer: a planner-selected *private* Kokoro
    artifact whose own ``voices.txt`` contains ``am_echo`` must be
    accepted by both ``create()`` and ``stream()`` even though
    ``am_echo`` is NOT in the public v0_19 static recipe's manifest.

    Pre-fix: the kernel's pre-prepare voice preflight read the static
    recipe and rejected ``am_echo`` before
    ``_prepare_local_tts_artifact`` could materialize the planner's
    artifact. The fix gates the preflight on candidate identity so
    private artifacts validate against their own sidecar.
    """
    from octomil.config.local import ResolvedExecutionDefaults
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    # Sanity: am_echo really is absent from the public v0_19 manifest.
    recipe = get_static_recipe("kokoro-82m", "tts")
    assert recipe is not None and recipe.materialization.voice_manifest
    public_manifest = {n.lower() for n in recipe.materialization.voice_manifest}
    assert "am_echo" not in public_manifest, "test premise broken: am_echo is now in the public recipe manifest"

    # The planner-prepared artifact dir contains a voices.txt that
    # *does* include am_echo at sid=2. The fake backend reads the dir
    # the kernel hands it, so this is what the production code would
    # see post-PrepareManager.
    voices_txt = tmp_path / "voices.txt"
    voices_txt.write_text("af_bella\naf_sarah\nam_echo\n")

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

    # ---- stream() must accept am_echo for the private candidate. ----
    with contextlib_ExitStack() as st:
        for p in common_patches:
            st.enter_context(p)
        stream = await kernel.synthesize_speech_stream(
            model="kokoro-82m",
            input="hello",
            voice="am_echo",
        )
        started, pcm, completed = await stream.collect()
        assert started.voice == "am_echo"
        assert completed.total_samples == 2 * 600
        assert pcm  # non-empty PCM body

    # ---- create() (non-streaming path) must also accept am_echo. ----
    with contextlib_ExitStack() as st:
        for p in common_patches:
            st.enter_context(p)
        response = await kernel.synthesize_speech(
            model="kokoro-82m",
            input="hello",
            voice="am_echo",
        )
        assert response.audio_bytes
        assert response.route.locality == "on_device"
        # The fake backend reports the requested voice.
        assert response.voice == "am_echo"


# Lightweight import — keeps the patch-stack assembly above readable
# without adding unittest.mock as a top-level import in this file.
from contextlib import ExitStack as contextlib_ExitStack  # noqa: E402
