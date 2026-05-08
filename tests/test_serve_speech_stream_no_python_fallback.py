"""v0.1.8 Lane C — server-side hard-cut tests for /v1/audio/speech/stream.

Pins:
  1. ``/v1/audio/speech/stream`` does NOT fall back to the Python
     sherpa engine when the native backend is unavailable. Instead
     it returns a typed RUNTIME_UNAVAILABLE error envelope.
  2. When the native backend IS available, the route uses it
     (NativeTtsStreamBackend.synthesize_with_chunks) and the
     legacy python-sherpa ``synthesize_stream`` is NEVER called.
  3. Voice validation runs BEFORE FastAPI emits HTTP 200 — bad
     voices surface as a typed 4xx body, not a mid-stream error
     after the consumer has attached.
  4. The honesty header ``X-Octomil-Streaming-Honesty: coalesced_after_synthesis``
     is on the response.

Tests reuse the same lifespan/app construction pattern as
``test_audio_speech_stream.py`` so the production code path
(``octomil/serve/app.py:create_app``) is exercised end-to-end.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from octomil.errors import OctomilError, OctomilErrorCode

pytestmark = [pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Helpers — fake backends + lifespan kickoff (mirrors test_audio_speech_stream)
# ---------------------------------------------------------------------------


class _FakeNativeTtsStreamBackend:
    """Stand-in for NativeTtsStreamBackend; emits synthetic chunks
    without any runtime dependency. Mirrors the real backend's
    public surface (``name``, ``load_model``, ``close``,
    ``validate_voice``, ``synthesize_with_chunks``)."""

    name = "native-sherpa-onnx-tts-stream"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.load_model_called = False
        self.closed = False

    def load_model(self, model_name: str) -> None:
        self.load_model_called = True

    def close(self) -> None:
        self.closed = True

    def validate_voice(self, voice: str | None) -> str:
        if voice is None or voice == "":
            return "0"
        v = str(voice).strip()
        if not v:
            return "0"
        if not v.isdigit():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"native TTS-stream: voice {voice!r} is not numeric",
            )
        return v

    def synthesize_with_chunks(self, text: str, *, voice_id: str | None = None) -> Any:
        self.calls.append({"text": text, "voice_id": voice_id})
        from octomil.runtime.native.tts_stream_backend import TtsAudioChunk

        # Two synthetic sentence chunks, ~100ms each at 22050 Hz.
        n_per_chunk = 2205
        cum = 0
        chunks = []
        for i in range(2):
            pcm = np.zeros(n_per_chunk, dtype=np.float32)
            cum += n_per_chunk
            chunks.append(
                TtsAudioChunk(
                    pcm_f32=pcm,
                    sample_rate_hz=22050,
                    chunk_index=i,
                    is_final=(i == 1),
                    cumulative_duration_ms=int(cum * 1000 / 22050),
                )
            )
        return iter(chunks)


class _FakePythonSherpaBackend:
    """Stand-in for the python-sherpa backend; raises on
    ``synthesize_stream`` so any accidental fallback fails the test."""

    name = "fake-python-sherpa"

    def __init__(self) -> None:
        self._sample_rate = 22050
        self._default_voice = "0"
        self._model_name = "kokoro-82m"

    @property
    def supports_streaming(self) -> bool:
        return True

    def load_model(self, model_name: str) -> None:
        pass

    def synthesize_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "v0.1.8 Lane C hard cut: python sherpa synthesize_stream "
            "MUST NOT be reached on the /v1/audio/speech/stream route."
        )

    def synthesize(self, *args: Any, **kwargs: Any) -> Any:
        return {"audio_bytes": b"", "content_type": "audio/wav"}


async def _start_lifespan(app: Any) -> None:
    cm = app.router.lifespan_context(app)
    await cm.__aenter__()


def _stub_kernel_warmup_for_tts(tmp_path: Any) -> Any:
    fake_prepare_outcome = MagicMock()
    fake_prepare_outcome.artifact_dir = str(tmp_path)
    return patch(
        "octomil.execution.kernel.ExecutionKernel.prepare",
        return_value=fake_prepare_outcome,
    )


def _make_app(
    *,
    native_factory: Any,
    sherpa_backend: Any,
    tmp_path: Any,
) -> Any:
    """Build the production ``create_app`` with the
    ``NativeTtsStreamBackend`` and ``SherpaTtsEngine`` factories
    patched. Returns the FastAPI app + the python-sherpa fake.
    """
    pytest.importorskip("fastapi")

    fake_engine = MagicMock()
    fake_engine.create_backend.return_value = sherpa_backend

    return (
        patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", return_value=fake_engine),
        patch("octomil.serve.app._is_sherpa_tts_model", return_value=True),
        _stub_kernel_warmup_for_tts(tmp_path),
        patch(
            "octomil.runtime.native.tts_stream_backend.NativeTtsStreamBackend",
            side_effect=native_factory,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_no_native_backend_returns_runtime_unavailable(tmp_path: Any) -> None:
    """When the native backend factory raises (no dylib / no
    runtime / sidecars missing) at startup, the stream route MUST
    return RUNTIME_UNAVAILABLE rather than fall back to the
    python-sherpa stream."""
    pytest.importorskip("httpx")
    from httpx import ASGITransport, AsyncClient

    sherpa_backend = _FakePythonSherpaBackend()

    def native_factory() -> Any:
        raise RuntimeError("simulated: no dylib advertising audio.tts.stream")

    patches = _make_app(
        native_factory=native_factory,
        sherpa_backend=sherpa_backend,
        tmp_path=tmp_path,
    )
    with patches[0], patches[1], patches[2], patches[3]:
        from octomil.serve.app import create_app

        app = create_app("kokoro-82m")
        await _start_lifespan(app)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/audio/speech/stream",
                json={"model": "kokoro-82m", "input": "hello world"},
            )
        assert resp.status_code in (400, 422, 500, 503), f"expected typed 4xx/5xx, got {resp.status_code}: {resp.text}"
        body_text = resp.text
        # Response surfaces RUNTIME_UNAVAILABLE somewhere in the
        # error envelope.
        assert "RUNTIME_UNAVAILABLE" in body_text or "runtime" in body_text.lower()


async def test_native_backend_drives_route_when_available(tmp_path: Any) -> None:
    """When the native backend IS available, the route MUST use
    it. The python-sherpa ``synthesize_stream`` is wired to raise
    AssertionError on call so any fallback fails the test."""
    pytest.importorskip("httpx")
    from httpx import ASGITransport, AsyncClient

    sherpa_backend = _FakePythonSherpaBackend()
    native_instance = _FakeNativeTtsStreamBackend()

    def native_factory() -> Any:
        return native_instance

    patches = _make_app(
        native_factory=native_factory,
        sherpa_backend=sherpa_backend,
        tmp_path=tmp_path,
    )
    with patches[0], patches[1], patches[2], patches[3]:
        from octomil.serve.app import create_app

        app = create_app("kokoro-82m")
        await _start_lifespan(app)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with client.stream(
                "POST",
                "/v1/audio/speech/stream",
                json={"model": "kokoro-82m", "input": "hello world", "voice": "0"},
            ) as resp:
                assert resp.status_code == 200, await resp.aread()
                # Honesty header pinned.
                assert resp.headers.get("x-octomil-streaming-honesty") == "coalesced_after_synthesis"
                # Backend tag must point at the native backend.
                assert "native" in resp.headers.get("x-octomil-backend", "")
                received = 0
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        received += 1
                assert received >= 1

        # The native backend was driven; python-sherpa was NOT.
        assert native_instance.load_model_called is True
        assert len(native_instance.calls) == 1
        assert native_instance.calls[0]["text"] == "hello world"
        assert native_instance.calls[0]["voice_id"] == "0"


async def test_invalid_voice_raises_before_streaming_response(tmp_path: Any) -> None:
    """Pre-stream voice validation: a non-numeric voice MUST
    surface as a typed 4xx body, not a mid-stream error."""
    pytest.importorskip("httpx")
    from httpx import ASGITransport, AsyncClient

    sherpa_backend = _FakePythonSherpaBackend()
    native_instance = _FakeNativeTtsStreamBackend()

    def native_factory() -> Any:
        return native_instance

    patches = _make_app(
        native_factory=native_factory,
        sherpa_backend=sherpa_backend,
        tmp_path=tmp_path,
    )
    with patches[0], patches[1], patches[2], patches[3]:
        from octomil.serve.app import create_app

        app = create_app("kokoro-82m")
        await _start_lifespan(app)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/audio/speech/stream",
                json={
                    "model": "kokoro-82m",
                    "input": "hello",
                    "voice": "af_bella",  # non-numeric — INVALID_INPUT.
                },
            )
        assert 400 <= resp.status_code < 500, f"expected typed 4xx, got {resp.status_code}: {resp.text}"
        body_text = resp.text
        assert "INVALID_INPUT" in body_text or "voice" in body_text.lower()
        # Synthesizer never called — voice rejected first.
        assert native_instance.calls == []


async def test_no_python_sherpa_synthesize_stream_in_module_path(tmp_path: Any) -> None:
    """Static / runtime guard: the production /v1/audio/speech/stream
    handler should NOT reach into the python-sherpa
    ``synthesize_stream`` codepath. We verify by replacing
    ``synthesize_stream`` with an asserting MagicMock and confirming
    a successful streaming request never touches it.

    (Belt-and-suspenders alongside the AssertionError side-effect
    on the fake — this test pins the audit invariant explicitly.)"""
    pytest.importorskip("httpx")
    from httpx import ASGITransport, AsyncClient

    sherpa_backend = _FakePythonSherpaBackend()
    sherpa_backend.synthesize_stream = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("python sherpa synthesize_stream MUST NOT be reachable")
    )
    native_instance = _FakeNativeTtsStreamBackend()

    def native_factory() -> Any:
        return native_instance

    patches = _make_app(
        native_factory=native_factory,
        sherpa_backend=sherpa_backend,
        tmp_path=tmp_path,
    )
    with patches[0], patches[1], patches[2], patches[3]:
        from octomil.serve.app import create_app

        app = create_app("kokoro-82m")
        await _start_lifespan(app)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with client.stream(
                "POST",
                "/v1/audio/speech/stream",
                json={"model": "kokoro-82m", "input": "hi"},
            ) as resp:
                assert resp.status_code == 200, await resp.aread()
                async for _ in resp.aiter_bytes():
                    pass

        sherpa_backend.synthesize_stream.assert_not_called()
