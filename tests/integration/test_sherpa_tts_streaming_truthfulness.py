"""Integration test — sherpa-onnx Kokoro streaming truthfulness contract.

This is the test that *would have caught* the v4.13.0 fake-realtime
regression. It exercises the real sherpa-onnx engine against a real
prepared Kokoro artifact and asserts the cadence advertised by the
SDK matches the cadence the engine actually delivers.

Skip-gated on:

  * ``sherpa-onnx`` importing cleanly (dlopen + ONNX runtime present),
  * a prepared Kokoro artifact under ``~/.cache/octomil/artifacts/``
    containing ``model.onnx``, ``tokens.txt``, ``voices.bin``, and
    ``espeak-ng-data/``.

When both are present, the test must FAIL if the engine ever returns
``chunk_count == 1`` while advertising ``sentence_chunk`` — that's the
exact condition the SDK is now contractually forbidden from claiming.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from octomil.audio.streaming import (
    SpeechAudioChunk,
    SpeechStreamCompleted,
    SpeechStreamStarted,
    TtsStreamingMode,
)

# ---------------------------------------------------------------------------
# Skip-gating
# ---------------------------------------------------------------------------


def _sherpa_loadable() -> bool:
    try:
        importlib.import_module("sherpa_onnx")
    except Exception:
        return False
    return True


def _find_prepared_kokoro_dir() -> Path | None:
    """Return a prepared Kokoro artifact dir or ``None``.

    Looks under the standard PrepareManager cache root. We require the
    full set of files the sherpa-onnx Kokoro config needs — partial
    bundles must NOT cause a flaky/red test.
    """
    cache_root = Path.home() / ".cache" / "octomil" / "artifacts" / "artifacts"
    if not cache_root.is_dir():
        return None
    required = ("model.onnx", "tokens.txt", "voices.bin")
    for child in sorted(cache_root.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("kokoro-"):
            continue
        if all((child / f).is_file() for f in required) and (child / "espeak-ng-data").is_dir():
            return child
    return None


_SKIP_REASON: str | None = None
if not _sherpa_loadable():
    _SKIP_REASON = "sherpa-onnx not importable (install octomil[tts] and ensure dylibs resolve)"
else:
    _PREPARED = _find_prepared_kokoro_dir()
    if _PREPARED is None:
        _SKIP_REASON = (
            "no prepared Kokoro artifact under ~/.cache/octomil/artifacts/ — "
            "run `octomil prepare kokoro-82m --capability tts` first"
        )

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MULTI_SENTENCE_TEXT = (
    "Welcome to the streaming truthfulness probe. "
    "This sentence is the second one. "
    "And this third sentence pushes the engine well past the single-callback "
    "threshold so the cadence claim can be verified honestly."
)
_SINGLE_SENTENCE_TEXT = "Hello there from the single-sentence probe."


def _build_real_sherpa_backend():
    """Construct the real sherpa-onnx Kokoro backend against the prepared dir.

    Skips at module-load time mean ``_PREPARED`` is non-None here.
    """
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend(model_dir=str(_PREPARED))  # type: ignore[name-defined]
    backend.load_model("kokoro-82m")
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_sentence_kokoro_actually_streams_multiple_chunks():
    """Long multi-sentence input MUST produce >1 audio chunk.

    This is the canary for fake-realtime: if Kokoro batches the whole
    utterance into one callback (the v4.13.0 bug), this test fails
    with ``observed_chunks == 1``. The fix is ``max_num_sentences=1``
    in OfflineTtsConfig — verified here by exercising the production
    path end-to-end."""
    import time

    from octomil.execution.kernel import _build_local_realtime_stream

    backend = _build_real_sherpa_backend()

    stream = _build_local_realtime_stream(
        backend=backend,
        text=_MULTI_SENTENCE_TEXT,
        voice="af_bella",
        speed=1.0,
        runtime_model="kokoro-82m",
        policy_preset="local_only",
        fallback_used=False,
        sdk_t0=time.monotonic(),
    )

    started, _pcm, completed = await stream.collect()

    assert isinstance(started, SpeechStreamStarted)
    assert isinstance(completed, SpeechStreamCompleted)

    # Advertised cadence for multi-sentence input must be sentence_chunk.
    assert (
        started.streaming_capability.mode == TtsStreamingMode.SENTENCE_CHUNK
    ), f"expected SENTENCE_CHUNK advertisement for multi-sentence input, got {started.streaming_capability.mode}"

    # The truthfulness contract: advertised sentence_chunk + observed
    # ``chunk_count == 1`` is the exact fake-realtime regression. If
    # this assertion fails, sherpa-onnx is batching everything again
    # (likely OfflineTtsConfig.max_num_sentences was reset).
    assert completed.observed_chunks > 1, (
        f"FAKE-REALTIME REGRESSION: SDK advertised sentence_chunk but engine "
        f"delivered {completed.observed_chunks} chunk(s). Check that "
        f"OfflineTtsConfig(max_num_sentences=1) is still set in "
        f"sherpa engine load_model()."
    )

    # Verification flag must be flipped on by the kernel completion path.
    assert completed.streaming_capability.mode == TtsStreamingMode.SENTENCE_CHUNK
    assert completed.streaming_capability.verified is True
    assert completed.capability_verified is True


@pytest.mark.asyncio
async def test_single_sentence_kokoro_advertises_final_chunk_truthfully():
    """Single-sentence input is honestly final_chunk — Kokoro will only
    invoke its callback once, so the SDK must not pretend otherwise."""
    import time

    from octomil.execution.kernel import _build_local_realtime_stream

    backend = _build_real_sherpa_backend()

    stream = _build_local_realtime_stream(
        backend=backend,
        text=_SINGLE_SENTENCE_TEXT,
        voice="af_bella",
        speed=1.0,
        runtime_model="kokoro-82m",
        policy_preset="local_only",
        fallback_used=False,
        sdk_t0=time.monotonic(),
    )
    started, _pcm, completed = await stream.collect()

    assert started.streaming_capability.mode == TtsStreamingMode.FINAL_CHUNK
    assert completed.observed_chunks == 1
    assert completed.streaming_capability.mode == TtsStreamingMode.FINAL_CHUNK
    assert completed.capability_verified is True


@pytest.mark.asyncio
async def test_first_chunk_arrives_strictly_before_completion():
    """A real engine must produce the first PCM bytes BEFORE
    ``SpeechStreamCompleted`` is emitted. If the gap collapses to
    zero, we are buffering the whole utterance — i.e. fake realtime
    even if multiple chunks eventually arrive."""
    import time

    from octomil.execution.kernel import _build_local_realtime_stream

    backend = _build_real_sherpa_backend()

    stream = _build_local_realtime_stream(
        backend=backend,
        text=_MULTI_SENTENCE_TEXT,
        voice="af_bella",
        speed=1.0,
        runtime_model="kokoro-82m",
        policy_preset="local_only",
        fallback_used=False,
        sdk_t0=time.monotonic(),
    )

    first_chunk_at: float | None = None
    completed_at: float | None = None
    t0 = time.monotonic()
    async for event in stream:
        if isinstance(event, SpeechAudioChunk) and first_chunk_at is None:
            first_chunk_at = time.monotonic() - t0
        elif isinstance(event, SpeechStreamCompleted):
            completed_at = time.monotonic() - t0

    assert first_chunk_at is not None
    assert completed_at is not None
    # Generous slack — even fast machines should leave at least 50ms
    # between first sentence boundary and final completion for this
    # ~3-sentence input. If they're equal, we're buffering.
    assert first_chunk_at < completed_at - 0.05, (
        f"first chunk {first_chunk_at:.3f}s should arrive well before "
        f"completion {completed_at:.3f}s — engine is buffering, not streaming"
    )


@pytest.mark.asyncio
async def test_completion_metrics_show_engine_ttfb_below_e2e_ttfb():
    """``engine_first_chunk_ms`` measures from inside the producer
    after Started emission; ``e2e_first_chunk_ms`` measures from the
    SDK call boundary. By construction engine TTFB <= e2e TTFB."""
    import time

    from octomil.execution.kernel import _build_local_realtime_stream

    backend = _build_real_sherpa_backend()
    stream = _build_local_realtime_stream(
        backend=backend,
        text=_MULTI_SENTENCE_TEXT,
        voice="af_bella",
        speed=1.0,
        runtime_model="kokoro-82m",
        policy_preset="local_only",
        fallback_used=False,
        sdk_t0=time.monotonic(),
    )
    _started, _pcm, completed = await stream.collect()

    assert completed.engine_first_chunk_ms is not None
    assert completed.e2e_first_chunk_ms is not None
    # Engine TTFB is a strict subset of e2e TTFB — give 1ms of slack
    # for the time.monotonic() ordering between the two measurements.
    assert completed.engine_first_chunk_ms <= completed.e2e_first_chunk_ms + 1.0
    assert completed.setup_ms <= completed.e2e_first_chunk_ms + 1.0
    assert completed.e2e_first_chunk_ms <= completed.total_latency_ms
