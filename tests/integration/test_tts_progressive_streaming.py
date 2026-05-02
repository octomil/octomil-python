"""Integration test — true progressive TTS streaming acceptance.

This test is the only place ``capability_verified=True`` for
:attr:`TtsStreamingMode.PROGRESSIVE` may be observed against a real
backend. Unit tests in ``tests/test_audio_speech_stream.py`` use a
``CapabilityAdvertisingBackend`` to drive the kernel's verification
branches; this file is what closes the loop against a real synthesis
path.

Skip-gated on:

  * ``OCTOMIL_RUN_PROGRESSIVE_TTS_INTEGRATION=1`` so the test never
    runs by accident on a dev box without an actual progressive
    backend wired up. Without a real backend that yields sub-sentence
    PCM during a single sentence, this test cannot pass.

  * ``OCTOMIL_PROGRESSIVE_TTS_BACKEND_FACTORY`` — dotted reference of
    the form ``module.path:attr`` resolving to a zero-argument
    callable that returns a ready-to-use backend instance whose
    ``streaming_capability(text)`` returns
    :func:`TtsStreamingCapability.progressive` for single-sentence
    input. This is the surface a real progressive backend (XTTS
    streaming, custom Kokoro patch, etc.) plugs into. Sherpa-onnx
    Kokoro / Piper / Pocket are sentence-boundary engines and will
    (correctly) be rejected by the assertions here, which is the
    point of the gate.

  * ``OCTOMIL_PROGRESSIVE_TTS_MODEL`` (optional) — runtime_model
    label passed to the kernel builder; defaults to
    ``"progressive-tts"`` so observability dashboards can filter.
  * ``OCTOMIL_PROGRESSIVE_TTS_VOICE`` (optional) — voice / speaker
    label to pass through. Defaults to ``None`` (backend default).
  * ``OCTOMIL_PROGRESSIVE_TTS_TEXT`` (optional) — synthesis input.
    Defaults to a single long sentence with no internal terminator
    boundaries, so the kernel's progressive verification gate
    (``count_sentences <= 1``) cannot be satisfied by sentence-
    boundary chunking.

Acceptance criteria (the "definition of done" for progressive TTS):

  * Started event advertises ``TtsStreamingMode.PROGRESSIVE`` with
    granularity ``FRAME``.
  * At least 2 :class:`SpeechAudioChunk` events arrive before
    completion.
  * Completed event reports
    ``streaming_capability.mode == PROGRESSIVE``,
    ``capability_verified=True``, ``observed_chunks > 1``, and
    ``e2e_first_chunk_ms < total_latency_ms``.

If the configured backend is sentence-boundary only (e.g. Kokoro),
this test will FAIL with the kernel having downgraded the completion
event to ``sentence_chunk`` — which is the correct behaviour. The
right response is "configure
``OCTOMIL_PROGRESSIVE_TTS_BACKEND_FACTORY`` to point at a real
progressive backend," NOT "loosen the assertions."

Pattern note
------------
This test bypasses the ``OctomilClient`` facade and builds the
stream directly via ``_build_local_realtime_stream`` — same approach
as ``test_sherpa_tts_streaming_truthfulness.py``. The facade requires
auth + kernel wiring that is irrelevant to the cadence-verification
question; calling the builder directly with the operator-supplied
backend is the cleanest way to exercise the progressive path without
provisioning a full client. ``FacadeSpeech.stream`` is also
synchronous (returns ``SpeechStream``), so a future facade-routed
variant of this test must NOT ``await`` it.
"""

from __future__ import annotations

import importlib
import os
import time

import pytest

from octomil.audio.streaming import (
    SpeechAudioChunk,
    SpeechStreamCompleted,
    SpeechStreamStarted,
    TtsStreamingGranularity,
    TtsStreamingMode,
)

_GATE_ENV = "OCTOMIL_RUN_PROGRESSIVE_TTS_INTEGRATION"
_FACTORY_ENV = "OCTOMIL_PROGRESSIVE_TTS_BACKEND_FACTORY"
_MODEL_ENV = "OCTOMIL_PROGRESSIVE_TTS_MODEL"
_VOICE_ENV = "OCTOMIL_PROGRESSIVE_TTS_VOICE"
_TEXT_ENV = "OCTOMIL_PROGRESSIVE_TTS_TEXT"

# A long single sentence with no internal terminator+space boundary,
# so a sentence-boundary backend cannot produce multi-chunk output
# from this prompt and accidentally satisfy the progressive gate.
_DEFAULT_SINGLE_SENTENCE_TEXT = (
    "the quick brown fox jumps over the lazy dog and then keeps running "
    "across the fields toward the horizon without ever once stopping or "
    "slowing down to catch its breath even as the daylight begins to fade"
)


pytestmark = pytest.mark.skipif(
    os.environ.get(_GATE_ENV) != "1",
    reason=(
        f"set {_GATE_ENV}=1 + {_FACTORY_ENV}=module.path:attr to run; "
        "this test requires a real progressive TTS backend"
    ),
)


def _resolve_backend_factory(spec: str):
    """Resolve a ``module.path:attr`` reference to a callable.

    Mirrors the entry-point pattern uvicorn / gunicorn use. Raises a
    pytest skip with a clear message when the reference is malformed
    or does not resolve, so the operator's first run produces an
    actionable error rather than a deep traceback.
    """
    if ":" not in spec:
        pytest.skip(f"{_FACTORY_ENV}={spec!r} is malformed; expected 'module.path:attr'")
    module_path, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.skip(f"{_FACTORY_ENV} module {module_path!r} not importable: {exc}")
    factory = getattr(module, attr, None)
    if not callable(factory):
        pytest.skip(f"{_FACTORY_ENV} attr {attr!r} on {module_path!r} is not callable")
    return factory


@pytest.mark.asyncio
async def test_progressive_tts_emits_multiple_chunks_for_single_sentence():
    factory_spec = os.environ.get(_FACTORY_ENV)
    if not factory_spec:
        pytest.skip(f"{_FACTORY_ENV} not set")

    backend_factory = _resolve_backend_factory(factory_spec)
    backend = backend_factory()

    runtime_model = os.environ.get(_MODEL_ENV) or "progressive-tts"
    voice = os.environ.get(_VOICE_ENV) or None
    text = os.environ.get(_TEXT_ENV) or _DEFAULT_SINGLE_SENTENCE_TEXT

    # Bypass the OctomilClient facade — same pattern as
    # ``test_sherpa_tts_streaming_truthfulness.py``. The facade's
    # ``stream`` is synchronous (returns SpeechStream); a future
    # facade-routed variant must not ``await`` it.
    from octomil.execution.kernel import _build_local_realtime_stream

    stream = _build_local_realtime_stream(
        backend=backend,
        text=text,
        voice=voice,
        speed=1.0,
        runtime_model=runtime_model,
        policy_preset="local_only",
        fallback_used=False,
        sdk_t0=time.monotonic(),
    )

    started: SpeechStreamStarted | None = None
    completed: SpeechStreamCompleted | None = None
    chunk_count = 0
    first_chunk_seen_before_completion = False

    async for event in stream:
        if isinstance(event, SpeechStreamStarted):
            started = event
        elif isinstance(event, SpeechAudioChunk):
            chunk_count += 1
            if completed is None:
                first_chunk_seen_before_completion = True
        elif isinstance(event, SpeechStreamCompleted):
            completed = event

    assert started is not None, "no SpeechStreamStarted event observed"
    assert completed is not None, "no SpeechStreamCompleted event observed"

    # 1. Backend must advertise progressive cadence on Started.
    assert started.streaming_capability.mode == TtsStreamingMode.PROGRESSIVE, (
        f"backend advertised {started.streaming_capability.mode!r}; "
        f"point {_FACTORY_ENV} at a backend whose ``streaming_capability(text)`` "
        "returns TtsStreamingCapability.progressive() for single-sentence input."
    )
    assert started.streaming_capability.granularity == TtsStreamingGranularity.FRAME

    # 2. At least 2 chunks must arrive — single-chunk runs ARE the
    # downgrade case the kernel's verification gate exists for.
    assert chunk_count > 1, (
        f"observed only {chunk_count} chunk(s) for a single-sentence input; "
        "this is the sentence-boundary fallback, not progressive cadence"
    )
    assert first_chunk_seen_before_completion, (
        "first SpeechAudioChunk must arrive before SpeechStreamCompleted " "for progressive cadence to be meaningful"
    )

    # 3. Completion event reports the verified progressive truth.
    assert completed.streaming_capability.mode == TtsStreamingMode.PROGRESSIVE, (
        f"completion downgraded to {completed.streaming_capability.mode!r}; "
        "the configured backend yielded sentence-boundary or single-chunk "
        "output, not sub-sentence progressive PCM"
    )
    assert completed.capability_verified is True
    assert completed.observed_chunks > 1
    assert completed.observed_chunks == chunk_count

    # 4. e2e first-chunk latency must be strictly less than total
    # latency — the consumer-visible TTFB advantage progressive
    # promises. Without this, "progressive" reduces to a label.
    assert completed.e2e_first_chunk_ms is not None
    assert completed.total_latency_ms > 0.0
    assert completed.e2e_first_chunk_ms < completed.total_latency_ms, (
        f"e2e_first_chunk_ms={completed.e2e_first_chunk_ms} "
        f">= total_latency_ms={completed.total_latency_ms}; "
        "progressive cadence requires first-chunk-before-completion"
    )
