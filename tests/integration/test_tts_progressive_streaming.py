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
  * ``OCTOMIL_PROGRESSIVE_TTS_MODEL`` — the model id (or ``@app/...``
    ref / artifact name) the SDK should route to. The selected model
    MUST resolve to a backend whose ``streaming_capability`` returns
    :func:`TtsStreamingCapability.progressive` for single-sentence
    input. Sherpa-onnx Kokoro / Piper does NOT — they're sentence-
    boundary, and will (correctly) be rejected by the assertions
    here, which is the point of the test.

  * ``OCTOMIL_PROGRESSIVE_TTS_VOICE`` (optional) — voice / speaker
    label to pass through. Falls back to backend default.
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

If the configured model/backend is sentence-boundary only (e.g.
Kokoro), this test will FAIL with the kernel having downgraded the
completion event to ``sentence_chunk`` — which is the correct
behavior. The right response is "configure
``OCTOMIL_PROGRESSIVE_TTS_MODEL`` to point at a real progressive
backend," NOT "loosen the assertions."
"""

from __future__ import annotations

import os

import pytest

from octomil.audio.streaming import (
    SpeechAudioChunk,
    SpeechStreamCompleted,
    SpeechStreamStarted,
    TtsStreamingGranularity,
    TtsStreamingMode,
)

_GATE_ENV = "OCTOMIL_RUN_PROGRESSIVE_TTS_INTEGRATION"
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
    reason=(f"set {_GATE_ENV}=1 and {_MODEL_ENV}=<model> to run; " "this test requires a real progressive TTS backend"),
)


@pytest.mark.asyncio
async def test_progressive_tts_emits_multiple_chunks_for_single_sentence():
    model = os.environ.get(_MODEL_ENV)
    if not model:
        pytest.skip(f"{_MODEL_ENV} not set")
    voice = os.environ.get(_VOICE_ENV) or None
    text = os.environ.get(_TEXT_ENV) or _DEFAULT_SINGLE_SENTENCE_TEXT

    # Lazy-import so a non-progressive checkout (no ``octomil.client``
    # extras installed) doesn't crash collection. The skip gate above
    # only allows entry when the operator has explicitly opted in.
    from octomil.client import Client  # type: ignore[import-not-found]

    client = Client()
    started: SpeechStreamStarted | None = None
    completed: SpeechStreamCompleted | None = None
    chunk_count = 0
    first_chunk_seen_before_completion = False

    stream = await client.audio.speech.stream(
        model=model,
        input=text,
        voice=voice,
        response_format="pcm",
    )
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
        f"set {_MODEL_ENV} to a model whose backend returns "
        "TtsStreamingCapability.progressive() for single-sentence input."
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
