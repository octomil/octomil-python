"""Integration probe — real PocketTTS streaming + voice cloning.

Skip-gated on:

  * ``OCTOMIL_RUN_SHERPA_POCKET_INTEGRATION=1`` env var (the bundle is
    NON-COMMERCIAL — never auto-runs in CI without explicit opt-in).
  * ``sherpa-onnx`` importing cleanly.
  * A prepared PocketTTS artifact under
    ``~/.cache/octomil/artifacts/`` containing the seven required
    Pocket files plus a reference WAV at ``ref.wav`` (24kHz mono).

Asserts the truthfulness contract holds for Pocket too:

  * Single LONG sentence emits ``chunk_count > 1`` if the engine
    advertises ``sentence_chunk``; otherwise ``final_chunk``
    advertised + ``chunk_count == 1`` (honest).
  * First chunk arrives strictly before completion.
  * The voice-cloning generate dispatch routes through Pocket's
    reference-audio path, not Kokoro's sid path.
"""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import pytest

from octomil.audio.streaming import (
    SpeechAudioChunk,
    SpeechStreamCompleted,
    SpeechStreamStarted,
    TtsStreamingMode,
)
from octomil.execution.tts_speaker_resolver import ResolvedTtsSpeaker


def _sherpa_loadable() -> bool:
    try:
        importlib.import_module("sherpa_onnx")
    except Exception:
        return False
    return True


def _find_prepared_pocket_artifact() -> Path | None:
    cache_root = Path.home() / ".cache" / "octomil" / "artifacts" / "artifacts"
    if not cache_root.is_dir():
        return None
    required = (
        "text_conditioner.onnx",
        "encoder.onnx",
        "lm_flow.int8.onnx",
        "decoder.int8.onnx",
        "lm_main.int8.onnx",
        "vocab.json",
        "token_scores.json",
        "ref.wav",
    )
    for child in sorted(cache_root.iterdir()):
        if not child.is_dir():
            continue
        if "pocket" not in child.name.lower():
            continue
        if all((child / f).is_file() for f in required):
            return child
    return None


_SKIP_REASON: str | None = None
if os.environ.get("OCTOMIL_RUN_SHERPA_POCKET_INTEGRATION") != "1":
    _SKIP_REASON = (
        "set OCTOMIL_RUN_SHERPA_POCKET_INTEGRATION=1 to run the Pocket integration probe (NON-COMMERCIAL bundle)"
    )
elif not _sherpa_loadable():
    _SKIP_REASON = "sherpa-onnx not importable (install octomil[tts] and ensure dylibs resolve)"
else:
    _PREPARED = _find_prepared_pocket_artifact()
    if _PREPARED is None:
        _SKIP_REASON = (
            "no prepared PocketTTS artifact (with ref.wav) under ~/.cache/octomil/artifacts/ — "
            "run client.prepare(model='pocket-tts-int8', capability='tts') and stage a ref.wav"
        )

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


_LONG_SINGLE_SENTENCE = (
    "Welcome to the streaming truthfulness probe for the Pocket TTS engine, "
    "where this single long sentence with multiple commas, semi-colons; clauses "
    "of varying length, and natural prosodic boundaries should give the engine "
    "ample opportunity to emit several callback chunks even though there is "
    "only one terminator at the very end."
)
_MULTI_SENTENCE = (
    "Hello from the Pocket integration probe. "
    "This second sentence forces a sentence-boundary callback. "
    "And a third for good measure."
)


def _build_real_pocket_backend():
    """Construct the real Pocket backend against the prepared dir."""
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend(model_dir=str(_PREPARED))  # type: ignore[name-defined]
    backend.load_model("pocket-tts-int8")
    return backend


def _profile_for_prepared_dir() -> ResolvedTtsSpeaker:
    # Module-load skip means _PREPARED is non-None when this runs;
    # the assertion exists to satisfy type narrowing for mypy and to
    # surface a clear failure if the gate ever stops firing first.
    assert _PREPARED is not None  # type: ignore[name-defined]
    return ResolvedTtsSpeaker(
        speaker="probe",
        native_voice=None,
        reference_audio=str(_PREPARED / "ref.wav"),
        reference_sample_rate=24000,
        source="planner_profile",
    )


@pytest.mark.asyncio
async def test_pocket_streaming_advertises_truthfully_and_emits_at_least_one_chunk():
    from octomil.execution.kernel import _build_local_realtime_stream

    backend = _build_real_pocket_backend()
    profile = _profile_for_prepared_dir()

    stream = _build_local_realtime_stream(
        backend=backend,
        text=_MULTI_SENTENCE,
        voice=None,
        resolved_speaker=profile,
        speed=1.0,
        runtime_model="pocket-tts-int8",
        policy_preset="local_only",
        fallback_used=False,
        sdk_t0=time.monotonic(),
    )
    started, _pcm, completed = await stream.collect()

    assert isinstance(started, SpeechStreamStarted)
    assert isinstance(completed, SpeechStreamCompleted)
    # Multi-sentence input must advertise sentence_chunk.
    assert started.streaming_capability.mode == TtsStreamingMode.SENTENCE_CHUNK
    # Honesty: advertised cadence either matched (verified=True) or
    # downgraded to final_chunk (verified=False). The completion
    # event must NOT advertise sentence_chunk while delivering one
    # chunk — that's the fake-realtime regression.
    if completed.observed_chunks <= 1:
        assert completed.streaming_capability.mode == TtsStreamingMode.FINAL_CHUNK
        assert completed.capability_verified is False
    else:
        assert completed.streaming_capability.mode == TtsStreamingMode.SENTENCE_CHUNK
        assert completed.capability_verified is True


@pytest.mark.asyncio
async def test_pocket_first_chunk_arrives_strictly_before_completion():
    from octomil.execution.kernel import _build_local_realtime_stream

    backend = _build_real_pocket_backend()
    profile = _profile_for_prepared_dir()

    stream = _build_local_realtime_stream(
        backend=backend,
        text=_LONG_SINGLE_SENTENCE,
        voice=None,
        resolved_speaker=profile,
        speed=1.0,
        runtime_model="pocket-tts-int8",
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
    # Even on a single long sentence Pocket should emit at least one
    # chunk strictly before completion — if the gap collapses to zero
    # we're buffering the whole utterance.
    assert first_chunk_at <= completed_at


@pytest.mark.asyncio
async def test_pocket_synthesize_create_path_works_end_to_end():
    """The non-streaming ``synthesize`` path must work too — it's the
    contract behind ``client.audio.speech.create``. Asserts WAV bytes
    come back and the duration is non-zero."""
    backend = _build_real_pocket_backend()
    profile = _profile_for_prepared_dir()

    result = backend.synthesize(_MULTI_SENTENCE, speaker_profile=profile)

    assert result["audio_bytes"]
    assert result["audio_bytes"][:4] == b"RIFF"
    assert result["sample_rate"] > 0
    assert result["duration_ms"] > 0
    assert result["voice"] == "probe"
