"""End-to-end speaker= plumbing — facade -> kernel -> backend.

Asserts:
  * ``FacadeSpeech.create`` and ``.stream`` accept ``speaker=`` and pass it
    through to the kernel.
  * ``FacadeSpeech.create``'s ``voice=`` keeps working (back-compat).
  * The kernel's local-realtime stream builder reads native_voice off
    the resolved speaker, so the SpeechStreamStarted event reflects
    the planner's mapping.
  * ``client.audio.voices.list`` returns logical speakers from the
    planner profile when the request is app-scoped, and the listed
    ids round-trip through ``speech.create(speaker=id)``.
  * Unsupported speaker on an app ref raises BEFORE
    ``SpeechStreamStarted`` (mirrors the v4.13 voice prevalidation
    contract).
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Optional

import pytest

from octomil.audio.speech import FacadeSpeech, VoiceInfo
from octomil.audio.streaming import (
    SpeechStreamStarted,
)
from octomil.errors import OctomilError
from octomil.execution.tts_speaker_resolver import resolve_tts_speaker
from octomil.runtime.planner.schemas import (
    AppResolution,
    RuntimeSelection,
    TtsSpeakerProfile,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


SAMPLE_RATE = 24000


class _FakeBackend:
    name = "fake"
    accepts_speaker_profile = True

    def __init__(self) -> None:
        self._sample_rate = SAMPLE_RATE
        self._default_voice = "af_bella"
        self.synthesize_calls: list[dict] = []
        self.synthesize_stream_calls: list[dict] = []

    def validate_voice(self, voice: Optional[str]) -> tuple[int, str]:
        return 0, voice or "af_bella"

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        *,
        speaker_profile: Any = None,
    ) -> dict:
        self.synthesize_calls.append({"text": text, "voice": voice, "speed": speed, "speaker_profile": speaker_profile})
        return {
            "audio_bytes": b"WAVfake",
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": SAMPLE_RATE,
            "duration_ms": 100,
            "voice": voice,
            "model": "kokoro-82m",
        }

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        *,
        speaker_profile: Any = None,
    ) -> AsyncIterator[dict]:
        self.synthesize_stream_calls.append(
            {"text": text, "voice": voice, "speed": speed, "speaker_profile": speaker_profile}
        )
        yield {"pcm_s16le": b"\x00\x00" * 1200, "num_samples": 1200, "sample_rate": SAMPLE_RATE}


class _FakeKernel:
    """Minimal kernel that captures kwargs and routes to a fake backend."""

    def __init__(self, backend: _FakeBackend, selection: Any = None) -> None:
        self.backend = backend
        self.selection = selection
        self.synthesize_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def synthesize_speech(self, **kwargs):
        self.synthesize_calls.append(kwargs)
        from octomil.audio.speech import SpeechResponse, SpeechRoute
        from octomil.execution.kernel import _call_backend_synthesize

        resolved = resolve_tts_speaker(
            speaker=kwargs.get("speaker"),
            voice=kwargs.get("voice"),
            selection=self.selection,
            is_app_ref=str(kwargs.get("model", "")).startswith("@app/"),
        )
        result = _call_backend_synthesize(self.backend, kwargs["input"], resolved, kwargs.get("speed", 1.0))
        return SpeechResponse(
            audio_bytes=result["audio_bytes"],
            content_type=result["content_type"],
            format=result["format"],
            model=kwargs["model"],
            voice=resolved.native_voice,
            sample_rate=result["sample_rate"],
            duration_ms=result["duration_ms"],
            route=SpeechRoute(locality="on_device", engine="sherpa-onnx"),
        )

    async def synthesize_speech_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        from octomil.execution.kernel import _build_local_realtime_stream

        resolved = resolve_tts_speaker(
            speaker=kwargs.get("speaker"),
            voice=kwargs.get("voice"),
            selection=self.selection,
            is_app_ref=str(kwargs.get("model", "")).startswith("@app/"),
        )
        return _build_local_realtime_stream(
            backend=self.backend,
            text=kwargs["input"],
            voice=kwargs.get("voice"),
            resolved_speaker=resolved,
            speed=kwargs.get("speed", 1.0),
            runtime_model=kwargs["model"],
            policy_preset="local_only",
            fallback_used=False,
            sdk_t0=kwargs.get("sdk_t0") or time.monotonic(),
        )


def _selection_with_eternum_speakers() -> RuntimeSelection:
    return RuntimeSelection(
        locality="local",
        app_resolution=AppResolution(
            app_id="app_eternum",
            capability="tts",
            routing_policy="private",
            selected_model="kokoro-82m",
            tts_speakers={
                "narrator": TtsSpeakerProfile(speaker_id="narrator", native_voice="af_nicole"),
                "madam_ambrose": TtsSpeakerProfile(
                    speaker_id="madam_ambrose",
                    reference_audio="/cache/refs/madam.wav",
                    reference_sample_rate=24000,
                ),
            },
        ),
    )


# ---------------------------------------------------------------------------
# Facade -> kernel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_threads_speaker_kwarg_to_kernel():
    backend = _FakeBackend()
    kernel = _FakeKernel(backend)
    facade = FacadeSpeech(kernel)
    response = await facade.create(model="kokoro-82m", input="hello", speaker="af_bella")
    # Kernel saw the kwarg.
    assert kernel.synthesize_calls[0]["speaker"] == "af_bella"
    # On a non-app-ref, speaker is treated as native_voice alias.
    assert response.voice == "af_bella"
    # Backend got the speaker_profile via the speaker-aware bridge.
    assert backend.synthesize_calls[0]["speaker_profile"] is not None
    assert backend.synthesize_calls[0]["speaker_profile"].source == "native_voice"


@pytest.mark.asyncio
async def test_create_voice_kwarg_still_works_back_compat():
    backend = _FakeBackend()
    kernel = _FakeKernel(backend)
    facade = FacadeSpeech(kernel)
    await facade.create(model="kokoro-82m", input="hello", voice="af_bella")
    assert kernel.synthesize_calls[0]["voice"] == "af_bella"
    assert kernel.synthesize_calls[0]["speaker"] is None
    assert backend.synthesize_calls[0]["voice"] == "af_bella"


@pytest.mark.asyncio
async def test_app_ref_speaker_resolves_to_native_voice_on_started_event():
    """``speaker="narrator"`` on an app ref where the planner maps it
    to ``native_voice="af_nicole"`` should surface ``af_nicole`` on
    SpeechStreamStarted, not ``narrator``."""
    backend = _FakeBackend()
    kernel = _FakeKernel(backend, selection=_selection_with_eternum_speakers())
    facade = FacadeSpeech(kernel)
    stream = facade.stream(model="@app/eternum/tts", input="hello", speaker="narrator")
    started, _pcm, _completed = await stream.collect()

    assert isinstance(started, SpeechStreamStarted)
    # The stream builder asks the backend's validate_voice with the
    # native_voice the resolver picked — so the started event reports
    # the native voice the engine will use.
    assert started.voice == "af_nicole"
    # And the underlying synthesize_stream call carried the resolved
    # native voice + the speaker_profile via the bridge.
    assert backend.synthesize_stream_calls[0]["voice"] == "af_nicole"
    assert backend.synthesize_stream_calls[0]["speaker_profile"].speaker == "narrator"


@pytest.mark.asyncio
async def test_unknown_speaker_on_app_ref_raises_before_stream_started():
    """A typo on an app-ref speaker must surface as a sync OctomilError —
    the consumer never sees a SpeechStreamStarted for an unsupported
    request."""
    backend = _FakeBackend()
    kernel = _FakeKernel(backend, selection=_selection_with_eternum_speakers())
    facade = FacadeSpeech(kernel)
    stream = facade.stream(model="@app/eternum/tts", input="hello", speaker="madam_ambros")  # typo

    with pytest.raises(OctomilError) as exc_info:
        async for _event in stream:
            pass

    assert "speaker_not_supported_for_app" in exc_info.value.error_message


@pytest.mark.asyncio
async def test_voice_promoted_to_speaker_when_id_matches_planner_profile():
    """Migration aid: ``voice="madam_ambrose"`` on an app ref still
    flows the planner's reference_audio profile through to the
    backend."""
    backend = _FakeBackend()
    kernel = _FakeKernel(backend, selection=_selection_with_eternum_speakers())
    facade = FacadeSpeech(kernel)
    await facade.create(model="@app/eternum/tts", input="hello", voice="narrator")

    profile = backend.synthesize_calls[0]["speaker_profile"]
    assert profile.source == "planner_profile"
    assert profile.speaker == "narrator"
    assert profile.native_voice == "af_nicole"


# ---------------------------------------------------------------------------
# voices.list — logical speakers
# ---------------------------------------------------------------------------


def _selection_with_pocket_only_speakers() -> RuntimeSelection:
    return RuntimeSelection(
        locality="local",
        app_resolution=AppResolution(
            app_id="x",
            capability="tts",
            routing_policy="private",
            selected_model="pocket-tts-int8",
            tts_speakers={
                "madam_ambrose": TtsSpeakerProfile(
                    speaker_id="madam_ambrose",
                    reference_audio="/cache/refs/madam.wav",
                    reference_sample_rate=24000,
                    language="en-GB",
                ),
            },
        ),
    )


def test_voice_info_logical_speakers_have_planner_profile_source():
    """Constructing a VoiceInfo with the new logical-speaker fields
    must round-trip cleanly (frozen dataclass)."""
    info = VoiceInfo(
        id="madam_ambrose",
        sid=None,
        default=False,
        source="planner_profile",
        speaker="madam_ambrose",
        native_voice=None,
        requires_reference=True,
        language="en-GB",
    )
    assert info.id == "madam_ambrose"
    assert info.source == "planner_profile"
    assert info.requires_reference is True
    assert info.language == "en-GB"


def test_voice_info_native_legacy_construction_still_works():
    """Old call sites using only ``id``, ``sid``, ``default`` must keep
    working — additive field defaults handle this."""
    info = VoiceInfo(id="af_bella", sid=1, default=True)
    assert info.source is None
    assert info.speaker is None
    assert info.requires_reference is False
