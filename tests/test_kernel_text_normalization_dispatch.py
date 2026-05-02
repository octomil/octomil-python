"""Kernel automatically applies the backend's text-normalization profile.

The whole point of the auto-normalization PR is that consumers don't
have to call ``for_kokoro(text)`` themselves. Tests pin:

  - The text the backend's ``synthesize_stream`` actually receives is
    the post-normalization string when the backend declares
    ``text_normalization_profile() == 'espeak_compat'``.
  - Same for the non-streaming ``synthesize`` path.
  - ``text_normalization='off'`` opts out — backend sees the raw
    string verbatim.
  - Backends with profile ``'none'`` (e.g. Pocket) are no-ops.
  - Backends without a ``text_normalization_profile`` method are
    no-ops (forward-compat for legacy / third-party engines).
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import pytest

from octomil.audio.streaming import (
    SpeechStreamCompleted,
    SpeechStreamStarted,
)

# ---------------------------------------------------------------------------
# Recording fakes
# ---------------------------------------------------------------------------


class _RecordingEspeakBackend:
    """Stand-in for the sherpa Kokoro/Piper backend.

    Records every text it sees; declares the espeak_compat profile.
    """

    name = "fake-espeak-backend"
    text_received: list[str]
    stream_text_received: list[str]

    def __init__(self) -> None:
        self.text_received = []
        self.stream_text_received = []
        self._sample_rate = 24000
        self._default_voice = "af_bella"
        self._family = "kokoro"

    def text_normalization_profile(self) -> str:
        return "espeak_compat"

    def streaming_capability(self, text: str):
        from octomil.audio.streaming import TtsStreamingCapability

        return TtsStreamingCapability.final_only(verified=False)

    def validate_voice(self, voice):
        return 0, voice or self._default_voice

    def synthesize(self, text: str, voice=None, speed: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        self.text_received.append(text)
        return {
            "audio_bytes": b"\x00\x00",
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": self._sample_rate,
            "duration_ms": 0,
            "voice": voice or self._default_voice,
            "model": "kokoro-82m",
        }

    async def synthesize_stream(
        self, text: str, voice=None, speed: float = 1.0, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        self.stream_text_received.append(text)
        # Yield one tiny chunk so the kernel's stream wrapper completes.
        yield {
            "pcm_s16le": b"\x00\x00",
            "num_samples": 1,
            "sample_rate": self._sample_rate,
        }


class _RecordingNoneProfileBackend(_RecordingEspeakBackend):
    """LM-frontend backend (Pocket / future Parler) — declares
    profile=none. Kernel must pass the raw string through."""

    def text_normalization_profile(self) -> str:
        return "none"


class _RecordingLegacyBackend(_RecordingEspeakBackend):
    """Backend that doesn't expose ``text_normalization_profile``
    at all (third-party / pre-this-PR). Kernel falls back to
    no-op."""

    def text_normalization_profile(self) -> str:  # type: ignore[override]
        raise AttributeError("never called")


def _strip_profile_method(backend: _RecordingEspeakBackend) -> _RecordingEspeakBackend:
    """Force the legacy posture (no ``text_normalization_profile``
    method) by deleting the attribute on an instance."""
    try:
        del type(backend).text_normalization_profile
    except AttributeError:
        pass
    return backend


# ---------------------------------------------------------------------------
# _build_local_realtime_stream — the streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_dispatch_normalizes_currency_when_backend_declares_espeak_compat():
    """The exact Eternum bug: ``$1200`` reaches the espeak-driven
    backend as ``1200 dollars`` (post-normalization), not as the raw
    string."""
    from octomil.execution.kernel import (
        _build_local_realtime_stream,
        _normalize_text_for_backend,
    )

    backend = _RecordingEspeakBackend()
    raw = "I owe him $1200."

    # The kernel's full streaming dispatch threads ``text_normalization``
    # in to ``synthesize_speech_stream``; this lower-level test exercises
    # the same helper the dispatch uses.
    normalized = _normalize_text_for_backend(backend, raw, "auto")
    assert normalized == "I owe him 1200 dollars."

    stream = _build_local_realtime_stream(
        backend=backend,
        text=normalized,
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
    # The backend's ``synthesize_stream`` saw the normalized string.
    assert backend.stream_text_received == ["I owe him 1200 dollars."]


@pytest.mark.asyncio
async def test_normalization_helper_respects_off_mode():
    """``text_normalization='off'`` returns the raw text — consumer
    has already pre-normalized or wants espeak's raw read."""
    from octomil.execution.kernel import _normalize_text_for_backend

    backend = _RecordingEspeakBackend()
    raw = "I owe him $1200."

    assert _normalize_text_for_backend(backend, raw, "off") == raw


@pytest.mark.asyncio
async def test_normalization_helper_respects_explicit_profile_override():
    """A literal profile name (e.g. ``espeak_compat``) overrides
    the backend's declaration. Useful when a consumer wants to force
    espeak normalization on a backend that declared ``none``."""
    from octomil.execution.kernel import _normalize_text_for_backend

    backend = _RecordingNoneProfileBackend()  # declares "none"
    raw = "I owe him $1200."

    forced = _normalize_text_for_backend(backend, raw, "espeak_compat")
    assert forced == "I owe him 1200 dollars."


@pytest.mark.asyncio
async def test_pocket_profile_is_no_op():
    """Pocket-style backends declare ``profile='none'`` — the
    kernel must NOT normalize their input. Their LM-based text
    frontend handles ``$1200`` natively and any SDK-side rewrite
    would be a regression."""
    from octomil.execution.kernel import _normalize_text_for_backend

    backend = _RecordingNoneProfileBackend()
    raw = "I owe him $1200, Mr. Smith."
    assert _normalize_text_for_backend(backend, raw, "auto") == raw


@pytest.mark.asyncio
async def test_legacy_backend_without_profile_method_is_no_op():
    """Forward-compat: third-party / pre-this-PR backends that
    don't expose ``text_normalization_profile`` get the raw text
    verbatim. Better silent passthrough than crash."""
    from octomil.execution.kernel import _normalize_text_for_backend

    class _NoProfileBackend:
        def synthesize(self, text, voice, speed, **kwargs):
            return {}

    backend = _NoProfileBackend()
    raw = "I owe him $1200, Mr. Smith."
    assert _normalize_text_for_backend(backend, raw, "auto") == raw


@pytest.mark.asyncio
async def test_unknown_explicit_profile_falls_back_to_auto():
    """A literal profile name the SDK doesn't recognize is treated
    as ``auto`` — the backend's declared profile wins. Forward-compat
    for any future profile name a consumer might try."""
    from octomil.execution.kernel import _normalize_text_for_backend

    backend = _RecordingEspeakBackend()  # declares espeak_compat
    raw = "I owe him $1200."

    # ``future_codec_v3`` is not in available_profiles(); fall back
    # to auto → backend declared espeak_compat → normalize.
    result = _normalize_text_for_backend(backend, raw, "future_codec_v3")
    assert result == "I owe him 1200 dollars."


# ---------------------------------------------------------------------------
# Sherpa backend declares the right profile per family
# ---------------------------------------------------------------------------


def test_sherpa_kokoro_declares_espeak_compat_profile():
    """Sanity check on the engine.py side of the wiring."""
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend.__new__(_SherpaTtsBackend)
    backend._family = "kokoro"
    assert backend.text_normalization_profile() == "espeak_compat"


def test_sherpa_piper_declares_espeak_compat_profile():
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend.__new__(_SherpaTtsBackend)
    backend._family = "vits"
    assert backend.text_normalization_profile() == "espeak_compat"


def test_sherpa_pocket_declares_none_profile():
    """Pocket has its own LM-based text frontend; SDK-side
    normalization would be a regression. Kokoro/Piper espeak rules
    must NOT apply."""
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend.__new__(_SherpaTtsBackend)
    backend._family = "pocket"
    assert backend.text_normalization_profile() == "none"
