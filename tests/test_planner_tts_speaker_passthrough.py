"""Planner ``tts_speakers`` round-trip — server JSON to typed schema.

The planner can attach logical TTS speaker profiles either at the
app-resolution level (one map for the app) or at the candidate level
(per-candidate overrides — e.g. only the PocketTTS candidate carries
reference-audio profiles, the Kokoro candidate carries native-voice
profiles). Both sites must rehydrate from a cached/server JSON dict
into typed :class:`TtsSpeakerProfile` entries; the previous behaviour
silently dropped the field, which would have made every speaker= call
on an app ref fall through to native-voice resolution.
"""

from __future__ import annotations

from octomil.runtime.planner.planner import (
    plan_dict_to_app_resolution,
    plan_dict_to_candidates,
)
from octomil.runtime.planner.schemas import TtsSpeakerProfile


def _server_app_resolution_with_speakers() -> dict:
    """Shape the server emits for an ``@app/eternum/tts`` resolution."""
    return {
        "app_id": "app_eternum",
        "capability": "tts",
        "routing_policy": "private",
        "selected_model": "pocket-tts-int8",
        "app_slug": "eternum",
        "selected_model_variant_id": None,
        "selected_model_version": None,
        "artifact_candidates": [],
        "preferred_engines": ["sherpa-onnx"],
        "fallback_policy": None,
        "plan_ttl_seconds": 604800,
        "tts_speakers": {
            "madam_ambrose": {
                "speaker_id": "madam_ambrose",
                "reference_audio": "https://eternum.example/voices/madam_ambrose_24k.wav",
                "reference_sample_rate": 24000,
                "language": "en-GB",
                "style": "narrator",
                "metadata": {"role": "antagonist"},
            },
            "narrator_default": {
                "speaker_id": "narrator_default",
                "native_voice": "af_bella",
                "language": "en-US",
            },
        },
    }


def test_app_resolution_rehydrates_tts_speaker_map():
    resolution = plan_dict_to_app_resolution(_server_app_resolution_with_speakers())

    assert resolution is not None
    assert set(resolution.tts_speakers.keys()) == {"madam_ambrose", "narrator_default"}

    ambrose = resolution.tts_speakers["madam_ambrose"]
    assert isinstance(ambrose, TtsSpeakerProfile)
    assert ambrose.speaker_id == "madam_ambrose"
    assert ambrose.reference_audio == "https://eternum.example/voices/madam_ambrose_24k.wav"
    assert ambrose.reference_sample_rate == 24000
    assert ambrose.language == "en-GB"
    assert ambrose.style == "narrator"
    assert ambrose.metadata == {"role": "antagonist"}
    # No native_voice on a few-shot voice-cloning profile.
    assert ambrose.native_voice is None

    narrator = resolution.tts_speakers["narrator_default"]
    assert narrator.native_voice == "af_bella"
    assert narrator.reference_audio is None
    assert narrator.metadata == {}


def test_app_resolution_without_speakers_yields_empty_map():
    """Forward-compat: a plan that omits the field must not raise."""
    payload = _server_app_resolution_with_speakers()
    payload.pop("tts_speakers")
    resolution = plan_dict_to_app_resolution(payload)

    assert resolution is not None
    assert resolution.tts_speakers == {}


def test_candidate_level_tts_speakers_rehydrate():
    """Per-candidate map — used when only one candidate engine
    (PocketTTS) carries reference-audio profiles."""
    candidates = plan_dict_to_candidates(
        [
            {
                "locality": "local",
                "priority": 0,
                "confidence": 0.9,
                "reason": "pocket candidate",
                "engine": "sherpa-onnx",
                "tts_speakers": {
                    "madam_ambrose": {
                        "speaker_id": "madam_ambrose",
                        "reference_audio": "/cache/refs/madam_ambrose.wav",
                        "reference_sample_rate": 24000,
                    }
                },
            },
            {
                "locality": "local",
                "priority": 1,
                "confidence": 0.7,
                "reason": "kokoro fallback",
                "engine": "sherpa-onnx",
                "tts_speakers": {
                    "narrator_default": {
                        "speaker_id": "narrator_default",
                        "native_voice": "af_bella",
                    },
                },
            },
        ]
    )

    assert len(candidates) == 2
    assert candidates[0].tts_speakers["madam_ambrose"].reference_audio == "/cache/refs/madam_ambrose.wav"
    assert candidates[0].tts_speakers["madam_ambrose"].reference_sample_rate == 24000
    assert candidates[1].tts_speakers["narrator_default"].native_voice == "af_bella"


def test_speaker_id_falls_back_to_dict_key_when_omitted():
    """Server-side bug-tolerance: if a profile entry omits speaker_id,
    use the dict key. Mirrors how ``voices.list`` will key the catalog."""
    resolution = plan_dict_to_app_resolution(
        {
            "app_id": "x",
            "capability": "tts",
            "routing_policy": "private",
            "selected_model": "pocket-tts-int8",
            "tts_speakers": {
                "madam_ambrose": {
                    # speaker_id missing
                    "reference_audio": "/r.wav",
                },
            },
        }
    )
    assert resolution is not None
    assert resolution.tts_speakers["madam_ambrose"].speaker_id == "madam_ambrose"


def test_malformed_speaker_entries_dropped_silently():
    """Non-dict entries must not crash the rehydrate path; the kernel
    falls through to native-voice resolution."""
    resolution = plan_dict_to_app_resolution(
        {
            "app_id": "x",
            "capability": "tts",
            "routing_policy": "private",
            "selected_model": "pocket-tts-int8",
            "tts_speakers": {
                "ok": {"speaker_id": "ok", "native_voice": "af_bella"},
                "bad": "not-a-dict",
                "blank": {"speaker_id": ""},
            },
        }
    )
    assert resolution is not None
    assert set(resolution.tts_speakers.keys()) == {"ok"}
