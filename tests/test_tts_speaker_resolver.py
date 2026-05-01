"""Unit tests for ``octomil.execution.tts_speaker_resolver``.

The resolver is small, pure, and on the synchronous voice-validation
path — every speaker/voice path through the kernel passes through it.
These tests pin the contract so we never silently re-bind a typo to
the wrong native voice or leak a reference-audio profile to a
native-voice engine.
"""

from __future__ import annotations

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.execution.tts_speaker_resolver import (
    ResolvedTtsSpeaker,
    list_logical_speakers,
    resolve_tts_speaker,
)
from octomil.runtime.planner.schemas import (
    AppResolution,
    RuntimeCandidatePlan,
    RuntimeSelection,
    TtsSpeakerProfile,
)


def _selection_with_app_speakers(speakers: dict) -> RuntimeSelection:
    return RuntimeSelection(
        locality="local",
        app_resolution=AppResolution(
            app_id="app_eternum",
            capability="tts",
            routing_policy="private",
            selected_model="pocket-tts-int8",
            tts_speakers=speakers,
        ),
    )


def _selection_with_candidate_speakers(speakers: dict) -> RuntimeSelection:
    return RuntimeSelection(
        locality="local",
        candidates=[
            RuntimeCandidatePlan(
                locality="local",
                priority=0,
                confidence=0.9,
                reason="primary",
                tts_speakers=speakers,
            ),
        ],
    )


def _profile(**kwargs: object) -> TtsSpeakerProfile:
    base: dict[str, object] = {"speaker_id": "x"}
    base.update(kwargs)
    return TtsSpeakerProfile(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# speaker= path
# ---------------------------------------------------------------------------


def test_speaker_resolves_via_planner_profile_for_app_ref():
    selection = _selection_with_app_speakers(
        {
            "madam_ambrose": _profile(
                speaker_id="madam_ambrose",
                reference_audio="/cache/refs/madam.wav",
                reference_sample_rate=24000,
            )
        }
    )
    resolved = resolve_tts_speaker(
        speaker="madam_ambrose",
        voice=None,
        selection=selection,
        is_app_ref=True,
    )
    assert resolved.source == "planner_profile"
    assert resolved.speaker == "madam_ambrose"
    assert resolved.reference_audio == "/cache/refs/madam.wav"
    assert resolved.reference_sample_rate == 24000
    assert resolved.has_reference is True
    # A few-shot voice-cloning profile has no native_voice.
    assert resolved.native_voice is None


def test_speaker_with_native_voice_profile_returns_native_label():
    selection = _selection_with_app_speakers({"narrator": _profile(speaker_id="narrator", native_voice="af_bella")})
    resolved = resolve_tts_speaker(
        speaker="narrator",
        voice=None,
        selection=selection,
        is_app_ref=True,
    )
    assert resolved.source == "planner_profile"
    assert resolved.native_voice == "af_bella"
    assert resolved.has_reference is False


def test_unknown_speaker_on_app_ref_raises_speaker_not_supported():
    """The exact bug the resolver prevents — a typo on an app ref
    must NOT silently fall through to native-voice matching."""
    selection = _selection_with_app_speakers(
        {"madam_ambrose": _profile(speaker_id="madam_ambrose", reference_audio="/r.wav")}
    )
    with pytest.raises(OctomilError) as exc_info:
        resolve_tts_speaker(
            speaker="madam_ambros",  # typo
            voice=None,
            selection=selection,
            is_app_ref=True,
        )
    assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
    assert "speaker_not_supported_for_app" in exc_info.value.error_message
    assert "madam_ambros" in exc_info.value.error_message


def test_speaker_lookup_is_case_insensitive_fallback():
    selection = _selection_with_app_speakers(
        {"Madam_Ambrose": _profile(speaker_id="Madam_Ambrose", reference_audio="/r.wav")}
    )
    resolved = resolve_tts_speaker(
        speaker="madam_ambrose",
        voice=None,
        selection=selection,
        is_app_ref=True,
    )
    assert resolved.source == "planner_profile"
    assert resolved.speaker == "madam_ambrose"
    assert resolved.reference_audio == "/r.wav"


def test_speaker_on_non_app_ref_treated_as_native_voice_alias():
    """For non-app refs (kokoro-82m by name), planner may not
    publish a speaker map at all. ``speaker=`` should be treated as
    a native-voice alias and let the backend's catalog gate enforce."""
    resolved = resolve_tts_speaker(
        speaker="af_bella",
        voice=None,
        selection=None,
        is_app_ref=False,
    )
    assert resolved.source == "native_voice"
    assert resolved.native_voice == "af_bella"
    assert resolved.speaker == "af_bella"


# ---------------------------------------------------------------------------
# voice= back-compat path
# ---------------------------------------------------------------------------


def test_voice_on_app_ref_promoted_to_speaker_when_id_matches():
    """Migration aid: callers that still pass ``voice="madam_ambrose"``
    on an app ref get the planner profile (so reference_audio flows)
    instead of an opaque native-voice mismatch."""
    selection = _selection_with_app_speakers(
        {"madam_ambrose": _profile(speaker_id="madam_ambrose", reference_audio="/r.wav")}
    )
    resolved = resolve_tts_speaker(
        speaker=None,
        voice="madam_ambrose",
        selection=selection,
        is_app_ref=True,
    )
    assert resolved.source == "planner_profile"
    assert resolved.speaker == "madam_ambrose"
    assert resolved.reference_audio == "/r.wav"


def test_voice_on_app_ref_falls_through_to_native_when_no_match():
    selection = _selection_with_app_speakers(
        {"madam_ambrose": _profile(speaker_id="madam_ambrose", reference_audio="/r.wav")}
    )
    resolved = resolve_tts_speaker(
        speaker=None,
        voice="af_bella",
        selection=selection,
        is_app_ref=True,
    )
    assert resolved.source == "native_voice"
    assert resolved.speaker is None
    assert resolved.native_voice == "af_bella"


def test_voice_on_non_app_ref_is_pure_native_voice():
    resolved = resolve_tts_speaker(
        speaker=None,
        voice="af_bella",
        selection=None,
        is_app_ref=False,
    )
    assert resolved.source == "native_voice"
    assert resolved.native_voice == "af_bella"
    assert resolved.speaker is None


# ---------------------------------------------------------------------------
# default path (neither supplied)
# ---------------------------------------------------------------------------


def test_neither_speaker_nor_voice_returns_default():
    resolved = resolve_tts_speaker(
        speaker=None,
        voice=None,
        selection=None,
        is_app_ref=False,
    )
    assert resolved.source == "default"
    assert resolved.native_voice is None
    assert resolved.speaker is None
    assert resolved.has_reference is False


# ---------------------------------------------------------------------------
# Speaker map merging — candidate-level overrides app-level
# ---------------------------------------------------------------------------


def test_candidate_level_speaker_overrides_app_level():
    """When both maps name the same speaker, the candidate-level entry
    wins. Use-case: app-level publishes a fallback ``native_voice``,
    the Pocket candidate publishes a ``reference_audio`` for the same
    logical id."""
    selection = RuntimeSelection(
        locality="local",
        app_resolution=AppResolution(
            app_id="x",
            capability="tts",
            routing_policy="private",
            selected_model="pocket-tts-int8",
            tts_speakers={
                "narrator": _profile(speaker_id="narrator", native_voice="af_bella"),
            },
        ),
        candidates=[
            RuntimeCandidatePlan(
                locality="local",
                priority=0,
                confidence=0.9,
                reason="pocket",
                tts_speakers={
                    "narrator": _profile(speaker_id="narrator", reference_audio="/r.wav"),
                },
            ),
        ],
    )
    resolved = resolve_tts_speaker(
        speaker="narrator",
        voice=None,
        selection=selection,
        is_app_ref=True,
    )
    # Candidate-level (reference_audio) wins over app-level (native_voice).
    assert resolved.reference_audio == "/r.wav"
    assert resolved.native_voice is None


# ---------------------------------------------------------------------------
# list_logical_speakers — used by voices.list
# ---------------------------------------------------------------------------


def test_list_logical_speakers_preserves_order_app_then_candidate():
    selection = RuntimeSelection(
        locality="local",
        app_resolution=AppResolution(
            app_id="x",
            capability="tts",
            routing_policy="private",
            selected_model="pocket-tts-int8",
            tts_speakers={
                "alpha": _profile(speaker_id="alpha", native_voice="af_bella"),
                "beta": _profile(speaker_id="beta", native_voice="af_nicole"),
            },
        ),
        candidates=[
            RuntimeCandidatePlan(
                locality="local",
                priority=0,
                confidence=0.9,
                reason="pocket",
                tts_speakers={
                    "gamma": _profile(speaker_id="gamma", reference_audio="/g.wav"),
                },
            ),
        ],
    )
    speakers = list_logical_speakers(selection)
    assert [s["speaker_id"] for s in speakers] == ["alpha", "beta", "gamma"]


def test_list_logical_speakers_returns_empty_when_no_planner():
    assert list_logical_speakers(None) == ()


def test_list_logical_speakers_returns_empty_when_no_speakers():
    selection = _selection_with_app_speakers({})
    assert list_logical_speakers(selection) == ()


# ---------------------------------------------------------------------------
# ResolvedTtsSpeaker contract
# ---------------------------------------------------------------------------


def test_resolved_speaker_is_frozen_dataclass():
    resolved = ResolvedTtsSpeaker(
        speaker="x",
        native_voice="af_bella",
        reference_audio=None,
        reference_sample_rate=None,
        source="planner_profile",
    )
    with pytest.raises((AttributeError, Exception)):
        resolved.speaker = "y"  # type: ignore[misc]
