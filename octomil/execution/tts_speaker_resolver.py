"""TTS speaker resolution.

Single source of truth for translating a caller's ``speaker`` (and
back-compat ``voice``) kwarg into the engine-facing payload — a
``ResolvedTtsSpeaker`` carrying both the native voice label (for
Kokoro / Piper) AND the reference-audio profile (for few-shot voice
cloning engines like PocketTTS).

Decoupling matters because:

  * The caller does not know the runtime engine. They pass a logical
    speaker id ("madam_ambrose") and let planner/app config map it.
  * Different engines need different inputs from the same logical
    speaker — Kokoro gets a sid, PocketTTS gets a reference WAV.
  * ``voices.list`` and ``speech.create`` / ``speech.stream`` MUST
    agree on what a given speaker means; they all funnel through
    this resolver.

The module is deliberately small and dependency-free at runtime
(only types from ``runtime.planner.schemas``) so the kernel can call
it on the synchronous voice-validation path without touching
``asyncio`` / ``httpx``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class ResolvedTtsSpeaker:
    """The engine-facing translation of a caller's speaker/voice request.

    ``source`` records *how* we got here so error messages, telemetry,
    and the ``voices.list`` catalog can reflect provenance:

      * ``"planner_profile"`` — speaker_id matched a profile in the
        app/candidate ``tts_speakers`` map.
      * ``"native_voice"`` — caller passed ``voice=`` and the engine's
        native catalog accepted it (no planner profile needed).
      * ``"default"`` — neither speaker nor voice supplied; the engine
        will use its built-in default speaker. ``native_voice`` may
        be ``None`` here; the backend resolves the default itself.
    """

    speaker: Optional[str]
    native_voice: Optional[str]
    reference_audio: Optional[str]
    reference_sample_rate: Optional[int]
    source: str  # "planner_profile" | "native_voice" | "default"
    # Planner-supplied side-channel fields. Backends that need
    # additional context per speaker (Pocket reads ``reference_text``
    # and ``num_steps``; future engines may carry style / language
    # tags) read them off this dict. Empty when the request resolved
    # via ``native_voice`` / ``default`` paths.
    metadata: Optional[dict[str, Any]] = None
    language: Optional[str] = None
    style: Optional[str] = None

    @property
    def has_reference(self) -> bool:
        """True iff this profile carries reference audio for cloning."""
        return bool(self.reference_audio)


def _merge_speaker_maps(
    app_speakers: Any,
    candidate_speakers: Any,
) -> dict[str, Any]:
    """Merge candidate-level speaker map over app-level (candidate wins).

    Candidate-level overrides exist for the common case where one
    candidate engine (e.g. Pocket) carries reference profiles and
    another (Kokoro) carries native-voice profiles for the same
    logical id — the kernel picks the candidate first, then the
    speaker map associated with it.
    """
    merged: dict[str, Any] = {}
    if isinstance(app_speakers, dict):
        merged.update(app_speakers)
    if isinstance(candidate_speakers, dict):
        merged.update(candidate_speakers)
    return merged


def _coerce_profile(profile: Any) -> Optional[dict[str, Any]]:
    """Pull the caller-relevant fields off a profile dataclass or dict.

    Defensive: planner schemas dataclass, cached dict, or duck-typed
    object — all flow into the same field set
    (``native_voice``, ``reference_audio``, ``reference_sample_rate``,
    plus the side-channel ``metadata`` / ``language`` / ``style``
    fields that some engines need to interpret a profile correctly).

    Pocket-specific note: ``metadata`` carries ``reference_text``
    (the prompt transcription) and ``num_steps`` (synthesis quality
    knob). Pre-fix the resolver dropped these on the floor, so a
    real planner profile would synthesize with empty prompt text and
    the engine's hard-coded ``num_steps=4`` default. Tests had been
    masking the gap by mutating ``ResolvedTtsSpeaker`` directly.
    """
    if profile is None:
        return None
    if isinstance(profile, dict):
        return {
            "native_voice": profile.get("native_voice"),
            "reference_audio": profile.get("reference_audio"),
            "reference_sample_rate": profile.get("reference_sample_rate"),
            "metadata": profile.get("metadata"),
            "language": profile.get("language"),
            "style": profile.get("style"),
        }
    return {
        "native_voice": getattr(profile, "native_voice", None),
        "reference_audio": getattr(profile, "reference_audio", None),
        "reference_sample_rate": getattr(profile, "reference_sample_rate", None),
        "metadata": getattr(profile, "metadata", None),
        "language": getattr(profile, "language", None),
        "style": getattr(profile, "style", None),
    }


def _speaker_lookup(
    speaker_id: str,
    speakers_map: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Case-sensitive lookup; falls back to lowercase match.

    Speaker ids on the wire are case-sensitive (the planner controls
    them) but we accept a lowercase input as a UX convenience for
    callers typing ``"Madam_Ambrose"`` etc. — same convention as the
    native voice catalog.
    """
    if speaker_id in speakers_map:
        return _coerce_profile(speakers_map[speaker_id])
    lowered = speaker_id.strip().lower()
    for key, profile in speakers_map.items():
        if key.lower() == lowered:
            return _coerce_profile(profile)
    return None


def resolve_tts_speaker(
    *,
    speaker: Optional[str],
    voice: Optional[str],
    selection: Any,
    is_app_ref: bool,
    selected_candidate: Any = None,
) -> ResolvedTtsSpeaker:
    """Translate caller-supplied speaker/voice into an engine payload.

    Resolution rules (mirrored in :func:`list_logical_speakers`):

      1. ``speaker`` is the canonical input. If supplied AND the
         planner published a matching profile, use that profile —
         ``source="planner_profile"``.
      2. ``speaker`` supplied but no profile match: this is the
         caller's mistake. Raise ``OctomilError`` with a precise
         ``speaker_not_supported_for_app`` message so they can fix
         it; do NOT fall through to native-voice matching, which
         would silently re-bind a typo to the wrong voice.
      3. ``voice`` supplied AND ``speaker`` absent AND request is
         app-scoped: try ``voice`` as a logical speaker first
         (so ``voice="madam_ambrose"`` keeps working for callers
         migrating from the old API). If found in the planner map,
         ``source="planner_profile"``. Otherwise fall through.
      4. ``voice`` supplied: pass to the engine as a native voice;
         backend's ``validate_voice`` does the catalog check.
         ``source="native_voice"``.
      5. Neither supplied: ``source="default"`` and the backend
         picks. The kernel may also set ``native_voice`` to a
         planner-supplied default speaker's native voice if one is
         flagged.

    The function is pure: it does NOT perform IO, does NOT touch
    voices.txt, and does NOT validate that ``native_voice`` exists
    in the engine's catalog. Those checks live in
    ``_validate_local_voice`` / ``backend.validate_voice``.

    ``selected_candidate`` MUST be the candidate the kernel actually
    chose for this synthesis. Earlier behaviour merged ``tts_speakers``
    from EVERY candidate in ``selection.candidates``, which let a
    non-selected candidate's profile silently override the running
    backend's profile for the same logical speaker id. The kernel
    now passes the selected candidate so only it (plus the app-level
    map) contributes to the resolved profile. ``None`` means "no
    candidate selected" — only the app-level map is consulted.
    """
    from octomil.errors import OctomilError, OctomilErrorCode

    speakers_map = _collect_speakers_map(selection, selected_candidate=selected_candidate)

    if speaker:
        profile = _speaker_lookup(speaker, speakers_map)
        if profile is not None:
            return ResolvedTtsSpeaker(
                speaker=speaker,
                native_voice=profile.get("native_voice"),
                reference_audio=profile.get("reference_audio"),
                reference_sample_rate=profile.get("reference_sample_rate"),
                source="planner_profile",
                metadata=profile.get("metadata"),
                language=profile.get("language"),
                style=profile.get("style"),
            )
        if is_app_ref:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"speaker_not_supported_for_app: speaker={speaker!r} is not "
                    f"in the planner's speaker map for this app. Call "
                    f"client.audio.voices.list(model=...) to see the supported "
                    f"speaker ids."
                ),
            )
        # Non-app refs: the planner may not publish a speaker map
        # at all (calling kokoro-82m by name). Treat speaker= as a
        # native-voice alias and let the backend's validate_voice
        # decide. Source flips to native_voice.
        return ResolvedTtsSpeaker(
            speaker=speaker,
            native_voice=speaker,
            reference_audio=None,
            reference_sample_rate=None,
            source="native_voice",
        )

    if voice:
        if is_app_ref and speakers_map:
            profile = _speaker_lookup(voice, speakers_map)
            if profile is not None:
                # voice= was a logical speaker id — promote to
                # speaker semantics so reference_audio etc. flow.
                return ResolvedTtsSpeaker(
                    speaker=voice,
                    native_voice=profile.get("native_voice"),
                    reference_audio=profile.get("reference_audio"),
                    reference_sample_rate=profile.get("reference_sample_rate"),
                    source="planner_profile",
                    metadata=profile.get("metadata"),
                    language=profile.get("language"),
                    style=profile.get("style"),
                )
        # Pure native-voice path. Backend.validate_voice gates
        # whether it actually exists in the catalog.
        return ResolvedTtsSpeaker(
            speaker=None,
            native_voice=voice,
            reference_audio=None,
            reference_sample_rate=None,
            source="native_voice",
        )

    # Neither supplied: backend picks the default.
    return ResolvedTtsSpeaker(
        speaker=None,
        native_voice=None,
        reference_audio=None,
        reference_sample_rate=None,
        source="default",
    )


def _collect_speakers_map(
    selection: Any,
    *,
    selected_candidate: Any = None,
) -> dict[str, Any]:
    """Return the merged ``app_level ⊕ selected_candidate`` ``tts_speakers``
    map.

    Earlier behaviour merged speakers from every candidate in the
    selection, which is unsafe: candidate B's profile for
    ``"narrator"`` could silently override candidate A's profile for
    the same id even when synthesis was actually running on A.
    Threading the selected candidate makes the resolved profile
    match the running backend exactly.

    ``selected_candidate=None`` collapses to "app-level only" so the
    listing and resolution paths stay deterministic when no
    candidate has been picked yet (e.g. the listing path is asked
    before the kernel routes the request).
    """
    if selection is None:
        return {}
    app_resolution = getattr(selection, "app_resolution", None)
    app_map = getattr(app_resolution, "tts_speakers", None)
    candidate_map: Any = None
    if selected_candidate is not None:
        candidate_map = getattr(selected_candidate, "tts_speakers", None)
    return _merge_speaker_maps(app_map, candidate_map)


def list_logical_speakers(
    selection: Any,
    *,
    selected_candidate: Any = None,
) -> tuple[dict[str, Any], ...]:
    """Return the ordered logical-speaker entries for ``voices.list``.

    Used by :func:`octomil.execution.kernel.list_speech_voices` to
    build a :class:`VoiceCatalog` for app refs whose planner
    publishes profiles. Entries preserve insertion order (app
    profiles first, then selected-candidate overrides; non-selected
    candidates contribute nothing — same constraint as the resolver).
    """
    merged = _collect_speakers_map(selection, selected_candidate=selected_candidate)
    out: list[dict[str, Any]] = []
    for key, profile in merged.items():
        coerced = _coerce_profile(profile) or {}
        # ``VoiceInfo`` (in octomil/audio/voices.py) exposes
        # ``language`` / ``style`` / ``metadata`` as public fields.
        # Pre-fix this loop dropped them, so listing always
        # advertised ``language=None`` even when the planner
        # profile carried one. The kernel ``list_speech_voices``
        # path projects the dict shape directly onto VoiceInfo.
        out.append(
            {
                "speaker_id": key,
                "native_voice": coerced.get("native_voice"),
                "reference_audio": coerced.get("reference_audio"),
                "reference_sample_rate": coerced.get("reference_sample_rate"),
                "language": coerced.get("language"),
                "style": coerced.get("style"),
                "metadata": coerced.get("metadata"),
            }
        )
    return tuple(out)


__all__ = [
    "ResolvedTtsSpeaker",
    "resolve_tts_speaker",
    "list_logical_speakers",
]


# Type-ignore: ``Iterable`` import is only used for annotations within
# the resolver helpers; left in for clarity even when unused at module
# import time. (Ruff will not flag this once the module gains its first
# annotated public iterable.)
_ = Iterable  # noqa: F841
