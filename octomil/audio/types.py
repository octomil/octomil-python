"""Audio data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from octomil.execution.kernel import RouteMetadata


@dataclass
class ChunkWindow:
    """One fixed decode window in a chunked (v=6) transcription.

    Public mirror of the runtime-side ``ChunkWindow`` (see
    :class:`octomil.runtime.native.stt_backend.ChunkWindow`). Defined
    here so the public ``octomil.audio`` surface never imports the
    native cffi module. ``off_ms`` is the window's start offset; the
    window OWNS the half-open span ``[off_ms, off_ms + step_ms)`` for
    segment-midpoint dedup (the last window owns through the end of the
    audio).
    """

    index: int
    start_ms: int
    end_ms: int
    off_ms: int
    is_last: bool


@dataclass
class ChunkDiagnostics:
    """Chunked-transcribe provenance surfaced on a transcription result.

    Public mirror of the runtime-side ``ChunkDiagnostics`` (see
    :class:`octomil.runtime.native.stt_backend.ChunkDiagnostics`),
    kept free of native cffi imports. ``tail_gap_ms`` > 0 means trailing
    audio produced no segment (e.g. a last-window collapse — the
    tail-drop class of bug). ``segment_owner_window[i]`` is the window
    index that owns ``segments[i]``.
    """

    window_ms: int
    overlap_ms: int
    step_ms: int
    window_count: int
    windows: list[ChunkWindow] = field(default_factory=list)
    segment_owner_window: list[int] = field(default_factory=list)
    audio_duration_ms: int = 0
    final_segment_end_ms: int = 0
    tail_gap_ms: int = 0


@dataclass
class TranscriptionSegment:
    """A single timestamped segment from a transcription."""

    text: str
    start_ms: int = 0
    end_ms: int = 0
    # v0.1.27 (OCT_EVENT_VERSION 4) per-segment decode diagnostics.
    # ``avg_logprob`` is the mean per-token log-probability (less-negative
    # = more confident); ``no_speech_prob`` is whisper's no-speech
    # probability in [0, 1]. Both default to 0.0 against runtimes/engines
    # that predate the getters (cloud / echo / legacy whisper).
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class TranscriptionPartial:
    """A provisional, revision-aware partial from a streaming transcription.

    Public mirror of the runtime's ``OCT_EVENT_TRANSCRIPT_PARTIAL``
    payload (see ``NativeEvent.partial_*`` in
    :mod:`octomil.runtime.native.loader`), kept free of native cffi
    imports. Emitted by :meth:`octomil.audio.transcriptions.AudioTranscriptions.stream_transcribe`
    while audio is still being fed.

    ``revision_id`` is a 1-based monotonic counter within the session —
    a higher ``revision_id`` supersedes every prior partial, so callers
    that render speculatively should always replace on the latest. A
    partial is NEVER authoritative: the final
    :class:`TranscriptionSegment` values yielded after ``end_input`` are
    the committed transcript. ``is_stable`` is ``True`` iff the whole
    partial is safe for speculative use (local-agreement converged);
    ``stable_prefix_bytes`` is the length of the stable UTF-8 prefix
    (0 when unavailable).
    """

    text: str
    revision_id: int
    is_stable: bool = False
    start_ms: int = 0
    end_ms: int = 0
    stable_prefix_bytes: int = 0


@dataclass
class TranscriptionResult:
    """Result of a non-streaming transcription."""

    text: str
    language: Optional[str] = None
    route: Optional["RouteMetadata"] = None
    # v0.1.27 facade surfacing — populated only on the native path.
    # ``segments`` is empty and ``chunk_diagnostics`` is None on the
    # legacy / cloud paths, so default behaviour is unchanged.
    segments: list[TranscriptionSegment] = field(default_factory=list)
    duration_ms: int = 0
    chunk_diagnostics: Optional[ChunkDiagnostics] = None


@dataclass
class VadResult:
    """Result of a native voice-activity-detection request."""

    transitions: list[Any]
    sample_rate_hz: int = 16000
    route: Optional["RouteMetadata"] = None


@dataclass
class SpeakerEmbeddingResult:
    """Result of a native speaker-embedding request."""

    embedding: list[float]
    model: str
    dimensions: int
    sample_rate_hz: int = 16000
    route: Optional["RouteMetadata"] = None


@dataclass
class DiarizationResult:
    """Result of a native speaker-diarization request."""

    segments: list[Any]
    sample_rate_hz: int = 16000
    route: Optional["RouteMetadata"] = None
