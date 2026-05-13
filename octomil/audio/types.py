"""Audio data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from octomil.execution.kernel import RouteMetadata


@dataclass
class TranscriptionResult:
    """Result of a non-streaming transcription."""

    text: str
    language: Optional[str] = None
    route: Optional["RouteMetadata"] = None


@dataclass
class TranscriptionSegment:
    """A single segment from streaming transcription."""

    text: str
    start_ms: int = 0
    end_ms: int = 0


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
