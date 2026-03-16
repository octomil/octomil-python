"""Audio data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResult:
    """Result of a non-streaming transcription."""

    text: str
    language: Optional[str] = None


@dataclass
class TranscriptionSegment:
    """A single segment from streaming transcription."""

    text: str
    start_ms: int = 0
    end_ms: int = 0
