"""``octomil.audio.diarization`` — native speaker diarization API.

This is the public Python surface for the native ``audio.diarization``
capability. It is deliberately thin: the runtime owns the pyannote
segmentation plus speaker-embedding pipeline, and this module only
provides an ergonomic context manager and stable exports.

There is no Python fallback. Missing native libraries or model artifacts
surface as bounded ``OctomilError`` values from
``octomil.runtime.native.diarization_backend``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from octomil.runtime.native.diarization_backend import (
    DiarizationSegment,
    NativeDiarizationBackend,
    runtime_advertises_audio_diarization,
)


@contextmanager
def open_diarization_backend() -> Iterator[NativeDiarizationBackend]:
    """Open the native diarization backend and close it on exit."""
    backend = NativeDiarizationBackend()
    try:
        backend.open()
        yield backend
    finally:
        backend.close()


__all__ = [
    "DiarizationSegment",
    "NativeDiarizationBackend",
    "open_diarization_backend",
    "runtime_advertises_audio_diarization",
]
