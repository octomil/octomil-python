"""OctomilAudio — audio namespace on OctomilClient."""

from __future__ import annotations

from typing import Callable, Optional

from octomil.audio.transcriptions import AudioTranscriptions
from octomil.audio.types import TranscriptionResult, TranscriptionSegment
from octomil.model_ref import ModelRef
from octomil.runtime.core.model_runtime import ModelRuntime


class OctomilAudio:
    """Namespace for audio APIs on OctomilClient.

    Usage::

        result = await client.audio.transcriptions.create(audio=data)
    """

    def __init__(
        self,
        runtime_resolver: Callable[[ModelRef], Optional[ModelRuntime]],
    ) -> None:
        self._transcriptions = AudioTranscriptions(runtime_resolver)

    @property
    def transcriptions(self) -> AudioTranscriptions:
        return self._transcriptions


__all__ = [
    "OctomilAudio",
    "AudioTranscriptions",
    "TranscriptionResult",
    "TranscriptionSegment",
]
