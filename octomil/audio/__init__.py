"""OctomilAudio — audio namespace.

Two shapes coexist:

* :class:`OctomilAudio` is the local-only namespace exposed on the legacy
  ``OctomilClient`` (``client.audio.transcriptions.create``).
* :class:`FacadeAudio` is the unified routed namespace exposed on the
  top-level :class:`octomil.Octomil` facade (``client.audio.speech.create``).
  It delegates to :class:`octomil.execution.kernel.ExecutionKernel` so a
  single code path resolves app refs and respects the routing policy.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from octomil.audio.speech import FacadeSpeech, SpeechResponse, SpeechRoute
from octomil.audio.transcriptions import AudioTranscriptions
from octomil.audio.types import TranscriptionResult, TranscriptionSegment
from octomil.model_ref import ModelRef
from octomil.runtime.core.model_runtime import ModelRuntime


class OctomilAudio:
    """Namespace for audio APIs on the legacy ``OctomilClient``.

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


class FacadeAudio:
    """Namespace for audio APIs on the top-level :class:`octomil.Octomil`.

    Wires :attr:`speech` against the execution kernel so app refs
    (``@app/<slug>/tts``) resolve through the routing policy.

    Usage::

        client = Octomil.from_env()
        await client.initialize()
        response = await client.audio.speech.create(
            model="@app/<slug>/tts",
            input="Hello from Octomil.",
        )
    """

    def __init__(self, kernel: Any) -> None:
        self._speech = FacadeSpeech(kernel)

    @property
    def speech(self) -> FacadeSpeech:
        return self._speech


__all__ = [
    "OctomilAudio",
    "FacadeAudio",
    "FacadeSpeech",
    "SpeechResponse",
    "SpeechRoute",
    "AudioTranscriptions",
    "TranscriptionResult",
    "TranscriptionSegment",
]
