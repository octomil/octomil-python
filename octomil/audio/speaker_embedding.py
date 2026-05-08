"""``octomil.audio.speaker_embedding`` — low-level speaker embedding API.

v0.1.5 introduces ``audio.speaker.embedding`` as a NEW capability
surface in the Python SDK. **No prior Python implementation existed;
this is the canonical API.** It is intentionally a thin pass-through
over the native runtime backend
(:mod:`octomil.runtime.native.speaker_backend`) — the runtime owns
the sherpa-onnx ERes2NetV2 inference, this module owns Python
ergonomics.

Design contract (cutover discipline):

* Hard-cut to native. The SDK does NOT fall through to a Python
  implementation when the runtime declines the capability.
* No new error codes are introduced; failures route through the
  bounded :class:`octomil.errors.OctomilErrorCode` taxonomy.
* Single-utterance per call: each :meth:`NativeSpeakerEmbeddingBackend.embed`
  call opens a fresh session, sends the clip, drains, closes.
* Embedding dimension is 512 (canonical ERes2NetV2 base, L2-normalized
  by the runtime). Future model updates may extend or shrink the dim;
  callers should NOT hardcode 512 in downstream comparisons — read it
  from the returned array's ``shape[0]``.

Example
-------
::

    import numpy as np
    from octomil.audio.speaker_embedding import open_speaker_embedding_backend

    with open_speaker_embedding_backend() as backend:
        emb_a = backend.embed(clip_a)        # np.ndarray[float32], shape (512,)
        emb_b = backend.embed(clip_b)
        cosine = float(np.dot(emb_a, emb_b))  # vectors are L2-normalized

Most callers pin to the default model
(``"sherpa-eres2netv2-base"``) — the runtime requires the canonical
SHA-256 to match anyway, so substituting a different name gets you a
fast ``UNSUPPORTED_MODALITY`` rejection rather than silent
substitution.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from octomil.runtime.native.speaker_backend import (
    NativeSpeakerEmbeddingBackend,
    runtime_advertises_audio_speaker_embedding,
)


@contextmanager
def open_speaker_embedding_backend(
    model_name: str = "sherpa-eres2netv2-base",
) -> Iterator[NativeSpeakerEmbeddingBackend]:
    """Context-managed entry point — opens the native backend, warms
    the model, yields the backend, and closes everything on exit.

    Parameters
    ----------
    model_name
        The canonical model identifier the runtime expects. v0.1.5
        only accepts ``"sherpa-eres2netv2-base"``; other names reject
        ``UNSUPPORTED_MODALITY``.

    Raises
    ------
    OctomilError
        Bounded codes per
        :mod:`octomil.runtime.native.speaker_backend`. Most commonly
        ``RUNTIME_UNAVAILABLE`` (dylib missing the engine,
        ``OCTOMIL_SHERPA_SPEAKER_MODEL`` unset) or
        ``CHECKSUM_MISMATCH`` (artifact digest drift).
    """
    backend = NativeSpeakerEmbeddingBackend()
    try:
        backend.load_model(model_name)
        yield backend
    finally:
        backend.close()


__all__ = [
    "NativeSpeakerEmbeddingBackend",
    "open_speaker_embedding_backend",
    "runtime_advertises_audio_speaker_embedding",
]
