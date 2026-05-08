"""``octomil.audio.vad`` — low-level voice-activity-detection API.

v0.1.5 introduces ``audio.vad`` as a NEW capability surface in the
Python SDK. **No prior Python implementation existed; this is the
canonical API.** It is intentionally a thin pass-through over the
native runtime backend (``octomil.runtime.native.vad_backend``) — the
runtime owns the silero VAD inference, this module owns the Python
ergonomics (context manager, dataclass, type hints).

Design contract (cutover discipline):

* Hard-cut to native. The SDK does NOT fall through to a Python
  implementation when the runtime declines the capability. There is
  no Python VAD implementation to fall through to.
* No new error codes are introduced. Failures route through the
  bounded :class:`octomil.errors.OctomilErrorCode` taxonomy
  (``RUNTIME_UNAVAILABLE``, ``CHECKSUM_MISMATCH``, ``INVALID_INPUT``,
  ``CANCELLED``, ``REQUEST_TIMEOUT``, ``INFERENCE_FAILED``).
* Lazy-fetch shape: open the backend once, open a session per
  utterance, drain transitions, close. Streaming use cases re-poll
  inside a single session.

Example
-------
::

    from octomil.audio.vad import open_vad_backend

    with open_vad_backend() as backend:
        with backend.open_session() as session:
            # Feed mono 16 kHz PCM-f32. Chunk size is up to the
            # caller; the runtime re-windows internally.
            session.feed_chunk(audio_clip)
            for transition in session.poll_transitions(drain_until_completed=True):
                print(f"{transition.kind} @ {transition.timestamp_ms} ms (conf={transition.confidence:.2f})")

The streaming flavor (interleaved feed / poll) is supported by
calling :meth:`VadStreamingSession.poll_transitions` with
``drain_until_completed=False``; that returns currently-pending
transitions and yields control back so the caller can buffer more
audio.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from octomil.runtime.native.vad_backend import (
    NativeVadBackend,
    VadStreamingSession,
    VadTransition,
    runtime_advertises_audio_vad,
)


@contextmanager
def open_vad_backend() -> Iterator[NativeVadBackend]:
    """Context-managed entry point — opens the native backend, runs
    capability-honesty check, yields the backend, and closes it on
    exit.

    Raises
    ------
    OctomilError
        Bounded codes per :mod:`octomil.runtime.native.vad_backend`.
        Most commonly ``RUNTIME_UNAVAILABLE`` when the runtime dylib
        was not built with ``OCT_ENABLE_ENGINE_SILERO_VAD=ON`` or
        ``OCTOMIL_SILERO_VAD_MODEL`` is unset, and
        ``CHECKSUM_MISMATCH`` when the artifact's SHA-256 doesn't
        match the canonical pin.
    """
    backend = NativeVadBackend()
    try:
        backend.open()
        yield backend
    finally:
        backend.close()


__all__ = [
    "NativeVadBackend",
    "VadStreamingSession",
    "VadTransition",
    "open_vad_backend",
    "runtime_advertises_audio_vad",
]
