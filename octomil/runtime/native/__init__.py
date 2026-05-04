"""Layer 2a embedded host — cffi binding to ``liboctomil-runtime``.

The cffi binding is ONE host path among two. The other is the daemon
host (slice 3b, gated on slice 2-proper). Both link the same dylib;
both implement the C ABI mirrored in ``octomil-contracts`` and
authoritatively defined in the private ``octomil-runtime`` repo's
``include/octomil/runtime.h``.

Slice 3 PR1 ships the loader + version handshake + last-error
wrappers + capabilities reader. Session entry points
(``oct_session_open`` etc.) intentionally NOT exposed in this PR —
the slice-2 stub returns ``OCT_STATUS_UNSUPPORTED`` for all of them
and there is no behavior to bind against until slice 2-proper lands.

Hard-cutover policy: this module is the EMBEDDED host path. The
daemon host (slice 3b) implements the same surface via UDS+protobuf.
The public SDK API (``octomil.audio.*``, etc.) is identical
regardless of host; ``OCTOMIL_RUNTIME_HOST={embedded,daemon,auto}``
is operator-facing only.

Public surface:
  * :class:`NativeRuntime` — open/close + capabilities + last-error.
  * :func:`abi_version` — version-handshake helper.
  * :exc:`NativeRuntimeError` — wraps non-OK status codes with
    ``last_error`` text from the runtime.
  * :data:`CAPABILITY_*` — canonical capability constants
    (re-exported from :mod:`octomil.runtime.native.capabilities`).
"""

from __future__ import annotations

from octomil.runtime.native.capabilities import (
    CAPABILITY_AUDIO_DIARIZATION,
    CAPABILITY_AUDIO_REALTIME_SESSION,
    CAPABILITY_AUDIO_SPEAKER_EMBEDDING,
    CAPABILITY_AUDIO_STT_BATCH,
    CAPABILITY_AUDIO_STT_STREAM,
    CAPABILITY_AUDIO_TRANSCRIPTION,
    CAPABILITY_AUDIO_TTS_BATCH,
    CAPABILITY_AUDIO_TTS_STREAM,
    CAPABILITY_AUDIO_VAD,
    CAPABILITY_CHAT_COMPLETION,
    CAPABILITY_CHAT_STREAM,
    CAPABILITY_EMBEDDINGS_IMAGE,
    CAPABILITY_EMBEDDINGS_TEXT,
    CAPABILITY_INDEX_VECTOR_QUERY,
    RUNTIME_CAPABILITIES,
)
from octomil.runtime.native.loader import (
    OCT_ACCEL_ANE,
    OCT_ACCEL_AUTO,
    OCT_ACCEL_CPU,
    OCT_ACCEL_CUDA,
    OCT_ACCEL_METAL,
    OCT_ERR_ACCELERATOR_UNAVAILABLE,
    OCT_ERR_ARTIFACT_DIGEST_MISMATCH,
    OCT_ERR_ENGINE_INIT_FAILED,
    OCT_ERR_INPUT_FORMAT_UNSUPPORTED,
    OCT_ERR_INPUT_OUT_OF_RANGE,
    OCT_ERR_INTERNAL,
    OCT_ERR_MODEL_LOAD_FAILED,
    OCT_ERR_OK,
    OCT_ERR_PREEMPTED,
    OCT_ERR_QUOTA_EXCEEDED,
    OCT_ERR_RAM_INSUFFICIENT,
    OCT_ERR_TIMEOUT,
    OCT_ERR_UNKNOWN,
    OCT_EVENT_AUDIO_CHUNK,
    # v0.4 step 2 — runtime-scope events.
    OCT_EVENT_CACHE_HIT,
    OCT_EVENT_CACHE_MISS,
    OCT_EVENT_MEMORY_PRESSURE,
    OCT_EVENT_METRIC,
    OCT_EVENT_MODEL_EVICTED,
    OCT_EVENT_MODEL_LOADED,
    OCT_EVENT_NONE,
    OCT_EVENT_PREEMPTED,
    OCT_EVENT_QUEUED,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
    OCT_EVENT_THERMAL_STATE,
    OCT_EVENT_TRANSCRIPT_CHUNK,
    OCT_EVENT_WATCHDOG_TIMEOUT,
    OCT_MODEL_CONFIG_VERSION,
    OCT_PRIORITY_FOREGROUND,
    OCT_PRIORITY_PREFETCH,
    OCT_PRIORITY_SPECULATIVE,
    OCT_STATUS_UNSUPPORTED,
    NativeEvent,
    NativeModel,
    NativeRuntime,
    NativeRuntimeError,
    NativeSession,
    abi_version,
)

__all__ = [
    "CAPABILITY_AUDIO_DIARIZATION",
    "CAPABILITY_AUDIO_REALTIME_SESSION",
    "CAPABILITY_AUDIO_SPEAKER_EMBEDDING",
    "CAPABILITY_AUDIO_STT_BATCH",
    "CAPABILITY_AUDIO_STT_STREAM",
    "CAPABILITY_AUDIO_TRANSCRIPTION",
    "CAPABILITY_AUDIO_TTS_BATCH",
    "CAPABILITY_AUDIO_TTS_STREAM",
    "CAPABILITY_AUDIO_VAD",
    "CAPABILITY_CHAT_COMPLETION",
    "CAPABILITY_CHAT_STREAM",
    "CAPABILITY_EMBEDDINGS_IMAGE",
    "CAPABILITY_EMBEDDINGS_TEXT",
    "CAPABILITY_INDEX_VECTOR_QUERY",
    "NativeEvent",
    "NativeModel",
    "NativeRuntime",
    "NativeRuntimeError",
    "NativeSession",
    "OCT_ACCEL_ANE",
    "OCT_ACCEL_AUTO",
    "OCT_ACCEL_CPU",
    "OCT_ACCEL_CUDA",
    "OCT_ACCEL_METAL",
    "OCT_ERR_ACCELERATOR_UNAVAILABLE",
    "OCT_ERR_ARTIFACT_DIGEST_MISMATCH",
    "OCT_ERR_ENGINE_INIT_FAILED",
    "OCT_ERR_INPUT_FORMAT_UNSUPPORTED",
    "OCT_ERR_INPUT_OUT_OF_RANGE",
    "OCT_ERR_INTERNAL",
    "OCT_ERR_MODEL_LOAD_FAILED",
    "OCT_ERR_OK",
    "OCT_ERR_PREEMPTED",
    "OCT_ERR_QUOTA_EXCEEDED",
    "OCT_ERR_RAM_INSUFFICIENT",
    "OCT_ERR_TIMEOUT",
    "OCT_ERR_UNKNOWN",
    "OCT_EVENT_AUDIO_CHUNK",
    "OCT_EVENT_CACHE_HIT",
    "OCT_EVENT_CACHE_MISS",
    "OCT_EVENT_MEMORY_PRESSURE",
    "OCT_EVENT_METRIC",
    "OCT_EVENT_MODEL_EVICTED",
    "OCT_EVENT_MODEL_LOADED",
    "OCT_EVENT_NONE",
    "OCT_EVENT_PREEMPTED",
    "OCT_EVENT_QUEUED",
    "OCT_EVENT_SESSION_COMPLETED",
    "OCT_EVENT_SESSION_STARTED",
    "OCT_EVENT_THERMAL_STATE",
    "OCT_EVENT_TRANSCRIPT_CHUNK",
    "OCT_EVENT_WATCHDOG_TIMEOUT",
    "OCT_MODEL_CONFIG_VERSION",
    "OCT_PRIORITY_FOREGROUND",
    "OCT_PRIORITY_PREFETCH",
    "OCT_PRIORITY_SPECULATIVE",
    "OCT_STATUS_UNSUPPORTED",
    "RUNTIME_CAPABILITIES",
    "abi_version",
]
