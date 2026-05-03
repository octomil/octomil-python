"""Layer 2a embedded host — cffi binding to ``liboctomil-runtime``.

The cffi binding is ONE host path among two. The other is the daemon
host (slice 3b, gated on slice 2-proper). Both link the same dylib;
both implement the C ABI defined in
``octomil/runtime-core/include/octomil/runtime.h``.

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
    CAPABILITY_AUDIO_REALTIME_SESSION,
    CAPABILITY_AUDIO_STT_BATCH,
    CAPABILITY_AUDIO_STT_STREAM,
    CAPABILITY_AUDIO_TRANSCRIPTION,
    CAPABILITY_AUDIO_TTS_BATCH,
    CAPABILITY_AUDIO_TTS_STREAM,
    CAPABILITY_CHAT_COMPLETION,
    CAPABILITY_CHAT_STREAM,
    RUNTIME_CAPABILITIES,
)
from octomil.runtime.native.loader import (
    NativeRuntime,
    NativeRuntimeError,
    abi_version,
)

__all__ = [
    "CAPABILITY_AUDIO_REALTIME_SESSION",
    "CAPABILITY_AUDIO_STT_BATCH",
    "CAPABILITY_AUDIO_STT_STREAM",
    "CAPABILITY_AUDIO_TRANSCRIPTION",
    "CAPABILITY_AUDIO_TTS_BATCH",
    "CAPABILITY_AUDIO_TTS_STREAM",
    "CAPABILITY_CHAT_COMPLETION",
    "CAPABILITY_CHAT_STREAM",
    "NativeRuntime",
    "NativeRuntimeError",
    "RUNTIME_CAPABILITIES",
    "abi_version",
]
