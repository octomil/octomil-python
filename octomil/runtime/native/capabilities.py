"""Canonical capability constants for the Layer 2a runtime ABI.

Mirrors ``octomil-contracts/schemas/core/runtime_capability.json``.
Source of truth is the contract; the constants here are codegen-
ready Python identifiers for the same strings. Drift is caught by
the contracts-side parity test and the Slice 3 PR2 conformance
harness's `requires_capability(...)` marker.

**Asymmetric reader rules** (per the contract):

  * Advertisement (runtime → SDK): forward-compatible. Bindings drop
    unknown advertised values from their parsed view.
  * Request (SDK → runtime): strict. Runtime returns
    ``OCT_STATUS_UNSUPPORTED`` on unknown values.

These constants are used in BOTH directions:
  * ``NativeRuntime.capabilities()`` filters advertised values
    against ``RUNTIME_CAPABILITIES`` and silently drops unknowns
    (forward-compat).
  * Future ``oct_session_open`` request paths use the constants
    directly; the runtime validates strictly.
"""

from __future__ import annotations

CAPABILITY_AUDIO_REALTIME_SESSION: str = "audio.realtime.session"
CAPABILITY_AUDIO_STT_BATCH: str = "audio.stt.batch"
CAPABILITY_AUDIO_STT_STREAM: str = "audio.stt.stream"
CAPABILITY_AUDIO_TRANSCRIPTION: str = "audio.transcription"
CAPABILITY_AUDIO_TTS_BATCH: str = "audio.tts.batch"
CAPABILITY_AUDIO_TTS_STREAM: str = "audio.tts.stream"
CAPABILITY_CHAT_COMPLETION: str = "chat.completion"
CAPABILITY_CHAT_STREAM: str = "chat.stream"

#: The canonical set of capabilities the Layer 2a runtime can claim.
#: Lex-sorted within each namespace, namespaces grouped — matches
#: the contract enum's order.
RUNTIME_CAPABILITIES: frozenset[str] = frozenset(
    {
        CAPABILITY_AUDIO_REALTIME_SESSION,
        CAPABILITY_AUDIO_STT_BATCH,
        CAPABILITY_AUDIO_STT_STREAM,
        CAPABILITY_AUDIO_TRANSCRIPTION,
        CAPABILITY_AUDIO_TTS_BATCH,
        CAPABILITY_AUDIO_TTS_STREAM,
        CAPABILITY_CHAT_COMPLETION,
        CAPABILITY_CHAT_STREAM,
    }
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
    "RUNTIME_CAPABILITIES",
]
