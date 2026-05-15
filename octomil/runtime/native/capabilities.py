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

# Original Layer 2a enum members (v0.3).
CAPABILITY_AUDIO_REALTIME_SESSION: str = "audio.realtime.session"
CAPABILITY_AUDIO_STT_BATCH: str = "audio.stt.batch"
CAPABILITY_AUDIO_STT_STREAM: str = "audio.stt.stream"
CAPABILITY_AUDIO_TRANSCRIPTION: str = "audio.transcription"
CAPABILITY_AUDIO_TTS_BATCH: str = "audio.tts.batch"
CAPABILITY_AUDIO_TTS_STREAM: str = "audio.tts.stream"
CAPABILITY_CHAT_COMPLETION: str = "chat.completion"
CAPABILITY_CHAT_STREAM: str = "chat.stream"

# v0.4 expansion (PE review consensus, octomil-workspace#27;
# contracts PR octomil-contracts#99). Strict-reject still applies;
# runtime advertises iff implemented.
CAPABILITY_AUDIO_DIARIZATION: str = "audio.diarization"
CAPABILITY_AUDIO_SPEAKER_EMBEDDING: str = "audio.speaker.embedding"
CAPABILITY_AUDIO_VAD: str = "audio.vad"
CAPABILITY_EMBEDDINGS_IMAGE: str = "embeddings.image"
CAPABILITY_EMBEDDINGS_TEXT: str = "embeddings.text"
CAPABILITY_INDEX_VECTOR_QUERY: str = "index.vector.query"

# v0.5 — Lane G cache ABI (octomil-contracts#129;
# octomil-runtime#53). Gates oct_runtime_cache_clear_*/introspect.
# Live conditional: runtimes advertise this only when at least one
# privacy-safe native cache provider is compiled in.
CAPABILITY_CACHE_INTROSPECT: str = "cache.introspect"

#: The canonical set of capabilities the Layer 2a runtime can claim.
#: Lex-sorted within each namespace, namespaces grouped — matches
#: the contract enum's order.
RUNTIME_CAPABILITIES: frozenset[str] = frozenset(
    {
        CAPABILITY_AUDIO_DIARIZATION,
        CAPABILITY_AUDIO_REALTIME_SESSION,
        CAPABILITY_AUDIO_SPEAKER_EMBEDDING,
        CAPABILITY_AUDIO_STT_BATCH,
        CAPABILITY_AUDIO_STT_STREAM,
        CAPABILITY_AUDIO_TRANSCRIPTION,
        CAPABILITY_AUDIO_TTS_BATCH,
        CAPABILITY_AUDIO_TTS_STREAM,
        CAPABILITY_AUDIO_VAD,
        CAPABILITY_CACHE_INTROSPECT,
        CAPABILITY_CHAT_COMPLETION,
        CAPABILITY_CHAT_STREAM,
        CAPABILITY_EMBEDDINGS_IMAGE,
        CAPABILITY_EMBEDDINGS_TEXT,
        CAPABILITY_INDEX_VECTOR_QUERY,
    }
)

CAPABILITY_STATUS_DONE_NATIVE_CUTOVER: str = "DONE_NATIVE_CUTOVER"
CAPABILITY_STATUS_LIVE_NATIVE_CONDITIONAL: str = "LIVE_NATIVE_CONDITIONAL"
CAPABILITY_STATUS_BLOCKED_WITH_PROOF: str = "BLOCKED_WITH_PROOF"

# Current Python SDK truth model for native runtime capabilities.
#
# DONE_NATIVE_CUTOVER means the Python product path is cut over to the
# native runtime and no Python-local product fallback is reachable.
DONE_NATIVE_CUTOVER_CAPABILITIES: frozenset[str] = frozenset(
    {
        CAPABILITY_CHAT_COMPLETION,
        CAPABILITY_CHAT_STREAM,
        CAPABILITY_EMBEDDINGS_TEXT,
    }
)

# LIVE_NATIVE_CONDITIONAL means a real native adapter exists, but
# oct_runtime_capabilities advertises it only when this runtime build's
# build/artifact/digest/sidecar gates pass. These are live, not
# scaffold, and not guaranteed to appear on every machine.
#
# v0.1.13: embeddings.image flipped from BLOCKED_WITH_PROOF to
# LIVE_NATIVE_CONDITIONAL. Runtime PR #91 (89d8005) merged the
# darwin-arm64 adapter — sherpa-onnx vendoring + Xenova SigLIP-base-
# patch16-224 uint8 ONNX (`vision_model_uint8.onnx`). Linux/Android
# remain refused by the runtime adapter; oct_runtime_capabilities
# does not advertise embeddings.image on those platforms. See
# docs/runtime/embeddings-image-abi-scope.md §12 for the platform
# matrix. The Python SDK forwards refusal honestly via the existing
# oct_runtime_capabilities advertisement gate in NativeSession.send_image.
LIVE_NATIVE_CONDITIONAL_CAPABILITIES: frozenset[str] = frozenset(
    {
        CAPABILITY_AUDIO_DIARIZATION,
        CAPABILITY_AUDIO_SPEAKER_EMBEDDING,
        CAPABILITY_AUDIO_STT_BATCH,
        CAPABILITY_AUDIO_STT_STREAM,
        CAPABILITY_AUDIO_TRANSCRIPTION,
        CAPABILITY_AUDIO_TTS_BATCH,
        CAPABILITY_AUDIO_TTS_STREAM,
        CAPABILITY_AUDIO_VAD,
        CAPABILITY_CACHE_INTROSPECT,
        CAPABILITY_EMBEDDINGS_IMAGE,
    }
)

# BLOCKED_WITH_PROOF means the name is legal and bounded by tests, but
# current runtimes must not advertise it. session_open rejects it with
# OCT_STATUS_UNSUPPORTED.
BLOCKED_WITH_PROOF_CAPABILITIES: frozenset[str] = frozenset(
    {
        CAPABILITY_AUDIO_REALTIME_SESSION,
        CAPABILITY_INDEX_VECTOR_QUERY,
    }
)

RUNTIME_CAPABILITY_STATUSES: dict[str, str] = {
    **{capability: CAPABILITY_STATUS_DONE_NATIVE_CUTOVER for capability in DONE_NATIVE_CUTOVER_CAPABILITIES},
    **{capability: CAPABILITY_STATUS_LIVE_NATIVE_CONDITIONAL for capability in LIVE_NATIVE_CONDITIONAL_CAPABILITIES},
    **{capability: CAPABILITY_STATUS_BLOCKED_WITH_PROOF for capability in BLOCKED_WITH_PROOF_CAPABILITIES},
}

assert frozenset(RUNTIME_CAPABILITY_STATUSES) == RUNTIME_CAPABILITIES


__all__ = [
    "BLOCKED_WITH_PROOF_CAPABILITIES",
    "CAPABILITY_STATUS_BLOCKED_WITH_PROOF",
    "CAPABILITY_STATUS_DONE_NATIVE_CUTOVER",
    "CAPABILITY_STATUS_LIVE_NATIVE_CONDITIONAL",
    "CAPABILITY_AUDIO_DIARIZATION",
    "CAPABILITY_AUDIO_REALTIME_SESSION",
    "CAPABILITY_AUDIO_SPEAKER_EMBEDDING",
    "CAPABILITY_AUDIO_STT_BATCH",
    "CAPABILITY_AUDIO_STT_STREAM",
    "CAPABILITY_AUDIO_TRANSCRIPTION",
    "CAPABILITY_AUDIO_TTS_BATCH",
    "CAPABILITY_AUDIO_TTS_STREAM",
    "CAPABILITY_AUDIO_VAD",
    "CAPABILITY_CACHE_INTROSPECT",
    "CAPABILITY_CHAT_COMPLETION",
    "CAPABILITY_CHAT_STREAM",
    "CAPABILITY_EMBEDDINGS_IMAGE",
    "CAPABILITY_EMBEDDINGS_TEXT",
    "CAPABILITY_INDEX_VECTOR_QUERY",
    "DONE_NATIVE_CUTOVER_CAPABILITIES",
    "LIVE_NATIVE_CONDITIONAL_CAPABILITIES",
    "RUNTIME_CAPABILITIES",
    "RUNTIME_CAPABILITY_STATUSES",
]
