"""Centralized ``oct_status_t`` → :class:`OctomilError` mapper.

v0.1.6 PR1: SDK error typing alignment. Until v0.1.5 every native
backend (chat / embeddings / stt / vad / speaker) carried its own
``_runtime_status_to_sdk_error`` helper; the bodies were nearly
identical except for the ``OCT_STATUS_UNSUPPORTED`` policy
(chat/embeddings → ``UNSUPPORTED_MODALITY``; audio capabilities →
``RUNTIME_UNAVAILABLE`` / ``CHECKSUM_MISMATCH`` disambiguated by
``last_error`` substring). The Lane G PR3 conformance generator
surfaced the divergence as Invariant 1 SKIPs — the contract-side
expectation is the typed ``OctomilError(<canonical_code>)``, not
``NativeRuntimeError`` with a numeric status field.

Boundary rule (load-bearing):

    * **Product paths raise canonical OctomilError.** Anything reachable
      through the public surface of a backend (``transcribe``,
      ``feed_chunk``, ``embed``, ``generate``) MUST raise
      :class:`octomil.errors.OctomilError` with a code drawn from the
      bounded SDK taxonomy. Use :func:`map_oct_status` at every
      ``except NativeRuntimeError`` site on the product path.
    * **Loader-level dlopen / ABI errors raise NativeRuntimeError.**
      The cffi wrapper methods inside ``loader.py`` (``oct_runtime_open``
      failure, ABI-MAJOR mismatch at dlopen, send_audio shape rejects,
      etc.) are allowed to raise raw :class:`NativeRuntimeError`. The
      backends catch and translate at the boundary; raw native errors
      are an explicit "you are looking inside the runtime wrapper, not
      the SDK product" signal.

The disambiguation rules below are derived from the runtime adapter
source (``octomil-runtime/src/{whisper_cpp,silero_vad,sherpa_*}_adapter.cpp``)
and the conformance YAMLs at
``octomil-contracts/fixtures/conformance/audio.*.yaml``.
"""

from __future__ import annotations

from ...errors import OctomilError, OctomilErrorCode
from .loader import (
    OCT_STATUS_BUSY,
    OCT_STATUS_CANCELLED,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_NOT_FOUND,
    OCT_STATUS_TIMEOUT,
    OCT_STATUS_UNSUPPORTED,
    OCT_STATUS_VERSION_MISMATCH,
)

__all__ = ["map_oct_status"]


# Substring matches for ``OCT_STATUS_UNSUPPORTED`` last_error
# disambiguation. Order matters — first match wins.
#
# Source-of-truth mapping (canonical, derived from the runtime adapter
# source and the conformance YAML expectations):
#
#   "digest"                       → CHECKSUM_MISMATCH (artifact integrity fail)
#   "audio.transcription"          → RUNTIME_UNAVAILABLE (capability not advertised)
#   "audio.vad"                    → RUNTIME_UNAVAILABLE
#   "audio.speaker.embedding"      → RUNTIME_UNAVAILABLE
#   "OCTOMIL_<X>_BIN" / "_MODEL"   → RUNTIME_UNAVAILABLE (artifact env unset)
#   "unset"                        → RUNTIME_UNAVAILABLE (env var unset)
#   "changed"                      → INVALID_INPUT (env drift between open/model_open)
#   "sample_rate" | "NaN" | "Inf" | "format"
#                                  → INVALID_INPUT (audio shape reject)
#   default UNSUPPORTED on the audio path → RUNTIME_UNAVAILABLE
#
# The "audio" capability strings appear in the runtime's
# ``oct_session_open`` last_error when the dispatcher rejects a
# session for a capability that the loaded adapter does NOT advertise
# (e.g. silero adapter is built but the operator forgot
# ``OCTOMIL_SILERO_VAD_BIN`` so the capability is filtered out).
_DIGEST_MARKERS = ("digest",)
_CAPABILITY_MARKERS = (
    "audio.transcription",
    "audio.vad",
    "audio.speaker.embedding",
)
_ENV_UNSET_MARKERS = (
    "octomil_whisper_bin",
    "octomil_silero_vad_bin",
    "octomil_sherpa_speaker_model",
    "_bin unset",
    "_model unset",
    " unset",
)
_CHANGED_MARKERS = ("changed",)
_INVALID_INPUT_MARKERS = ("sample_rate", "nan", "inf", "format")


def map_oct_status(
    status: int,
    last_error: str = "",
    *,
    message: str = "",
    default_unsupported_code: OctomilErrorCode = OctomilErrorCode.RUNTIME_UNAVAILABLE,
) -> OctomilError:
    """Map a non-OK ``oct_status_t`` (+ ``last_error`` text) to a typed
    :class:`OctomilError` from the bounded SDK taxonomy.

    Parameters
    ----------
    status:
        The numeric ``oct_status_t`` returned by the runtime ABI.
    last_error:
        ``oct_runtime_last_error()`` text (or
        :attr:`NativeRuntimeError.last_error`). May be empty; the
        substring rules below are case-insensitive and skip cleanly
        when no marker is present.
    message:
        SDK-side context to prepend to the ``OctomilError`` message
        (e.g. ``"native STT backend failed to open runtime"``).
    default_unsupported_code:
        Policy knob for ``OCT_STATUS_UNSUPPORTED`` when no substring
        rule matches. Defaults to ``RUNTIME_UNAVAILABLE`` (the audio
        backends' policy: capability filtered out by adapter
        ``is_loadable_now`` check). Chat and embeddings backends pass
        ``UNSUPPORTED_MODALITY`` instead — for them, UNSUPPORTED comes
        from request shape (grammar / pooling / unsupported role),
        not capability availability.

    Returns
    -------
    A typed :class:`OctomilError`. Caller is expected to ``raise`` it.

    Notes
    -----
    The runtime returns ``OCT_STATUS_UNSUPPORTED`` for at least three
    distinct conditions, separable only by ``last_error`` text:

        1. Artifact digest mismatch ("digest" substring)
           → :attr:`OctomilErrorCode.CHECKSUM_MISMATCH`
        2. Capability not advertised because env var unset
           (env var name appears in last_error)
           → :attr:`OctomilErrorCode.RUNTIME_UNAVAILABLE`
        3. Audio sample shape reject (sample_rate / NaN / format)
           → :attr:`OctomilErrorCode.INVALID_INPUT`

    See also: ``sdk_probe_session_pattern.md`` for the probe-session
    pattern bindings use to surface (1) vs (2) before the user even
    calls ``transcribe()``.
    """
    last_error_lc = (last_error or "").lower()

    if status == OCT_STATUS_NOT_FOUND:
        # Adapter missing artifact path → MODEL_NOT_FOUND. If
        # last_error mentions a capability name, it's the
        # capability-unloadable variant (artifact path is set but
        # not pointing at a real file) — RUNTIME_UNAVAILABLE.
        if any(marker in last_error_lc for marker in _CAPABILITY_MARKERS):
            code = OctomilErrorCode.RUNTIME_UNAVAILABLE
        else:
            code = OctomilErrorCode.MODEL_NOT_FOUND

    elif status == OCT_STATUS_INVALID_INPUT:
        code = OctomilErrorCode.INVALID_INPUT

    elif status == OCT_STATUS_UNSUPPORTED:
        # Order: digest > env-drift > shape-reject > capability/env-unset.
        if any(marker in last_error_lc for marker in _DIGEST_MARKERS):
            code = OctomilErrorCode.CHECKSUM_MISMATCH
        elif any(marker in last_error_lc for marker in _CHANGED_MARKERS):
            # Env drift between runtime_open and model_open. Bounded
            # INVALID_INPUT reject per v0.1.5 capability gating regime.
            code = OctomilErrorCode.INVALID_INPUT
        elif any(marker in last_error_lc for marker in _INVALID_INPUT_MARKERS):
            code = OctomilErrorCode.INVALID_INPUT
        elif any(marker in last_error_lc for marker in _CAPABILITY_MARKERS):
            code = OctomilErrorCode.RUNTIME_UNAVAILABLE
        elif any(marker in last_error_lc for marker in _ENV_UNSET_MARKERS):
            code = OctomilErrorCode.RUNTIME_UNAVAILABLE
        else:
            code = default_unsupported_code

    elif status == OCT_STATUS_VERSION_MISMATCH:
        # ABI skew at oct_runtime_open. The dlopen-time skew is caught
        # earlier in loader._build_lib() and surfaces as
        # NativeRuntimeError; we can still see VERSION_MISMATCH on the
        # product path if a fresh runtime open detects a divergent
        # session config version etc. → RUNTIME_UNAVAILABLE.
        code = OctomilErrorCode.RUNTIME_UNAVAILABLE

    elif status == OCT_STATUS_CANCELLED:
        # Canonical taxonomy uses ``CANCELLED`` (not "OPERATION_CANCELLED").
        code = OctomilErrorCode.CANCELLED

    elif status == OCT_STATUS_TIMEOUT:
        # Canonical taxonomy uses ``REQUEST_TIMEOUT`` (not bare "TIMEOUT").
        code = OctomilErrorCode.REQUEST_TIMEOUT

    elif status == OCT_STATUS_BUSY:
        # Defensive — should not reach the public path. The runtime
        # serializes session_open behind a session pool; OCT_STATUS_BUSY
        # surfaces only if the pool is saturated, which the SDK queues
        # rather than propagates. Map to SERVER_ERROR so it surfaces as
        # 500 if it ever leaks.
        code = OctomilErrorCode.SERVER_ERROR

    else:
        # Catch-all for OCT_STATUS_INTERNAL / unknown codes / future
        # additions. INFERENCE_FAILED is the bounded "something broke
        # while running the model" code.
        code = OctomilErrorCode.INFERENCE_FAILED

    if message and last_error:
        full_message = f"{message}: {last_error}"
    elif message:
        full_message = message
    elif last_error:
        full_message = last_error
    else:
        full_message = f"oct_status_t={status}"
    return OctomilError(code=code, message=full_message)
