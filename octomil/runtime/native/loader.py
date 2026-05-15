"""cffi loader for ``liboctomil-runtime``.

Embedded host (in-process) binding. Locates the dylib via:

  1. ``OCTOMIL_RUNTIME_DYLIB`` env var (operator override). Wins if set.
  2. Most-recently-fetched dev artifact under
     ``~/.cache/octomil-runtime/<version>/lib/``, populated by
     ``scripts/fetch_runtime_dev.py``.
  3. ``ImportError`` naming both resolution paths.

The runtime SOURCE lives in the private ``octomil-runtime`` repo.
This module never builds from source and never searches for an
in-tree ``runtime-core`` subtree (that subtree was extracted out of
this repo). SDK builds consume the runtime via a binary release
artifact; the fetch script + env override ARE the supported
resolution paths.

This module exposes:

  * Version inspection (``oct_runtime_abi_version_*``).
  * Runtime open/close (``oct_runtime_open``, ``oct_runtime_close``).
  * Capabilities (``oct_runtime_capabilities``,
    ``oct_runtime_capabilities_free``).
  * Last-error (``oct_runtime_last_error``,
    ``oct_last_thread_error``).
  * Session lifecycle bindings: ``NativeSession`` wraps
    ``oct_session_open / send_audio / send_text / poll_event /
    cancel / close``. Live native capabilities use these entry points
    when the runtime advertises them; blocked capability names reject
    with bounded ``OCT_STATUS_UNSUPPORTED``.

**Forward-compat advertisement parsing:** capabilities returned by
the runtime that do NOT appear in :data:`RUNTIME_CAPABILITIES` are
silently dropped from the parsed view (with a DEBUG log). This
matches the asymmetric reader rule from the contract.
"""

from __future__ import annotations

import logging
import os
import struct as _emb_struct  # used by poll_event for OCT_EVENT_EMBEDDING_VECTOR
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status codes — mirrors the C ABI header.
# ---------------------------------------------------------------------------

#: Operation succeeded.
OCT_STATUS_OK: int = 0
#: Malformed config / NULL out / etc.
OCT_STATUS_INVALID_INPUT: int = 1
#: Capability / locality / engine not supported by this build.
OCT_STATUS_UNSUPPORTED: int = 2
#: Model URI / artifact missing on disk.
OCT_STATUS_NOT_FOUND: int = 3
#: Input queue full (realtime backpressure).
OCT_STATUS_BUSY: int = 4
#: ``poll_event`` timeout.
OCT_STATUS_TIMEOUT: int = 5
#: Session was cancelled.
OCT_STATUS_CANCELLED: int = 6
#: Runtime invariant violated.
OCT_STATUS_INTERNAL: int = 7
#: ``config.version`` unknown to this runtime build.
OCT_STATUS_VERSION_MISMATCH: int = 8

_STATUS_NAMES: dict[int, str] = {
    OCT_STATUS_OK: "OCT_STATUS_OK",
    OCT_STATUS_INVALID_INPUT: "OCT_STATUS_INVALID_INPUT",
    OCT_STATUS_UNSUPPORTED: "OCT_STATUS_UNSUPPORTED",
    OCT_STATUS_NOT_FOUND: "OCT_STATUS_NOT_FOUND",
    OCT_STATUS_BUSY: "OCT_STATUS_BUSY",
    OCT_STATUS_TIMEOUT: "OCT_STATUS_TIMEOUT",
    OCT_STATUS_CANCELLED: "OCT_STATUS_CANCELLED",
    OCT_STATUS_INTERNAL: "OCT_STATUS_INTERNAL",
    OCT_STATUS_VERSION_MISMATCH: "OCT_STATUS_VERSION_MISMATCH",
}


# ---------------------------------------------------------------------------
# Priority + event-type constants (slice 2A) — mirror runtime.h.
# ---------------------------------------------------------------------------

OCT_PRIORITY_SPECULATIVE: int = 0
OCT_PRIORITY_PREFETCH: int = 1
OCT_PRIORITY_FOREGROUND: int = 2

OCT_EVENT_NONE: int = 0
OCT_EVENT_SESSION_STARTED: int = 1
OCT_EVENT_AUDIO_CHUNK: int = 2
OCT_EVENT_TRANSCRIPT_CHUNK: int = 3
OCT_EVENT_USER_SPEECH_DETECTED: int = 4
OCT_EVENT_TURN_ENDED: int = 5
OCT_EVENT_CAPABILITY_VERIFIED: int = 6
OCT_EVENT_ERROR: int = 7
OCT_EVENT_SESSION_COMPLETED: int = 8
OCT_EVENT_INPUT_DROPPED: int = 9
# v0.4 step 2 — runtime-scope events delivered via the
# oct_telemetry_sink_fn callback.
OCT_EVENT_MODEL_LOADED: int = 10
OCT_EVENT_MODEL_EVICTED: int = 11
OCT_EVENT_CACHE_HIT: int = 12
OCT_EVENT_CACHE_MISS: int = 13
OCT_EVENT_QUEUED: int = 14
OCT_EVENT_PREEMPTED: int = 15
OCT_EVENT_MEMORY_PRESSURE: int = 16
OCT_EVENT_THERMAL_STATE: int = 17
OCT_EVENT_WATCHDOG_TIMEOUT: int = 18
OCT_EVENT_METRIC: int = 19
# v0.1.3 — embeddings.text result event. Payload is `data.embedding_vector`
# { values, n_dim, n_input_tokens, index, pooling_type, is_normalized }.
# `values` is a runtime-owned float32 buffer with `n_dim` elements;
# lifetime = until next poll on this session (same contract as
# transcript_chunk text). Bindings MUST copy before issuing another poll.
OCT_EVENT_EMBEDDING_VECTOR: int = 20
# v0.1.5 PR-2 — STT events. TRANSCRIPT_SEGMENT carries one timestamped
# segment (utf8, n_bytes, start_ms, end_ms, segment_index, is_final).
# TRANSCRIPT_FINAL is the end-of-transcript event with the concatenated
# UTF-8 plus n_segments + duration_ms; followed immediately by
# OCT_EVENT_SESSION_COMPLETED(OK). All inner pointer fields are
# runtime-owned; lifetime = until next poll. Bindings MUST copy.
OCT_EVENT_TRANSCRIPT_SEGMENT: int = 21
OCT_EVENT_TRANSCRIPT_FINAL: int = 22
# v0.1.8 Lane A — TTS audio-chunk event (audio.tts.batch + audio.tts.stream).
# Carries PCM bytes for one synthesized chunk; granularity depends on the
# capability:
#   * audio.tts.batch  — ONE chunk per utterance (is_final=1).
#   * audio.tts.stream — ONE chunk per sentence batch (Option C, sherpa
#     `OfflineTts::Generate(callback=)`); last chunk per utterance carries
#     is_final=1, followed by OCT_EVENT_SESSION_COMPLETED(OK).
# Layout: ``data.tts_audio_chunk { pcm, n_bytes, sample_rate, sample_format,
# channels, is_final }`` — runtime-owned ``pcm`` lifetime = until next poll.
# Bindings MUST copy bytes out at poll time (this binding does so via
# numpy.frombuffer(...).copy()).
OCT_EVENT_TTS_AUDIO_CHUNK: int = 23
# v0.1.5 PR-2N — VAD transition event (audio.vad capability, silero VAD
# adapter PR-2F at runtime). One event per transition edge; payload is
# `data.vad_transition` { transition_kind, timestamp_ms, confidence }.
# Multiple transitions per session are normal — bindings pair START
# with the next END to derive a span. confidence ∈ [0.0, 1.0] is the
# average per-window silero probability across the span.
OCT_EVENT_VAD_TRANSITION: int = 24
# audio.diarization segment event. One event per speaker turn,
# followed by OCT_EVENT_SESSION_COMPLETED.
OCT_EVENT_DIARIZATION_SEGMENT: int = 25

# v0.1.5 PR-2N — closed-enum sentinels for
# data.vad_transition.transition_kind. UNKNOWN is a future-compat
# sentinel; never emitted by v0.1.5 runtime but bindings MUST handle it
# (treat as "skip event") rather than crash, per the runtime header
# contract. SPEECH_START is the leading edge; SPEECH_END is the
# trailing edge. Bindings derive a [start, end] span by pairing a
# SPEECH_START with the next SPEECH_END on the same session.
OCT_VAD_TRANSITION_UNKNOWN: int = 0
OCT_VAD_TRANSITION_SPEECH_START: int = 1
OCT_VAD_TRANSITION_SPEECH_END: int = 2

OCT_SAMPLE_FORMAT_PCM_S16LE: int = 1
OCT_SAMPLE_FORMAT_PCM_F32LE: int = 2

# v0.1.3 — pooling-type discriminator carried on
# data.embedding_vector.pooling_type. Mirrors llama.cpp's
# LLAMA_POOLING_TYPE_* enum 1:1. UNKNOWN(0) is a forward-compat
# sentinel for any value the runtime doesn't recognize. The runtime
# REJECTS NONE/UNSPECIFIED (decoder-only chat models) and RANK
# (re-rankers; output is n_cls_out, not n_embd) at session_open with
# OCT_STATUS_UNSUPPORTED — see the per-context pooling-type gate.
OCT_EMBED_POOLING_UNKNOWN: int = 0
OCT_EMBED_POOLING_MEAN: int = 1
OCT_EMBED_POOLING_CLS: int = 2
OCT_EMBED_POOLING_LAST: int = 3
OCT_EMBED_POOLING_RANK: int = 4

# audio.diarization speaker_id sentinel.
OCT_DIARIZATION_SPEAKER_UNKNOWN: int = 65535

# v0.1.1 — bumped lockstep with runtime.h. session_config v=3 adds
# the appended `oct_model_t* model` field; chat.completion sessions
# REQUIRE non-NULL config.model on v=3 (runtime returns INVALID_INPUT
# otherwise — bindings MUST upgrade or stay on non-chat capabilities).
OCT_SESSION_CONFIG_VERSION: int = 3
OCT_EVENT_VERSION: int = 2

# v0.4 step 1 — model lifecycle config struct (unchanged in v0.1.1).
OCT_MODEL_CONFIG_VERSION: int = 1

OCT_ACCEL_AUTO: int = 0
OCT_ACCEL_METAL: int = 1
OCT_ACCEL_CUDA: int = 2
OCT_ACCEL_CPU: int = 3
OCT_ACCEL_ANE: int = 4

# v0.1.11 Lane G — cache scope discriminators for
# oct_runtime_cache_clear_scope. Mirrors OCT_CACHE_SCOPE_* in runtime.h.
# Broader scopes subsume narrower: APP > RUNTIME > SESSION > REQUEST.
# Scope codes match the canonical CacheScope enum (request|session|runtime|app)
# in octomil-contracts v1.24.0 (enums/cache_scope.yaml, #123). The integer
# ordering here is a binding-internal detail; the canonical wire form is
# the string code emitted in CacheEntrySnapshot.scope.
OCT_CACHE_SCOPE_REQUEST: int = 0  # clear entries created within a single request
OCT_CACHE_SCOPE_SESSION: int = 1  # clear entries scoped to a session
OCT_CACHE_SCOPE_RUNTIME: int = 2  # clear all entries scoped to this runtime handle
OCT_CACHE_SCOPE_APP: int = 3  # clear all entries including persistent on-disk

# Capability constant that gates the cache introspect API.
# CANONICAL: registered in octomil-contracts v1.25.0
# (schemas/core/runtime_capability.json, #129).
OCT_CAPABILITY_CACHE_INTROSPECT: str = "cache.introspect"

# v0.4 — bounded error code taxonomy. Wire-value enum (uint32_t).
# Mirrors octomil-contracts/fixtures/runtime_error_code/canonical_error_codes.json.
OCT_ERR_OK: int = 0
OCT_ERR_MODEL_LOAD_FAILED: int = 1
OCT_ERR_ARTIFACT_DIGEST_MISMATCH: int = 2
OCT_ERR_ENGINE_INIT_FAILED: int = 3
OCT_ERR_RAM_INSUFFICIENT: int = 4
OCT_ERR_ACCELERATOR_UNAVAILABLE: int = 5
OCT_ERR_INPUT_OUT_OF_RANGE: int = 6
OCT_ERR_INPUT_FORMAT_UNSUPPORTED: int = 7
OCT_ERR_TIMEOUT: int = 8
OCT_ERR_PREEMPTED: int = 9
OCT_ERR_QUOTA_EXCEEDED: int = 10
OCT_ERR_INTERNAL: int = 11
OCT_ERR_UNKNOWN: int = 0xFFFFFFFF  # forward-compat sentinel — UINT32_MAX

# v0.1.12 / ABI minor 11 — image-input MIME discriminator for
# ``oct_image_view_t.mime``. Closed enum with a forward-compat
# sentinel at 0 (same rule as OCT_EMBED_POOLING_UNKNOWN and
# OCT_VAD_TRANSITION_UNKNOWN). PNG/JPEG/WEBP are encoded forms
# decoded engine-side via the vendored stb_image; RGB8 is a raw
# decoded uint8 RGB pixel buffer (dimensions inferred from
# n_bytes once the adapter contract pins width*height).
#
# Symbol presence in the dylib is gated by ABI minor >= 11;
# capability use is gated by ``oct_runtime_capabilities``
# advertising "embeddings.image". See
# docs/runtime/embeddings-image-abi-scope.md (#85).
OCT_IMAGE_MIME_UNKNOWN: int = 0  # future-compat sentinel; never set by callers
OCT_IMAGE_MIME_PNG: int = 1  # image/png — encoded
OCT_IMAGE_MIME_JPEG: int = 2  # image/jpeg — encoded
OCT_IMAGE_MIME_WEBP: int = 3  # image/webp — encoded
OCT_IMAGE_MIME_RGB8: int = 4  # raw decoded uint8 RGB pixel buffer

# v0.1.12 / ABI minor 11 — appended embedding pooling discriminator
# for CLIP/SigLIP-style image embeddings (single pooled vector per
# image, NOT a text mean-pool). Disambiguates image vs text
# embeddings at the consumer side. Existing OCT_EMBED_POOLING_*
# values 0..4 are unchanged; this is a pure append.
OCT_EMBED_POOLING_IMAGE_CLIP: int = 5


def _status_name(code: int) -> str:
    return _STATUS_NAMES.get(code, f"OCT_STATUS_UNKNOWN({code})")


def _validate_correlation_id(
    label: str,
    value: str | None,
    *,
    max_bytes: int,
) -> None:
    """Validate a v0.4-step-2 correlation ID against the
    runtime.h contract: caller-owned UTF-8 string, ≤ max_bytes,
    ASCII-printable (codepoints 0x21..0x7E), no whitespace, no
    control characters. NULL/None is allowed (runtime echoes ""
    on events for un-correlated slots).

    Codex R1 fix: previous version only length-checked; the
    full contract is enforced here so invalid IDs cannot land
    in trace/audit envelope labels."""
    if value is None:
        return
    # Codex R2 nit: raw `str.encode("utf-8")` raises UnicodeEncodeError
    # on lone surrogates / unencodable strings. Translate to the
    # ABI's typed error so callers don't have to handle two
    # exception types for the same "bad correlation ID" bug class.
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise NativeRuntimeError(
            OCT_STATUS_INVALID_INPUT,
            f"{label} contains unencodable characters (UTF-8): {exc}",
        ) from exc
    if len(encoded) > max_bytes:
        raise NativeRuntimeError(
            OCT_STATUS_INVALID_INPUT,
            f"{label} exceeds {max_bytes}-byte limit ({len(encoded)} bytes)",
        )
    # ASCII-printable, no whitespace, no control chars.
    for i, ch in enumerate(value):
        cp = ord(ch)
        if cp < 0x21 or cp > 0x7E:
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                f"{label} contains invalid character at offset {i} "
                f"(codepoint U+{cp:04X}); ABI requires ASCII-printable "
                f"(0x21..0x7E), no whitespace, no control chars",
            )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NativeRuntimeError(RuntimeError):
    """Raised when an OCT_API call returns a non-OK status. Carries
    the numeric status code AND the runtime's ``last_error`` text
    (when available)."""

    def __init__(self, status: int, message: str, last_error: str = "") -> None:
        super().__init__(
            f"{_status_name(status)} ({status}): {message}"
            + (f" — runtime last_error: {last_error!r}" if last_error else "")
        )
        self.status: int = status
        self.last_error: str = last_error


class OctomilUnsupportedError(NativeRuntimeError):
    """Raised by binding-side gates when a capability is not actually
    usable on the loaded runtime — either the dylib's ABI minor lacks
    the necessary symbol table, or ``oct_runtime_capabilities`` does
    not advertise the capability string.

    Carries ``OCT_STATUS_UNSUPPORTED`` so callers that already match
    on the numeric status code see the same value. Distinct subclass
    so capability gates can be caught precisely without swallowing
    every NativeRuntimeError.
    """

    def __init__(self, capability: str, message: str) -> None:
        super().__init__(OCT_STATUS_UNSUPPORTED, message)
        self.capability: str = capability


# ---------------------------------------------------------------------------
# Dylib resolution
# ---------------------------------------------------------------------------

ENV_DYLIB_OVERRIDE: str = "OCTOMIL_RUNTIME_DYLIB"

# Env var that selects a specific runtime flavor when set.
# Valid values: "chat", "stt".  Authoritative — no fallback when set.
_ENV_FLAVOR: str = "OCTOMIL_RUNTIME_FLAVOR"

# Preference order for flavor selection when no env override is set.
# chat covers the most common consumer paths (chat completion, embeddings);
# stt is opt-in.  First entry is most preferred (tried first by resolver).
_FLAVOR_PREFERENCE: tuple[str, ...] = ("chat", "stt")

# Default location for a fetched dev artifact. The fetch script
# (`scripts/fetch_runtime_dev.py`) extracts release tarballs into
# `~/.cache/octomil-runtime/<version>/lib/liboctomil-runtime.dylib`.
# This is the ONLY fallback after the explicit env override; we
# do NOT walk up to a sibling `runtime-core/` subtree any more —
# that source code now lives in the private `octomil-runtime`
# repo and is consumed via signed (eventually) binary releases.
_FETCH_CACHE_ROOT = Path.home() / ".cache" / "octomil-runtime"
_RUNTIME_LIBNAMES = (
    "liboctomil-runtime.dylib",  # macOS
    "liboctomil-runtime.so",  # Linux
    "octomil-runtime.dll",  # Windows
)


def _version_sort_key(p: Path) -> tuple:
    """Parse ``v0.0.1`` / ``v0.10.2`` / ``v1.2.3-rc1`` style cache
    directory names into a sortable tuple. Falls back to a string
    sort for unparseable names so the function never raises.

    Codex R1 fix: lexicographic sort would put ``v0.0.10`` BEFORE
    ``v0.0.2`` and the most-recently-fetched-wins rule would pick
    the wrong release. Parse numeric components when possible."""
    name = p.name
    if name.startswith("v"):
        name = name[1:]
    parts = name.split("-", 1)
    core = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    nums: list[int] = []
    for chunk in core.split("."):
        try:
            nums.append(int(chunk))
        except ValueError:
            # Bail on first non-numeric chunk; everything after is
            # alphabetic (release suffix) and gets compared as string.
            return (0, p.name)
    # Pre-release suffix sorts BEFORE the release of the same core
    # (e.g. v0.0.1-rc1 < v0.0.1). Empty suffix is "release" and gets
    # the high sentinel.
    suffix_key = suffix if suffix else "\uffff"
    return (1, tuple(nums), suffix_key)


_EXTRACTION_SENTINEL = ".extracted-ok"


def _fetched_dylib_candidates() -> list[Path]:
    """Return dev-cache dylibs found under ``~/.cache/octomil-runtime``,
    ordered most-preferred first (newest version, preferred flavor).

    **Flavor selection:**

    * If ``OCTOMIL_RUNTIME_FLAVOR`` is set, only that flavor's candidates
      are returned.  An unrecognised value raises ``ImportError`` immediately
      — there is no silent fallback.
    * When unset, flavors are ordered by :data:`_FLAVOR_PREFERENCE`
      (``chat`` before ``stt``).  Unknown flavor names sort after all known
      ones.

    **Version ordering:** newest version first (``_version_sort_key``
    reversed).  Combining both axes gives a list ordered as:

        ``[v0.1.5-chat, v0.1.5-stt, v0.1.4-chat, v0.1.4-stt, ...]``

    Callers iterate *forward* — the first candidate that exists and loads
    wins (no ``reversed()`` needed).

    **Two cache layouts are supported:**

    *New (flavor-keyed, written by fetch_runtime_dev.py post flavor-cache
    fix):*

        ``<version>/<flavor>/lib/liboctomil-runtime.*``
        ``<version>/<flavor>/lib/.extracted-ok``

    *Legacy (flavor-blind, written by pre-fix fetch_runtime_dev.py):*

        ``<version>/lib/liboctomil-runtime.*``
        ``<version>/lib/.extracted-ok``

    Legacy slices are treated as ``flavor="chat"`` for ordering purposes.

    **Sentinel requirement:** the loader MUST honor the sentinel written by
    the fetch script.  A crash mid-extraction can leave a truncated dylib on
    disk; without the sentinel check the SDK would happily load it.  Caches
    that lack a matching sentinel are silently skipped — re-run the fetch
    script to fix."""
    # --- Flavor env-var override ---
    env_flavor = os.environ.get(_ENV_FLAVOR)
    if env_flavor is not None:
        valid = set(_FLAVOR_PREFERENCE)
        if env_flavor not in valid:
            raise ImportError(
                f"{_ENV_FLAVOR}={env_flavor!r} is not a recognised flavor.\n"
                f"Valid values: {sorted(valid)}.\n"
                f"Unset the variable to use the default flavor preference."
            )

    def _flavor_sort_key(flavor_name: str) -> tuple:
        try:
            return (0, _FLAVOR_PREFERENCE.index(flavor_name))
        except ValueError:
            return (1, flavor_name)

    if not _FETCH_CACHE_ROOT.is_dir():
        return []
    out: list[Path] = []
    # Sort newest-first so we can append in order without reversing later.
    for version_dir in sorted(_FETCH_CACHE_ROOT.iterdir(), key=_version_sort_key, reverse=True):
        if not version_dir.is_dir():
            continue

        # Collect (flavor_sort_key, Path) pairs for this version, then sort.
        version_candidates: list[tuple[tuple, Path]] = []

        # --- Legacy layout: <version>/lib/ ---
        # Detect by checking <version>/lib/.extracted-ok directly.
        # Treated as flavor="chat" for sorting purposes.
        legacy_lib = version_dir / "lib"
        legacy_sentinel = legacy_lib / _EXTRACTION_SENTINEL
        if legacy_sentinel.is_file():
            if env_flavor is None or env_flavor == "chat":
                for name in _RUNTIME_LIBNAMES:
                    candidate = legacy_lib / name
                    if candidate.is_file():
                        version_candidates.append((_flavor_sort_key("chat"), candidate))

        # --- New flavor-keyed layout: <version>/<flavor>/lib/ ---
        # Walk immediate subdirs that look like flavor names (non-hidden,
        # no "lib" or "include" to avoid treating legacy layout dirs as
        # flavor subdirs, and no "_download*" scratch dirs).
        for flavor_dir in version_dir.iterdir():
            if not flavor_dir.is_dir():
                continue
            if flavor_dir.name.startswith((".", "_")) or flavor_dir.name in ("lib", "include"):
                continue
            if env_flavor is not None and flavor_dir.name != env_flavor:
                continue
            flavor_lib = flavor_dir / "lib"
            flavor_sentinel = flavor_lib / _EXTRACTION_SENTINEL
            if not flavor_sentinel.is_file():
                continue
            for name in _RUNTIME_LIBNAMES:
                candidate = flavor_lib / name
                if candidate.is_file():
                    version_candidates.append((_flavor_sort_key(flavor_dir.name), candidate))

        # Sort within this version by flavor preference; most-preferred first.
        version_candidates.sort(key=lambda x: x[0])
        out.extend(p for _, p in version_candidates)

    return out


def _resolve_dylib() -> Path:
    """Find a usable dylib path or raise ImportError with a precise
    pointer to the documented setup paths.

    Resolution order:
      1. ``OCTOMIL_RUNTIME_DYLIB`` env var. Authoritative when set —
         if the path is missing we raise immediately (silent fallback
         would mask deployment bugs).
      2. Most-recently-fetched dev artifact under
         ``~/.cache/octomil-runtime/<version>/<flavor>/lib/``
         (new flavor-keyed layout) or
         ``~/.cache/octomil-runtime/<version>/lib/``
         (legacy flavor-blind layout). Populated by
         ``scripts/fetch_runtime_dev.py``.

    Flavor selection within the cache is governed by
    ``OCTOMIL_RUNTIME_FLAVOR`` (explicit) or :data:`_FLAVOR_PREFERENCE`
    (default: chat first).  See :func:`_fetched_dylib_candidates`.

    There is NO in-tree source-build fallback. The runtime source
    lives in the private ``octomil-runtime`` repo. SDK builds consume
    it via the binary release artifact."""
    override = os.environ.get(ENV_DYLIB_OVERRIDE)
    if override:
        override_path = Path(override)
        if override_path.is_file():
            return override_path
        raise ImportError(
            f"{ENV_DYLIB_OVERRIDE} points at {override!r} which does not exist.\n"
            f"Operator override is authoritative — fix the path or unset the\n"
            f"env var to use the dev-cache fallback.\n"
            f"For local dev, run: python scripts/fetch_runtime_dev.py"
        )
    # `_fetched_dylib_candidates()` returns candidates in preference order:
    # newest version first, most-preferred flavor (chat) first within a
    # version.  Iterate forward — first candidate that exists wins.
    tried: list[str] = []
    for path in _fetched_dylib_candidates():
        if path.is_file():
            return path
        tried.append(str(path))
    raise ImportError(
        "Could not locate liboctomil-runtime.\n"
        "Operator override (preferred): set "
        f"{ENV_DYLIB_OVERRIDE}=/abs/path/to/liboctomil-runtime.dylib\n"
        "Local dev cache: run `python scripts/fetch_runtime_dev.py` to\n"
        "fetch the latest dev artifact from the private octomil-runtime\n"
        "repo's GitHub Releases. The runtime source is no longer in\n"
        "this repo; do not search for an in-tree build directory.\n"
        + ("Tried dev-cache paths:\n" + "\n".join(f"  - {t}" for t in tried) if tried else "")
    )


# ---------------------------------------------------------------------------
# cffi cdef + lib singleton
# ---------------------------------------------------------------------------

# Mirrors runtime.h. ANY drift caught by the size-parity tests in
# tests/test_runtime_native_loader.py — every struct in the cdef has
# a corresponding `oct_<struct>_size()` helper that the runtime
# returns; the test asserts cdef sizeof matches dylib sizeof.
_CDEF: str = """
typedef uint32_t oct_status_t;
typedef uint32_t oct_priority_t;
typedef uint32_t oct_event_type_t;
typedef uint32_t oct_error_code_t;  /* v0.4 step 2 — used inside oct_event_t.data.error.error_code */

typedef struct oct_runtime oct_runtime_t;
typedef struct oct_session oct_session_t;
/* v0.1.1: forward-decl so oct_session_config_t.model can name the
 * type. Full struct is opaque to bindings; the runtime owns layout. */
typedef struct oct_model   oct_model_t;

typedef struct {
    uint32_t      version;
    size_t        size;
    const char**  supported_engines;
    const char**  supported_capabilities;
    const char**  supported_archs;
    uint64_t      ram_total_bytes;
    uint64_t      ram_available_bytes;
    uint8_t       has_apple_silicon;
    uint8_t       has_cuda;
    uint8_t       has_metal;
    uint8_t       _reserved0;
} oct_capabilities_t;

/* Slice 2A: full event-envelope cdef. Mirrors runtime.h verbatim;
 * if the C compiler reorders or pads any inner field differently
 * than cffi computes, the size-parity test in test_runtime_native_loader
 * fails immediately. */
typedef struct oct_event {
    uint32_t           version;
    size_t             size;
    oct_event_type_t   type;
    uint64_t           monotonic_ns;
    void*              user_data;
    union {
        struct {
            const uint8_t* pcm;
            uint32_t       n_bytes;
            uint32_t       sample_rate;
            uint32_t       sample_format;
            uint16_t       channels;
            uint8_t        is_final;
            uint8_t        _reserved0;
        } audio_chunk;
        struct {
            const char* utf8;
            uint32_t    n_bytes;
        } transcript_chunk;
        struct {
            const char* code;
            const char* message;
            /* v0.4 step 2 — APPENDED inside the existing inner struct
             * after slice-2A's two strings. */
            oct_error_code_t error_code;
            uint32_t         _reserved0;
        } error;
        struct {
            const char* engine;
            const char* model_digest;
            const char* locality;
            const char* streaming_mode;
            const char* runtime_build_tag;
        } session_started;
        struct {
            float        setup_ms;
            float        engine_first_chunk_ms;
            float        e2e_first_chunk_ms;
            float        total_latency_ms;
            float        queued_ms;
            uint32_t     observed_chunks;
            uint8_t      capability_verified;
            uint8_t      _reserved0;
            uint16_t     _reserved1;
            oct_status_t terminal_status;
        } session_completed;
        struct {
            uint32_t     n_frames_dropped;
            uint32_t     sample_rate;
            uint16_t     channels;
            uint16_t     _reserved0;
            const char*  reason;
            uint64_t     dropped_at_ns;
        } input_dropped;

        /* v0.4 step 2 — runtime-scope event payloads (appended). */
        struct {
            const char* engine;
            const char* model_id;
            const char* artifact_digest;
            uint64_t    load_ms;
            uint64_t    warm_ms;
            const char* policy_preset;
            void*       config_user_data;
            const char* source;
        } model_loaded;
        struct {
            const char* engine;
            const char* model_id;
            const char* artifact_digest;
            uint64_t    freed_bytes;
            const char* reason;
            void*       config_user_data;
        } model_evicted;
        struct {
            const char* layer;
            uint32_t    saved_tokens;
            uint32_t    _reserved0;
        } cache;
        struct {
            uint32_t    queue_position;
            uint32_t    queue_depth;
        } queued;
        struct {
            uint32_t    preempted_by_priority;
            uint32_t    _reserved0;
            const char* reason;
        } preempted;
        struct {
            uint64_t    ram_available_bytes;
            uint8_t     severity;
            uint8_t     _reserved0;
            uint16_t    _reserved1;
            uint32_t    _reserved2;
        } memory_pressure;
        struct {
            uint8_t     state;
            uint8_t     _reserved0;
            uint16_t    _reserved1;
            uint32_t    _reserved2;
        } thermal_state;
        struct {
            uint32_t    timeout_ms;
            uint32_t    _reserved0;
            const char* phase;
        } watchdog_timeout;
        struct {
            const char* name;
            double      value;
        } metric;
        /* v0.1.3 — embeddings.text pooled vector payload. Emitted
         * once per input string in input order. `values` lifetime
         * = until next poll on this session. Bindings copy before
         * the next poll. */
        struct {
            const float* values;
            uint32_t     n_dim;
            uint32_t     n_input_tokens;
            uint32_t     index;
            uint32_t     pooling_type;
            uint8_t      is_normalized;
            uint8_t      _reserved0;
            uint16_t     _reserved1;
        } embedding_vector;

        /* v0.1.5 — VAD transition (capability not advertised in PR-2B
         * SDK consumption path; declared so the union sizeof matches
         * the dylib's). */
        struct {
            uint32_t    transition_kind;
            uint32_t    timestamp_ms;
            float       confidence;
            uint32_t    _reserved0;
        } vad_transition;

        /* v0.1.5 PR-2 — STT transcript segment. utf8 lifetime = until
         * next poll. The SDK copies the bytes out at poll time before
         * returning to the caller. */
        struct {
            const char* utf8;
            uint32_t    n_bytes;
            uint32_t    start_ms;
            uint32_t    end_ms;
            uint32_t    segment_index;
            uint8_t     is_final;
            uint8_t     _reserved0;
            uint16_t    _reserved1;
        } transcript_segment;

        /* v0.1.5 PR-2 — STT end-of-transcript. Followed by
         * SESSION_COMPLETED(OK). utf8 lifetime contract identical to
         * transcript_segment. */
        struct {
            const char* utf8;
            uint32_t    n_bytes;
            uint32_t    n_segments;
            uint32_t    duration_ms;
            uint32_t    _reserved0;
            uint32_t    _reserved1;
        } transcript_final;

        /* audio.diarization segment. */
        struct {
            uint32_t    start_ms;
            uint32_t    end_ms;
            uint16_t    speaker_id;
            uint16_t    _reserved0;
            uint32_t    _reserved1;
            const char* speaker_label;
        } diarization_segment;

        /* TTS audio chunk. Emitted by audio.tts.batch and
         * audio.tts.stream when the native TTS gate advertises. */
        struct {
            const uint8_t* pcm;
            uint32_t       n_bytes;
            uint32_t       sample_rate;
            uint32_t       sample_format;
            uint16_t       channels;
            uint8_t        is_final;
            uint8_t        _reserved0;
        } tts_audio_chunk;
    } data;

    /* ──────── v0.4 step 2 — operational envelope APPENDED ──────── */
    const char*        request_id;
    const char*        route_id;
    const char*        trace_id;
    const char*        engine_version;
    const char*        adapter_version;
    const char*        accelerator;
    const char*        artifact_digest;
    uint8_t            cache_was_hit;
    uint8_t            _reserved0;
    uint16_t           _reserved1;
    uint32_t           _reserved2;
} oct_event_t;

typedef void (*oct_telemetry_sink_fn)(const oct_event_t* event, void* user_data);

typedef struct {
    uint32_t              version;
    const char*           artifact_root;
    oct_telemetry_sink_fn telemetry_sink;
    void*                 telemetry_user_data;
    uint32_t              max_sessions;
} oct_runtime_config_t;

typedef struct {
    uint32_t       version;
    const char*    model_uri;
    const char*    capability;
    const char*    locality;
    const char*    policy_preset;
    const char*    speaker_id;
    uint32_t       sample_rate_in;
    uint32_t       sample_rate_out;
    oct_priority_t priority;
    void*          user_data;
    /* v0.4 step 2 — appended. NULL ok; runtime echoes "" on events. */
    const char*    request_id;
    const char*    route_id;
    const char*    trace_id;
    const char*    kv_prefix_key;
    /* v0.1.1 (session_config v=3) — pre-warmed model handle from
     * oct_model_open + oct_model_warm. chat.completion REQUIRES
     * non-NULL model on v=3; runtime returns INVALID_INPUT
     * otherwise. Other capabilities still resolve via model_uri.
     * Caller retains ownership; the binding MUST keep the model
     * alive until the session has been closed. */
    oct_model_t*   model;
} oct_session_config_t;

typedef struct {
    const float* samples;
    uint32_t     n_frames;
    uint32_t     sample_rate;
    uint16_t     channels;
    uint16_t     _reserved0;
} oct_audio_view_t;

uint32_t oct_runtime_abi_version_major(void);
uint32_t oct_runtime_abi_version_minor(void);
uint32_t oct_runtime_abi_version_patch(void);
uint64_t oct_runtime_abi_version_packed(void);

oct_status_t oct_runtime_open(
    const oct_runtime_config_t* config,
    oct_runtime_t** out
);
void oct_runtime_close(oct_runtime_t* runtime);

oct_status_t oct_runtime_capabilities(
    oct_runtime_t* runtime,
    oct_capabilities_t* out
);
void oct_runtime_capabilities_free(oct_capabilities_t* caps);

int oct_runtime_last_error(
    oct_runtime_t* runtime,
    char* buf,
    size_t buflen
);
int oct_last_thread_error(
    char* buf,
    size_t buflen
);

/* Session lifecycle entry points. Live capabilities are adapter-backed;
 * blocked or unavailable capabilities return OCT_STATUS_UNSUPPORTED. */
oct_status_t oct_session_open(
    oct_runtime_t* runtime,
    const oct_session_config_t* config,
    oct_session_t** out
);
void oct_session_close(oct_session_t* session);

oct_status_t oct_session_send_audio(
    oct_session_t* session,
    const oct_audio_view_t* audio
);
oct_status_t oct_session_send_text(
    oct_session_t* session,
    const char* utf8
);
oct_status_t oct_session_poll_event(
    oct_session_t* session,
    oct_event_t* out,
    uint32_t timeout_ms
);
oct_status_t oct_session_cancel(oct_session_t* session);

/* ABI-parity introspection — sizeof as computed by the dylib's C
 * compiler. The cffi cdef declarations above MUST round-trip
 * through these getters; drift fails the size-parity test. */
size_t oct_runtime_config_size(void);
size_t oct_capabilities_size(void);
size_t oct_session_config_size(void);
size_t oct_audio_view_size(void);
size_t oct_event_size(void);

/* v0.1.1 — model lifecycle (oct_model_t forward-declared earlier).
 * Real implementation; no longer stubs. Engine adapters
 * (Slice 2C and following) extend the engine_hint matrix. */
typedef uint32_t oct_accelerator_pref_t;

typedef struct {
    uint32_t              version;
    const char*           model_uri;
    const char*           artifact_digest;
    const char*           engine_hint;
    const char*           policy_preset;
    uint32_t              accelerator_pref;
    uint64_t              ram_budget_bytes;
    void*                 user_data;
} oct_model_config_t;

oct_status_t oct_model_open(
    oct_runtime_t* runtime,
    const oct_model_config_t* config,
    oct_model_t** out_model
);
oct_status_t oct_model_warm(oct_model_t* model);
oct_status_t oct_model_evict(oct_model_t* model);
/* v0.1.1: return type changed void→oct_status_t. Returns BUSY when
 * sessions still borrow the model (handle remains valid; binding
 * retries after closing sessions); INVALID_INPUT on NULL or already-
 * closed handle; OK on successful free. */
oct_status_t oct_model_close(oct_model_t* model);
size_t       oct_model_config_size(void);

/* v0.1.11 Lane G — cache clear/introspection ABI skeleton.
 * All four entry points return OCT_STATUS_UNSUPPORTED until Lanes
 * B/C/F wire real caches. Added to cdef so bindings can call them
 * and conformance tests can exercise the stub contracts.
 *
 * Feature probe: check for "cache.introspect" in
 * oct_runtime_capabilities().supported_capabilities before calling.
 * ABI minor is NOT bumped — these are probe-gated additive symbols.
 *
 * CANONICAL: capability "cache.introspect" registered in
 * octomil-contracts v1.25.0 (#129); cache scope codes match
 * enums/cache_scope.yaml from v1.24.0 (#123). */
typedef uint32_t oct_cache_scope_t;

oct_status_t oct_runtime_cache_clear_all(
    oct_runtime_t* runtime
);
oct_status_t oct_runtime_cache_clear_capability(
    oct_runtime_t* runtime,
    const char*    capability_id
);
oct_status_t oct_runtime_cache_clear_scope(
    oct_runtime_t*    runtime,
    oct_cache_scope_t scope_id
);
oct_status_t oct_runtime_cache_introspect(
    oct_runtime_t* runtime,
    char*          out_json_buf,
    size_t         buf_len
);

/* v0.1.12 / ABI minor 11 — embeddings.image input surface (STUB on
 * the runtime side until the SigLIP-base int8 adapter PR lands).
 *
 * Symbol presence is gated by the dylib's reported minor; capability
 * usage is gated by ``oct_runtime_capabilities`` advertising
 * "embeddings.image". This binding cdef's the symbols unconditionally
 * so the build is stable across both minor=10 and minor=11 dylibs;
 * the lib singleton resolves the actual function pointers
 * conditionally (see `_resolve_image_symbols` below) and any caller
 * is double-gated: capability advertised AND symbol resolved.
 *
 * Per docs/runtime/embeddings-image-abi-scope.md (#85). */
typedef struct {
    const uint8_t* bytes;
    size_t         n_bytes;
    uint32_t       mime;
    uint32_t       _reserved0;
} oct_image_view_t;

size_t oct_image_view_size(void);

oct_status_t oct_session_send_image(
    oct_session_t*          session,
    const oct_image_view_t* view
);
"""


# v0.4 — minimum ABI version this binding requires. The cdef calls
# v0.4 symbols (oct_model_open, oct_model_config_size, etc.); loading
# a v0.3 dylib would fail later with a missing-symbol cffi error
# rather than a typed compatibility error. Codex R1 fix: fail fast
# at load time with NativeRuntimeError(VERSION_MISMATCH).
_REQUIRED_ABI_MAJOR: int = 0
_REQUIRED_ABI_MINOR: int = 10  # v0.1.11: cache introspection ABI symbols and native audio event parity.


def _build_lib() -> tuple[Any, Any]:
    """Construct the cffi (FFI, lib) pair. Imported lazily so test
    discovery doesn't fail when cffi or the dylib is unavailable.

    Codex R1 fix: enforces (major == _REQUIRED_ABI_MAJOR && minor >=
    _REQUIRED_ABI_MINOR) at load time so a v0.4 binding loading a
    v0.3 dylib raises a typed NativeRuntimeError IMMEDIATELY rather
    than failing later on the first call to a v0.4-only symbol."""
    try:
        from cffi import FFI  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "cffi is required for octomil.runtime.native. "
            "Install via `pip install octomil[native]` (slice 3 PR1 extra) "
            "or `pip install octomil[test]` (development)."
        ) from exc
    ffi = FFI()
    ffi.cdef(_CDEF)
    dylib = _resolve_dylib()
    lib = ffi.dlopen(str(dylib))
    # ABI compatibility check — fail fast on dylib/binding skew.
    dylib_major = int(lib.oct_runtime_abi_version_major())
    dylib_minor = int(lib.oct_runtime_abi_version_minor())
    if (dylib_major, dylib_minor) < (_REQUIRED_ABI_MAJOR, _REQUIRED_ABI_MINOR):
        raise NativeRuntimeError(
            OCT_STATUS_VERSION_MISMATCH,
            f"liboctomil-runtime ABI version {dylib_major}.{dylib_minor} loaded "
            f"from {dylib} is older than the {_REQUIRED_ABI_MAJOR}."
            f"{_REQUIRED_ABI_MINOR} this binding requires. Rebuild the dylib "
            "from a matching octomil-python checkout, or downgrade the binding.",
        )
    if dylib_major != _REQUIRED_ABI_MAJOR:
        raise NativeRuntimeError(
            OCT_STATUS_VERSION_MISMATCH,
            f"liboctomil-runtime ABI MAJOR {dylib_major} is incompatible with "
            f"binding MAJOR {_REQUIRED_ABI_MAJOR}. Major bumps require side-by-"
            "side dylibs and a binding rebuild.",
        )
    logger.debug(
        "loaded liboctomil-runtime from %s (ABI %d.%d, binding requires >= %d.%d)",
        dylib,
        dylib_major,
        dylib_minor,
        _REQUIRED_ABI_MAJOR,
        _REQUIRED_ABI_MINOR,
    )
    # v0.1.12 / ABI minor 11 — optional image-input symbols.
    # _REQUIRED_ABI_MINOR stays at 10; these symbols are bound only
    # when the dylib advertises minor >= 11. On older minor=10
    # runtimes the resolved entry is None and the binding NEVER raises
    # during load — callers double-gate via capability advertisement
    # before invoking. Symbol-presence drives whether the C-side
    # symbol table even contains the export; capability advertisement
    # drives whether the runtime will accept the call (the runtime
    # itself stubs the call to OCT_STATUS_UNSUPPORTED until the
    # SigLIP-base int8 adapter lands and removes embeddings.image
    # from the BLOCKED_WITH_PROOF set).
    _resolve_optional_image_symbols(lib, dylib_minor)
    return ffi, lib  # noqa: RET504 — explicit ffi/lib pair documents both


# v0.1.12 / ABI minor 11 — sidecar storing the resolved optional
# image-input function pointers (or None when the loaded dylib's
# minor is < 11). Indexed by the cdef'd symbol name. Populated by
# ``_resolve_optional_image_symbols`` once at load time.
#
# This is deliberately a plain dict instead of monkey-patching
# attributes onto the cffi ``lib`` object: cffi raises AttributeError
# on attribute access when a cdef'd symbol is absent from the
# loaded dylib, so a sentinel store decouples the binding's
# probe path from cffi's lazy resolution failure mode.
_OPTIONAL_IMAGE_SYMBOLS: dict[str, Any] = {
    "oct_image_view_size": None,
    "oct_session_send_image": None,
}


def _resolve_optional_image_symbols(lib: Any, dylib_minor: int) -> None:
    """Conditionally resolve the ABI minor 11 image-input symbols.

    NEVER raises on a minor=10 dylib — image-input is OPT-IN and the
    required ABI floor stays at 10. When the runtime advertises
    minor >= 11 but the symbol is unexpectedly missing (corrupt
    build, partial relink) the slot stays None and the capability
    gate at call time surfaces the same UNSUPPORTED error path.
    """
    # Reset to None on each load — _build_lib may run multiple times
    # in tests that reset the singleton.
    for name in _OPTIONAL_IMAGE_SYMBOLS:
        _OPTIONAL_IMAGE_SYMBOLS[name] = None
    if dylib_minor < 11:
        logger.debug(
            "skipping ABI minor 11 image-input symbol bind: dylib advertises minor=%d (< 11)",
            dylib_minor,
        )
        return
    for name in _OPTIONAL_IMAGE_SYMBOLS:
        try:
            _OPTIONAL_IMAGE_SYMBOLS[name] = getattr(lib, name)
        except AttributeError:
            # Symbol cdef'd by the binding but missing from the
            # loaded dylib's export table. Leave as None; capability
            # gate at call time will reject UNSUPPORTED.
            logger.debug(
                "ABI minor 11 image-input symbol %s missing from dylib despite minor=%d",
                name,
                dylib_minor,
            )


_FFI: Optional[Any] = None
_LIB: Optional[Any] = None


def _get_lib() -> tuple[Any, Any]:
    """Singleton accessor. Builds the FFI once per process."""
    global _FFI, _LIB
    if _FFI is None or _LIB is None:
        _FFI, _LIB = _build_lib()
    return _FFI, _LIB


# ---------------------------------------------------------------------------
# Public API — version handshake + last-error
# ---------------------------------------------------------------------------


def abi_version() -> tuple[int, int, int]:
    """Return ``(major, minor, patch)`` of the loaded dylib's ABI.

    Bindings call this immediately after loading to verify compat.
    A pinned major version + ``minor >= compiled-against`` is the
    lockstep upgrade rule."""
    _, lib = _get_lib()
    return (
        int(lib.oct_runtime_abi_version_major()),
        int(lib.oct_runtime_abi_version_minor()),
        int(lib.oct_runtime_abi_version_patch()),
    )


def last_thread_error(buflen: int = 512) -> str:
    """Read the thread-scoped last-error buffer. Used after a failed
    ``oct_runtime_open`` (no runtime handle available yet)."""
    ffi, lib = _get_lib()
    buf = ffi.new(f"char[{buflen}]")
    n = lib.oct_last_thread_error(buf, buflen)
    if n <= 0:
        return ""
    return ffi.string(buf, n).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# NativeRuntime — handle wrapper
# ---------------------------------------------------------------------------


class NativeRuntime:
    """RAII-style wrapper over ``oct_runtime_t``.

    Construct with ``NativeRuntime.open(...)`` (factory) or use as a
    context manager. Always closes the handle even if a downstream
    call raises.
    """

    def __init__(self, ffi: Any, lib: Any, handle: Any) -> None:
        # Internal: callers should use NativeRuntime.open() instead.
        self._ffi = ffi
        self._lib = lib
        self._handle = handle  # cffi `oct_runtime_t*`
        self._closed = False
        # Track live child sessions via weakref so NativeRuntime.close()
        # can mark them closed before the dylib's `oct_runtime_close`
        # implicitly tears them down. Without this, a Python wrapper
        # would call into a freed `oct_session_t*` on its next
        # send_audio / poll_event / __del__.
        # Codex R1 blocker fix.
        import weakref

        self._sessions: "weakref.WeakSet[NativeSession]" = weakref.WeakSet()
        # v0.4 step 1: same WeakSet pattern for NativeModel children.
        self._models: "weakref.WeakSet[NativeModel]" = weakref.WeakSet()

    @classmethod
    def open(
        cls,
        *,
        artifact_root: str = "",
        max_sessions: int = 0,
    ) -> "NativeRuntime":
        """Open a runtime with the v1 config shape (matches
        ``runtime.h``'s ``oct_runtime_config_t``).

        Codex R1 fix: previous signature used invented field names
        (``cache_root``, ``runtime_build_tag``); the actual config
        has ``artifact_root`` and ``max_sessions``. The kwargs-only
        signature lets future minor-bumps add fields painlessly.

        ``artifact_root`` is the directory where prepared artifacts
        live on disk; the runtime COPIES the string during open, so
        the caller can free its buffer immediately after.

        ``max_sessions`` is a hard cap (0 = unbounded).

        The telemetry sink + user-data are not exposed in this v0
        wrapper — set NULL. Slice 2-proper / 3 PR2 will add a
        Python-side telemetry callback path."""
        ffi, lib = _get_lib()
        cfg = ffi.new("oct_runtime_config_t*")
        cfg.version = 1
        # String needs to outlive the cffi call; the runtime copies
        # internally per the header's STRING LIFETIME contract, so
        # the local ref is fine.
        artifact_root_buf = ffi.new("char[]", artifact_root.encode("utf-8"))
        cfg.artifact_root = artifact_root_buf
        cfg.telemetry_sink = ffi.NULL
        cfg.telemetry_user_data = ffi.NULL
        cfg.max_sessions = max_sessions
        out = ffi.new("oct_runtime_t**")
        status = int(lib.oct_runtime_open(cfg, out))
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                "oct_runtime_open failed",
                last_error=last_thread_error(),
            )
        return cls(ffi, lib, out[0])

    def close(self) -> None:
        """Close the handle. Idempotent.

        v0.1.1: oct_runtime_close REFUSES (with last_error) when any
        model is still open. The binding therefore MUST drain
        sessions (decrement model.in_use) and close models BEFORE
        runtime_close. We do this in three ordered passes:

          1. oct_session_close every live session — drops the
             borrowed model refcount.
          2. oct_model_close every live model — frees engine state.
             A model that returns BUSY despite the session drain
             above is a binding bug; we surface it via last_error
             and continue (the runtime_close will refuse).
          3. oct_runtime_close — frees the runtime allocation IFF
             open_models is now empty.

        Slice-2A invalidation pattern preserved: every wrapper is
        marked invalid AFTER the C close call so any racing user
        code raises NativeRuntimeError instead of dereferencing
        freed memory.
        """
        if self._closed:
            return
        # 1. Close sessions first — drops their model in_use ref.
        for sess in list(self._sessions):
            try:
                sess.close()
            except Exception:  # noqa: BLE001 — best-effort drain
                pass
            sess._invalidate_after_runtime_close()  # noqa: SLF001
        self._sessions.clear()
        # 2. Close models. Status is observed best-effort: BUSY
        # means a session somehow survived the drain above (binding
        # bug); the runtime_close below will refuse.
        for mdl in list(self._models):
            try:
                mdl.close()
            except Exception:  # noqa: BLE001 — best-effort drain
                pass
            mdl._invalidate_after_runtime_close()  # noqa: SLF001
        self._models.clear()
        # 3. Close runtime. v0.1.1 docstring: void return preserved;
        # the runtime sets last_error if it refused (open_models
        # non-empty). The wrapper marks itself closed regardless —
        # if the C side leaked, the binding's last_error is the
        # diagnostic, not a hung wrapper.
        self._lib.oct_runtime_close(self._handle)
        self._closed = True

    def __enter__(self) -> "NativeRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort safety net; explicit close() / context manager
        # is the documented path.
        try:
            self.close()
        except Exception:
            pass

    def _check_open(self) -> None:
        if self._closed:
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                "runtime handle is closed",
            )

    def last_error(self, buflen: int = 512) -> str:
        """Read the runtime-scoped last-error buffer. Use this when
        the prior call had a runtime handle in scope; pre-open
        failures use :func:`last_thread_error`."""
        self._check_open()
        buf = self._ffi.new(f"char[{buflen}]")
        n = self._lib.oct_runtime_last_error(self._handle, buf, buflen)
        if n <= 0:
            return ""
        return self._ffi.string(buf, n).decode("utf-8", errors="replace")

    def capabilities(self) -> "RuntimeCapabilities":
        """Read the advertised capability set.

        Forward-compatible: capability strings the runtime advertises
        that are NOT in :data:`RUNTIME_CAPABILITIES` are silently
        dropped from the parsed view (with a DEBUG log). This matches
        the asymmetric reader rule from the contract — newer-runtime
        + older-SDK upgrades don't fail."""
        from octomil.runtime.native.capabilities import RUNTIME_CAPABILITIES

        self._check_open()
        ffi = self._ffi
        lib = self._lib
        caps = ffi.new("oct_capabilities_t*")
        caps.size = ffi.sizeof("oct_capabilities_t")
        status = int(lib.oct_runtime_capabilities(self._handle, caps))
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                "oct_runtime_capabilities failed",
                last_error=self.last_error(),
            )
        try:
            advertised = _read_string_array(ffi, caps.supported_capabilities)
            # Forward-compat: drop unknowns.
            known: list[str] = []
            unknown: list[str] = []
            for cap in advertised:
                if cap in RUNTIME_CAPABILITIES:
                    known.append(cap)
                else:
                    unknown.append(cap)
            if unknown:
                logger.debug(
                    "runtime advertised %d unknown capability string(s); dropping (forward-compat): %s",
                    len(unknown),
                    unknown,
                )
            return RuntimeCapabilities(
                version=int(caps.version),
                supported_engines=tuple(_read_string_array(ffi, caps.supported_engines)),
                supported_capabilities=tuple(known),
                supported_archs=tuple(_read_string_array(ffi, caps.supported_archs)),
                ram_total_bytes=int(caps.ram_total_bytes),
                ram_available_bytes=int(caps.ram_available_bytes),
                has_apple_silicon=bool(caps.has_apple_silicon),
                has_cuda=bool(caps.has_cuda),
                has_metal=bool(caps.has_metal),
            )
        finally:
            lib.oct_runtime_capabilities_free(caps)

    def open_session(
        self,
        *,
        capability: str,
        model_uri: str = "",
        locality: str = "on_device",
        policy_preset: str = "",
        speaker_id: str = "",
        sample_rate_in: int = 0,
        sample_rate_out: int = 0,
        priority: int = OCT_PRIORITY_FOREGROUND,
        # v0.4 step 2 — appended correlation IDs. Caller-owned strings
        # copied by the runtime at open. None ⇒ runtime echoes "" on
        # events. Length limits per runtime.h: kv_prefix_key ≤256 B;
        # request/route/trace_id ≤128 B each; ASCII printable, no
        # whitespace. Out-of-bounds returns OCT_STATUS_INVALID_INPUT.
        request_id: str | None = None,
        route_id: str | None = None,
        trace_id: str | None = None,
        kv_prefix_key: str | None = None,
        # v0.1.1 — pre-warmed model handle. chat.completion REQUIRES
        # non-NULL model on a v=3 session_config. Other capabilities
        # may pass model=None and resolve via model_uri (slice-2A).
        model: "NativeModel | None" = None,
    ) -> "NativeSession":
        """Open a session against this runtime.

        This is the canonical embedded-host entry point for realtime /
        TTS / STT / chat-style capabilities. The runtime applies
        strict-reject on ``capability``: any value not in
        :data:`octomil.runtime.native.capabilities.RUNTIME_CAPABILITIES`
        returns ``OCT_STATUS_UNSUPPORTED``. Canonical but blocked or
        unavailable capabilities also reject until a runtime advertises
        them via :meth:`capabilities`.

        ``capability`` MUST be one of the canonical strings
        (``audio.realtime.session``, ``audio.tts.stream`` etc.); the
        wrapper does NOT validate client-side because the runtime is
        the source of truth on what its build supports.

        ``locality`` is informational (only ``on_device`` is honored;
        anything else returns UNSUPPORTED). Cloud routing happens
        above this ABI.

        Strings are copied by the runtime during open; caller is free
        to drop the references on return.
        """
        self._check_open()
        ffi = self._ffi
        lib = self._lib
        cfg = ffi.new("oct_session_config_t*")
        cfg.version = OCT_SESSION_CONFIG_VERSION
        # Keep the encoded buffers alive across the FFI call. The
        # runtime copies internally per the header's STRING LIFETIME
        # contract; the local refs satisfy that.
        keepalive: list[Any] = []

        def _cstr(s: str) -> Any:
            buf = ffi.new("char[]", s.encode("utf-8"))
            keepalive.append(buf)
            return buf

        cfg.model_uri = _cstr(model_uri)
        cfg.capability = _cstr(capability)
        cfg.locality = _cstr(locality)
        cfg.policy_preset = _cstr(policy_preset)
        cfg.speaker_id = _cstr(speaker_id) if speaker_id else ffi.NULL
        cfg.sample_rate_in = sample_rate_in
        cfg.sample_rate_out = sample_rate_out
        cfg.priority = priority
        cfg.user_data = ffi.NULL
        # v0.4 step 2 — appended correlation IDs. Codex R1 fix:
        # validate the FULL contract from runtime.h (length AND
        # ASCII-printable, no whitespace, no control chars), not
        # just byte length. Shared validator for all four slots.
        _validate_correlation_id("request_id", request_id, max_bytes=128)
        _validate_correlation_id("route_id", route_id, max_bytes=128)
        _validate_correlation_id("trace_id", trace_id, max_bytes=128)
        _validate_correlation_id("kv_prefix_key", kv_prefix_key, max_bytes=256)
        cfg.request_id = _cstr(request_id) if request_id else ffi.NULL
        cfg.route_id = _cstr(route_id) if route_id else ffi.NULL
        cfg.trace_id = _cstr(trace_id) if trace_id else ffi.NULL
        cfg.kv_prefix_key = _cstr(kv_prefix_key) if kv_prefix_key else ffi.NULL
        # v0.1.1 — wire the warmed model handle. The model's lifetime
        # MUST extend past oct_session_close; we hold a strong
        # reference on the resulting NativeSession (see Codex R1 P1
        # fix in NativeSession.__init__) so a temporary-model
        # pattern like `open_session(model=open_model(...))` doesn't
        # GC the wrapper out from under the live session.
        if model is not None:
            if model._closed or model._handle_invalid:
                raise NativeRuntimeError(
                    OCT_STATUS_INVALID_INPUT,
                    "open_session: model handle is closed or invalidated",
                )
            # Codex R1 missed-case: cross-runtime model misuse.
            # `model` MUST have been opened against THIS runtime.
            # The runtime side may also reject mismatched parents,
            # but defense-in-depth at the binding layer surfaces a
            # precise typed error before crossing the FFI boundary.
            if model._owner is not self:
                raise NativeRuntimeError(
                    OCT_STATUS_INVALID_INPUT,
                    "open_session: model was opened on a different "
                    "NativeRuntime — cross-runtime model handles are "
                    "not supported",
                )
            cfg.model = model._handle
        else:
            cfg.model = ffi.NULL
        out = ffi.new("oct_session_t**")
        status = int(lib.oct_session_open(self._handle, cfg, out))
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                f"oct_session_open(capability={capability!r}) failed",
                last_error=self.last_error(),
            )
        sess = NativeSession(ffi, lib, out[0], owner=self, borrowed_model=model)
        # Register so NativeRuntime.close() can pre-invalidate the
        # wrapper before the dylib implicitly closes its handle.
        self._sessions.add(sess)
        return sess

    def open_model(
        self,
        *,
        model_uri: str,
        artifact_digest: str = "",
        engine_hint: str = "",
        policy_preset: str = "",
        accelerator_pref: int = OCT_ACCEL_AUTO,
        ram_budget_bytes: int = 0,
    ) -> "NativeModel":
        """v0.4 step 1: open a warm model handle.

        Slice-2A invariants preserved: caller-owned strings are
        copied at open per the STRING LIFETIME rule. Stub returns
        OCT_STATUS_UNSUPPORTED until engine adapters land (Slice 2C
        Moshi/MLX, then llama.cpp / sherpa-onnx / whisper.cpp / ONNX).

        Layer-2b invariants preserved: `model_uri` MUST be a local
        URI / digest. The runtime does NOT resolve `@app/...` refs.
        """
        self._check_open()
        ffi = self._ffi
        lib = self._lib
        cfg = ffi.new("oct_model_config_t*")
        cfg.version = OCT_MODEL_CONFIG_VERSION
        keepalive: list[Any] = []

        def _cstr(s: str) -> Any:
            buf = ffi.new("char[]", s.encode("utf-8"))
            keepalive.append(buf)
            return buf

        cfg.model_uri = _cstr(model_uri)
        cfg.artifact_digest = _cstr(artifact_digest) if artifact_digest else ffi.NULL
        cfg.engine_hint = _cstr(engine_hint) if engine_hint else ffi.NULL
        cfg.policy_preset = _cstr(policy_preset) if policy_preset else ffi.NULL
        cfg.accelerator_pref = accelerator_pref
        cfg.ram_budget_bytes = ram_budget_bytes
        cfg.user_data = ffi.NULL
        out = ffi.new("oct_model_t**")
        status = int(lib.oct_model_open(self._handle, cfg, out))
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                f"oct_model_open(model_uri={model_uri!r}) failed",
                last_error=self.last_error(),
            )
        mdl = NativeModel(ffi, lib, out[0], owner=self)
        self._models.add(mdl)
        return mdl


def _read_string_array(ffi: Any, ptr: Any) -> list[str]:
    """Walk a NULL-terminated ``const char**`` from the runtime.

    Per the contract, an empty list is a non-NULL pointer to a
    single-element array whose only element is NULL. A NULL outer
    pointer is treated as ``OCT_STATUS_INTERNAL`` per the header;
    we surface that as an empty list rather than crashing the
    binding (defense in depth)."""
    if ptr == ffi.NULL:
        return []
    result: list[str] = []
    i = 0
    while True:
        entry = ptr[i]
        if entry == ffi.NULL:
            break
        result.append(ffi.string(entry).decode("utf-8", errors="replace"))
        i += 1
    return result


# ---------------------------------------------------------------------------
# RuntimeCapabilities — parsed view of oct_capabilities_t
# ---------------------------------------------------------------------------


class RuntimeCapabilities:
    """Parsed, owned snapshot of the runtime's capability descriptor.

    All fields are immutable and own their data — the underlying
    ``oct_capabilities_t`` has been freed by the time this object
    is returned to the caller.
    """

    __slots__ = (
        "version",
        "supported_engines",
        "supported_capabilities",
        "supported_archs",
        "ram_total_bytes",
        "ram_available_bytes",
        "has_apple_silicon",
        "has_cuda",
        "has_metal",
    )

    def __init__(
        self,
        *,
        version: int,
        supported_engines: tuple[str, ...],
        supported_capabilities: tuple[str, ...],
        supported_archs: tuple[str, ...],
        ram_total_bytes: int,
        ram_available_bytes: int,
        has_apple_silicon: bool,
        has_cuda: bool,
        has_metal: bool,
    ) -> None:
        self.version = version
        self.supported_engines = supported_engines
        self.supported_capabilities = supported_capabilities
        self.supported_archs = supported_archs
        self.ram_total_bytes = ram_total_bytes
        self.ram_available_bytes = ram_available_bytes
        self.has_apple_silicon = has_apple_silicon
        self.has_cuda = has_cuda
        self.has_metal = has_metal

    def claims(self, capability: str) -> bool:
        """Convenience: True iff the runtime advertises ``capability``.
        The slice-3 PR2 conformance harness's
        ``requires_capability(...)`` marker uses this."""
        return capability in self.supported_capabilities

    def __repr__(self) -> str:
        return (
            f"RuntimeCapabilities(version={self.version}, "
            f"capabilities={self.supported_capabilities!r}, "
            f"engines={self.supported_engines!r}, "
            f"archs={self.supported_archs!r})"
        )


# ---------------------------------------------------------------------------
# NativeSession
# ---------------------------------------------------------------------------


class NativeEvent:
    """Parsed snapshot of an ``oct_event_t``.

    The cffi event buffer is reused across :meth:`NativeSession.poll_event`
    calls (the runtime's lifetime contract makes the inner pointer
    fields valid only until the next poll). This view extracts the
    primitive fields the binding cares about into Python-owned data
    so callers can hold onto it across polls.

    Event payloads are populated only by live advertised capabilities;
    blocked capabilities reject before producing synthetic events."""

    __slots__ = (
        "type",
        "version",
        "monotonic_ns",
        "user_data_ptr",
        # v0.4 step 2 — operational envelope (always-non-NULL strings;
        # runtime echoes "" when the source slot was NULL).
        "request_id",
        "route_id",
        "trace_id",
        "engine_version",
        "adapter_version",
        "accelerator",
        "artifact_digest",
        "cache_was_hit",
        # Cutover: TRANSCRIPT_CHUNK text payload, copied out of the
        # cffi event buffer at poll time (the runtime owns the
        # storage only until the next poll). Empty for non-chunk
        # event types.
        "text",
        # Cutover R1 (Codex): SESSION_COMPLETED.terminal_status
        # (oct_status_t) and OCT_EVENT_ERROR.error_code
        # (oct_error_code_t) — copied from the inner-payload union
        # at poll time so the SDK binding can map runtime-side
        # rejects to bounded OctomilError codes precisely
        # (UNSUPPORTED stays UNSUPPORTED, not flattened to
        # INVALID_INPUT). Zero for event types that don't carry
        # these fields.
        "terminal_status",
        "error_code",
        # Cutover follow-up #73: cache + session-completed telemetry.
        # The runtime emits CACHE_HIT / CACHE_MISS events with
        # cache.layer + cache.saved_tokens, and SESSION_COMPLETED with
        # setup_ms / engine_first_chunk_ms / e2e_first_chunk_ms /
        # total_latency_ms / queued_ms. Expose them so the SDK's
        # InferenceMetrics + get_verbose_metadata can surface real
        # telemetry instead of just chunk count + ttfc.
        "cache_layer",
        "cache_saved_tokens",
        "setup_ms",
        "engine_first_chunk_ms",
        "e2e_first_chunk_ms",
        "total_latency_ms",
        "queued_ms",
        # v0.1.3 — embeddings.text payload. Copied out of the cffi
        # buffer at poll time (lifetime contract: runtime owns the
        # float buffer only until the next poll). The list itself
        # is Python-owned and survives across polls. n_dim is the
        # vector dimension (model's pooled output size); index is
        # the input position in the batch; pooling_type maps to
        # OCT_EMBED_POOLING_*; is_normalized is 1 iff the runtime
        # already L2-normalized.
        "values",
        "n_dim",
        "n_input_tokens",
        "index",
        "pooling_type",
        "is_normalized",
        # v0.1.5 PR-2 — STT events. transcript-segment fields are
        # populated only on OCT_EVENT_TRANSCRIPT_SEGMENT; transcript-
        # final fields only on OCT_EVENT_TRANSCRIPT_FINAL. `text` is
        # the segment / final UTF-8 (already copied out of the
        # runtime-owned buffer at poll time so the caller can hold it
        # across subsequent polls).
        "segment_start_ms",
        "segment_end_ms",
        "segment_index",
        "segment_is_final",
        "final_n_segments",
        "final_duration_ms",
        # v0.1.5 PR-2N — VAD transition payload. Populated only on
        # OCT_EVENT_VAD_TRANSITION. transition_kind is one of
        # OCT_VAD_TRANSITION_* constants; confidence is the silero
        # average per-window speech probability across the span,
        # clamped to [0.0, 1.0]; timestamp_ms is the runtime-monotonic
        # event time at the transition edge.
        "vad_transition_kind",
        "vad_timestamp_ms",
        "vad_confidence",
        # audio.diarization segment payload. Populated only on
        # OCT_EVENT_DIARIZATION_SEGMENT. speaker_label is copied out
        # of the runtime-owned string at poll time.
        "diarization_start_ms",
        "diarization_end_ms",
        "diarization_speaker_id",
        "diarization_speaker_label",
        # v0.1.8 Lane A — TTS audio chunk fields. Populated only on
        # OCT_EVENT_TTS_AUDIO_CHUNK. ``tts_pcm_bytes`` is a Python bytes
        # object copied out of the runtime-owned ``data.tts_audio_chunk.pcm``
        # buffer at poll time (lifetime contract: runtime owns the buffer
        # only until the next poll on this session). Other fields are
        # primitive copies.
        "tts_pcm_bytes",
        "tts_sample_rate",
        "tts_sample_format",
        "tts_channels",
        "tts_is_final",
    )

    def __init__(
        self,
        *,
        type: int,
        version: int,
        monotonic_ns: int,
        user_data_ptr: int,
        request_id: str = "",
        route_id: str = "",
        trace_id: str = "",
        engine_version: str = "",
        adapter_version: str = "",
        accelerator: str = "",
        artifact_digest: str = "",
        cache_was_hit: bool = False,
        text: str = "",
        terminal_status: int = 0,
        error_code: int = 0,
        cache_layer: str = "",
        cache_saved_tokens: int = 0,
        setup_ms: float = 0.0,
        engine_first_chunk_ms: float = 0.0,
        e2e_first_chunk_ms: float = 0.0,
        total_latency_ms: float = 0.0,
        queued_ms: float = 0.0,
        values: list[float] | None = None,
        n_dim: int = 0,
        n_input_tokens: int = 0,
        index: int = 0,
        pooling_type: int = 0,
        is_normalized: bool = False,
        segment_start_ms: int = 0,
        segment_end_ms: int = 0,
        segment_index: int = 0,
        segment_is_final: bool = False,
        final_n_segments: int = 0,
        final_duration_ms: int = 0,
        vad_transition_kind: int = 0,
        vad_timestamp_ms: int = 0,
        vad_confidence: float = 0.0,
        diarization_start_ms: int = 0,
        diarization_end_ms: int = 0,
        diarization_speaker_id: int = OCT_DIARIZATION_SPEAKER_UNKNOWN,
        diarization_speaker_label: str = "",
        tts_pcm_bytes: bytes = b"",
        tts_sample_rate: int = 0,
        tts_sample_format: int = 0,
        tts_channels: int = 0,
        tts_is_final: bool = False,
    ) -> None:
        self.type = type
        self.version = version
        self.monotonic_ns = monotonic_ns
        self.user_data_ptr = user_data_ptr
        self.request_id = request_id
        self.route_id = route_id
        self.trace_id = trace_id
        self.engine_version = engine_version
        self.adapter_version = adapter_version
        self.accelerator = accelerator
        self.artifact_digest = artifact_digest
        self.cache_was_hit = cache_was_hit
        self.text = text
        self.terminal_status = terminal_status
        self.error_code = error_code
        self.cache_layer = cache_layer
        self.cache_saved_tokens = cache_saved_tokens
        self.setup_ms = setup_ms
        self.engine_first_chunk_ms = engine_first_chunk_ms
        self.e2e_first_chunk_ms = e2e_first_chunk_ms
        self.total_latency_ms = total_latency_ms
        self.queued_ms = queued_ms
        self.values = values if values is not None else []
        self.n_dim = n_dim
        self.n_input_tokens = n_input_tokens
        self.index = index
        self.pooling_type = pooling_type
        self.is_normalized = is_normalized
        self.segment_start_ms = segment_start_ms
        self.segment_end_ms = segment_end_ms
        self.segment_index = segment_index
        self.segment_is_final = segment_is_final
        self.final_n_segments = final_n_segments
        self.final_duration_ms = final_duration_ms
        self.vad_transition_kind = vad_transition_kind
        self.vad_timestamp_ms = vad_timestamp_ms
        self.vad_confidence = vad_confidence
        self.diarization_start_ms = diarization_start_ms
        self.diarization_end_ms = diarization_end_ms
        self.diarization_speaker_id = diarization_speaker_id
        self.diarization_speaker_label = diarization_speaker_label
        self.tts_pcm_bytes = tts_pcm_bytes
        self.tts_sample_rate = tts_sample_rate
        self.tts_sample_format = tts_sample_format
        self.tts_channels = tts_channels
        self.tts_is_final = tts_is_final

    @property
    def is_none(self) -> bool:
        return self.type == OCT_EVENT_NONE

    def __repr__(self) -> str:
        return f"NativeEvent(type={self.type}, monotonic_ns={self.monotonic_ns})"


class NativeSession:
    """RAII-style wrapper over ``oct_session_t``.

    Lifetime is bound to the parent :class:`NativeRuntime`; closing
    the runtime implicitly cancels and closes any live sessions, but
    bindings should call :meth:`close` explicitly for clean drain.

    Single-thread-affine: ``send_audio``, ``send_text``, ``poll_event``
    MUST NOT race against each other on the same session. ``cancel``
    is the only entry point safe from any thread.

    Blocked or unavailable capabilities return ``OCT_STATUS_UNSUPPORTED``
    (raised as :class:`NativeRuntimeError`). The wrapper exists so:
      * Conformance tests have a stable target.
      * Live conditional adapters can use the real ABI surface.
      * Capability-gated tests auto-skip via
        :meth:`NativeRuntime.capabilities` rather than crashing.
    """

    def __init__(
        self,
        ffi: Any,
        lib: Any,
        handle: Any,
        *,
        owner: NativeRuntime,
        borrowed_model: "NativeModel | None" = None,
    ) -> None:
        self._ffi = ffi
        self._lib = lib
        self._handle = handle
        self._owner = owner
        self._closed = False
        # Set by NativeRuntime.close() right before it calls
        # `oct_runtime_close()`. Once flipped, every entry point
        # raises rather than dereferencing the now-freed handle, and
        # session_close becomes a no-op (dylib already closed it).
        # Codex R1 blocker fix.
        self._handle_invalid = False
        # v0.1.1 Codex R1 P1 fix: hold a STRONG reference to the
        # borrowed NativeModel for the session's lifetime. Without
        # this, the user pattern `rt.open_session(model=rt.open_model(...))`
        # leaves the model wrapper unreferenced (NativeRuntime._models
        # is a WeakSet); GC could call NativeModel.__del__ →
        # close() → BUSY (runtime refuses; handle stays alive
        # C-side), but our wrapper is gone. Subsequent
        # NativeRuntime.close() then sees an empty WeakSet, calls
        # oct_runtime_close, which refuses-with-last-error because
        # the C-side open_models list is non-empty — leaking the
        # runtime allocation. The strong ref keeps the wrapper
        # alive through the session's lifetime so close() and the
        # WeakSet membership stay coherent.
        self._borrowed_model = borrowed_model
        # Reusable event buffer — runtime fills the same struct on every
        # poll; the inner pointer fields are valid only until the next
        # poll call per the header's lifetime contract.
        self._event_buf: Any = ffi.new("oct_event_t*")
        self._event_buf.size = ffi.sizeof("oct_event_t")

    def _check_open(self) -> None:
        if self._handle_invalid:
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                "session handle invalidated by parent NativeRuntime.close()",
            )
        if self._closed:
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                "session handle is closed",
            )

    def send_audio(
        self,
        samples: bytes,
        *,
        sample_rate: int,
        channels: int = 1,
    ) -> None:
        """Push interleaved float32 PCM to the session's input queue.

        ``samples`` is a Python ``bytes`` (or buffer-protocol object)
        containing interleaved float32 values; total float count is
        ``len(samples) // 4``, frames per channel is that count divided
        by ``channels``. Caller-owned; the runtime copies internally
        if it needs to retain.

        Returns on OCT_STATUS_OK; raises on any non-OK status —
        OCT_STATUS_BUSY surfaces as a NativeRuntimeError so the
        caller can decide drop-vs-retry, mirroring the C contract."""
        self._check_open()
        ffi = self._ffi
        # Reject malformed float32 buffers explicitly. Codex+Gemini R2
        # nit: `len(samples) // 4` would silently truncate a trailing
        # byte (e.g. len==5 → 1 float + dropped byte), letting a
        # misaligned buffer cross the FFI.
        if len(samples) % 4 != 0:
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                f"send_audio: buffer length {len(samples)} is not a multiple of float32 size (4)",
            )
        n_floats = len(samples) // 4
        if channels <= 0 or n_floats == 0 or n_floats % channels != 0:
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                f"send_audio: bad shape (len={len(samples)}, channels={channels})",
            )
        n_frames = n_floats // channels
        view = ffi.new("oct_audio_view_t*")
        # cffi: cast a bytes-like object directly to const float*.
        view.samples = ffi.cast("const float*", ffi.from_buffer(samples))
        view.n_frames = n_frames
        view.sample_rate = sample_rate
        view.channels = channels
        view._reserved0 = 0
        status = int(self._lib.oct_session_send_audio(self._handle, view))
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                "oct_session_send_audio failed",
                last_error=self._owner.last_error(),
            )

    def send_image(
        self,
        image_bytes: bytes,
        *,
        mime: int,
    ) -> None:
        """v0.1.12 / ABI minor 11 — push an image to the session's
        input queue. STUB on the runtime side until the SigLIP-base
        int8 adapter PR lands; on the binding side this is gated
        double-blind:

          1. The loaded dylib MUST have advertised ABI minor >= 11.
             Older runtimes don't export ``oct_session_send_image``.
          2. The runtime MUST advertise the ``embeddings.image``
             capability via ``oct_runtime_capabilities``. While
             ``embeddings.image`` is BLOCKED_WITH_PROOF, no real
             runtime should advertise it; this binding refuses the
             call rather than letting the runtime stub return
             UNSUPPORTED with a less specific error envelope.

        Both gates raise :class:`OctomilUnsupportedError` so capability-
        sensitive callers can catch precisely without swallowing every
        :class:`NativeRuntimeError`. There is NO public ``embeddings_image()``
        SDK surface in this PR — see the module docstring for the
        sequencing constraint.
        """
        self._check_open()
        capability = "embeddings.image"
        # Gate 1: symbol resolved at load time.
        send_image_fn = _OPTIONAL_IMAGE_SYMBOLS.get("oct_session_send_image")
        if send_image_fn is None:
            major, minor, _patch = abi_version()
            raise OctomilUnsupportedError(
                capability,
                f"{capability} is not available: loaded liboctomil-runtime "
                f"ABI {major}.{minor} does not export oct_session_send_image "
                f"(requires minor >= 11). This is the optional image-input "
                f"surface added in v0.1.12; the required ABI floor for this "
                f"binding stays at {_REQUIRED_ABI_MAJOR}.{_REQUIRED_ABI_MINOR}.",
            )
        # Gate 2: capability advertised by runtime.
        advertised = self._owner.capabilities().supported_capabilities
        if capability not in advertised:
            major, minor, _patch = abi_version()
            raise OctomilUnsupportedError(
                capability,
                f"{capability} is not advertised by this runtime "
                f"(v{major}.{minor}). The image-input symbol table is "
                f"present but the capability is not live — "
                f"oct_runtime_capabilities does not include "
                f"{capability!r}. While the capability is "
                f"BLOCKED_WITH_PROOF the SigLIP-base int8 adapter has "
                f"not yet wired the embeddings.image session class.",
            )
        # NOTE: Even if both gates pass, the runtime itself currently
        # stubs oct_session_send_image to OCT_STATUS_UNSUPPORTED until
        # the adapter PR lands. We forward the call so any future
        # capability-flipped runtime gets exercised through the same
        # path the conformance harness uses.
        ffi = self._ffi
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                f"send_image: image_bytes must be a bytes-like object, got {type(image_bytes).__name__}",
            )
        # Normalize to a byte-format memoryview. For non-byte memoryviews
        # (e.g., memoryview over array.array('I') with itemsize=4),
        # ``len(image_bytes)`` returns the ELEMENT count, not the BYTE
        # count — using that for ``oct_image_view_t.n_bytes`` would
        # under-report the buffer length by a factor of itemsize. The
        # ``mv.cast("B")`` forces byte semantics so ``mv.nbytes`` equals
        # the true byte length AND the downstream ``uint8_t*`` cast on
        # ``ffi.from_buffer(mv)`` is safe regardless of original itemsize.
        if isinstance(image_bytes, memoryview):
            mv = image_bytes
        else:
            mv = memoryview(image_bytes)
        mv = mv.cast("B")
        n_bytes = mv.nbytes
        if n_bytes == 0:
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                "send_image: image_bytes is empty (n_bytes=0)",
            )
        if mime == OCT_IMAGE_MIME_UNKNOWN or mime not in (
            OCT_IMAGE_MIME_PNG,
            OCT_IMAGE_MIME_JPEG,
            OCT_IMAGE_MIME_WEBP,
            OCT_IMAGE_MIME_RGB8,
        ):
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                f"send_image: mime must be one of OCT_IMAGE_MIME_PNG/JPEG/WEBP/RGB8 (got {mime})",
            )
        view = ffi.new("oct_image_view_t*")
        view.bytes = ffi.cast("const uint8_t*", ffi.from_buffer(mv))
        view.n_bytes = n_bytes
        view.mime = mime
        view._reserved0 = 0
        status = int(send_image_fn(self._handle, view))
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                "oct_session_send_image failed",
                last_error=self._owner.last_error(),
            )

    def embeddings_image(self, image_bytes: bytes, *, mime: int) -> None:
        """Public-facing image embedding entry point.

        BLOCKED: ``embeddings.image`` is BLOCKED_WITH_PROOF. The
        runtime exports ``oct_session_send_image`` as of ABI minor 11
        but the SigLIP-base int8 adapter has not been wired, so this
        method raises :class:`NotImplementedError` rather than
        forwarding to the stubbed C call. The lower-level
        :meth:`send_image` method exists for the conformance harness
        and future adapter-flip testing; production callers should
        treat ``embeddings.image`` as unavailable until this method's
        body actually forwards to ``oct_session_send_image``.

        TODO(v0.1.13): wire to :meth:`send_image` once the adapter
        flips ``embeddings.image`` out of ``BLOCKED_WITH_PROOF_CAPABILITIES``
        and the runtime advertises the capability.
        """
        raise NotImplementedError(
            "embeddings.image is BLOCKED_WITH_PROOF: ABI minor 11 binding "
            "stubs are in place, but no public SDK surface forwards to "
            "oct_session_send_image until the SigLIP-base int8 adapter "
            "lands and removes embeddings.image from the blocked set."
        )

    def send_text(self, utf8: str) -> None:
        """Push a UTF-8 text turn to the session.

        Caller-owned and copied by the runtime. Same error surface as
        :meth:`send_audio`."""
        self._check_open()
        ffi = self._ffi
        encoded = ffi.new("char[]", utf8.encode("utf-8"))
        status = int(self._lib.oct_session_send_text(self._handle, encoded))
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                "oct_session_send_text failed",
                last_error=self._owner.last_error(),
            )

    def send_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        """v0.1.2: send a `chat.completion` turn with caller-controlled
        generation options.

        Emits the wrapped JSON shape on `oct_session_send_text`:
        ``{"messages":[...], "options":{...}}``. The runtime applies
        the model's chat template, tokenizes, prefills, and runs the
        decode loop bounded by ``max_tokens`` (or ``max_completion_tokens``
        if both are set — the latter wins, matching OpenAI's spec).

        Bounded errors propagate via the resulting event stream:
            - out-of-range max_tokens → SESSION_COMPLETED(INVALID_INPUT)
              with OCT_EVENT_ERROR(error_code=OCT_ERR_INPUT_FORMAT_UNSUPPORTED).
            - non-zero temperature / non-1.0 top_p →
              SESSION_COMPLETED(UNSUPPORTED) with the same error envelope.
            - prompt tokenizing past n_ctx →
              SESSION_COMPLETED(INVALID_INPUT) with
              OCT_EVENT_ERROR(error_code=OCT_ERR_INPUT_OUT_OF_RANGE).

        Parameters
        ----------
        messages
            Canonical chat-messages list. Each entry MUST have exactly
            ``role`` (∈ {system, user, assistant}) and ``content``
            (string). v0.1.2 rejects unknown keys, extra roles, etc.
        max_tokens
            OpenAI-legacy alias. 1..4096 (= n_ctx). When unset, runtime
            applies its default cap (256).
        max_completion_tokens
            OpenAI-current alias. Same range. When BOTH max_tokens
            and max_completion_tokens are set, max_completion_tokens
            wins (runtime-side resolution).
        temperature
            v0.1.2 ships greedy-only; only 0.0 is accepted. Non-zero
            values reject UNSUPPORTED.
        top_p
            v0.1.2 ships greedy-only; only 1.0 is accepted. Non-1.0
            values reject UNSUPPORTED.
        """
        import json as _json

        options: dict[str, Any] = {}
        if max_tokens is not None:
            options["max_tokens"] = int(max_tokens)
        if max_completion_tokens is not None:
            options["max_completion_tokens"] = int(max_completion_tokens)
        if temperature is not None:
            options["temperature"] = float(temperature)
        if top_p is not None:
            options["top_p"] = float(top_p)
        payload: dict[str, Any] = {"messages": messages}
        if options:
            payload["options"] = options
        self.send_text(_json.dumps(payload))

    def send_embed(self, inputs: str | list[str]) -> None:
        """v0.1.3: send an `embeddings.text` request.

        Emits ``{"input": <str | [str, ...]>}`` on
        ``oct_session_send_text``. The runtime tokenizes, runs
        ``llama_decode`` in embedding mode, pulls the pooled vector
        via ``llama_get_embeddings_seq``, L2-normalizes, and emits
        one ``OCT_EVENT_EMBEDDING_VECTOR`` per input in input order
        followed by ``OCT_EVENT_SESSION_COMPLETED(OK)``.

        Atomic-batch failure: per-input failure produces one
        ``OCT_EVENT_ERROR`` followed immediately by
        ``OCT_EVENT_SESSION_COMPLETED`` with the matching bounded
        status; subsequent inputs are NOT processed. Bindings derive
        the failed index from the count of EMBEDDING_VECTOR events
        received before the ERROR (emission is in input order).

        Parameters
        ----------
        inputs
            Either a single string OR a non-empty list of strings.
            Empty / whitespace-only strings reject INVALID_INPUT
            via the runtime-side validator (privacy-preserved error
            messages — no input bytes echoed).
        """
        import json as _json

        if isinstance(inputs, str):
            payload: dict[str, Any] = {"input": inputs}
        elif isinstance(inputs, list):
            payload = {"input": list(inputs)}
        else:
            raise TypeError(f"send_embed: inputs must be str or list[str], got {type(inputs).__name__}")
        self.send_text(_json.dumps(payload))

    def poll_event(self, timeout_ms: int = 0) -> NativeEvent:
        """Wait up to ``timeout_ms`` for the next event from the session.

        Slice 2A: the stub returns OCT_STATUS_UNSUPPORTED but still
        zeros the event buffer respecting the versioned-output-struct
        contract. We translate UNSUPPORTED into a raise; OCT_STATUS_OK
        with type==OCT_EVENT_NONE is the documented timeout shape."""
        self._check_open()
        ffi = self._ffi
        ev = self._event_buf
        ev.size = ffi.sizeof("oct_event_t")
        ev.version = OCT_EVENT_VERSION
        status = int(self._lib.oct_session_poll_event(self._handle, ev, timeout_ms))

        def _envelope(ev_buf: Any) -> dict[str, Any]:
            """v0.4 step 2: harvest the operational envelope. Runtime
            ALWAYS writes non-NULL pointers (empty strings on
            uncorrelated slots), so the cffi.string calls are safe."""
            return {
                "request_id": ffi.string(ev_buf.request_id).decode("utf-8", errors="replace")
                if ev_buf.request_id != ffi.NULL
                else "",
                "route_id": ffi.string(ev_buf.route_id).decode("utf-8", errors="replace")
                if ev_buf.route_id != ffi.NULL
                else "",
                "trace_id": ffi.string(ev_buf.trace_id).decode("utf-8", errors="replace")
                if ev_buf.trace_id != ffi.NULL
                else "",
                "engine_version": ffi.string(ev_buf.engine_version).decode("utf-8", errors="replace")
                if ev_buf.engine_version != ffi.NULL
                else "",
                "adapter_version": ffi.string(ev_buf.adapter_version).decode("utf-8", errors="replace")
                if ev_buf.adapter_version != ffi.NULL
                else "",
                "accelerator": ffi.string(ev_buf.accelerator).decode("utf-8", errors="replace")
                if ev_buf.accelerator != ffi.NULL
                else "",
                "artifact_digest": ffi.string(ev_buf.artifact_digest).decode("utf-8", errors="replace")
                if ev_buf.artifact_digest != ffi.NULL
                else "",
                "cache_was_hit": bool(ev_buf.cache_was_hit),
            }

        if status == OCT_STATUS_TIMEOUT:
            return NativeEvent(
                type=OCT_EVENT_NONE,
                version=int(ev.version),
                monotonic_ns=int(ev.monotonic_ns),
                user_data_ptr=int(ffi.cast("uintptr_t", ev.user_data)),
                **_envelope(ev),
            )
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                "oct_session_poll_event failed",
                last_error=self._owner.last_error(),
            )
        # Cutover: copy out the TRANSCRIPT_CHUNK text BEFORE we
        # return — the runtime's contract is "inner pointers valid
        # only until the next poll", so by the time the caller
        # iterates we'd be reading freed memory. n_bytes bounds the
        # read; the buffer is utf-8 by spec.
        text_payload = ""
        terminal_status = 0
        error_code = 0
        cache_layer = ""
        cache_saved_tokens = 0
        setup_ms = 0.0
        engine_first_chunk_ms = 0.0
        e2e_first_chunk_ms = 0.0
        total_latency_ms = 0.0
        queued_ms = 0.0
        # v0.1.3 — embeddings.text payload defaults.
        emb_values: list[float] = []
        emb_n_dim = 0
        emb_n_input_tokens = 0
        emb_index = 0
        emb_pooling_type = 0
        emb_is_normalized = False
        # v0.1.5 PR-2 — STT payload defaults.
        seg_start_ms = 0
        seg_end_ms = 0
        seg_index = 0
        seg_is_final = False
        final_n_segments = 0
        final_duration_ms = 0
        # v0.1.5 PR-2N — VAD transition payload defaults.
        vad_kind = 0
        vad_ts_ms = 0
        vad_conf = 0.0
        # audio.diarization payload defaults.
        diar_start_ms = 0
        diar_end_ms = 0
        diar_speaker_id = OCT_DIARIZATION_SPEAKER_UNKNOWN
        diar_speaker_label = ""
        # v0.1.8 Lane A — TTS audio-chunk payload defaults.
        tts_pcm_bytes = b""
        tts_sample_rate = 0
        tts_sample_format = 0
        tts_channels = 0
        tts_is_final = False
        ev_type = int(ev.type)
        if ev_type == OCT_EVENT_TRANSCRIPT_CHUNK:
            chunk = ev.data.transcript_chunk
            if chunk.utf8 != ffi.NULL and chunk.n_bytes > 0:
                text_payload = ffi.buffer(chunk.utf8, int(chunk.n_bytes))[:].decode("utf-8", errors="replace")
        elif ev_type == OCT_EVENT_SESSION_COMPLETED:
            # Cutover R1 (Codex): expose typed terminal_status so
            # the SDK can map UNSUPPORTED rejects to
            # UNSUPPORTED_MODALITY (rather than flatten everything
            # post-error to INVALID_INPUT).
            sc = ev.data.session_completed
            terminal_status = int(sc.terminal_status)
            # Cutover follow-up #73: latency telemetry from
            # SESSION_COMPLETED — surface to InferenceMetrics +
            # get_verbose_metadata so callers see real timing
            # rather than just a chunk count.
            setup_ms = float(sc.setup_ms)
            engine_first_chunk_ms = float(sc.engine_first_chunk_ms)
            e2e_first_chunk_ms = float(sc.e2e_first_chunk_ms)
            total_latency_ms = float(sc.total_latency_ms)
            queued_ms = float(sc.queued_ms)
        elif ev_type == OCT_EVENT_ERROR:
            # Cutover R1 (Codex): typed error_code from the inner
            # payload.
            error_code = int(ev.data.error.error_code)
        elif ev_type in (OCT_EVENT_CACHE_HIT, OCT_EVENT_CACHE_MISS):
            # Cutover follow-up #73: cache telemetry. layer string is
            # owned by the runtime only until the next poll, so copy
            # it out now (same pattern as transcript_chunk text).
            cache = ev.data.cache
            cache_saved_tokens = int(cache.saved_tokens)
            if cache.layer != ffi.NULL:
                cache_layer = ffi.string(cache.layer).decode("utf-8", errors="replace")
        elif ev_type == OCT_EVENT_TRANSCRIPT_SEGMENT:
            # v0.1.5 PR-2 — STT segment. utf8 + timestamps + ordinal.
            # Copy bytes out of the runtime-owned buffer NOW per the
            # "valid until next poll" lifetime contract.
            seg = ev.data.transcript_segment
            if seg.utf8 != ffi.NULL and seg.n_bytes > 0:
                text_payload = ffi.buffer(seg.utf8, int(seg.n_bytes))[:].decode("utf-8", errors="replace")
            seg_start_ms = int(seg.start_ms)
            seg_end_ms = int(seg.end_ms)
            seg_index = int(seg.segment_index)
            seg_is_final = bool(seg.is_final)
        elif ev_type == OCT_EVENT_TRANSCRIPT_FINAL:
            # v0.1.5 PR-2 — STT end-of-transcript. utf8 + n_segments +
            # duration_ms. Same lifetime rule as segment.
            fin = ev.data.transcript_final
            if fin.utf8 != ffi.NULL and fin.n_bytes > 0:
                text_payload = ffi.buffer(fin.utf8, int(fin.n_bytes))[:].decode("utf-8", errors="replace")
            final_n_segments = int(fin.n_segments)
            final_duration_ms = int(fin.duration_ms)
        elif ev_type == OCT_EVENT_EMBEDDING_VECTOR:
            # v0.1.3 — pooled embedding vector. The runtime owns the
            # float buffer only until the next poll on this session,
            # so we MUST copy it out here. ffi.buffer reads n_dim *
            # sizeof(float) bytes; struct.unpack converts to a
            # Python list (binding-owned, survives across polls).
            emb = ev.data.embedding_vector
            emb_n_dim = int(emb.n_dim)
            emb_n_input_tokens = int(emb.n_input_tokens)
            emb_index = int(emb.index)
            emb_pooling_type = int(emb.pooling_type)
            emb_is_normalized = bool(emb.is_normalized)
            if emb.values != ffi.NULL and emb_n_dim > 0:
                # Each fp32 = 4 bytes; copy via cffi.buffer slice +
                # struct.unpack. R1 Codex/Claude nit: ``struct`` is
                # imported at module level (`_emb_struct`) so the
                # poll hot path doesn't pay lazy-import overhead.
                buf = ffi.buffer(emb.values, emb_n_dim * 4)[:]
                emb_values = list(_emb_struct.unpack(f"{emb_n_dim}f", buf))
        elif ev_type == OCT_EVENT_TTS_AUDIO_CHUNK:
            # v0.1.8 Lane A — TTS audio chunk. Layout matches
            # data.audio_chunk: { pcm, n_bytes, sample_rate, sample_format,
            # channels, is_final }. The PCM buffer is runtime-owned with
            # lifetime "until next poll on this session" — copy bytes out
            # NOW. n_bytes bounds the read.
            tts = ev.data.tts_audio_chunk
            n_bytes = int(tts.n_bytes)
            if tts.pcm != ffi.NULL and n_bytes > 0:
                tts_pcm_bytes = bytes(ffi.buffer(tts.pcm, n_bytes))
            else:
                tts_pcm_bytes = b""
            tts_sample_rate = int(tts.sample_rate)
            tts_sample_format = int(tts.sample_format)
            tts_channels = int(tts.channels)
            tts_is_final = bool(tts.is_final)
        elif ev_type == OCT_EVENT_VAD_TRANSITION:
            # v0.1.5 PR-2N — silero VAD transition edge. Payload is
            # primitive-typed (kind enum + ms timestamp + f32
            # confidence); no runtime-owned pointer fields, so no
            # copy-out lifetime concern. transition_kind is one of
            # OCT_VAD_TRANSITION_* (UNKNOWN=0 sentinel; SPEECH_START=1;
            # SPEECH_END=2). The runtime header guarantees confidence
            # ∈ [0.0, 1.0]; we surface it as-is and let the caller
            # decide whether to enforce that range (defensive callers
            # may want to clamp).
            vad = ev.data.vad_transition
            vad_kind = int(vad.transition_kind)
            vad_ts_ms = int(vad.timestamp_ms)
            vad_conf = float(vad.confidence)
        elif ev_type == OCT_EVENT_DIARIZATION_SEGMENT:
            # audio.diarization — speaker-turn segment. Primitive
            # timestamps plus a runtime-owned speaker_label pointer;
            # copy the label now because it is valid only until the
            # next poll on this session.
            diar = ev.data.diarization_segment
            diar_start_ms = int(diar.start_ms)
            diar_end_ms = int(diar.end_ms)
            diar_speaker_id = int(diar.speaker_id)
            if diar.speaker_label != ffi.NULL:
                diar_speaker_label = ffi.string(diar.speaker_label).decode("utf-8", errors="replace")
        return NativeEvent(
            type=ev_type,
            version=int(ev.version),
            monotonic_ns=int(ev.monotonic_ns),
            user_data_ptr=int(ffi.cast("uintptr_t", ev.user_data)),
            text=text_payload,
            terminal_status=terminal_status,
            error_code=error_code,
            cache_layer=cache_layer,
            cache_saved_tokens=cache_saved_tokens,
            setup_ms=setup_ms,
            engine_first_chunk_ms=engine_first_chunk_ms,
            e2e_first_chunk_ms=e2e_first_chunk_ms,
            total_latency_ms=total_latency_ms,
            queued_ms=queued_ms,
            values=emb_values,
            n_dim=emb_n_dim,
            n_input_tokens=emb_n_input_tokens,
            index=emb_index,
            pooling_type=emb_pooling_type,
            is_normalized=emb_is_normalized,
            segment_start_ms=seg_start_ms,
            segment_end_ms=seg_end_ms,
            segment_index=seg_index,
            segment_is_final=seg_is_final,
            final_n_segments=final_n_segments,
            final_duration_ms=final_duration_ms,
            vad_transition_kind=vad_kind,
            vad_timestamp_ms=vad_ts_ms,
            vad_confidence=vad_conf,
            diarization_start_ms=diar_start_ms,
            diarization_end_ms=diar_end_ms,
            diarization_speaker_id=diar_speaker_id,
            diarization_speaker_label=diar_speaker_label,
            tts_pcm_bytes=tts_pcm_bytes,
            tts_sample_rate=tts_sample_rate,
            tts_sample_format=tts_sample_format,
            tts_channels=tts_channels,
            tts_is_final=tts_is_final,
            **_envelope(ev),
        )

    def cancel(self) -> int:
        """Request cancellation. Safe from any thread.

        Returns the raw status code (OCT_STATUS_OK on first call,
        OCT_STATUS_CANCELLED on idempotent re-cancel) — these are NOT
        treated as errors. Any other code raises."""
        if self._handle_invalid or self._closed:
            return OCT_STATUS_CANCELLED
        status = int(self._lib.oct_session_cancel(self._handle))
        if status not in (OCT_STATUS_OK, OCT_STATUS_CANCELLED, OCT_STATUS_UNSUPPORTED):
            raise NativeRuntimeError(
                status,
                "oct_session_cancel failed",
                last_error=self._owner.last_error(),
            )
        return status

    def close(self) -> None:
        """Close the session. Idempotent.

        If the parent runtime has already been closed, the underlying
        ``oct_session_t*`` was implicitly closed by the dylib (per
        the runtime.h contract); we MUST NOT call ``oct_session_close``
        on a freed handle. ``_invalidate_after_runtime_close`` flips
        this flag so the cleanup path becomes a no-op."""
        if self._closed:
            return
        if not self._handle_invalid:
            self._lib.oct_session_close(self._handle)
        self._closed = True
        # Codex R2 nit: drop the strong ref to the borrowed model so
        # the model wrapper can be reaped now (rather than waiting
        # for NativeRuntime.close() or wrapper GC). The C-side in_use
        # decrement happened inside oct_session_close above; releasing
        # the Python ref releases the wrapper for the WeakSet to
        # observe. Tighter "session lifetime" for resource hygiene.
        self._borrowed_model = None

    def _invalidate_after_runtime_close(self) -> None:
        """Marker called by NativeRuntime.close() before the dylib
        tears down the runtime. Subsequent operations on this session
        raise (via _check_open); subsequent close() is a no-op (the
        dylib already freed the handle implicitly)."""
        self._handle_invalid = True
        self._closed = True
        # Match the close()-path behavior so the borrowed-model ref
        # doesn't keep the wrapper alive past the runtime cascade.
        self._borrowed_model = None

    def __enter__(self) -> "NativeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# v0.4 step 1 — NativeModel
# ---------------------------------------------------------------------------


class NativeModel:
    """RAII-style wrapper over ``oct_model_t`` (v0.4 step 1).

    The model handle is the warm-handle abstraction the runtime uses
    for pool-keyed caching, eviction, and signed-manifest identity.
    A session may open against a `model_uri` (slice-2A behavior) or
    against a pre-warmed `NativeModel` (future v0.4 step that
    appends `oct_session_config_t.model`).

    v0.4 step 1: stubs only. Every method calls into the runtime's
    UNSUPPORTED stub and raises `NativeRuntimeError`. The wrapper
    exists so:
      * Conformance harness has a stable target NOW.
      * Layer-2b warm-pool plumbing can be designed against this
        surface immediately.
      * Engine adapters (Slice 2C Moshi, future llama.cpp /
        sherpa-onnx / whisper.cpp / ONNX) replace stubs file-by-file
        with tests already pinned.

    Lifetime tracking mirrors `NativeSession`'s slice-2A
    `_handle_invalid` pattern. `NativeRuntime.close()` invalidates
    every live `NativeModel` it spawned before calling
    `oct_runtime_close`.
    """

    def __init__(self, ffi: Any, lib: Any, handle: Any, *, owner: "NativeRuntime") -> None:
        self._ffi = ffi
        self._lib = lib
        self._handle = handle
        self._owner = owner
        self._closed = False
        self._handle_invalid = False

    def _check_open(self) -> None:
        if self._handle_invalid:
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                "model handle invalidated by parent NativeRuntime.close()",
            )
        if self._closed:
            raise NativeRuntimeError(
                OCT_STATUS_INVALID_INPUT,
                "model handle is closed",
            )

    def warm(self) -> None:
        """Run engine warmup. Stub raises UNSUPPORTED."""
        self._check_open()
        status = int(self._lib.oct_model_warm(self._handle))
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                "oct_model_warm failed",
                last_error=self._owner.last_error(),
            )

    def evict(self) -> int:
        """Request eviction. Returns raw status code; OCT_STATUS_BUSY
        and OCT_STATUS_UNSUPPORTED are NOT treated as errors so
        cleanup paths can call evict() without try/except."""
        if self._closed or self._handle_invalid:
            return OCT_STATUS_OK
        status = int(self._lib.oct_model_evict(self._handle))
        if status not in (OCT_STATUS_OK, OCT_STATUS_BUSY, OCT_STATUS_UNSUPPORTED):
            raise NativeRuntimeError(
                status,
                "oct_model_evict failed",
                last_error=self._owner.last_error(),
            )
        return status

    def close(self) -> int:
        """Close the model. v0.1.1: status-bearing.

        Returns the raw ``oct_status_t`` so the binding can observe
        ``OCT_STATUS_BUSY`` when sessions still borrow the handle.
        On BUSY: the model is NOT freed; the binding closes its
        sessions and retries. On OK: handle freed; this wrapper marks
        itself closed.

        Idempotent on the wrapper side: a second call after OK or
        after a parent runtime close returns ``OCT_STATUS_OK``
        without re-entering the dylib. Slice-2A invalidation pattern
        preserved (parent runtime close pre-invalidates).

        Mirrors :meth:`evict`'s status-return shape so cleanup
        contexts can call ``model.close()`` without try/except —
        BUSY is recoverable, not exceptional.
        """
        if self._closed:
            return OCT_STATUS_OK
        if self._handle_invalid:
            self._closed = True
            return OCT_STATUS_OK
        status = int(self._lib.oct_model_close(self._handle))
        if status == OCT_STATUS_OK:
            self._closed = True
            return status
        if status == OCT_STATUS_BUSY:
            # Handle is still alive on the runtime side. Do NOT mark
            # _closed; the binding can retry after closing sessions.
            return status
        # Anything else (INVALID_INPUT, INTERNAL) — surface as a
        # typed error. The handle's runtime-side state is ambiguous
        # but our wrapper stays unmarked so the binding can decide.
        raise NativeRuntimeError(
            status,
            "oct_model_close failed",
            last_error=self._owner.last_error(),
        )

    def _invalidate_after_runtime_close(self) -> None:
        self._handle_invalid = True
        self._closed = True

    def __enter__(self) -> "NativeModel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "ENV_DYLIB_OVERRIDE",
    "_ENV_FLAVOR",
    "_FLAVOR_PREFERENCE",
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
    "OCT_DIARIZATION_SPEAKER_UNKNOWN",
    "OCT_MODEL_CONFIG_VERSION",
    "OCT_EVENT_AUDIO_CHUNK",
    "OCT_EVENT_CACHE_HIT",
    "OCT_EVENT_CACHE_MISS",
    "OCT_EVENT_CAPABILITY_VERIFIED",
    "OCT_EVENT_DIARIZATION_SEGMENT",
    "OCT_EVENT_EMBEDDING_VECTOR",
    "OCT_EVENT_ERROR",
    "OCT_EVENT_INPUT_DROPPED",
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
    "OCT_EVENT_TRANSCRIPT_FINAL",
    "OCT_EVENT_TRANSCRIPT_SEGMENT",
    "OCT_EVENT_TTS_AUDIO_CHUNK",
    "OCT_EVENT_TURN_ENDED",
    "OCT_EVENT_USER_SPEECH_DETECTED",
    "OCT_EVENT_VAD_TRANSITION",
    "OCT_EVENT_VERSION",
    "OCT_EVENT_WATCHDOG_TIMEOUT",
    "OCT_PRIORITY_FOREGROUND",
    "OCT_PRIORITY_PREFETCH",
    "OCT_PRIORITY_SPECULATIVE",
    "OCT_SAMPLE_FORMAT_PCM_F32LE",
    "OCT_SAMPLE_FORMAT_PCM_S16LE",
    "OCT_SESSION_CONFIG_VERSION",
    "OCT_STATUS_BUSY",
    "OCT_STATUS_CANCELLED",
    "OCT_STATUS_INTERNAL",
    "OCT_STATUS_INVALID_INPUT",
    "OCT_STATUS_NOT_FOUND",
    "OCT_STATUS_OK",
    "OCT_STATUS_TIMEOUT",
    "OCT_STATUS_UNSUPPORTED",
    "OCT_STATUS_VERSION_MISMATCH",
    "OCT_VAD_TRANSITION_SPEECH_END",
    "OCT_VAD_TRANSITION_SPEECH_START",
    "OCT_VAD_TRANSITION_UNKNOWN",
    "RuntimeCapabilities",
    "abi_version",
    "last_thread_error",
]
