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
  * Slice 2A — session lifecycle bindings: ``NativeSession`` wraps
    ``oct_session_open / send_audio / send_text / poll_event /
    cancel / close``. The stub returns ``OCT_STATUS_UNSUPPORTED``
    for every call until slice 2-proper lands a real session
    adapter; the wrapper exists so bindings, conformance tests, and
    higher-level routing code can compile against the real surface
    NOW and have it auto-skip on capability advertisement.

**Forward-compat advertisement parsing:** capabilities returned by
the runtime that do NOT appear in :data:`RUNTIME_CAPABILITIES` are
silently dropped from the parsed view (with a DEBUG log). This
matches the asymmetric reader rule from the contract.
"""

from __future__ import annotations

import logging
import os
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

OCT_SAMPLE_FORMAT_PCM_S16LE: int = 1
OCT_SAMPLE_FORMAT_PCM_F32LE: int = 2

# v0.4 step 2 — bumped lockstep with runtime.h.
OCT_SESSION_CONFIG_VERSION: int = 2
OCT_EVENT_VERSION: int = 2

# v0.4 step 1 — model lifecycle.
OCT_MODEL_CONFIG_VERSION: int = 1

OCT_ACCEL_AUTO: int = 0
OCT_ACCEL_METAL: int = 1
OCT_ACCEL_CUDA: int = 2
OCT_ACCEL_CPU: int = 3
OCT_ACCEL_ANE: int = 4

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


# ---------------------------------------------------------------------------
# Dylib resolution
# ---------------------------------------------------------------------------

ENV_DYLIB_OVERRIDE: str = "OCTOMIL_RUNTIME_DYLIB"

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


def _fetched_dylib_candidates() -> list[Path]:
    """Return any dev-cache dylibs found under ``~/.cache/octomil-runtime``.

    Sorted newest-version-last so the most-recently-fetched release
    wins when multiple are cached. The fetch script populates this
    directory; nothing else writes to it."""
    if not _FETCH_CACHE_ROOT.is_dir():
        return []
    out: list[Path] = []
    for version_dir in sorted(_FETCH_CACHE_ROOT.iterdir(), key=_version_sort_key):
        if not version_dir.is_dir():
            continue
        for name in _RUNTIME_LIBNAMES:
            candidate = version_dir / "lib" / name
            if candidate.is_file():
                out.append(candidate)
    return out


def _resolve_dylib() -> Path:
    """Find a usable dylib path or raise ImportError with a precise
    pointer to the documented setup paths.

    Resolution order:
      1. ``OCTOMIL_RUNTIME_DYLIB`` env var. Authoritative when set —
         if the path is missing we raise immediately (silent fallback
         would mask deployment bugs).
      2. Most-recently-fetched dev artifact under
         ``~/.cache/octomil-runtime/<version>/lib/``. Populated by
         ``scripts/fetch_runtime_dev.py``.

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
    # `_fetched_dylib_candidates()` returns oldest-version-first.
    # Iterate in reverse so the most-recently-fetched release wins —
    # otherwise the version-tuple sort fix is defeated by the
    # iteration order. Codex R2 blocker fix.
    tried: list[str] = []
    for path in reversed(_fetched_dylib_candidates()):
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

/* Slice 2A — session lifecycle entry points. The stub returns
 * OCT_STATUS_UNSUPPORTED for everything until slice 2-proper lands. */
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

/* v0.4 step 1 — model lifecycle. Stubs return OCT_STATUS_UNSUPPORTED.
 * Engine adapters (Slice 2C and following) replace these. */
typedef struct oct_model oct_model_t;
typedef uint32_t oct_accelerator_pref_t;
/* oct_error_code_t typedef moved to top of cdef in v0.4 step 2 (must
 * appear before oct_event_t inner struct uses it). */

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
void         oct_model_close(oct_model_t* model);
size_t       oct_model_config_size(void);
"""


# v0.4 — minimum ABI version this binding requires. The cdef calls
# v0.4 symbols (oct_model_open, oct_model_config_size, etc.); loading
# a v0.3 dylib would fail later with a missing-symbol cffi error
# rather than a typed compatibility error. Codex R1 fix: fail fast
# at load time with NativeRuntimeError(VERSION_MISMATCH).
_REQUIRED_ABI_MAJOR: int = 0
_REQUIRED_ABI_MINOR: int = 5


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
    return ffi, lib  # noqa: RET504 — explicit ffi/lib pair documents both


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

        Codex R1 blocker: per `runtime.h`, live sessions are
        CANCELLED and closed implicitly by `oct_runtime_close`. We
        tell every Python wrapper "the C handle behind you is gone"
        BEFORE the dylib closes the runtime, so any later call on
        a stale `NativeSession` raises NativeRuntimeError instead of
        dereferencing freed memory."""
        if self._closed:
            return
        # Snapshot first; closing each wrapper removes itself from
        # the WeakSet which would mutate during iteration.
        for sess in list(self._sessions):
            sess._invalidate_after_runtime_close()  # noqa: SLF001
        self._sessions.clear()
        # v0.4 step 1: explicit model close BEFORE invalidation. Codex
        # R2 fix: oct_runtime_close documents implicit model cleanup
        # (defense at C ABI), but we also close handles binding-side
        # so engine adapters with expensive resources (mmap'd weights,
        # KV buffers, accelerator contexts) get the explicit close
        # path. Order matters: close (real C call) first, THEN
        # invalidate the wrapper so subsequent operations raise.
        for mdl in list(self._models):
            try:
                mdl.close()
            except Exception:  # noqa: BLE001 — best-effort drain
                pass
            mdl._invalidate_after_runtime_close()  # noqa: SLF001
        self._models.clear()
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
    ) -> "NativeSession":
        """Open a session against this runtime.

        Slice 2A: this is the canonical embedded-host entry point for
        every realtime / TTS / STT / chat capability. The runtime
        applies strict-reject on ``capability``: any value not in
        :data:`octomil.runtime.native.capabilities.RUNTIME_CAPABILITIES`
        returns ``OCT_STATUS_UNSUPPORTED``. Against the slice-2 stub
        EVERY call returns UNSUPPORTED — bindings should treat that
        as the design state until a runtime advertises the capability
        via :meth:`capabilities`.

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
        out = ffi.new("oct_session_t**")
        status = int(lib.oct_session_open(self._handle, cfg, out))
        if status != OCT_STATUS_OK:
            raise NativeRuntimeError(
                status,
                f"oct_session_open(capability={capability!r}) failed",
                last_error=self.last_error(),
            )
        sess = NativeSession(ffi, lib, out[0], owner=self)
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
# NativeSession — slice 2A
# ---------------------------------------------------------------------------


class NativeEvent:
    """Parsed snapshot of an ``oct_event_t``.

    The cffi event buffer is reused across :meth:`NativeSession.poll_event`
    calls (the runtime's lifetime contract makes the inner pointer
    fields valid only until the next poll). This view extracts the
    primitive fields the binding cares about into Python-owned data
    so callers can hold onto it across polls.

    Slice 2A: the stub never produces a non-NONE event; this class
    exists so the slice-2-proper session adapter has a stable
    surface to write tests against immediately."""

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

    Against the slice-2 stub every method returns
    ``OCT_STATUS_UNSUPPORTED`` (raised as :class:`NativeRuntimeError`).
    The wrapper exists so:
      * Conformance tests have a stable target NOW.
      * The Layer-2b session adapter can be written against the real
        surface without further binding churn.
      * Capability-gated tests auto-skip on the stub via
        :meth:`NativeRuntime.capabilities` rather than crashing.
    """

    def __init__(self, ffi: Any, lib: Any, handle: Any, *, owner: NativeRuntime) -> None:
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
        return NativeEvent(
            type=int(ev.type),
            version=int(ev.version),
            monotonic_ns=int(ev.monotonic_ns),
            user_data_ptr=int(ffi.cast("uintptr_t", ev.user_data)),
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

    def _invalidate_after_runtime_close(self) -> None:
        """Marker called by NativeRuntime.close() before the dylib
        tears down the runtime. Subsequent operations on this session
        raise (via _check_open); subsequent close() is a no-op (the
        dylib already freed the handle implicitly)."""
        self._handle_invalid = True
        self._closed = True

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

    def close(self) -> None:
        """Close the model. Idempotent. No-op if parent runtime
        already closed (mirrors NativeSession's invalidation pattern)."""
        if self._closed:
            return
        if not self._handle_invalid:
            self._lib.oct_model_close(self._handle)
        self._closed = True

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
    "OCT_MODEL_CONFIG_VERSION",
    "OCT_EVENT_AUDIO_CHUNK",
    "OCT_EVENT_CACHE_HIT",
    "OCT_EVENT_CACHE_MISS",
    "OCT_EVENT_CAPABILITY_VERIFIED",
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
    "OCT_EVENT_TURN_ENDED",
    "OCT_EVENT_USER_SPEECH_DETECTED",
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
    "RuntimeCapabilities",
    "abi_version",
    "last_thread_error",
]
