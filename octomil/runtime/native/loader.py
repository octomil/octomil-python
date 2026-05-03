"""cffi loader for ``liboctomil-runtime``.

Embedded host (in-process) binding. Locates the dylib via:

  1. ``OCTOMIL_RUNTIME_DYLIB`` env var (operator override).
  2. Sibling ``runtime-core/build/liboctomil-runtime.{dylib,so,dll}``
     (dev path; works when the user ran ``cmake --build build`` in
     ``octomil/runtime-core/``).
  3. ``ImportError`` with a message pointing at
     ``octomil/runtime-core/BUILD.md``.

This module exposes ONLY:

  * Version inspection (``oct_runtime_abi_version_*``).
  * Runtime open/close (``oct_runtime_open``, ``oct_runtime_close``).
  * Capabilities (``oct_runtime_capabilities``,
    ``oct_runtime_capabilities_free``).
  * Last-error (``oct_runtime_last_error``,
    ``oct_last_thread_error``).

Session entry points (``oct_session_*``) are intentionally NOT
exposed in slice 3 PR1 — the slice-2 stub returns
``OCT_STATUS_UNSUPPORTED`` for all of them and there is no behavior
to bind against. They land alongside slice 2-proper (or its first
real session adapter).

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


def _status_name(code: int) -> str:
    return _STATUS_NAMES.get(code, f"OCT_STATUS_UNKNOWN({code})")


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


def _candidate_dylib_paths() -> list[Path]:
    """Return the ordered list of paths to try.

    The env-var override wins; otherwise we walk up from this module's
    file to find the workspace's ``runtime-core/build/`` directory.
    The dev path covers the common case: cloned the repo, ran
    ``cmake --build build`` in ``octomil/runtime-core/``."""
    candidates: list[Path] = []
    override = os.environ.get(ENV_DYLIB_OVERRIDE)
    if override:
        candidates.append(Path(override))
    here = Path(__file__).resolve()
    # octomil/runtime/native/loader.py → octomil-python/octomil/runtime-core/build/
    repo_root = here.parents[2]  # octomil-python/octomil
    runtime_core_build = repo_root / "runtime-core" / "build"
    for name in (
        "liboctomil-runtime.dylib",  # macOS
        "liboctomil-runtime.so",  # Linux
        "octomil-runtime.dll",  # Windows
    ):
        candidates.append(runtime_core_build / name)
    return candidates


def _resolve_dylib() -> Path:
    """Find a usable dylib path or raise ImportError pointing at
    BUILD.md.

    Override semantics: if ``OCTOMIL_RUNTIME_DYLIB`` is set, the
    override path is authoritative. If it doesn't exist, raise
    immediately rather than falling through to the dev-path fallback
    — silently ignoring an explicit operator override would mask
    deployment configuration bugs."""
    override = os.environ.get(ENV_DYLIB_OVERRIDE)
    if override:
        override_path = Path(override)
        if override_path.is_file():
            return override_path
        raise ImportError(
            f"{ENV_DYLIB_OVERRIDE} points at {override!r} which does not exist.\n"
            f"Operator override is authoritative — fix the path or unset the\n"
            f"env var to use the dev-path fallback.\n"
            f"Build instructions: octomil/runtime-core/BUILD.md"
        )
    tried: list[str] = []
    for path in _candidate_dylib_paths():
        if path.is_file():
            return path
        tried.append(str(path))
    raise ImportError(
        "Could not locate liboctomil-runtime.\n"
        "Tried (in order):\n"
        + "\n".join(f"  - {t}" for t in tried)
        + f"\n\nBuild instructions: octomil/runtime-core/BUILD.md\n"
        f"Operator override: set {ENV_DYLIB_OVERRIDE} to an absolute path."
    )


# ---------------------------------------------------------------------------
# cffi cdef + lib singleton
# ---------------------------------------------------------------------------

# Keeping the cdef minimal — only the symbols slice 3 PR1 binds.
# Slice 2-proper / 3 PR2 will extend this when session entries land.
_CDEF: str = """
typedef uint32_t oct_status_t;

typedef struct oct_runtime oct_runtime_t;

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

typedef void (*oct_telemetry_sink_fn)(const char* event_json, void* user_data);

typedef struct {
    uint32_t              version;
    const char*           cache_root;
    const char*           runtime_build_tag;
    oct_telemetry_sink_fn telemetry_sink;
    void*                 telemetry_user_data;
} oct_runtime_config_t;

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
"""


def _build_lib() -> tuple[Any, Any]:
    """Construct the cffi (FFI, lib) pair. Imported lazily so test
    discovery doesn't fail when cffi or the dylib is unavailable."""
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
    logger.debug("loaded liboctomil-runtime from %s", dylib)
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

    @classmethod
    def open(
        cls,
        *,
        cache_root: str = "",
        runtime_build_tag: str = "",
    ) -> "NativeRuntime":
        """Open a runtime with the slice-1 v1 config shape.

        Slice-2-proper extends ``oct_runtime_config_t`` with
        additional fields; v1 is what the stub accepts today.
        ``cache_root`` and ``runtime_build_tag`` are passed through
        as-is; empty strings are valid (the runtime falls back to
        platform defaults)."""
        ffi, lib = _get_lib()
        cfg = ffi.new("oct_runtime_config_t*")
        cfg.version = 1
        # Strings need to outlive the call; keep references.
        cache_root_buf = ffi.new("char[]", cache_root.encode("utf-8"))
        build_tag_buf = ffi.new("char[]", runtime_build_tag.encode("utf-8"))
        cfg.cache_root = cache_root_buf
        cfg.runtime_build_tag = build_tag_buf
        cfg.telemetry_sink = ffi.NULL
        cfg.telemetry_user_data = ffi.NULL
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
        """Close the handle. Idempotent."""
        if self._closed:
            return
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


__all__ = [
    "ENV_DYLIB_OVERRIDE",
    "NativeRuntime",
    "NativeRuntimeError",
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
