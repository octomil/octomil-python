"""Native runtime cache clear / introspection API — Lane G skeleton.

Exposes four operations that map 1:1 to the ABI entry points added by
Lane G (see ``octomil-runtime`` ``include/octomil/runtime.h``):

  * :func:`clear_all`          — clear every cache on the runtime handle.
  * :func:`clear_capability`   — clear caches for one canonical capability.
  * :func:`clear_scope`        — clear caches at a given scope level.
  * :func:`introspect`         — return a privacy-safe :class:`CacheSnapshot`.

All four operations are **stubs** until Lanes B/C/F wire real cache
implementations.  The stubs raise :class:`CacheNotImplementedError`
(a subclass of ``NotImplementedError``) so callers can explicitly catch
the stub state.  Once real caches land, the same functions return
normally.

Privacy contract (enforced by binding tests in
``tests/test_native_cache_api.py``):

  * :func:`introspect` returns **only** bounded numeric fields per cache
    entry: ``capability``, ``scope``, ``entries``, ``bytes``, ``hit``,
    ``miss``.
  * No SHA-256 hex keys, no token IDs, no prompt text, no audio bytes,
    no file paths, no model paths appear anywhere in the output.

Feature probe:

  Before calling any function here the caller SHOULD verify the runtime
  advertises ``"cache.introspect"`` in its capabilities list.  If not
  advertised, all functions raise :class:`CacheNotImplementedError`.
  The probe is advisory — the functions perform it automatically when
  a runtime handle is supplied, and raise rather than silently succeed.

Canonical references:

  * Capability ``cache.introspect`` registered in
    ``octomil-contracts/schemas/core/runtime_capability.json`` (v1.25.0,
    #129).
  * Cache scope codes ``request|session|runtime|app`` match
    ``octomil-contracts/enums/cache_scope.yaml`` (v1.24.0, #123).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from typing import Any

from .loader import (
    OCT_CACHE_SCOPE_APP,
    OCT_CACHE_SCOPE_REQUEST,
    OCT_CACHE_SCOPE_RUNTIME,
    OCT_CACHE_SCOPE_SESSION,
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_NOT_FOUND,
    OCT_STATUS_OK,
    OCT_STATUS_UNSUPPORTED,
    NativeRuntimeError,
    _get_lib,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exported scope constants (re-export for convenience)
# ---------------------------------------------------------------------------

SCOPE_REQUEST: int = OCT_CACHE_SCOPE_REQUEST
SCOPE_SESSION: int = OCT_CACHE_SCOPE_SESSION
SCOPE_RUNTIME: int = OCT_CACHE_SCOPE_RUNTIME
SCOPE_APP: int = OCT_CACHE_SCOPE_APP

# Canonical set — used for validation.
_VALID_SCOPES: frozenset[int] = frozenset({SCOPE_REQUEST, SCOPE_SESSION, SCOPE_RUNTIME, SCOPE_APP})

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CacheNotImplementedError(NotImplementedError):
    """Raised when the cache API stub has not yet been implemented.

    Subclasses ``NotImplementedError`` so callers can catch the stub
    state explicitly without importing this module's private error type.
    Once Lanes B/C/F wire real cache implementations, this error will
    no longer be raised from the happy path.
    """


# ---------------------------------------------------------------------------
# Privacy-safe return type for introspect()
# ---------------------------------------------------------------------------


class CacheEntrySnapshot:
    """Bounded, privacy-safe snapshot of a single cache's stats.

    Only numeric and enum fields are exposed.  No keys, no hashes,
    no text, no audio, no paths.

    Field names match the canonical CachePolicy / CacheScope schemas in
    octomil-contracts (v1.24.0 #123, v1.25.0 #129).
    """

    __slots__ = ("capability", "scope", "entries", "bytes", "hit", "miss")

    def __init__(
        self,
        *,
        capability: str,
        scope: str,
        entries: int,
        bytes: int,  # noqa: A002 — matches the JSON field name from the ABI
        hit: int,
        miss: int,
    ) -> None:
        #: Canonical capability name (e.g. ``"chat.completion"``).
        self.capability: str = capability
        #: Scope level: one of ``"request"``, ``"session"``,
        #: ``"runtime"``, ``"app"``.
        self.scope: str = scope
        #: Number of live cache entries.
        self.entries: int = int(entries)
        #: Estimated bytes consumed.
        self.bytes: int = int(bytes)
        #: Hit counter since last reset.
        self.hit: int = int(hit)
        #: Miss counter since last reset.
        self.miss: int = int(miss)

    def __repr__(self) -> str:
        return (
            f"CacheEntrySnapshot(capability={self.capability!r}, "
            f"scope={self.scope!r}, entries={self.entries}, "
            f"bytes={self.bytes}, hit={self.hit}, miss={self.miss})"
        )


class CacheSnapshot:
    """Privacy-safe introspection snapshot returned by :func:`introspect`.

    Contains only the bounded fields documented in ``runtime.h``.
    The ``is_stub`` flag is ``True`` until Lanes B/C/F land.

    Field names match the canonical CachePolicy / CacheScope schemas in
    octomil-contracts (v1.24.0 #123, v1.25.0 #129).
    """

    __slots__ = ("version", "is_stub", "entries")

    def __init__(
        self,
        *,
        version: int,
        is_stub: bool,
        entries: list[CacheEntrySnapshot],
    ) -> None:
        #: Schema version of the introspect output (currently 1).
        self.version: int = int(version)
        #: True while real caches are not yet wired.
        self.is_stub: bool = bool(is_stub)
        #: Per-cache stats, one entry per registered cache.
        self.entries: list[CacheEntrySnapshot] = entries

    def __repr__(self) -> str:
        return f"CacheSnapshot(version={self.version}, is_stub={self.is_stub}, entries={self.entries!r})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_INTROSPECT_BUF_BYTES: int = 65536  # 64 KiB max for the JSON blob


def _check_status(
    status: int,
    op: str,
    *,
    runtime_handle: Optional[Any] = None,
    allow_not_found: bool = False,
) -> None:
    """Raise on non-OK ABI status codes.

    Maps OCT_STATUS_UNSUPPORTED → :class:`CacheNotImplementedError`.
    Maps OCT_STATUS_NOT_FOUND → :class:`CacheNotImplementedError` when
    ``allow_not_found`` is False (clear_capability treats NOT_FOUND as
    "nothing to clear" and returns normally).
    All other non-OK statuses raise :class:`NativeRuntimeError`.
    """
    if status == OCT_STATUS_OK:
        return
    if status == OCT_STATUS_NOT_FOUND and allow_not_found:
        # clear_capability: "recognized name but no cache registered" is
        # semantically "nothing to clear" — not an error.
        logger.debug("%s: no cache registered (NOT_FOUND — treated as empty)", op)
        return
    if status == OCT_STATUS_UNSUPPORTED:
        raise CacheNotImplementedError(
            f"{op}: cache API not yet implemented (real caches land in Lanes B/C/F; OCT_STATUS_UNSUPPORTED)"
        )
    if status == OCT_STATUS_INVALID_INPUT:
        raise NativeRuntimeError(
            status,
            f"{op}: invalid input (NULL runtime or bad argument)",
        )
    raise NativeRuntimeError(status, f"{op}: unexpected ABI status")


def _parse_snapshot(raw_json: str) -> CacheSnapshot:
    """Parse the bounded JSON from oct_runtime_cache_introspect.

    Validates that the parsed object matches the allowed schema:
    only numeric and enum fields, no keys/hashes/paths/text.

    Raises ValueError if the JSON fails schema validation.
    """
    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"oct_runtime_cache_introspect returned invalid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError("introspect JSON root must be a JSON object")

    # Allowed top-level keys only — no arbitrary fields that could carry
    # leaked data.
    allowed_top = {"version", "is_stub", "entries"}
    unexpected = set(obj.keys()) - allowed_top
    if unexpected:
        raise ValueError(f"introspect JSON contains unexpected keys: {unexpected!r}")

    # Codex B3 (post-hoc Lane G sweep): require version + is_stub keys
    # explicitly (don't silently default). A malformed runtime emitting
    # `{}` would otherwise parse as version=0, is_stub=False, masking
    # real wire-shape regressions.
    for required in ("version", "is_stub"):
        if required not in obj:
            raise ValueError(f"introspect JSON missing required key: {required!r}")

    version = obj["version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("introspect JSON 'version' must be int")
    if version < 0:
        raise ValueError(f"introspect JSON 'version' must be >= 0; got {version}")

    if not isinstance(obj["is_stub"], bool):
        raise ValueError("introspect JSON 'is_stub' must be bool")
    is_stub = obj["is_stub"]

    raw_entries = obj.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ValueError("introspect JSON 'entries' must be a list")

    # Privacy-bound enum sets — capability and scope MUST be canonical.
    # Codex B3 (post-hoc Lane G sweep): without enum validation, a buggy
    # runtime emitting capability='/Users/leak/path' or
    # scope='/etc/passwd' would pass schema since the field is allowed.
    # Imported lazily to avoid a circular import via octomil.runtime.native.
    from octomil.runtime.native.capabilities import RUNTIME_CAPABILITIES

    _VALID_SCOPE_STRINGS: frozenset[str] = frozenset({"request", "session", "runtime", "app"})

    allowed_entry_keys = {"capability", "scope", "entries", "bytes", "hit", "miss"}
    parsed: list[CacheEntrySnapshot] = []
    for i, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"introspect entries[{i}] must be a JSON object")
        unexpected_entry = set(entry.keys()) - allowed_entry_keys
        if unexpected_entry:
            raise ValueError(f"introspect entries[{i}] contains unexpected keys: {unexpected_entry!r}")

        # Type validation — coerce-via-int-and-str hides garbage. Reject
        # non-canonical types loudly so leaks can't slip through.
        for str_field in ("capability", "scope"):
            val = entry.get(str_field, None)
            if not isinstance(val, str):
                raise ValueError(f"introspect entries[{i}].{str_field} must be a string; " f"got {type(val).__name__}")
        for num_field in ("entries", "bytes", "hit", "miss"):
            val = entry.get(num_field, None)
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(f"introspect entries[{i}].{num_field} must be an int; " f"got {type(val).__name__}")
            if val < 0:
                raise ValueError(f"introspect entries[{i}].{num_field} must be >= 0; got {val}")

        # Enum validation — capability and scope must be canonical.
        cap = entry["capability"]
        scope = entry["scope"]
        if cap not in RUNTIME_CAPABILITIES:
            raise ValueError(f"introspect entries[{i}].capability is not a canonical " f"runtime capability: {cap!r}")
        if scope not in _VALID_SCOPE_STRINGS:
            raise ValueError(
                f"introspect entries[{i}].scope is not a canonical "
                f"cache_scope (request|session|runtime|app): {scope!r}"
            )

        parsed.append(
            CacheEntrySnapshot(
                capability=cap,
                scope=scope,
                entries=entry["entries"],
                bytes=entry["bytes"],
                hit=entry["hit"],
                miss=entry["miss"],
            )
        )

    return CacheSnapshot(version=version, is_stub=is_stub, entries=parsed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clear_all(runtime_handle: Any) -> None:
    """Clear ALL caches across ALL capabilities for this runtime handle.

    Idempotent — calling when caches are empty always returns normally.

    Args:
        runtime_handle: A live ``oct_runtime_t*`` cffi pointer obtained
            from :func:`~octomil.runtime.native.loader.NativeRuntime.open`.

    Raises:
        CacheNotImplementedError: The cache API stub is not yet wired
            (``OCT_STATUS_UNSUPPORTED``). This is expected until Lanes
            B/C/F land.
        NativeRuntimeError: Any other ABI error.
        ValueError: ``runtime_handle`` is None.

    Capability ``cache.introspect`` is registered in
    octomil-contracts v1.25.0 (#129); cache scope codes match
    ``enums/cache_scope.yaml`` from v1.24.0 (#123).
    """
    if runtime_handle is None:
        raise ValueError("clear_all: runtime_handle is None")
    _, lib = _get_lib()
    status = int(lib.oct_runtime_cache_clear_all(runtime_handle))
    _check_status(status, "clear_all", runtime_handle=runtime_handle)


def clear_capability(runtime_handle: Any, capability_name: str) -> None:
    """Clear caches for a single capability.

    Args:
        runtime_handle: A live ``oct_runtime_t*`` cffi pointer.
        capability_name: A canonical capability string, e.g.
            ``"chat.completion"`` or ``"audio.transcription"``.

    Raises:
        CacheNotImplementedError: Stub not yet wired.
        NativeRuntimeError: Any other ABI error.
        ValueError: ``runtime_handle`` is None or ``capability_name``
            is empty.

    Note:
        The runtime returns ``OCT_STATUS_NOT_FOUND`` when the capability
        name is canonical but no cache is registered for it.  This is
        treated as "nothing to clear" and does NOT raise an exception.

    Capability ``cache.introspect`` is registered in
    octomil-contracts v1.25.0 (#129); cache scope codes match
    ``enums/cache_scope.yaml`` from v1.24.0 (#123).
    """
    if runtime_handle is None:
        raise ValueError("clear_capability: runtime_handle is None")
    if not capability_name:
        raise ValueError("clear_capability: capability_name must be non-empty")
    _, lib = _get_lib()
    cap_bytes = capability_name.encode("utf-8")
    status = int(lib.oct_runtime_cache_clear_capability(runtime_handle, cap_bytes))
    _check_status(
        status,
        f"clear_capability({capability_name!r})",
        runtime_handle=runtime_handle,
        allow_not_found=True,
    )


def clear_scope(runtime_handle: Any, scope: int) -> None:
    """Clear cache entries matching a scope level.

    Broader scopes subsume narrower ones::

        SCOPE_APP > SCOPE_RUNTIME > SCOPE_SESSION > SCOPE_REQUEST

    A ``SCOPE_RUNTIME`` clear also clears all ``SCOPE_SESSION`` and
    ``SCOPE_REQUEST`` entries.

    Args:
        runtime_handle: A live ``oct_runtime_t*`` cffi pointer.
        scope: One of :data:`SCOPE_REQUEST`, :data:`SCOPE_SESSION`,
            :data:`SCOPE_RUNTIME`, :data:`SCOPE_APP`.

    Raises:
        CacheNotImplementedError: Stub not yet wired.
        NativeRuntimeError: Any other ABI error.
        ValueError: ``runtime_handle`` is None or ``scope`` is not a
            valid scope constant.

    Capability ``cache.introspect`` is registered in
    octomil-contracts v1.25.0 (#129); cache scope codes match
    ``enums/cache_scope.yaml`` from v1.24.0 (#123).
    """
    if runtime_handle is None:
        raise ValueError("clear_scope: runtime_handle is None")
    if scope not in _VALID_SCOPES:
        raise ValueError(
            f"clear_scope: scope {scope!r} is not a valid OCT_CACHE_SCOPE_* "
            f"constant; expected one of {sorted(_VALID_SCOPES)}"
        )
    _, lib = _get_lib()
    status = int(lib.oct_runtime_cache_clear_scope(runtime_handle, scope))
    _check_status(status, f"clear_scope({scope})", runtime_handle=runtime_handle)


def introspect(runtime_handle: Any) -> CacheSnapshot:
    """Return a privacy-safe snapshot of current cache state.

    The returned :class:`CacheSnapshot` contains **only** bounded
    numeric fields: ``capability``, ``scope``, ``entries``, ``bytes``,
    ``hit``, ``miss``.  No cache keys, no hashed keys, no token IDs,
    no prompt text, no audio bytes, no file paths, and no model paths
    appear in the output.

    While the stubs are active (``is_stub=True``), the returned
    snapshot has an empty ``entries`` list.  Real data appears once
    Lanes B/C/F wire their cache implementations.

    Args:
        runtime_handle: A live ``oct_runtime_t*`` cffi pointer.

    Returns:
        A :class:`CacheSnapshot` with bounded fields only.

    Raises:
        CacheNotImplementedError: Stub not yet wired (the snapshot is
            still returned with ``is_stub=True`` — callers can inspect
            the stub shape even in this state).
        NativeRuntimeError: ABI returned a hard error (not UNSUPPORTED).
        ValueError: ``runtime_handle`` is None, or the ABI emitted JSON
            that fails the privacy schema check.

    Capability ``cache.introspect`` is registered in
    octomil-contracts v1.25.0 (#129); cache scope codes match
    ``enums/cache_scope.yaml`` from v1.24.0 (#123).
    """
    if runtime_handle is None:
        raise ValueError("introspect: runtime_handle is None")
    ffi, lib = _get_lib()
    buf = ffi.new(f"char[{_INTROSPECT_BUF_BYTES}]")
    status = int(lib.oct_runtime_cache_introspect(runtime_handle, buf, _INTROSPECT_BUF_BYTES))
    raw = ffi.string(buf).decode("utf-8", errors="replace")

    # B4 fix (Codex post-hoc Lane G sweep): hard ABI errors must surface
    # as NativeRuntimeError, NOT as ValueError from a JSON parse against
    # an empty/garbage buffer. The runtime ABI contract for INVALID_INPUT
    # writes either nothing or a NUL to out_json_buf — calling
    # _parse_snapshot('') here would mask the real status as
    # "invalid JSON" and lose the OCT_STATUS_INVALID_INPUT signal.
    # Status check ordering: hard errors first, then UNSUPPORTED (stub),
    # then OK (parse + return).
    if status not in (OCT_STATUS_OK, OCT_STATUS_UNSUPPORTED):
        raise NativeRuntimeError(
            status,
            f"introspect: ABI returned status={status} (not OK / not UNSUPPORTED)",
        )

    # Parse + validate the bounded JSON. Privacy-schema check runs on
    # stub output too so the assertion is exercised before real caches
    # land.
    snapshot = _parse_snapshot(raw)

    if status == OCT_STATUS_UNSUPPORTED:
        # Stub path: snapshot is valid, entries is empty, is_stub=True.
        # Raise CacheNotImplementedError but carry the snapshot so the
        # caller can inspect the stub shape (e.g. in privacy tests).
        err = CacheNotImplementedError(
            "introspect: cache API stub — no real caches registered yet (OCT_STATUS_UNSUPPORTED); snapshot.is_stub=True"
        )
        err.snapshot = snapshot  # type: ignore[attr-defined]
        raise err
    return snapshot
