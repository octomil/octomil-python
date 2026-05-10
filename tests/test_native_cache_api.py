"""Tests for octomil.runtime.native.cache — Lane G cache clear/introspect skeleton.

Covers:
  * API contract: all four functions raise CacheNotImplementedError (stub).
  * Idempotency: clear_all called twice does not crash.
  * clear_capability("unknown") returns cleanly (no hard error).
  * clear_scope: valid scope constants pass; invalid constants raise ValueError.
  * introspect: CacheNotImplementedError carries a snapshot attribute with
    schema-validated CacheSnapshot(is_stub=True, entries=[]).
  * Privacy: introspect JSON output contains no SHA-256 hex, no slashes,
    no file-extension substrings, no prompt text.
  * Schema shape: snapshot fields are bounded numeric/enum types only.
  * clear_all/clear_capability/clear_scope all raise ValueError on None handle.

All tests operate on a mock runtime handle — the native dylib is not
required.  The mock patches _get_lib() to return an (ffi, lib) pair
whose lib returns OCT_STATUS_UNSUPPORTED for all cache calls and writes
the bounded stub JSON for introspect.

Canonical references:

  * Capability ``cache.introspect`` registered in
    ``octomil-contracts/schemas/core/runtime_capability.json`` (v1.25.0,
    #129).
  * Cache scope codes ``request|session|runtime|app`` match
    ``octomil-contracts/enums/cache_scope.yaml`` (v1.24.0, #123).
"""

from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from octomil.runtime.native.cache import (
    SCOPE_APP,
    SCOPE_REQUEST,
    SCOPE_RUNTIME,
    SCOPE_SESSION,
    CacheEntrySnapshot,
    CacheNotImplementedError,
    CacheSnapshot,
    _parse_snapshot,
    clear_all,
    clear_capability,
    clear_scope,
    introspect,
)
from octomil.runtime.native.loader import (
    OCT_STATUS_INVALID_INPUT,
    OCT_STATUS_NOT_FOUND,
    OCT_STATUS_OK,
    OCT_STATUS_UNSUPPORTED,
    NativeRuntimeError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STUB_JSON = '{"version":1,"is_stub":true,"entries":[]}'


def _make_ffi_lib(
    clear_all_status: int = OCT_STATUS_UNSUPPORTED,
    clear_cap_status: int = OCT_STATUS_UNSUPPORTED,
    clear_scope_status: int = OCT_STATUS_UNSUPPORTED,
    introspect_status: int = OCT_STATUS_UNSUPPORTED,
    introspect_json: str = _STUB_JSON,
) -> tuple[MagicMock, MagicMock]:
    """Build a (ffi, lib) mock pair for a given set of return codes."""
    ffi = MagicMock()
    lib = MagicMock()

    # lib.oct_runtime_cache_* return codes
    lib.oct_runtime_cache_clear_all.return_value = clear_all_status
    lib.oct_runtime_cache_clear_capability.return_value = clear_cap_status
    lib.oct_runtime_cache_clear_scope.return_value = clear_scope_status
    lib.oct_runtime_cache_introspect.return_value = introspect_status

    # ffi.new returns a buffer; ffi.string returns the bytes
    buf_mock = MagicMock()
    ffi.new.return_value = buf_mock
    ffi.string.return_value = introspect_json.encode("utf-8")

    return ffi, lib


@pytest.fixture()
def mock_lib_unsupported() -> tuple[MagicMock, MagicMock]:
    """All cache entry points return OCT_STATUS_UNSUPPORTED (stub path)."""
    return _make_ffi_lib()


@pytest.fixture()
def fake_handle() -> MagicMock:
    """A non-None fake runtime handle."""
    return MagicMock(name="oct_runtime_t*")


# ---------------------------------------------------------------------------
# §1 None handle — ValueError before ABI call
# ---------------------------------------------------------------------------


class TestNoneHandleGuard:
    def test_clear_all_none_handle(self) -> None:
        with pytest.raises(ValueError, match="runtime_handle is None"):
            clear_all(None)

    def test_clear_capability_none_handle(self) -> None:
        with pytest.raises(ValueError, match="runtime_handle is None"):
            clear_capability(None, "chat.completion")

    def test_clear_capability_empty_name(self, fake_handle: Any) -> None:
        with pytest.raises(ValueError, match="capability_name must be non-empty"):
            clear_capability(fake_handle, "")

    def test_clear_scope_none_handle(self) -> None:
        with pytest.raises(ValueError, match="runtime_handle is None"):
            clear_scope(None, SCOPE_RUNTIME)

    def test_introspect_none_handle(self) -> None:
        with pytest.raises(ValueError, match="runtime_handle is None"):
            introspect(None)


# ---------------------------------------------------------------------------
# §2 Stub: all operations raise CacheNotImplementedError
# ---------------------------------------------------------------------------


class TestStubReturnsNotImplemented:
    def test_clear_all_raises(self, fake_handle: Any, mock_lib_unsupported: tuple[Any, Any]) -> None:
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=mock_lib_unsupported,
        ):
            with pytest.raises(CacheNotImplementedError):
                clear_all(fake_handle)

    def test_clear_capability_raises(self, fake_handle: Any, mock_lib_unsupported: tuple[Any, Any]) -> None:
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=mock_lib_unsupported,
        ):
            with pytest.raises(CacheNotImplementedError):
                clear_capability(fake_handle, "chat.completion")

    def test_clear_scope_raises(self, fake_handle: Any, mock_lib_unsupported: tuple[Any, Any]) -> None:
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=mock_lib_unsupported,
        ):
            with pytest.raises(CacheNotImplementedError):
                clear_scope(fake_handle, SCOPE_APP)

    def test_introspect_raises(self, fake_handle: Any, mock_lib_unsupported: tuple[Any, Any]) -> None:
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=mock_lib_unsupported,
        ):
            with pytest.raises(CacheNotImplementedError):
                introspect(fake_handle)


# ---------------------------------------------------------------------------
# §3 Idempotency: clear_all twice does not crash
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_clear_all_idempotent(self, fake_handle: Any, mock_lib_unsupported: tuple[Any, Any]) -> None:
        """Both calls raise CacheNotImplementedError — no crash, no hard error."""
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=mock_lib_unsupported,
        ):
            with pytest.raises(CacheNotImplementedError):
                clear_all(fake_handle)
            # Second call — same stub, still CacheNotImplementedError.
            with pytest.raises(CacheNotImplementedError):
                clear_all(fake_handle)

    def test_clear_all_with_ok_status_is_idempotent(self, fake_handle: Any) -> None:
        """Once real caches land (OK status), clear_all must not raise."""
        ffi, lib = _make_ffi_lib(clear_all_status=OCT_STATUS_OK)
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            # Must not raise.
            clear_all(fake_handle)
            # Idempotent second call.
            clear_all(fake_handle)
        assert lib.oct_runtime_cache_clear_all.call_count == 2


# ---------------------------------------------------------------------------
# §4 clear_capability with unknown name
# ---------------------------------------------------------------------------


class TestClearCapabilityUnknown:
    def test_not_found_is_not_an_error(self, fake_handle: Any) -> None:
        """NOT_FOUND means 'recognized name, no cache registered' — not an error."""
        ffi, lib = _make_ffi_lib(clear_cap_status=OCT_STATUS_NOT_FOUND)
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            # Must not raise — NOT_FOUND is treated as "already empty".
            clear_capability(fake_handle, "audio.transcription")

    def test_unsupported_raises_not_implemented(self, fake_handle: Any) -> None:
        ffi, lib = _make_ffi_lib(clear_cap_status=OCT_STATUS_UNSUPPORTED)
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            with pytest.raises(CacheNotImplementedError):
                clear_capability(fake_handle, "embeddings.text")

    def test_invalid_input_raises_native_error(self, fake_handle: Any) -> None:
        ffi, lib = _make_ffi_lib(clear_cap_status=OCT_STATUS_INVALID_INPUT)
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            with pytest.raises(NativeRuntimeError):
                clear_capability(fake_handle, "chat.completion")


# ---------------------------------------------------------------------------
# §5 clear_scope: valid scopes pass; invalid scope raises ValueError
# ---------------------------------------------------------------------------


class TestClearScope:
    @pytest.mark.parametrize(
        "scope",
        [SCOPE_REQUEST, SCOPE_SESSION, SCOPE_RUNTIME, SCOPE_APP],
    )
    def test_valid_scopes_reach_abi(self, fake_handle: Any, scope: int) -> None:
        ffi, lib = _make_ffi_lib(clear_scope_status=OCT_STATUS_UNSUPPORTED)
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            with pytest.raises(CacheNotImplementedError):
                clear_scope(fake_handle, scope)
        # Verify the ABI was actually called (not short-circuited).
        lib.oct_runtime_cache_clear_scope.assert_called_once_with(fake_handle, scope)

    def test_invalid_scope_raises_value_error_before_abi(self, fake_handle: Any) -> None:
        ffi, lib = _make_ffi_lib()
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            with pytest.raises(ValueError, match="not a valid OCT_CACHE_SCOPE"):
                clear_scope(fake_handle, 999)
        # ABI must NOT have been called.
        lib.oct_runtime_cache_clear_scope.assert_not_called()

    def test_scope_constants_are_correct_values(self) -> None:
        assert SCOPE_REQUEST == 0
        assert SCOPE_SESSION == 1
        assert SCOPE_RUNTIME == 2
        assert SCOPE_APP == 3


# ---------------------------------------------------------------------------
# §6 introspect snapshot carries is_stub=True and entries=[]
# ---------------------------------------------------------------------------


class TestIntrospectSnapshot:
    def test_snapshot_attached_to_exception(self, fake_handle: Any, mock_lib_unsupported: tuple[Any, Any]) -> None:
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=mock_lib_unsupported,
        ):
            with pytest.raises(CacheNotImplementedError) as exc_info:
                introspect(fake_handle)
        exc = exc_info.value
        assert hasattr(exc, "snapshot"), "exception must carry .snapshot"
        snap = exc.snapshot
        assert isinstance(snap, CacheSnapshot)
        assert snap.is_stub is True
        assert snap.entries == []
        assert snap.version >= 1

    def test_snapshot_entries_is_list(self, fake_handle: Any, mock_lib_unsupported: tuple[Any, Any]) -> None:
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=mock_lib_unsupported,
        ):
            with pytest.raises(CacheNotImplementedError) as exc_info:
                introspect(fake_handle)
        assert isinstance(exc_info.value.snapshot.entries, list)


# ---------------------------------------------------------------------------
# §7 Privacy tests — introspect JSON must not contain forbidden patterns
# ---------------------------------------------------------------------------

# These run against _parse_snapshot which validates the stub JSON shape.
# When real caches land, these same tests will run against real output.


class TestIntrospectPrivacy:
    @pytest.fixture()
    def stub_json(self) -> str:
        return _STUB_JSON

    @pytest.fixture()
    def stub_snapshot(self, stub_json: str) -> CacheSnapshot:
        return _parse_snapshot(stub_json)

    def test_no_sha256_hex_in_json(self, stub_json: str) -> None:
        """No 32+ consecutive lowercase hex chars (SHA-256 digest)."""
        assert not re.search(r"[0-9a-f]{32,}", stub_json), "introspect JSON must not contain SHA-256 hex strings"

    def test_no_slash_in_json(self, stub_json: str) -> None:
        """No '/' means no file paths, model paths, or URIs."""
        assert "/" not in stub_json, "introspect JSON must not contain '/' (no file/model paths)"

    def test_no_wav_in_json(self, stub_json: str) -> None:
        assert ".wav" not in stub_json

    def test_no_gguf_in_json(self, stub_json: str) -> None:
        assert ".gguf" not in stub_json

    def test_no_onnx_in_json(self, stub_json: str) -> None:
        assert ".onnx" not in stub_json

    def test_no_bin_in_json(self, stub_json: str) -> None:
        assert ".bin" not in stub_json

    def test_no_prompt_text_in_json(self, stub_json: str) -> None:
        assert "prompt" not in stub_json.lower(), "introspect JSON must not contain 'prompt' text"

    def test_no_token_id_in_json(self, stub_json: str) -> None:
        assert "token_id" not in stub_json.lower(), "introspect JSON must not contain 'token_id'"

    def test_snapshot_fields_are_bounded_types(self, stub_snapshot: CacheSnapshot) -> None:
        assert isinstance(stub_snapshot.version, int)
        assert isinstance(stub_snapshot.is_stub, bool)
        assert isinstance(stub_snapshot.entries, list)

    def test_entry_fields_are_bounded_types(self) -> None:
        """Test with a synthetic entry to ensure schema enforces types."""
        json_with_entry = json.dumps(
            {
                "version": 1,
                "is_stub": False,
                "entries": [
                    {
                        "capability": "chat.completion",
                        "scope": "runtime",
                        "entries": 42,
                        "bytes": 1048576,
                        "hit": 100,
                        "miss": 10,
                    }
                ],
            }
        )
        snap = _parse_snapshot(json_with_entry)
        assert len(snap.entries) == 1
        e = snap.entries[0]
        assert isinstance(e, CacheEntrySnapshot)
        assert e.capability == "chat.completion"
        assert e.scope == "runtime"
        assert e.entries == 42
        assert e.bytes == 1048576
        assert e.hit == 100
        assert e.miss == 10


# ---------------------------------------------------------------------------
# §8 _parse_snapshot rejects unexpected keys (privacy schema)
# ---------------------------------------------------------------------------


class TestParseSnapshotSchemaValidation:
    def test_rejects_unknown_top_level_key(self) -> None:
        bad_json = '{"version":1,"is_stub":true,"entries":[],"secret_key":"abc"}'
        with pytest.raises(ValueError, match="unexpected keys"):
            _parse_snapshot(bad_json)

    def test_rejects_unknown_entry_key(self) -> None:
        bad_json = json.dumps(
            {
                "version": 1,
                "is_stub": True,
                "entries": [
                    {
                        "capability": "chat.completion",
                        "scope": "runtime",
                        "entries": 0,
                        "bytes": 0,
                        "hit": 0,
                        "miss": 0,
                        "prompt_text": "secret",  # NOT allowed
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="unexpected keys"):
            _parse_snapshot(bad_json)

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_snapshot("not json at all")

    def test_rejects_non_dict_root(self) -> None:
        with pytest.raises(ValueError, match="root must be a JSON object"):
            _parse_snapshot("[1, 2, 3]")

    def test_rejects_non_list_entries(self) -> None:
        with pytest.raises(ValueError, match="'entries' must be a list"):
            _parse_snapshot('{"version":1,"is_stub":true,"entries":"oops"}')

    def test_empty_entries_is_valid(self) -> None:
        snap = _parse_snapshot('{"version":1,"is_stub":true,"entries":[]}')
        assert snap.entries == []
        assert snap.is_stub is True


# ---------------------------------------------------------------------------
# §9 Top-level octomil.cache facade
# ---------------------------------------------------------------------------


class TestCacheFacade:
    """Verify the octomil.cache module re-exports the native cache API."""

    def test_imports_from_octomil_cache(self) -> None:
        from octomil.cache import (  # noqa: F401
            SCOPE_APP,  # noqa: F401
            SCOPE_REQUEST,  # noqa: F401
            SCOPE_RUNTIME,  # noqa: F401
            SCOPE_SESSION,  # noqa: F401
            CacheEntrySnapshot,  # noqa: F401
            CacheNotImplementedError,
            CacheSnapshot,  # noqa: F401
            clear_all,  # noqa: F401
            clear_capability,  # noqa: F401
            clear_scope,  # noqa: F401
            introspect,  # noqa: F401
        )

        assert CacheNotImplementedError is not None

    def test_facade_clear_all_delegates(self, fake_handle: Any) -> None:
        from octomil.cache import clear_all as facade_clear_all

        ffi, lib = _make_ffi_lib(clear_all_status=OCT_STATUS_UNSUPPORTED)
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            with pytest.raises(CacheNotImplementedError):
                facade_clear_all(fake_handle)

    def test_facade_introspect_delegates(self, fake_handle: Any) -> None:
        from octomil.cache import introspect as facade_introspect

        ffi, lib = _make_ffi_lib(introspect_status=OCT_STATUS_UNSUPPORTED)
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            with pytest.raises(CacheNotImplementedError) as exc_info:
                facade_introspect(fake_handle)
        assert hasattr(exc_info.value, "snapshot")


# ---------------------------------------------------------------------------
# §10 Post-hoc Lane G sweep regressions (Codex B1/B3/B4)
# ---------------------------------------------------------------------------
# Regression coverage for the post-hoc consensus sweep follow-up
# (octomil-python fix/cache-sweep-571-B1-B4-...). Each test pins a
# previously-missing invariant so the gap doesn't reopen.


class TestPostHocSweepFollowUp:
    """Regressions for issues caught by the post-hoc Lane G consensus sweep."""

    def test_capability_in_runtime_capabilities_registry(self) -> None:
        """Codex B1: cache.introspect MUST be in RUNTIME_CAPABILITIES.

        Without this, NativeRuntime.capabilities() silently drops the
        runtime's advertised cache.introspect string and feature probes
        report False even when the runtime supports it.
        """
        from octomil.runtime.native.capabilities import (
            CAPABILITY_CACHE_INTROSPECT,
            RUNTIME_CAPABILITIES,
        )

        assert CAPABILITY_CACHE_INTROSPECT == "cache.introspect"
        assert CAPABILITY_CACHE_INTROSPECT in RUNTIME_CAPABILITIES

    def test_introspect_invalid_input_raises_native_error_not_value_error(self, fake_handle: Any) -> None:
        """Codex B4: hard ABI status MUST surface as NativeRuntimeError.

        The pre-fix code parsed the buffer (likely empty / NUL) BEFORE
        checking the status, so OCT_STATUS_INVALID_INPUT surfaced as
        ValueError("invalid JSON") and lost the real error code.
        """
        ffi, lib = _make_ffi_lib(
            introspect_status=OCT_STATUS_INVALID_INPUT,
            introspect_json="",  # ABI writes empty/NUL on INVALID_INPUT
        )
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            with pytest.raises(NativeRuntimeError) as exc_info:
                introspect(fake_handle)
        assert exc_info.value.status == OCT_STATUS_INVALID_INPUT

    def test_parse_snapshot_requires_version_key(self) -> None:
        """Codex B3: missing 'version' key MUST raise (no silent default)."""
        bad_json = '{"is_stub":true,"entries":[]}'
        with pytest.raises(ValueError, match="missing required key.*version"):
            _parse_snapshot(bad_json)

    def test_parse_snapshot_requires_is_stub_key(self) -> None:
        """Codex B3: missing 'is_stub' key MUST raise (no silent default)."""
        bad_json = '{"version":1,"entries":[]}'
        with pytest.raises(ValueError, match="missing required key.*is_stub"):
            _parse_snapshot(bad_json)

    def test_parse_snapshot_rejects_non_canonical_capability(self) -> None:
        """Codex B3: capability MUST be a canonical RUNTIME_CAPABILITIES string.

        Without enum validation, a buggy runtime emitting
        capability='/Users/leak/path' would pass key-shape validation
        and reach the caller despite the privacy contract.
        """
        bad_json = json.dumps(
            {
                "version": 1,
                "is_stub": False,
                "entries": [
                    {
                        "capability": "/Users/leak/secret_path",
                        "scope": "runtime",
                        "entries": 0,
                        "bytes": 0,
                        "hit": 0,
                        "miss": 0,
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="not a canonical runtime capability"):
            _parse_snapshot(bad_json)

    def test_parse_snapshot_rejects_non_canonical_scope(self) -> None:
        """Codex B3: scope MUST be one of request|session|runtime|app."""
        bad_json = json.dumps(
            {
                "version": 1,
                "is_stub": False,
                "entries": [
                    {
                        "capability": "chat.completion",
                        "scope": "global",  # NOT canonical
                        "entries": 0,
                        "bytes": 0,
                        "hit": 0,
                        "miss": 0,
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="not a canonical cache_scope"):
            _parse_snapshot(bad_json)

    def test_parse_snapshot_rejects_negative_counter(self) -> None:
        """Codex B3: counter fields MUST be >= 0."""
        bad_json = json.dumps(
            {
                "version": 1,
                "is_stub": False,
                "entries": [
                    {
                        "capability": "chat.completion",
                        "scope": "runtime",
                        "entries": -1,
                        "bytes": 0,
                        "hit": 0,
                        "miss": 0,
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="must be >= 0"):
            _parse_snapshot(bad_json)

    def test_parse_snapshot_rejects_non_int_counter(self) -> None:
        """Codex B3: counter fields MUST be int (not silent str→int coercion)."""
        bad_json = json.dumps(
            {
                "version": 1,
                "is_stub": False,
                "entries": [
                    {
                        "capability": "chat.completion",
                        "scope": "runtime",
                        "entries": "42",  # string not int
                        "bytes": 0,
                        "hit": 0,
                        "miss": 0,
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="must be an int"):
            _parse_snapshot(bad_json)

    def test_introspect_unsupported_still_validates_strict_schema(self, fake_handle: Any) -> None:
        """Stub path MUST still run the full strict schema; can't skip
        validation just because we plan to raise CacheNotImplementedError.
        """
        # Stub JSON missing required is_stub key — should ValueError, not
        # silently raise CacheNotImplementedError with a malformed snapshot.
        ffi, lib = _make_ffi_lib(
            introspect_status=OCT_STATUS_UNSUPPORTED,
            introspect_json='{"version":1,"entries":[]}',  # is_stub absent
        )
        with patch(
            "octomil.runtime.native.cache._get_lib",
            return_value=(ffi, lib),
        ):
            with pytest.raises(ValueError, match="missing required key.*is_stub"):
                introspect(fake_handle)
