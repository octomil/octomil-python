"""Tests for ``octomil.runtime.native`` cffi loader.

Consumes an external ``liboctomil-runtime`` artifact resolved via:

  1. ``OCTOMIL_RUNTIME_DYLIB`` env var (operator override).
  2. The dev cache populated by ``scripts/fetch_runtime_dev.py``
     under ``~/.cache/octomil-runtime/<version>/lib/``.

The runtime source no longer lives in this repo — see the private
``octomil-runtime`` repo. Tests skip cleanly when neither resolution
path produces a dylib (CI without a token, or a contributor who
hasn't run the fetch script). Unit tests not requiring the dylib
remain green either way.

Exercises:
  * Version handshake.
  * ``oct_runtime_open`` v1 success + v0 / v2 / NULL-out error paths.
  * ``oct_runtime_close`` + idempotent context-manager close.
  * ``oct_runtime_capabilities`` honors out->size, returns empty
    sentinel arrays from the stub.
  * Forward-compat: capabilities() drops unknown advertised strings.
  * Thread-error and runtime-error read-back paths.
  * Slice 2A: ABI parity for the session-level structs
    (oct_session_config_t, oct_audio_view_t, oct_event_t) plus
    NativeSession stub-behavior (open/send_audio/send_text/poll_event/
    cancel/close all surface OCT_STATUS_UNSUPPORTED cleanly).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Skip the entire module if cffi isn't available — this matches the
# `native` extra opt-in surface.
cffi = pytest.importorskip("cffi", reason="cffi extra not installed")  # noqa: F841


def _resolve_external_dylib() -> Path | None:
    """Return the dylib path resolved via env override or dev cache,
    or None when neither is populated. Used by test gates without
    forcing a session-wide skip — see the comment on the fixture
    restructure below."""
    from octomil.runtime.native import loader as _loader

    override = os.environ.get(_loader.ENV_DYLIB_OVERRIDE)
    if override:
        path = Path(override)
        if not path.is_file():
            pytest.fail(
                f"{_loader.ENV_DYLIB_OVERRIDE}={override!r} does not exist; " f"operator override is authoritative."
            )
        return path
    candidates = _loader._fetched_dylib_candidates()
    if candidates:
        return candidates[-1]
    return None


# Module-level resolution: do this once per session and cache so
# every fixture sees the same answer.
_EXTERNAL_DYLIB: Path | None = None


def _external_dylib() -> Path | None:
    global _EXTERNAL_DYLIB
    if _EXTERNAL_DYLIB is not None:
        return _EXTERNAL_DYLIB
    _EXTERNAL_DYLIB = _resolve_external_dylib()
    return _EXTERNAL_DYLIB


# Codex R1 blocker fix: previously the autouse `_isolate_loader`
# fixture took `runtime_dylib` as a parameter, so a `pytest.skip`
# in `runtime_dylib` cascaded to EVERY test — including the
# structural regression guards (no in-tree subtree, error-message
# regression, version sort key) that don't actually need a dylib.
# CI without a fetched dylib silently skipped the whole boundary
# enforcement. Now: tests that need a working runtime carry the
# `requires_runtime` marker (or are auto-marked by the heuristic
# in `pytest_collection_modifyitems` below); the autouse only
# applies env override + FFI reset for those.
@pytest.fixture(scope="session")
def runtime_dylib() -> Path:
    """Resolve an external ``liboctomil-runtime`` dylib. Skip cleanly
    when neither the env override nor the dev cache is populated.

    Populate the dev cache:    python scripts/fetch_runtime_dev.py
    Or pin a specific binary:  export OCTOMIL_RUNTIME_DYLIB=...
    """
    path = _external_dylib()
    if path is None:
        pytest.skip(
            "no external liboctomil-runtime available — set "
            "OCTOMIL_RUNTIME_DYLIB or run "
            "`python scripts/fetch_runtime_dev.py` to populate "
            "~/.cache/octomil-runtime/."
        )
        raise AssertionError("unreachable: pytest.skip raises")
    return path


# NOTE: marker registration + collection auto-marker live in
# tests/conftest.py — pytest doesn't run the relevant hooks from
# test modules. The autouse fixture below reads the marker.
@pytest.fixture(autouse=True)
def _isolate_loader(request, monkeypatch):
    """Tests carrying ``@pytest.mark.requires_runtime`` (or
    auto-marked above) get the FFI/Lib singletons reset and the env
    override pointed at the resolved dylib; if no dylib is available
    they skip cleanly. Tests without the marker run unmodified —
    structural guards run regardless of whether a dylib is staged."""
    if request.node.get_closest_marker("requires_runtime") is None:
        yield
        return
    dylib = _external_dylib()
    if dylib is None:
        pytest.skip(
            "no external liboctomil-runtime available — set "
            "OCTOMIL_RUNTIME_DYLIB or run "
            "`python scripts/fetch_runtime_dev.py`."
        )
        return
    monkeypatch.setenv("OCTOMIL_RUNTIME_DYLIB", str(dylib))
    import octomil.runtime.native.loader as _loader

    monkeypatch.setattr(_loader, "_FFI", None)
    monkeypatch.setattr(_loader, "_LIB", None)
    yield


# ---------------------------------------------------------------------------
# Version handshake
# ---------------------------------------------------------------------------


def test_abi_version_returns_three_tuple():
    from octomil.runtime.native import abi_version

    major, minor, patch = abi_version()
    assert isinstance(major, int)
    assert isinstance(minor, int)
    assert isinstance(patch, int)
    # ABI v0.4 step 2: header bumped to MINOR=5 (additive — added
    # operational envelope on oct_event_t (APPENDED after union);
    # error_code field on OCT_EVENT_ERROR (APPENDED inside inner
    # struct); 10 runtime-scope event types (MODEL_LOADED/EVICTED,
    # CACHE_HIT/MISS, QUEUED, PREEMPTED, MEMORY_PRESSURE,
    # THERMAL_STATE, WATCHDOG_TIMEOUT, METRIC); session_config v=2
    # with appended request_id/route_id/trace_id/kv_prefix_key.
    # Existing readers stay 0.3/0.4-step-1-compat via versioned-
    # output size handshake.
    assert (major, minor) == (0, 5)
    assert patch == 0


# ---------------------------------------------------------------------------
# oct_runtime_open / _close
# ---------------------------------------------------------------------------


def test_open_v1_succeeds():
    from octomil.runtime.native import NativeRuntime

    rt = NativeRuntime.open()
    rt.close()


def test_open_with_artifact_root_succeeds():
    """The actual `oct_runtime_config_t` field is `artifact_root`,
    not the misnamed `cache_root` an earlier draft used. Codex R1
    blocker fix."""
    from octomil.runtime.native import NativeRuntime

    rt = NativeRuntime.open(artifact_root="/tmp/test-artifact-root", max_sessions=4)
    rt.close()


def test_open_idempotent_close():
    from octomil.runtime.native import NativeRuntime

    rt = NativeRuntime.open()
    rt.close()
    rt.close()  # second close is a no-op


# ---------------------------------------------------------------------------
# ABI struct-layout parity (Codex R1 blocker)
# ---------------------------------------------------------------------------


def test_oct_runtime_config_t_size_matches_runtime():
    """The cffi cdef MUST match the C compiler's view of
    `oct_runtime_config_t` exactly. ABI mode does NOT catch
    struct-layout mismatch at parse time; only the runtime-side
    reader would, and only when it touched a non-version field.
    Pin the size match here so future drift fails loudly."""
    from octomil.runtime.native.loader import _get_lib

    ffi, lib = _get_lib()
    cffi_size = ffi.sizeof("oct_runtime_config_t")
    runtime_size = int(lib.oct_runtime_config_size())
    assert cffi_size == runtime_size, (
        f"oct_runtime_config_t struct-layout drift: "
        f"cffi cdef sizeof={cffi_size}, C compiler sizeof={runtime_size}. "
        f"The Python cdef in loader.py has drifted from runtime.h."
    )


def test_oct_capabilities_t_size_matches_runtime():
    from octomil.runtime.native.loader import _get_lib

    ffi, lib = _get_lib()
    cffi_size = ffi.sizeof("oct_capabilities_t")
    runtime_size = int(lib.oct_capabilities_size())
    assert (
        cffi_size == runtime_size
    ), f"oct_capabilities_t struct-layout drift: cffi cdef sizeof={cffi_size}, C compiler sizeof={runtime_size}."


def test_oct_session_config_t_size_matches_runtime():
    """Slice 2A: pin struct-layout parity for oct_session_config_t.
    The cdef in loader.py mirrors runtime.h's oct_session_config_t
    field-for-field; if the runtime grows or reorders a field and
    the cdef doesn't follow, this fails immediately rather than
    crashing on the first non-version field touch."""
    from octomil.runtime.native.loader import _get_lib

    ffi, lib = _get_lib()
    cffi_size = ffi.sizeof("oct_session_config_t")
    runtime_size = int(lib.oct_session_config_size())
    assert (
        cffi_size == runtime_size
    ), f"oct_session_config_t struct-layout drift: cffi cdef sizeof={cffi_size}, C compiler sizeof={runtime_size}."


def test_oct_audio_view_t_size_matches_runtime():
    from octomil.runtime.native.loader import _get_lib

    ffi, lib = _get_lib()
    cffi_size = ffi.sizeof("oct_audio_view_t")
    runtime_size = int(lib.oct_audio_view_size())
    assert (
        cffi_size == runtime_size
    ), f"oct_audio_view_t struct-layout drift: cffi cdef sizeof={cffi_size}, C compiler sizeof={runtime_size}."


def test_oct_event_t_size_matches_runtime():
    """oct_event_t is the largest struct in the ABI (tagged union of
    every event payload). A cdef that omits one inner struct field
    would shrink the cffi sizeof; a runtime that grows a payload
    without a binding update would grow the C sizeof. Either way,
    this test is the canonical drift detector."""
    from octomil.runtime.native.loader import _get_lib

    ffi, lib = _get_lib()
    cffi_size = ffi.sizeof("oct_event_t")
    runtime_size = int(lib.oct_event_size())
    assert (
        cffi_size == runtime_size
    ), f"oct_event_t struct-layout drift: cffi cdef sizeof={cffi_size}, C compiler sizeof={runtime_size}."


def test_context_manager_closes_on_exit():
    from octomil.runtime.native import NativeRuntime

    with NativeRuntime.open() as rt:
        assert not rt._closed  # noqa: SLF001
    assert rt._closed  # noqa: SLF001


def test_capabilities_after_close_raises():
    from octomil.runtime.native import NativeRuntime, NativeRuntimeError

    rt = NativeRuntime.open()
    rt.close()
    with pytest.raises(NativeRuntimeError) as exc_info:
        rt.capabilities()
    assert "closed" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# oct_runtime_capabilities — slice-2 stub returns empty sentinel arrays
# ---------------------------------------------------------------------------


def test_capabilities_returns_empty_known_set_against_stub():
    """The slice-2 stub advertises NO capabilities. The harness MUST
    see an empty `supported_capabilities` tuple (forward-compat
    parsing — unknown advertised strings drop from the parsed view,
    but the stub doesn't advertise anything)."""
    from octomil.runtime.native import NativeRuntime

    with NativeRuntime.open() as rt:
        caps = rt.capabilities()
        assert caps.supported_capabilities == ()
        assert caps.supported_engines == ()
        assert caps.supported_archs == ()
        # Version field round-trips.
        assert caps.version == 1


def test_capabilities_claims_check_returns_false_for_canonical_names():
    """`requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)` is
    the slice-3 PR2 marker pattern. Against the stub, every check
    must return False — no native capabilities yet."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        CAPABILITY_AUDIO_TTS_STREAM,
        CAPABILITY_CHAT_COMPLETION,
        NativeRuntime,
    )

    with NativeRuntime.open() as rt:
        caps = rt.capabilities()
        assert not caps.claims(CAPABILITY_AUDIO_REALTIME_SESSION)
        assert not caps.claims(CAPABILITY_AUDIO_TTS_STREAM)
        assert not caps.claims(CAPABILITY_CHAT_COMPLETION)


# ---------------------------------------------------------------------------
# Last-error read-back
# ---------------------------------------------------------------------------


def test_thread_error_after_invalid_open():
    """Trigger a thread-scoped error path: NULL-out on
    oct_runtime_open. The runtime sets a thread-error string that
    `last_thread_error()` reads back."""
    from octomil.runtime.native.loader import (
        OCT_STATUS_INVALID_INPUT,
        NativeRuntimeError,
        _get_lib,
        last_thread_error,
    )

    ffi, lib = _get_lib()
    cfg = ffi.new("oct_runtime_config_t*")
    cfg.version = 1
    cfg.artifact_root = ffi.NULL
    cfg.telemetry_sink = ffi.NULL
    cfg.telemetry_user_data = ffi.NULL
    cfg.max_sessions = 0
    status = int(lib.oct_runtime_open(cfg, ffi.NULL))
    assert status == OCT_STATUS_INVALID_INPUT
    err = last_thread_error()
    assert "out parameter" in err.lower() or "out " in err.lower()
    # Constructor wrapper should also raise NativeRuntimeError on
    # configuration failure.
    _ = NativeRuntimeError  # noqa: F841 — imported for symbol-existence check


# ---------------------------------------------------------------------------
# Capabilities constants
# ---------------------------------------------------------------------------


def test_capabilities_constants_match_contract_set():
    """Sanity: the Python constants mirror the contract enum
    exactly. Drift here would make `requires_capability(...)`
    silently never match. v0.4: enum +6
    (octomil-contracts#99)."""
    from octomil.runtime.native import RUNTIME_CAPABILITIES

    expected = {
        # Slice 2A baseline (v0.3)
        "audio.realtime.session",
        "audio.stt.batch",
        "audio.stt.stream",
        "audio.transcription",
        "audio.tts.batch",
        "audio.tts.stream",
        "chat.completion",
        "chat.stream",
        # ABI v0.4 expansion
        "audio.diarization",
        "audio.speaker.embedding",
        "audio.vad",
        "embeddings.image",
        "embeddings.text",
        "index.vector.query",
    }
    assert set(RUNTIME_CAPABILITIES) == expected


def test_v0_4_capabilities_present():
    """ABI v0.4: 6 new capabilities admitted to the runtime enum.
    embeddings.text in particular replaces the v0.3 'intentionally
    absent' rule."""
    from octomil.runtime.native import RUNTIME_CAPABILITIES

    for cap in (
        "audio.diarization",
        "audio.speaker.embedding",
        "audio.vad",
        "embeddings.image",
        "embeddings.text",
        "index.vector.query",
    ):
        assert cap in RUNTIME_CAPABILITIES, f"v0.4 capability {cap!r} missing"


# ---------------------------------------------------------------------------
# Dylib resolution
# ---------------------------------------------------------------------------


def test_dylib_override_env_var_used(monkeypatch, tmp_path: Path):
    """The OCTOMIL_RUNTIME_DYLIB env var must take precedence over
    the dev-cache fallback. Set it to a non-existent path; loader
    should error pointing at the override path AND name the
    fetch-script fallback (NOT the deprecated in-tree build path)."""
    monkeypatch.setenv("OCTOMIL_RUNTIME_DYLIB", str(tmp_path / "does-not-exist.dylib"))
    import octomil.runtime.native.loader as loader

    monkeypatch.setattr(loader, "_FFI", None)
    monkeypatch.setattr(loader, "_LIB", None)
    with pytest.raises(ImportError) as exc_info:
        loader._build_lib()
    msg = str(exc_info.value)
    assert "does-not-exist.dylib" in msg
    assert "fetch_runtime_dev.py" in msg
    # Hard guard against regression to the in-tree fallback that this
    # PR removed. If this string ever reappears, the loader is
    # silently falling back into a deleted runtime-core subtree.
    assert "runtime-core" not in msg
    assert "BUILD.md" not in msg


def test_dylib_resolution_message_when_nothing_resolves(monkeypatch, tmp_path: Path):
    """When neither the env override nor the dev cache produces a
    dylib, the error must name both resolution paths explicitly so
    an operator can fix it without guessing."""
    monkeypatch.delenv("OCTOMIL_RUNTIME_DYLIB", raising=False)
    # Point the dev-cache root at an empty directory.
    import octomil.runtime.native.loader as loader

    monkeypatch.setattr(loader, "_FETCH_CACHE_ROOT", tmp_path / "empty-cache")
    monkeypatch.setattr(loader, "_FFI", None)
    monkeypatch.setattr(loader, "_LIB", None)

    with pytest.raises(ImportError) as exc_info:
        loader._resolve_dylib()
    msg = str(exc_info.value)
    assert "OCTOMIL_RUNTIME_DYLIB" in msg
    assert "fetch_runtime_dev.py" in msg
    # Same regression guard as above.
    assert "runtime-core" not in msg


def test_no_in_tree_runtime_core_subtree(monkeypatch):
    """Guardrail: octomil-python must NOT contain an in-tree
    runtime-core source subtree. The runtime source lives in the
    private octomil-runtime repo. This test fails loudly if anyone
    re-introduces the directory by accident or via a bad cherry-pick."""
    repo_root = Path(__file__).resolve().parent.parent
    forbidden = repo_root / "octomil" / "runtime-core"
    assert not forbidden.exists(), (
        f"in-tree runtime-core subtree found at {forbidden}. "
        f"Layer 2a runtime is owned by the private octomil-runtime "
        f"repo and consumed via OCTOMIL_RUNTIME_DYLIB / dev-cache."
    )


def test_version_sort_key_handles_double_digit_minor(tmp_path: Path):
    """Codex R1 missed-case: lexicographic sort would put v0.0.10
    BEFORE v0.0.2 and the most-recently-fetched-wins rule would
    pick the wrong release. Pin the parsed-tuple sort here so a
    revert to lex sort fails loudly."""
    from octomil.runtime.native.loader import _version_sort_key

    versions = [
        tmp_path / "v0.0.2",
        tmp_path / "v0.0.10",
        tmp_path / "v0.1.0",
        tmp_path / "v0.0.1-rc1",
        tmp_path / "v0.0.1",
    ]
    sorted_paths = sorted(versions, key=_version_sort_key)
    sorted_names = [p.name for p in sorted_paths]
    # v0.0.1-rc1 sorts before v0.0.1; v0.0.10 sorts AFTER v0.0.2.
    assert sorted_names == ["v0.0.1-rc1", "v0.0.1", "v0.0.2", "v0.0.10", "v0.1.0"], sorted_names


def test_resolve_dylib_returns_newest_cached_version(monkeypatch, tmp_path: Path):
    """Codex R2 blocker fix: `_resolve_dylib()` previously took
    `candidates[0]` (oldest) instead of `[-1]` (newest), defeating
    the version-tuple sort. Build a fake cache with v0.0.2 and
    v0.0.10 and confirm `_resolve_dylib()` picks v0.0.10."""
    monkeypatch.delenv("OCTOMIL_RUNTIME_DYLIB", raising=False)
    import octomil.runtime.native.loader as loader

    def make_cached_dylib(version: str, with_sentinel: bool = True) -> Path:
        version_dir = tmp_path / version
        lib = version_dir / "lib"
        lib.mkdir(parents=True)
        dylib = lib / "liboctomil-runtime.dylib"
        dylib.write_bytes(b"\x00")  # not actually loaded; just must exist
        if with_sentinel:
            (lib / loader._EXTRACTION_SENTINEL).write_text(version + "\n")
        return dylib

    older = make_cached_dylib("v0.0.2")
    newer = make_cached_dylib("v0.0.10")
    monkeypatch.setattr(loader, "_FETCH_CACHE_ROOT", tmp_path)

    resolved = loader._resolve_dylib()
    assert resolved == newer, (
        f"_resolve_dylib should pick the newest cached version, got {resolved}; "
        f"older was {older}, newer was {newer}"
    )


def test_resolve_dylib_skips_cache_without_sentinel(monkeypatch, tmp_path: Path):
    """Codex R3 blocker fix: a partial/corrupt extraction can leave
    a dylib on disk WITHOUT the `.extracted-ok` sentinel that the
    fetch script writes only after a full successful extraction.
    The loader MUST refuse such caches. Otherwise the SDK loads a
    truncated artifact on the next import.

    Set up: a `v0.0.10` cache with a dylib but NO sentinel; confirm
    `_resolve_dylib()` raises ImportError instead of loading it."""
    monkeypatch.delenv("OCTOMIL_RUNTIME_DYLIB", raising=False)
    import octomil.runtime.native.loader as loader

    version_dir = tmp_path / "v0.0.10"
    lib = version_dir / "lib"
    lib.mkdir(parents=True)
    (lib / "liboctomil-runtime.dylib").write_bytes(b"\x00")  # no sentinel
    monkeypatch.setattr(loader, "_FETCH_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(loader, "_FFI", None)
    monkeypatch.setattr(loader, "_LIB", None)

    with pytest.raises(ImportError) as exc_info:
        loader._resolve_dylib()
    msg = str(exc_info.value)
    assert "fetch_runtime_dev.py" in msg or "OCTOMIL_RUNTIME_DYLIB" in msg


def test_safe_extract_refuses_symlink_member(tmp_path: Path):
    """Codex R2 blocker fix: `_safe_extract` must refuse symlinks,
    hardlinks, and device entries (parity with Python 3.12's
    `filter='data'`). Pin via a synthetic tar containing a symlink
    that targets `/etc/passwd` — extraction must SystemExit."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib

    spec = importlib.util.spec_from_file_location(
        "fetch_runtime_dev",
        Path(__file__).resolve().parent.parent / "scripts" / "fetch_runtime_dev.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import tarfile as _tarfile

    bad_tar = tmp_path / "evil.tar.gz"
    with _tarfile.open(bad_tar, "w:gz") as tf:
        info = _tarfile.TarInfo("evil-link")
        info.type = _tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)

    target = tmp_path / "extract"
    target.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        mod._safe_extract(bad_tar, target)
    assert "link entry" in str(exc_info.value).lower() or "symlinks" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Slice 2A — NativeSession stub behavior
#
# Every session entry point on the slice-2 stub returns
# OCT_STATUS_UNSUPPORTED. These tests pin that behavior so the slice
# 2-proper Moshi-on-MLX adapter can replace stubs file-by-file with
# tests that fail loudly on regressions.
#
# Critical property: the stub MUST NEVER yield a fake event. Any
# poll_event call returns UNSUPPORTED (raised) before reaching the
# event-loop path; bindings that mistake "stub" for "working" because
# poll_event silently produced an OCT_EVENT_NONE would silently drop
# payloads in the slice 2-proper hand-off.
# ---------------------------------------------------------------------------


def test_session_open_unadvertised_capability_returns_unsupported():
    """Under v0.1.0, capabilities NOT advertised by the runtime
    (because no engine adapter is loadable for them) return
    UNSUPPORTED on session_open. `audio.realtime.session` is the
    Moshi capability; no engine ships it in v0.1.0, so it's
    always UNSUPPORTED."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        OCT_STATUS_UNSUPPORTED,
        NativeRuntime,
        NativeRuntimeError,
    )

    with NativeRuntime.open() as rt:
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION)
        assert exc_info.value.status == OCT_STATUS_UNSUPPORTED
        # last_error round-trips a descriptive runtime message — bindings
        # rely on this for diagnostic reporting in production.
        assert "session_open" in exc_info.value.last_error.lower()


def test_session_open_unknown_capability_under_stub_returns_unsupported():
    """Even for non-canonical capability strings the stub returns
    UNSUPPORTED (rather than crashing or producing a fake handle).
    Real runtime applies strict-reject against the canonical enum
    here; the stub shape stays consistent."""
    from octomil.runtime.native import OCT_STATUS_UNSUPPORTED, NativeRuntime, NativeRuntimeError

    with NativeRuntime.open() as rt:
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_session(capability="does.not.exist")
        assert exc_info.value.status == OCT_STATUS_UNSUPPORTED


def test_session_open_never_returns_a_handle_against_stub():
    """Bindings rely on `*out == NULL` after a non-OK session_open.
    Without it the wrapper would attempt to NativeSession.close() a
    garbage handle on the cleanup path — the C contract guarantees
    NULL on failure."""
    from octomil.runtime.native.loader import (
        OCT_STATUS_UNSUPPORTED,
        _get_lib,
    )

    ffi, lib = _get_lib()
    cfg = ffi.new("oct_runtime_config_t*")
    cfg.version = 1
    cfg.artifact_root = ffi.NULL
    cfg.telemetry_sink = ffi.NULL
    cfg.telemetry_user_data = ffi.NULL
    cfg.max_sessions = 0
    rt_out = ffi.new("oct_runtime_t**")
    assert int(lib.oct_runtime_open(cfg, rt_out)) == 0
    rt = rt_out[0]
    try:
        sess_cfg = ffi.new("oct_session_config_t*")
        sess_cfg.version = 1
        cap_buf = ffi.new("char[]", b"audio.realtime.session")
        sess_cfg.capability = cap_buf
        loc_buf = ffi.new("char[]", b"on_device")
        sess_cfg.locality = loc_buf
        sess_out = ffi.new("oct_session_t**")
        # Pre-poison sess_out so we'd notice if the stub fails to NULL it.
        sess_out[0] = ffi.cast("oct_session_t*", 0xDEADBEEF)
        status = int(lib.oct_session_open(rt, sess_cfg, sess_out))
        assert status == OCT_STATUS_UNSUPPORTED
        assert sess_out[0] == ffi.NULL, "stub session_open must NULL out on failure"
    finally:
        lib.oct_runtime_close(rt)


def test_session_send_audio_against_stub_returns_unsupported():
    """A NativeSession constructed against a NULL handle exercises
    the entry-point shim. The stub returns UNSUPPORTED; the wrapper
    raises NativeRuntimeError."""
    from octomil.runtime.native.loader import (
        OCT_STATUS_UNSUPPORTED,
        NativeRuntimeError,
        NativeSession,
        _get_lib,
    )

    ffi, lib = _get_lib()
    # Construct a NativeSession against NULL — the stub send_audio
    # explicitly handles this by setting a thread error and returning
    # UNSUPPORTED. We need a fake "owner" that only exposes last_error;
    # the simplest is a real (open) NativeRuntime.
    from octomil.runtime.native import NativeRuntime

    with NativeRuntime.open() as rt:
        sess = NativeSession(ffi, lib, ffi.NULL, owner=rt)
        # 4 frames of mono float32 — 16 bytes.
        samples = b"\x00\x00\x00\x00" * 4
        with pytest.raises(NativeRuntimeError) as exc_info:
            sess.send_audio(samples, sample_rate=16000, channels=1)
        assert exc_info.value.status == OCT_STATUS_UNSUPPORTED


def test_session_send_text_with_null_session_returns_invalid_input():
    """v0.1.0 tightened the NULL-session contract: send_text on a
    NULL session pointer returns INVALID_INPUT, not UNSUPPORTED.
    The previous slice-2A stub returned UNSUPPORTED uniformly;
    with real engines wired in, NULL-session is now a caller bug."""
    from octomil.runtime.native import NativeRuntime
    from octomil.runtime.native.loader import (
        NativeRuntimeError,
        NativeSession,
        _get_lib,
    )

    OCT_STATUS_INVALID_INPUT = 1  # from runtime.h

    ffi, lib = _get_lib()
    with NativeRuntime.open() as rt:
        sess = NativeSession(ffi, lib, ffi.NULL, owner=rt)
        with pytest.raises(NativeRuntimeError) as exc_info:
            sess.send_text("hello")
        assert exc_info.value.status == OCT_STATUS_INVALID_INPUT


def test_session_poll_event_against_stub_never_yields_fake_event():
    """The most-load-bearing stub-behavior assertion. The slice-2
    poll_event MUST NOT silently return a OCT_STATUS_OK event. If it
    did, a binding would mistake the stub for a working runtime and
    drop real payloads on the slice-2-proper handover. The stub
    returns UNSUPPORTED → wrapper raises."""
    from octomil.runtime.native import NativeRuntime
    from octomil.runtime.native.loader import (
        OCT_STATUS_UNSUPPORTED,
        NativeRuntimeError,
        NativeSession,
        _get_lib,
    )

    ffi, lib = _get_lib()
    with NativeRuntime.open() as rt:
        sess = NativeSession(ffi, lib, ffi.NULL, owner=rt)
        with pytest.raises(NativeRuntimeError) as exc_info:
            sess.poll_event(timeout_ms=0)
        assert exc_info.value.status == OCT_STATUS_UNSUPPORTED


def test_session_cancel_with_null_session_raises_invalid_input():
    """v0.1.0 tightened the NULL-session contract: cancel() on a
    NULL session raises NativeRuntimeError(INVALID_INPUT). The
    slice-2A "UNSUPPORTED is fine, return as status" idempotent
    semantic was appropriate when every session entry point was
    a stub; with real engines wired in, NULL-session at cancel
    is a programming error."""
    from octomil.runtime.native import NativeRuntime
    from octomil.runtime.native.loader import (
        NativeRuntimeError,
        NativeSession,
        _get_lib,
    )

    OCT_STATUS_INVALID_INPUT = 1

    ffi, lib = _get_lib()
    with NativeRuntime.open() as rt:
        sess = NativeSession(ffi, lib, ffi.NULL, owner=rt)
        with pytest.raises(NativeRuntimeError) as exc_info:
            sess.cancel()
        assert exc_info.value.status == OCT_STATUS_INVALID_INPUT


def test_session_close_idempotent():
    """close() is the cleanup path; it must be a no-op on the second
    call (mirrors NativeRuntime.close behavior)."""
    from octomil.runtime.native import NativeRuntime
    from octomil.runtime.native.loader import NativeSession, _get_lib

    ffi, lib = _get_lib()
    with NativeRuntime.open() as rt:
        sess = NativeSession(ffi, lib, ffi.NULL, owner=rt)
        sess.close()
        sess.close()


def test_session_send_audio_rejects_bad_shape():
    """Client-side shape validation — channels must divide n_floats,
    n_floats must be > 0. Surfacing OCT_STATUS_INVALID_INPUT before
    crossing the FFI is cheaper and gives a Python stack frame for
    the diagnostic."""
    from octomil.runtime.native import NativeRuntime
    from octomil.runtime.native.loader import (
        OCT_STATUS_INVALID_INPUT,
        NativeRuntimeError,
        NativeSession,
        _get_lib,
    )

    ffi, lib = _get_lib()
    with NativeRuntime.open() as rt:
        sess = NativeSession(ffi, lib, ffi.NULL, owner=rt)
        with pytest.raises(NativeRuntimeError) as exc_info:
            sess.send_audio(b"", sample_rate=16000, channels=1)
        assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
        with pytest.raises(NativeRuntimeError):
            # 5 floats but 2 channels — not a frame boundary.
            sess.send_audio(b"\x00" * 4 * 5, sample_rate=16000, channels=2)
        # Codex+Gemini R2 nit: trailing byte (len % 4 != 0) is rejected
        # rather than silently truncated. 5 bytes would have been parsed
        # as 1 float with a dropped trailing byte previously.
        with pytest.raises(NativeRuntimeError) as exc_info:
            sess.send_audio(b"\x00" * 5, sample_rate=16000, channels=1)
        assert "multiple of float32" in str(exc_info.value)


def test_runtime_close_invalidates_live_sessions():
    """Codex R1 blocker fix regression: per runtime.h, live sessions
    are CANCELLED and closed implicitly by oct_runtime_close. A
    Python NativeSession wrapper that survives that teardown must
    NOT call into the freed handle on its next operation or its
    __del__."""
    from octomil.runtime.native import (
        NativeRuntime,
        NativeRuntimeError,
        NativeSession,
    )
    from octomil.runtime.native.loader import OCT_STATUS_INVALID_INPUT, _get_lib

    ffi, lib = _get_lib()
    rt = NativeRuntime.open()
    # Construct a live NativeSession via the wrapper, registered with
    # the runtime. The slice-2 stub returns UNSUPPORTED on real opens,
    # so we synthesize the registration directly via the path that
    # NativeRuntime.open_session would take after a successful call.
    sess = NativeSession(ffi, lib, ffi.NULL, owner=rt)
    rt._sessions.add(sess)  # noqa: SLF001 — internal wiring under test

    rt.close()

    # After runtime close, every operation on the session wrapper
    # raises rather than dereferencing the freed handle.
    with pytest.raises(NativeRuntimeError) as exc_info:
        sess.send_text("after-close")
    assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
    assert "invalidated" in str(exc_info.value).lower()

    # close() on the invalidated session is a no-op (must NOT call
    # oct_session_close on a freed handle).
    sess.close()
    sess.close()

    # cancel() falls through to the idempotent CANCELLED return so
    # cleanup paths can call it without try/except.
    from octomil.runtime.native.loader import OCT_STATUS_CANCELLED

    assert sess.cancel() == OCT_STATUS_CANCELLED


def test_session_open_session_after_runtime_close_raises():
    """Ordering invariant: session_open after runtime_close fails
    cleanly via the wrapper rather than crashing on a stale handle."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        NativeRuntime,
        NativeRuntimeError,
    )

    rt = NativeRuntime.open()
    rt.close()
    with pytest.raises(NativeRuntimeError) as exc_info:
        rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION)
    assert "closed" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# ABI v0.4 step 1 — model lifecycle stub behavior
# ---------------------------------------------------------------------------


def test_oct_model_config_t_size_matches_runtime():
    """v0.4 step 1: pin struct-layout parity for oct_model_config_t.
    Same regression-detector pattern as session_config / audio_view /
    event."""
    from octomil.runtime.native.loader import _get_lib

    ffi, lib = _get_lib()
    cffi_size = ffi.sizeof("oct_model_config_t")
    runtime_size = int(lib.oct_model_config_size())
    assert (
        cffi_size == runtime_size
    ), f"oct_model_config_t struct-layout drift: cffi cdef sizeof={cffi_size}, C compiler sizeof={runtime_size}."


def test_open_model_against_stub_returns_unsupported():
    """v0.4 step 1: every model entry point returns UNSUPPORTED on
    the stub. NativeRuntime.open_model raises NativeRuntimeError."""
    from octomil.runtime.native import OCT_STATUS_UNSUPPORTED, NativeRuntime, NativeRuntimeError

    with NativeRuntime.open() as rt:
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_model(model_uri="local:///tmp/does-not-exist")
        assert exc_info.value.status == OCT_STATUS_UNSUPPORTED
        assert "model_open" in exc_info.value.last_error.lower()
        assert "v0.4 step 1" in exc_info.value.last_error.lower()


def test_open_model_after_runtime_close_raises():
    """Ordering invariant: open_model after runtime close raises a
    typed exception (binding-level invalidation; UB at C level)."""
    from octomil.runtime.native import NativeRuntime, NativeRuntimeError

    rt = NativeRuntime.open()
    rt.close()
    with pytest.raises(NativeRuntimeError) as exc_info:
        rt.open_model(model_uri="local:///tmp/x")
    assert "closed" in str(exc_info.value).lower()


def test_open_model_v0_returns_version_mismatch():
    """v0.4 step 1: model_open with config.version=0xFF returns
    OCT_STATUS_VERSION_MISMATCH cleanly."""
    from octomil.runtime.native import NativeRuntime
    from octomil.runtime.native.loader import (
        OCT_STATUS_VERSION_MISMATCH,
        NativeRuntimeError,
        _get_lib,
    )

    ffi, lib = _get_lib()
    with NativeRuntime.open() as rt:
        cfg = ffi.new("oct_model_config_t*")
        cfg.version = 0xFF  # invalid
        cfg.model_uri = ffi.new("char[]", b"local:///tmp/x")
        out = ffi.new("oct_model_t**")
        status = int(lib.oct_model_open(rt._handle, cfg, out))  # noqa: SLF001
        assert status == OCT_STATUS_VERSION_MISMATCH
        assert out[0] == ffi.NULL
        # Sanity: runtime invariants — wrapper would surface this as
        # NativeRuntimeError, but this test calls the C ABI directly
        # to verify the version-mismatch path exists.
        _ = NativeRuntimeError  # symbol-existence check


def test_native_model_invalidated_on_runtime_close():
    """v0.4 step 1: NativeModel children participate in the same
    WeakSet invalidation cascade that protects NativeSession (Codex
    R1 blocker fix from slice 2A — extended to models)."""
    from octomil.runtime.native import NativeModel, NativeRuntime, NativeRuntimeError
    from octomil.runtime.native.loader import OCT_STATUS_INVALID_INPUT, _get_lib

    ffi, lib = _get_lib()
    rt = NativeRuntime.open()
    # Synthesize a NativeModel + register it (mirrors what
    # open_model would do after a successful C call; stub never
    # produces a real handle).
    mdl = NativeModel(ffi, lib, ffi.NULL, owner=rt)
    rt._models.add(mdl)  # noqa: SLF001 — internal wiring under test

    rt.close()

    with pytest.raises(NativeRuntimeError) as exc_info:
        mdl.warm()
    assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
    assert "invalidated" in str(exc_info.value).lower()

    # close() on invalidated model is a no-op (must NOT call
    # oct_model_close on a freed handle).
    mdl.close()
    mdl.close()
    # evict() falls through to OK so cleanup paths are safe.
    from octomil.runtime.native.loader import OCT_STATUS_OK

    assert mdl.evict() == OCT_STATUS_OK


def test_oct_err_constants_stable():
    """v0.4: OCT_ERR_* numeric assignments are stable forever per
    the contracts schema (canonical_error_codes.json). Drift here
    breaks SDK codegen across every binding. Codex R3 missed-case:
    full parity coverage instead of spot-checking OK/UNKNOWN only."""
    from octomil.runtime.native.loader import (
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
    )

    # Mirrors octomil-contracts/fixtures/runtime_error_code/canonical_error_codes.json.
    expected = {
        "OCT_ERR_OK": 0,
        "OCT_ERR_MODEL_LOAD_FAILED": 1,
        "OCT_ERR_ARTIFACT_DIGEST_MISMATCH": 2,
        "OCT_ERR_ENGINE_INIT_FAILED": 3,
        "OCT_ERR_RAM_INSUFFICIENT": 4,
        "OCT_ERR_ACCELERATOR_UNAVAILABLE": 5,
        "OCT_ERR_INPUT_OUT_OF_RANGE": 6,
        "OCT_ERR_INPUT_FORMAT_UNSUPPORTED": 7,
        "OCT_ERR_TIMEOUT": 8,
        "OCT_ERR_PREEMPTED": 9,
        "OCT_ERR_QUOTA_EXCEEDED": 10,
        "OCT_ERR_INTERNAL": 11,
        "OCT_ERR_UNKNOWN": 0xFFFFFFFF,
    }
    actual = {
        "OCT_ERR_OK": OCT_ERR_OK,
        "OCT_ERR_MODEL_LOAD_FAILED": OCT_ERR_MODEL_LOAD_FAILED,
        "OCT_ERR_ARTIFACT_DIGEST_MISMATCH": OCT_ERR_ARTIFACT_DIGEST_MISMATCH,
        "OCT_ERR_ENGINE_INIT_FAILED": OCT_ERR_ENGINE_INIT_FAILED,
        "OCT_ERR_RAM_INSUFFICIENT": OCT_ERR_RAM_INSUFFICIENT,
        "OCT_ERR_ACCELERATOR_UNAVAILABLE": OCT_ERR_ACCELERATOR_UNAVAILABLE,
        "OCT_ERR_INPUT_OUT_OF_RANGE": OCT_ERR_INPUT_OUT_OF_RANGE,
        "OCT_ERR_INPUT_FORMAT_UNSUPPORTED": OCT_ERR_INPUT_FORMAT_UNSUPPORTED,
        "OCT_ERR_TIMEOUT": OCT_ERR_TIMEOUT,
        "OCT_ERR_PREEMPTED": OCT_ERR_PREEMPTED,
        "OCT_ERR_QUOTA_EXCEEDED": OCT_ERR_QUOTA_EXCEEDED,
        "OCT_ERR_INTERNAL": OCT_ERR_INTERNAL,
        "OCT_ERR_UNKNOWN": OCT_ERR_UNKNOWN,
    }
    assert actual == expected, f"OCT_ERR_* assignment drift:\n  expected: {expected}\n  actual:   {actual}"
    # Numeric values must be unique (collisions silently mis-route errors).
    values = list(expected.values())
    assert len(values) == len(set(values)), "duplicate OCT_ERR_* values"


def test_loader_rejects_dylib_with_old_abi_minor(monkeypatch, tmp_path: Path):
    """Codex R1 blocker: v0.4 binding loading a v0.3 dylib should fail
    fast at load time with OCT_STATUS_VERSION_MISMATCH instead of
    crashing later on a missing v0.4 symbol. Simulate by monkey-
    patching the major/minor accessors after dlopen."""
    from octomil.runtime.native.loader import (
        OCT_STATUS_VERSION_MISMATCH,
        NativeRuntimeError,
        _build_lib,
    )

    # Pre-warm: real load works.
    ffi, lib = _build_lib()
    assert int(lib.oct_runtime_abi_version_minor()) >= 4

    # Now simulate an old dylib by monkey-patching the binding's
    # required-version constants UPWARD (equivalent test path).
    import octomil.runtime.native.loader as loader

    monkeypatch.setattr(loader, "_REQUIRED_ABI_MINOR", 99)
    monkeypatch.setattr(loader, "_FFI", None)
    monkeypatch.setattr(loader, "_LIB", None)
    with pytest.raises(NativeRuntimeError) as exc_info:
        _build_lib()
    assert exc_info.value.status == OCT_STATUS_VERSION_MISMATCH
    assert "older than" in str(exc_info.value).lower()


def test_native_runtime_close_invokes_model_close_before_invalidation():
    """Codex R2 fix: NativeRuntime.close() must call NativeModel.close()
    on each tracked model BEFORE flipping the invalidation flag, so
    engine adapters with expensive resources get the explicit close
    path. Defense-in-depth alongside the C ABI's documented implicit
    cleanup contract."""
    from octomil.runtime.native import NativeModel, NativeRuntime
    from octomil.runtime.native.loader import _get_lib

    ffi, lib = _get_lib()
    rt = NativeRuntime.open()

    closed_count = {"n": 0}

    class _ProbeModel(NativeModel):
        def close(self):  # type: ignore[override]
            closed_count["n"] += 1
            super().close()

    mdl = _ProbeModel(ffi, lib, ffi.NULL, owner=rt)
    rt._models.add(mdl)  # noqa: SLF001

    rt.close()

    assert closed_count["n"] == 1, "NativeRuntime.close() must call model.close() on each tracked model"
    # Subsequent ops on the wrapper raise (invalidated).
    from octomil.runtime.native import NativeRuntimeError

    with pytest.raises(NativeRuntimeError):
        mdl.warm()


# ---------------------------------------------------------------------------
# ABI v0.4 step 2 — operational envelope + new event types
# ---------------------------------------------------------------------------


def test_oct_event_t_size_includes_v0_4_envelope():
    """v0.4 step 2: oct_event_t grew by the operational envelope.
    Size parity test catches accidental cdef/runtime drift."""
    from octomil.runtime.native.loader import _get_lib

    ffi, lib = _get_lib()
    cffi_size = ffi.sizeof("oct_event_t")
    runtime_size = int(lib.oct_event_size())
    assert cffi_size == runtime_size, f"oct_event_t v0.4 step 2 drift: cffi={cffi_size}, runtime={runtime_size}"


def test_oct_session_config_t_v0_4_appended_fields_present():
    """v0.4 step 2: oct_session_config_t v=2 appends request_id /
    route_id / trace_id / kv_prefix_key. The cdef and runtime must
    agree on the new struct size."""
    from octomil.runtime.native.loader import _get_lib

    ffi, lib = _get_lib()
    cffi_size = ffi.sizeof("oct_session_config_t")
    runtime_size = int(lib.oct_session_config_size())
    assert cffi_size == runtime_size


def test_session_config_version_bumped_to_2():
    """v0.4 step 2 bumps session_config version 1 → 2."""
    from octomil.runtime.native.loader import OCT_SESSION_CONFIG_VERSION

    assert OCT_SESSION_CONFIG_VERSION == 2


def test_event_version_bumped_to_2():
    """v0.4 step 2 bumps event version 1 → 2."""
    from octomil.runtime.native.loader import OCT_EVENT_VERSION

    assert OCT_EVENT_VERSION == 2


def test_v0_4_step_2_event_type_constants_assigned_correctly():
    """v0.4 step 2 event-type numeric assignments are STABLE forever."""
    from octomil.runtime.native.loader import (
        OCT_EVENT_CACHE_HIT,
        OCT_EVENT_CACHE_MISS,
        OCT_EVENT_MEMORY_PRESSURE,
        OCT_EVENT_METRIC,
        OCT_EVENT_MODEL_EVICTED,
        OCT_EVENT_MODEL_LOADED,
        OCT_EVENT_PREEMPTED,
        OCT_EVENT_QUEUED,
        OCT_EVENT_THERMAL_STATE,
        OCT_EVENT_WATCHDOG_TIMEOUT,
    )

    assert OCT_EVENT_MODEL_LOADED == 10
    assert OCT_EVENT_MODEL_EVICTED == 11
    assert OCT_EVENT_CACHE_HIT == 12
    assert OCT_EVENT_CACHE_MISS == 13
    assert OCT_EVENT_QUEUED == 14
    assert OCT_EVENT_PREEMPTED == 15
    assert OCT_EVENT_MEMORY_PRESSURE == 16
    assert OCT_EVENT_THERMAL_STATE == 17
    assert OCT_EVENT_WATCHDOG_TIMEOUT == 18
    assert OCT_EVENT_METRIC == 19


def test_open_session_with_correlation_ids_stub():
    """v0.4 step 2: open_session accepts the new correlation kwargs.
    Stub still returns UNSUPPORTED (slice-2 stub semantics
    preserved); the kwargs flow through cleanly."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        OCT_STATUS_UNSUPPORTED,
        NativeRuntime,
        NativeRuntimeError,
    )

    with NativeRuntime.open() as rt:
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_session(
                capability=CAPABILITY_AUDIO_REALTIME_SESSION,
                request_id="req-abc-123",
                route_id="route-xyz-789",
                trace_id="00f067aa0ba902b7",
                kv_prefix_key="app:scribe:system_v3",
            )
        assert exc_info.value.status == OCT_STATUS_UNSUPPORTED


def test_open_session_rejects_oversized_correlation_ids():
    """v0.4 step 2 PE review §1.9.5: correlation IDs have explicit
    length caps. Bindings reject before the FFI call."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        NativeRuntime,
        NativeRuntimeError,
    )
    from octomil.runtime.native.loader import OCT_STATUS_INVALID_INPUT

    with NativeRuntime.open() as rt:
        # request_id > 128 bytes
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_session(
                capability=CAPABILITY_AUDIO_REALTIME_SESSION,
                request_id="x" * 129,
            )
        assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
        assert "request_id" in str(exc_info.value)

        # kv_prefix_key > 256 bytes
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_session(
                capability=CAPABILITY_AUDIO_REALTIME_SESSION,
                kv_prefix_key="y" * 257,
            )
        assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
        assert "kv_prefix_key" in str(exc_info.value)


def test_native_event_envelope_populated_on_stub_poll():
    """v0.4 step 2: stub poll_event populates envelope with empty-
    string sentinels (NEVER NULL). Bindings can strlen() safely
    without NULL-checking each field."""
    from octomil.runtime.native import NativeRuntime
    from octomil.runtime.native.loader import NativeSession, _get_lib

    ffi, lib = _get_lib()
    with NativeRuntime.open() as rt:
        sess = NativeSession(ffi, lib, ffi.NULL, owner=rt)
        # Stub poll_event raises on UNSUPPORTED. To exercise the
        # envelope population path directly, call the C poll_event
        # with a fresh event buffer and inspect.
        ev = ffi.new("oct_event_t*")
        ev.size = ffi.sizeof("oct_event_t")
        ev.version = 2
        status = int(lib.oct_session_poll_event(ffi.NULL, ev, 0))
        # Stub returns UNSUPPORTED; envelope is populated regardless
        # because the population happens in the C-side write path
        # before the status return.
        assert status != 0
        # envelope strings are non-NULL (empty sentinel) on the stub.
        for field in (
            "request_id",
            "route_id",
            "trace_id",
            "engine_version",
            "adapter_version",
            "accelerator",
            "artifact_digest",
        ):
            ptr = getattr(ev, field)
            assert ptr != ffi.NULL, f"envelope field {field!r} is NULL (must be empty string)"
            assert ffi.string(ptr) == b"", f"envelope field {field!r} should be empty string sentinel"
        assert int(ev.cache_was_hit) == 0
        # Avoid the no-handle NativeSession close path returning unexpected status.
        sess.close()


def test_native_event_class_exposes_envelope_fields():
    """v0.4 step 2: NativeEvent class carries the envelope as Python
    attributes — no NULL handling required at the binding edge."""
    from octomil.runtime.native import OCT_EVENT_NONE, NativeEvent

    ev = NativeEvent(
        type=OCT_EVENT_NONE,
        version=2,
        monotonic_ns=12345,
        user_data_ptr=0,
        request_id="req-1",
        route_id="route-1",
        trace_id="trace-1",
        engine_version="moshi-mlx@0.2.6",
        adapter_version="adapter-sha-deadbeef",
        accelerator="metal",
        artifact_digest="sha256:" + "a" * 64,
        cache_was_hit=True,
    )
    assert ev.request_id == "req-1"
    assert ev.route_id == "route-1"
    assert ev.trace_id == "trace-1"
    assert ev.engine_version == "moshi-mlx@0.2.6"
    assert ev.accelerator == "metal"
    assert ev.cache_was_hit is True


def test_v0_4_step_2_envelope_slots_are_pinned():
    """Privacy invariant from PE review §2: the operational envelope
    fields are id/version/digest strings ONLY. No prompts, audio
    bytes, transcript text, file paths, or PHI/PII may be carried
    via the envelope.

    Codex R2 nit: pin the envelope SUBSET, not the full __slots__.
    Future legitimate payload-parsing fields (audio_chunk pcm,
    transcript_chunk utf8, model_loaded engine, etc.) are NOT
    envelope fields and should NOT trip this guard. This test
    asserts only that:
      (a) every named envelope slot exists (no accidental removal)
      (b) any net-new field with an envelope-like name shape (id,
          version, digest, accelerator) was reviewed.
    Privacy gates on payload fields live in their own per-payload
    tests once those payloads ship."""
    from octomil.runtime.native import NativeEvent

    expected_envelope_slots = {
        "request_id",
        "route_id",
        "trace_id",
        "engine_version",
        "adapter_version",
        "accelerator",
        "artifact_digest",
        "cache_was_hit",
    }
    actual_slots = set(NativeEvent.__slots__)
    # (a) every expected envelope slot MUST exist (no removal).
    missing = expected_envelope_slots - actual_slots
    assert not missing, f"envelope slots missing from NativeEvent: {missing}"
    # (b) base slots (id/version) are part of the structure too.
    base_slots = {"type", "version", "monotonic_ns", "user_data_ptr"}
    documented = expected_envelope_slots | base_slots
    # If a future PR wants to ADD a new envelope-class field
    # (id-shaped / version-shaped / digest-shaped), it must update
    # expected_envelope_slots here AND get explicit PE review since
    # that broadens the privacy surface. Other (payload) slots may
    # come and go without this test failing.
    envelope_like_extras = {
        s for s in actual_slots - documented if any(token in s for token in ("_id", "version", "digest", "accelerator"))
    }
    assert not envelope_like_extras, (
        f"NativeEvent grew envelope-class slots without PE review: "
        f"{envelope_like_extras}. Privacy boundary requires explicit "
        f"approval for any new id/version/digest field."
    )


def test_open_session_rejects_correlation_ids_with_whitespace():
    """Codex R1 fix: ABI contract requires ASCII-printable
    (0x21..0x7E), no whitespace, no control chars. Bindings reject
    pre-FFI."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        NativeRuntime,
        NativeRuntimeError,
    )
    from octomil.runtime.native.loader import OCT_STATUS_INVALID_INPUT

    with NativeRuntime.open() as rt:
        # whitespace
        for bad in ("with space", "tab\there", "line\nbreak"):
            with pytest.raises(NativeRuntimeError) as exc_info:
                rt.open_session(
                    capability=CAPABILITY_AUDIO_REALTIME_SESSION,
                    request_id=bad,
                )
            assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
            assert "request_id" in str(exc_info.value)


def test_open_session_rejects_correlation_ids_with_control_chars():
    """Codex R1 fix: control chars (codepoints < 0x21) rejected."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        NativeRuntime,
        NativeRuntimeError,
    )
    from octomil.runtime.native.loader import OCT_STATUS_INVALID_INPUT

    with NativeRuntime.open() as rt:
        # control chars
        for bad in ("\x00null", "bell\x07here", "del\x7fchar"):
            with pytest.raises(NativeRuntimeError) as exc_info:
                rt.open_session(
                    capability=CAPABILITY_AUDIO_REALTIME_SESSION,
                    route_id=bad,
                )
            assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
            assert "route_id" in str(exc_info.value)


def test_open_session_rejects_correlation_ids_with_non_ascii():
    """Codex R1 fix: non-ASCII codepoints (> 0x7E) rejected — the
    envelope is a bounded-cardinality label surface, not a
    user-string carrier."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        NativeRuntime,
        NativeRuntimeError,
    )
    from octomil.runtime.native.loader import OCT_STATUS_INVALID_INPUT

    with NativeRuntime.open() as rt:
        for bad in ("café", "日本語", "\u202erlt-override"):
            with pytest.raises(NativeRuntimeError) as exc_info:
                rt.open_session(
                    capability=CAPABILITY_AUDIO_REALTIME_SESSION,
                    trace_id=bad,
                )
            assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
            assert "trace_id" in str(exc_info.value)


def test_open_session_accepts_canonical_correlation_id_shapes():
    """Sanity: canonical W3C-style trace ids and URL-safe IDs pass.
    Stub still returns UNSUPPORTED but the validator allows the
    well-formed cases through."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        OCT_STATUS_UNSUPPORTED,
        NativeRuntime,
        NativeRuntimeError,
    )

    with NativeRuntime.open() as rt:
        for good in (
            "00f067aa0ba902b7",  # 16-char hex
            "abcdef12-3456-7890-abcd-ef1234567890",
            "req:scribe:2026-05-04T00:00:00Z",
            "app/scribe/system_v3",
        ):
            with pytest.raises(NativeRuntimeError) as exc_info:
                rt.open_session(
                    capability=CAPABILITY_AUDIO_REALTIME_SESSION,
                    request_id=good,
                )
            assert (
                exc_info.value.status == OCT_STATUS_UNSUPPORTED
            ), f"valid id {good!r} should reach the C ABI and stub return UNSUPPORTED, not be rejected client-side"


def test_v0_4_step_2_event_constants_exported_publicly():
    """Codex R1 missed-case: every new OCT_EVENT_* constant must be
    importable from `octomil.runtime.native` (the public surface)
    not just `loader.py`."""
    import octomil.runtime.native as native

    for name in (
        "OCT_EVENT_MODEL_LOADED",
        "OCT_EVENT_MODEL_EVICTED",
        "OCT_EVENT_CACHE_HIT",
        "OCT_EVENT_CACHE_MISS",
        "OCT_EVENT_QUEUED",
        "OCT_EVENT_PREEMPTED",
        "OCT_EVENT_MEMORY_PRESSURE",
        "OCT_EVENT_THERMAL_STATE",
        "OCT_EVENT_WATCHDOG_TIMEOUT",
        "OCT_EVENT_METRIC",
    ):
        assert hasattr(native, name), f"public surface missing {name}"


def test_open_session_translates_unicode_encode_error():
    """Codex R2 nit: lone surrogates / unencodable strings should
    raise NativeRuntimeError(INVALID_INPUT), not raw
    UnicodeEncodeError. Callers handle one exception type for
    'bad correlation ID', not two."""
    from octomil.runtime.native import (
        CAPABILITY_AUDIO_REALTIME_SESSION,
        NativeRuntime,
        NativeRuntimeError,
    )
    from octomil.runtime.native.loader import OCT_STATUS_INVALID_INPUT

    with NativeRuntime.open() as rt:
        # Lone surrogate is unencodable as UTF-8 (without surrogateescape).
        bad = "\ud800"  # high surrogate alone
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_session(
                capability=CAPABILITY_AUDIO_REALTIME_SESSION,
                request_id=bad,
            )
        assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
        assert "request_id" in str(exc_info.value)
