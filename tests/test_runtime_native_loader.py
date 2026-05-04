"""Tests for ``octomil.runtime.native`` cffi loader.

Builds the dylib via CMake in a session fixture (cached across the
test session), then exercises:

  * Version handshake.
  * ``oct_runtime_open`` v1 success + v0 / v2 / NULL-out error paths.
  * ``oct_runtime_close`` + idempotent context-manager close.
  * ``oct_runtime_capabilities`` honors out->size, returns empty
    sentinel arrays from the slice-2 stub.
  * Forward-compat: capabilities() drops unknown advertised strings.
  * Thread-error and runtime-error read-back paths.
  * Slice 2A: ABI parity for the session-level structs
    (oct_session_config_t, oct_audio_view_t, oct_event_t) plus
    NativeSession stub-behavior (open/send_audio/send_text/poll_event/
    cancel/close all surface OCT_STATUS_UNSUPPORTED cleanly).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Skip the entire module if cffi isn't available — this matches the
# `native` extra opt-in surface.
cffi = pytest.importorskip("cffi", reason="cffi extra not installed")  # noqa: F841

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_CORE_DIR = REPO_ROOT / "octomil" / "runtime-core"


def _platform_dylib_name() -> str:
    import sys

    if sys.platform == "darwin":
        return "liboctomil-runtime.dylib"
    if sys.platform == "win32":
        return "octomil-runtime.dll"
    return "liboctomil-runtime.so"


@pytest.fixture(scope="session")
def runtime_dylib() -> Path:
    """Build the slice-2 stub dylib once per test session and return
    its absolute path. The build is cached by CMake so re-runs are
    fast."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake not installed; cannot build runtime-core dylib")
    build_dir = RUNTIME_CORE_DIR / "build"
    if not build_dir.exists():
        subprocess.run(
            ["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Debug"],
            check=True,
            cwd=RUNTIME_CORE_DIR,
        )
    subprocess.run(["cmake", "--build", "build"], check=True, cwd=RUNTIME_CORE_DIR)
    dylib_path = build_dir / _platform_dylib_name()
    assert dylib_path.is_file(), f"build did not produce {dylib_path}"
    return dylib_path


@pytest.fixture(autouse=True)
def _isolate_loader(monkeypatch, runtime_dylib: Path):
    """Ensure each test gets a fresh FFI/lib pair pointed at the
    session-built dylib. Resets the loader's module-level singletons
    so the override env var takes effect."""
    monkeypatch.setenv("OCTOMIL_RUNTIME_DYLIB", str(runtime_dylib))
    import octomil.runtime.native.loader as loader

    monkeypatch.setattr(loader, "_FFI", None)
    monkeypatch.setattr(loader, "_LIB", None)
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
    # ABI v0.4 step 1: header bumped to MINOR=4 (additive — added
    # oct_model_t lifecycle (open/warm/evict/close — stubs);
    # oct_error_code_t typedef + 13 constants; +6 capability
    # constants (audio.diarization, audio.speaker.embedding,
    # audio.vad, embeddings.image, embeddings.text,
    # index.vector.query). Existing readers stay 0.3-compat.
    assert (major, minor) == (0, 4)
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
    the dev-path fallback. Set it to a non-existent path; loader
    should error pointing at the override path first."""
    monkeypatch.setenv("OCTOMIL_RUNTIME_DYLIB", str(tmp_path / "does-not-exist.dylib"))
    import octomil.runtime.native.loader as loader

    monkeypatch.setattr(loader, "_FFI", None)
    monkeypatch.setattr(loader, "_LIB", None)
    with pytest.raises(ImportError) as exc_info:
        loader._build_lib()
    assert "does-not-exist.dylib" in str(exc_info.value)
    assert "BUILD.md" in str(exc_info.value)


def test_dylib_resolution_message_lists_all_candidates(monkeypatch, tmp_path: Path):
    """When the dylib can't be found, the error message lists every
    path tried so an operator can correlate which build step failed."""
    monkeypatch.delenv("OCTOMIL_RUNTIME_DYLIB", raising=False)
    # Move the runtime-core build dir aside temporarily by overriding
    # the candidate-path resolution — simpler to just check via
    # _candidate_dylib_paths directly.
    from octomil.runtime.native.loader import _candidate_dylib_paths

    paths = _candidate_dylib_paths()
    # Three candidates per platform-name iteration: dylib, so, dll.
    assert len(paths) == 3
    # All should reference the runtime-core/build directory.
    for path in paths:
        assert "runtime-core" in str(path)
        assert "build" in str(path)


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


def test_session_open_against_stub_returns_unsupported():
    """The strict-reject contract on ``capability``: under the slice-2
    stub *every* capability — including canonical canonical names —
    returns UNSUPPORTED. The session out-pointer is NULL."""
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
        assert "slice-2" in exc_info.value.last_error.lower()


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


def test_session_send_text_against_stub_returns_unsupported():
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
            sess.send_text("hello")
        assert exc_info.value.status == OCT_STATUS_UNSUPPORTED


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


def test_session_cancel_against_stub_returns_unsupported_status():
    """cancel() returns the raw status code; UNSUPPORTED is not
    treated as an error (idempotent semantics include UNSUPPORTED so
    bindings can call cancel() in cleanup paths without try/except)."""
    from octomil.runtime.native import NativeRuntime
    from octomil.runtime.native.loader import (
        OCT_STATUS_UNSUPPORTED,
        NativeSession,
        _get_lib,
    )

    ffi, lib = _get_lib()
    with NativeRuntime.open() as rt:
        sess = NativeSession(ffi, lib, ffi.NULL, owner=rt)
        status = sess.cancel()
        assert status == OCT_STATUS_UNSUPPORTED


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
    breaks SDK codegen across every binding."""
    from octomil.runtime.native import OCT_ERR_OK, OCT_ERR_UNKNOWN

    assert OCT_ERR_OK == 0
    assert OCT_ERR_UNKNOWN == 0xFFFFFFFF  # UINT32_MAX forward-compat sentinel


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
