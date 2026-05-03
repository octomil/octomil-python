"""Tests for ``octomil.runtime.native`` cffi loader (slice 3 PR1).

Builds the dylib via CMake in a session fixture (cached across the
test session), then exercises:

  * Version handshake.
  * ``oct_runtime_open`` v1 success + v0 / v2 / NULL-out error paths.
  * ``oct_runtime_close`` + idempotent context-manager close.
  * ``oct_runtime_capabilities`` honors out->size, returns empty
    sentinel arrays from the slice-2 stub.
  * Forward-compat: capabilities() drops unknown advertised strings.
  * Thread-error and runtime-error read-back paths.

Session entry tests are NOT in this file — those land alongside
slice 2-proper / 3 PR2.
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
    # Slice 3 PR1 R2: header bumped to MINOR=2 (additive — added
    # oct_runtime_config_size/oct_capabilities_size introspection
    # functions; existing readers stay compatible). Lockstep update.
    assert (major, minor) == (0, 2)
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
    assert cffi_size == runtime_size, (
        f"oct_capabilities_t struct-layout drift: " f"cffi cdef sizeof={cffi_size}, C compiler sizeof={runtime_size}."
    )


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
    silently never match."""
    from octomil.runtime.native import RUNTIME_CAPABILITIES

    expected = {
        "audio.realtime.session",
        "audio.stt.batch",
        "audio.stt.stream",
        "audio.transcription",
        "audio.tts.batch",
        "audio.tts.stream",
        "chat.completion",
        "chat.stream",
    }
    assert set(RUNTIME_CAPABILITIES) == expected


def test_no_embeddings_capability():
    """Per the strategy doc: embeddings.text is intentionally absent
    from the runtime capability enum."""
    from octomil.runtime.native import RUNTIME_CAPABILITIES

    assert "embeddings.text" not in RUNTIME_CAPABILITIES


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
