"""Conformance harness — backend selector + capability marker.

Backend is selected via ``OCTOMIL_CONFORMANCE_BACKEND={python,native,both}``
(default ``python``). Tests that exercise a runtime capability use
``@requires_capability(CAP_NAME)`` to enforce ownership: tests for
capabilities Python doesn't own as oracle skip on python with
``no-python-oracle``; tests for capabilities the native runtime
doesn't advertise skip on native with ``runtime_capabilities``.

Result: a slice-2-stub native run skips every capability-gated test;
a python-only run skips native-first tests. Neither backend can
silently pass a test it doesn't actually own.

`PYTHON_ORACLE_CAPABILITIES` is the explicit set Python actually
owns as a behavioral oracle. Each migration slice updates this set
when a Python kernel becomes the comparison reference (or ceases to
be, post-migration).
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest

# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

ENV_BACKEND: str = "OCTOMIL_CONFORMANCE_BACKEND"
BACKEND_PYTHON: str = "python"
BACKEND_NATIVE: str = "native"
BACKEND_BOTH: str = "both"
_VALID_BACKENDS = {BACKEND_PYTHON, BACKEND_NATIVE, BACKEND_BOTH}


def _resolve_backends() -> list[str]:
    raw = os.environ.get(ENV_BACKEND, BACKEND_PYTHON)
    if raw not in _VALID_BACKENDS:
        raise RuntimeError(
            f"{ENV_BACKEND}={raw!r} is not one of {sorted(_VALID_BACKENDS)!r}; "
            f"defaulting to {BACKEND_PYTHON!r} would mask the misconfiguration."
        )
    if raw == BACKEND_BOTH:
        return [BACKEND_PYTHON, BACKEND_NATIVE]
    return [raw]


@pytest.fixture(scope="session")
def selected_backends() -> list[str]:
    """The backends this session will exercise."""
    return _resolve_backends()


# ---------------------------------------------------------------------------
# Capability ownership — Python-side oracle set
# ---------------------------------------------------------------------------

#: Capabilities Python owns as oracle. Updated lockstep with each
#: migration slice. Today's set: TTS streaming/batch, STT
#: streaming/batch, transcription, chat completion/stream — the
#: existing Python-kernel-backed behaviors.
#:
#: NOT in this set:
#:   * `audio.realtime.session` — native-first; no Python
#:     implementation. Skips on python with `no-python-oracle`.
#:
#: Updated by:
#:   * Slice 5 PRs (TTS migration) — keeps tts entries here while
#:     the Python kernel is the oracle; eventually drops them
#:     after the one-release-window oracle period elapses.
#:   * Slice 6 PRs (STT/transcription/chat migration) — same
#:     pattern.
PYTHON_ORACLE_CAPABILITIES: frozenset[str] = frozenset(
    {
        "audio.tts.stream",
        "audio.tts.batch",
        "audio.stt.stream",
        "audio.stt.batch",
        "audio.transcription",
        "chat.completion",
        "chat.stream",
    }
)


# ---------------------------------------------------------------------------
# Native runtime fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def native_runtime_capabilities() -> frozenset[str]:
    """Snapshot of the live runtime's advertised capability set.

    Returns an empty frozenset on the slice-2 stub (no capabilities
    advertised). Callers MUST treat this as forward-compatible — a
    newer runtime advertising more capabilities than the SDK knows
    about silently drops the unknowns (handled inside
    `NativeRuntime.capabilities()`).

    Skipped if the native binding can't load the dylib (cffi missing,
    OCTOMIL_RUNTIME_DYLIB unset and dev cache empty); the harness
    itself never crashes."""
    try:
        from octomil.runtime.native import NativeRuntime
    except ImportError as exc:
        pytest.skip(f"native binding unavailable: {exc}")
        return frozenset()  # unreachable; pytest.skip raises

    try:
        with NativeRuntime.open() as rt:
            caps = rt.capabilities()
            return frozenset(caps.supported_capabilities)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"native runtime failed to open: {exc}")
        return frozenset()  # unreachable


# ---------------------------------------------------------------------------
# Capability marker
# ---------------------------------------------------------------------------


def requires_capability(capability: str):
    """Decorator/marker. Used on test functions that exercise a
    specific runtime capability.

    Usage:

      @requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
      def test_realtime_session_starts(backend, ...):
          ...

    The test function's first parameter MUST be ``backend`` (the
    parametrize fixture below). The marker reads the active backend
    from that parameter and applies the right skip rule.
    """
    return pytest.mark.requires_capability(capability)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_capability(capability): mark test as requiring a "
        "specific runtime capability (per-backend skip rules in "
        "tests/runtime_conformance/conftest.py)",
    )


# ---------------------------------------------------------------------------
# Backend fixture — parametrized via `params=` so the fixture body
# (which applies the capability-aware skip rules) actually runs.
# `pytest_generate_tests` would inject the parameter directly without
# invoking any fixture function, so the skip logic would never fire.
# ---------------------------------------------------------------------------


@pytest.fixture(params=_resolve_backends(), ids=lambda b: f"backend={b}")
def backend(request: pytest.FixtureRequest) -> Iterator[str]:
    """The current backend for this test instance. Applies the
    capability-aware skip rules based on the test's
    ``@requires_capability(...)`` marker (if any).

    Codex R1 blocker fix: ``native_runtime_capabilities`` is fetched
    lazily inside the native branch via ``request.getfixturevalue``.
    Previous version took it as a direct fixture param, which made
    pytest resolve it eagerly — and on a Python-only run with the
    dylib unavailable, that fixture's skip propagated through every
    backend-parametrized test, hiding Python-oracle regressions.

    Skip semantics:

    * Python backend + cap not in `PYTHON_ORACLE_CAPABILITIES` →
      skip ``no-python-oracle: <cap>`` (native-first by design).
    * Native backend + cap not advertised by the runtime →
      skip ``runtime_capabilities: <cap>`` (slice-2-stub state).
    """
    backend_value: str = request.param

    # Read the requires_capability marker (if any) from the test.
    marker = request.node.get_closest_marker("requires_capability")
    if marker is not None:
        capability = marker.args[0]
        if backend_value == BACKEND_PYTHON:
            if capability not in PYTHON_ORACLE_CAPABILITIES:
                pytest.skip(f"no-python-oracle: {capability} is native-first; Python has no implementation")
        elif backend_value == BACKEND_NATIVE:
            # Lazy fetch — only on native branch. Python-only runs
            # never touch the native runtime fixture.
            native_caps: frozenset[str] = request.getfixturevalue("native_runtime_capabilities")
            if capability not in native_caps:
                pytest.skip(f"runtime_capabilities: native runtime does not advertise {capability}")
        else:  # pragma: no cover — guarded above
            raise RuntimeError(f"unknown backend {backend_value!r}")

    yield backend_value


__all__ = [
    "BACKEND_BOTH",
    "BACKEND_NATIVE",
    "BACKEND_PYTHON",
    "ENV_BACKEND",
    "PYTHON_ORACLE_CAPABILITIES",
    "requires_capability",
]
