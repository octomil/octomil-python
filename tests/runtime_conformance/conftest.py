"""Conformance harness — backend selector + capability marker.

Backend is selected via ``OCTOMIL_CONFORMANCE_BACKEND={python,native,both}``
(default ``python``). Tests that exercise a runtime capability use
``@requires_capability(CAP_NAME)`` to enforce the current native
capability truth model:

* ``DONE_NATIVE_CUTOVER`` and ``LIVE_NATIVE_CONDITIONAL`` capabilities
  may run on the python backend when a Python conformance path exists.
* ``BLOCKED_WITH_PROOF`` capabilities skip on python with
  ``no-python-oracle`` because the SDK must not treat reserved names as
  live behavior.
* Native backend runs only when the live runtime actually advertises
  the capability; conditional gates that fail skip with
  ``runtime_capabilities``.

Neither backend can silently pass a test it doesn't actually own.

`PYTHON_ORACLE_CAPABILITIES` is kept as the backwards-compatible name
for the conformance-capable set, now derived from the status partition
instead of migration-slice prose.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest

from octomil.runtime.native.capabilities import (
    CAPABILITY_STATUS_BLOCKED_WITH_PROOF,
    CAPABILITY_STATUS_DONE_NATIVE_CUTOVER,
    CAPABILITY_STATUS_LIVE_NATIVE_CONDITIONAL,
    DONE_NATIVE_CUTOVER_CAPABILITIES,
    LIVE_NATIVE_CONDITIONAL_CAPABILITIES,
    RUNTIME_CAPABILITY_STATUSES,
)

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

#: Backwards-compatible name for capabilities that the python backend
#: may exercise in conformance. It is status-derived: blocked enum names
#: never run on the python backend, while done and live-conditional native
#: surfaces can be compared through the Python SDK harness.
PYTHON_ORACLE_CAPABILITIES: frozenset[str] = frozenset(
    DONE_NATIVE_CUTOVER_CAPABILITIES | LIVE_NATIVE_CONDITIONAL_CAPABILITIES
)

_PYTHON_RUNNABLE_STATUSES: frozenset[str] = frozenset(
    {
        CAPABILITY_STATUS_DONE_NATIVE_CUTOVER,
        CAPABILITY_STATUS_LIVE_NATIVE_CONDITIONAL,
    }
)


# ---------------------------------------------------------------------------
# Native runtime fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def native_runtime_capabilities() -> frozenset[str]:
    """Snapshot of the live runtime's advertised capability set.

    Callers MUST treat this as forward-compatible — a newer runtime
    advertising more capabilities than the SDK knows about silently
    drops the unknowns (handled inside `NativeRuntime.capabilities()`).

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

    * Python backend + BLOCKED_WITH_PROOF cap →
      skip ``no-python-oracle: <cap>``.
    * Native backend + cap not advertised by the runtime →
      skip ``runtime_capabilities: <cap>``.
    """
    backend_value: str = request.param

    # Read the requires_capability marker (if any) from the test.
    marker = request.node.get_closest_marker("requires_capability")
    if marker is not None:
        capability = marker.args[0]
        status = RUNTIME_CAPABILITY_STATUSES.get(capability)
        if status is None:
            raise RuntimeError(
                f"requires_capability({capability!r}) is not in the canonical " "RUNTIME_CAPABILITY_STATUSES partition."
            )
        if backend_value == BACKEND_PYTHON:
            if status not in _PYTHON_RUNNABLE_STATUSES:
                pytest.skip(
                    f"no-python-oracle: {capability} is {CAPABILITY_STATUS_BLOCKED_WITH_PROOF}; "
                    "Python has no live conformance path"
                )
        elif backend_value == BACKEND_NATIVE:
            # Lazy fetch — only on native branch. Python-only runs
            # never touch the native runtime fixture.
            native_caps: frozenset[str] = request.getfixturevalue("native_runtime_capabilities")
            if capability not in native_caps:
                pytest.skip(f"runtime_capabilities: native runtime does not advertise {capability} " f"({status})")
        else:  # pragma: no cover — guarded above
            raise RuntimeError(f"unknown backend {backend_value!r}")

    yield backend_value


__all__ = [
    "BACKEND_BOTH",
    "BACKEND_NATIVE",
    "BACKEND_PYTHON",
    "CAPABILITY_STATUS_BLOCKED_WITH_PROOF",
    "CAPABILITY_STATUS_DONE_NATIVE_CUTOVER",
    "CAPABILITY_STATUS_LIVE_NATIVE_CONDITIONAL",
    "ENV_BACKEND",
    "PYTHON_ORACLE_CAPABILITIES",
    "RUNTIME_CAPABILITY_STATUSES",
    "requires_capability",
]
