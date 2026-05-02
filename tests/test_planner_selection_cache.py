"""Per-process planner-selection cache (cross-model perf PR).

The planner is deterministic in ``(model, capability, policy_preset)``
within one process, so re-running ``planner.resolve`` on every dispatch
is wasted work. This module pins:

  - The cache hit (second call with same key skips ``planner.resolve``).
  - Negative caching (failed planner returns None; second call doesn't
    re-pay the failure-path cost within the TTL).
  - Different keys go to the planner independently.
  - ``OCTOMIL_PLANNER_SELECTION_CACHE_TTL_SECONDS=0`` disables caching.
  - ``OCTOMIL_RUNTIME_PLANNER_CACHE=0`` short-circuits before the cache
    even consults the planner (matches existing env-var posture).
  - ``release_planner_selection_cache`` clears every entry.
  - ``ExecutionKernel.release_warmed_backends`` also clears the
    planner cache (single public "drop my caches" surface).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from octomil.execution.kernel import ExecutionKernel
from octomil.execution.planner_resolution import (
    _PLANNER_SELECTION_CACHE,
    _resolve_planner_selection,
    release_planner_selection_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    release_planner_selection_cache()
    yield
    release_planner_selection_cache()


@pytest.fixture(autouse=True)
def _reset_bootstrap_warned():
    # The "planner unavailable" warning is once-per-process; reset so
    # tests that exercise the bootstrap-failure path don't depend on
    # whichever ran first.
    import octomil.execution.planner_resolution as pr

    pr._PLANNER_BOOTSTRAP_WARNED = False
    yield
    pr._PLANNER_BOOTSTRAP_WARNED = False


class _StubSelection:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def test_repeated_call_with_same_key_skips_planner_resolve():
    """First call hits the planner; second call returns the cached
    selection without re-resolving."""
    sel = _StubSelection("hot")
    call_count = 0

    class _StubPlanner:
        def resolve(self, *, model, capability, routing_policy):
            nonlocal call_count
            call_count += 1
            return sel

    with patch(
        "octomil.runtime.planner.planner.RuntimePlanner",
        return_value=_StubPlanner(),
    ):
        first = _resolve_planner_selection("kokoro", "tts", "local_first")
        second = _resolve_planner_selection("kokoro", "tts", "local_first")

    assert first is sel
    assert second is sel
    assert call_count == 1, "second call should hit the cache, not the planner"


def test_negative_results_are_cached_too():
    """A planner that raises returns None; the next call doesn't
    re-attempt within the TTL. This is the offline-planner / Ren'Py
    case — without negative caching every dispatch pays the full
    bootstrap-and-fail roundtrip."""
    call_count = 0

    class _FailingPlanner:
        def resolve(self, *, model, capability, routing_policy):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("offline planner")

    with patch(
        "octomil.runtime.planner.planner.RuntimePlanner",
        return_value=_FailingPlanner(),
    ):
        first = _resolve_planner_selection("kokoro", "tts", "local_first")
        second = _resolve_planner_selection("kokoro", "tts", "local_first")

    assert first is None
    assert second is None
    assert call_count == 1, "second call should hit the negative-cached None"


def test_different_keys_resolve_independently():
    """Cache keys are ``(model, capability, policy_preset)``; changing
    any of the three triggers a fresh resolve."""
    seen_keys = []

    class _Recorder:
        def resolve(self, *, model, capability, routing_policy):
            seen_keys.append((model, capability, routing_policy))
            return _StubSelection(f"{model}/{capability}/{routing_policy}")

    with patch(
        "octomil.runtime.planner.planner.RuntimePlanner",
        return_value=_Recorder(),
    ):
        _resolve_planner_selection("kokoro", "tts", "local_first")
        _resolve_planner_selection("piper", "tts", "local_first")
        _resolve_planner_selection("kokoro", "transcription", "local_first")
        _resolve_planner_selection("kokoro", "tts", "cloud_first")

    assert len(seen_keys) == 4
    assert len({k for k in seen_keys}) == 4


def test_ttl_zero_disables_cache_but_still_calls_planner(monkeypatch):
    """``OCTOMIL_PLANNER_SELECTION_CACHE_TTL_SECONDS=0`` is the
    bypass switch that still goes to the planner — distinct from
    ``OCTOMIL_RUNTIME_PLANNER_CACHE=0`` which short-circuits before
    the planner is consulted at all."""
    monkeypatch.setenv("OCTOMIL_PLANNER_SELECTION_CACHE_TTL_SECONDS", "0")
    call_count = 0

    class _Counter:
        def resolve(self, *, model, capability, routing_policy):
            nonlocal call_count
            call_count += 1
            return _StubSelection(str(call_count))

    with patch(
        "octomil.runtime.planner.planner.RuntimePlanner",
        return_value=_Counter(),
    ):
        _resolve_planner_selection("kokoro", "tts", "local_first")
        _resolve_planner_selection("kokoro", "tts", "local_first")
        _resolve_planner_selection("kokoro", "tts", "local_first")

    assert call_count == 3
    assert _PLANNER_SELECTION_CACHE == {}


def test_runtime_planner_cache_env_var_short_circuits_before_cache(monkeypatch):
    """``OCTOMIL_RUNTIME_PLANNER_CACHE=0`` returns ``None`` without
    consulting either the cache or the planner — pre-existing
    semantics of that env var, preserved."""
    monkeypatch.setenv("OCTOMIL_RUNTIME_PLANNER_CACHE", "0")

    class _BoomPlanner:
        def resolve(self, *, model, capability, routing_policy):
            raise AssertionError("planner must not be consulted")

    with patch(
        "octomil.runtime.planner.planner.RuntimePlanner",
        return_value=_BoomPlanner(),
    ):
        first = _resolve_planner_selection("kokoro", "tts", "local_first")

    assert first is None
    assert _PLANNER_SELECTION_CACHE == {}


def test_release_planner_selection_cache_clears_entries():
    sel = _StubSelection("a")
    call_count = 0

    class _Stub:
        def resolve(self, *, model, capability, routing_policy):
            nonlocal call_count
            call_count += 1
            return sel

    with patch(
        "octomil.runtime.planner.planner.RuntimePlanner",
        return_value=_Stub(),
    ):
        _resolve_planner_selection("kokoro", "tts", "local_first")
        assert call_count == 1
        release_planner_selection_cache()
        assert _PLANNER_SELECTION_CACHE == {}
        _resolve_planner_selection("kokoro", "tts", "local_first")
        assert call_count == 2, "post-release call must re-resolve"


def test_release_warmed_backends_also_clears_planner_cache():
    """The kernel's public 'drop my caches' surface clears both
    the warmup cache and the planner cache. Embedded callers that
    call ``release_warmed_backends`` to free GPU memory shouldn't
    need to know the planner cache exists separately."""
    kernel = ExecutionKernel()
    sel = _StubSelection("a")

    class _Stub:
        def resolve(self, *, model, capability, routing_policy):
            return sel

    with patch(
        "octomil.runtime.planner.planner.RuntimePlanner",
        return_value=_Stub(),
    ):
        _resolve_planner_selection("kokoro", "tts", "local_first")

    assert _PLANNER_SELECTION_CACHE  # populated
    kernel._warmed_backends[("tts", "kokoro", None, None, None)] = object()

    kernel.release_warmed_backends()

    assert kernel._warmed_backends == {}
    assert _PLANNER_SELECTION_CACHE == {}


def test_negative_cache_does_not_mask_recovery_after_release():
    """Operator scenario: planner was offline → negative-cached None.
    Network comes back; operator calls ``release_warmed_backends``
    (or the cache TTL elapses). Next call must reach the planner."""
    fail_count = 0
    succeed_count = 0
    sel = _StubSelection("recovered")

    class _ToggleablePlanner:
        def __init__(self, mode: str) -> None:
            self._mode = mode

        def resolve(self, *, model, capability, routing_policy):
            nonlocal fail_count, succeed_count
            if self._mode == "fail":
                fail_count += 1
                raise RuntimeError("offline")
            succeed_count += 1
            return sel

    with patch(
        "octomil.runtime.planner.planner.RuntimePlanner",
        return_value=_ToggleablePlanner("fail"),
    ):
        first = _resolve_planner_selection("kokoro", "tts", "local_first")
    assert first is None
    assert fail_count == 1

    release_planner_selection_cache()

    with patch(
        "octomil.runtime.planner.planner.RuntimePlanner",
        return_value=_ToggleablePlanner("ok"),
    ):
        second = _resolve_planner_selection("kokoro", "tts", "local_first")
    assert second is sel
    assert succeed_count == 1
