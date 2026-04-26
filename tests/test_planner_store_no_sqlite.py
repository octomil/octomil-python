"""PR A: planner cache must work in environments where ``_sqlite3`` is missing.

Some Python distributions (Ren'Py's bundled CPython, certain
PyInstaller builds, sandboxed embeds) ship without the ``_sqlite3``
extension. Top-level ``import octomil`` previously crashed with
``ImportError: No module named '_sqlite3'`` because ``store.py``
imported sqlite3 unconditionally at module load. This test suite
pins the recovery path:

  - ``import octomil.runtime.planner.store`` must succeed when
    ``sqlite3`` import raises ``ImportError``;
  - ``build_runtime_planner_store`` must return a working
    in-memory backend in that case;
  - the SDK must log ONE actionable WARNING (not silent fallback,
    not "planner disabled");
  - the planner-resolution helper must still attempt resolution
    rather than crash, and bootstrap failures must surface at
    WARNING (not DEBUG) so the operator notices.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import pytest


def _reload_store_module(monkeypatch):
    """Reimport store.py with our sqlite3 patch active."""
    if "octomil.runtime.planner.store" in sys.modules:
        del sys.modules["octomil.runtime.planner.store"]
    import octomil.runtime.planner.store as store

    return store


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_factory_picks_sqlite_when_available(tmp_path, monkeypatch):
    """Happy path: real CPython, ``_sqlite3`` present → SQLite backend."""
    monkeypatch.setenv("OCTOMIL_RUNTIME_PLANNER_DB", str(tmp_path / "planner.sqlite3"))
    monkeypatch.delenv("OCTOMIL_RUNTIME_PLANNER_CACHE", raising=False)
    from octomil.runtime.planner.store import (
        SQLiteRuntimePlannerStore,
        build_runtime_planner_store,
    )

    store = build_runtime_planner_store()
    assert isinstance(store, SQLiteRuntimePlannerStore)
    store.close()


def test_factory_falls_back_to_memory_when_sqlite3_missing(monkeypatch, caplog):
    """The Ren'Py / PyInstaller failure mode: ``_sqlite3`` import
    raises ``ImportError``. The SDK MUST still construct a working
    cache (in-memory) and log ONE warning."""
    import octomil.runtime.planner.store as store_module

    # Reset the once-flag so the warning fires for this test.
    store_module._warned_about_missing_sqlite = False

    with patch.object(store_module, "_try_import_sqlite3", return_value=None):
        with caplog.at_level(logging.WARNING, logger="octomil.runtime.planner.store"):
            store = store_module.build_runtime_planner_store()

    assert isinstance(store, store_module.MemoryRuntimePlannerStore)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("sqlite cache unavailable" in r.message for r in warnings), [r.message for r in warnings]
    # Reviewer's specific bar: the message MUST NOT say "planner disabled".
    assert not any("planner disabled" in r.message.lower() for r in warnings)


def test_factory_warns_only_once_across_calls(monkeypatch, caplog):
    import octomil.runtime.planner.store as store_module

    store_module._warned_about_missing_sqlite = False

    with patch.object(store_module, "_try_import_sqlite3", return_value=None):
        with caplog.at_level(logging.WARNING, logger="octomil.runtime.planner.store"):
            store_module.build_runtime_planner_store()
            store_module.build_runtime_planner_store()
            store_module.build_runtime_planner_store()

    warnings = [r for r in caplog.records if "sqlite cache unavailable" in r.message]
    assert len(warnings) == 1, [r.message for r in warnings]


def test_factory_returns_null_when_disabled_via_env(monkeypatch):
    monkeypatch.setenv("OCTOMIL_RUNTIME_PLANNER_CACHE", "0")
    from octomil.runtime.planner.store import (
        NullRuntimePlannerStore,
        build_runtime_planner_store,
    )

    store = build_runtime_planner_store()
    assert isinstance(store, NullRuntimePlannerStore)


def test_null_store_is_not_chosen_automatically_when_sqlite_missing(monkeypatch):
    """The null store is an explicit escape hatch. Auto fallback for
    missing sqlite3 must be the in-memory store, not the null store
    — the planner still benefits from intra-process memoization."""
    import octomil.runtime.planner.store as store_module

    monkeypatch.delenv("OCTOMIL_RUNTIME_PLANNER_CACHE", raising=False)
    store_module._warned_about_missing_sqlite = False

    with patch.object(store_module, "_try_import_sqlite3", return_value=None):
        store = store_module.build_runtime_planner_store()

    assert not isinstance(store, store_module.NullRuntimePlannerStore)
    assert isinstance(store, store_module.MemoryRuntimePlannerStore)


# ---------------------------------------------------------------------------
# Memory backend behaviour
# ---------------------------------------------------------------------------


def test_memory_store_round_trips_plans():
    from octomil.runtime.planner.store import MemoryRuntimePlannerStore

    store = MemoryRuntimePlannerStore()
    key = "abc"
    store.put_plan(
        key,
        model="kokoro-en-v0_19",
        capability="tts",
        policy="local_first",
        plan_json='{"foo": "bar"}',
        source="planner",
        ttl_seconds=600,
    )
    assert store.get_plan(key) == {"foo": "bar"}


def test_memory_store_evicts_lru_when_full():
    from octomil.runtime.planner.store import MemoryRuntimePlannerStore

    store = MemoryRuntimePlannerStore(max_entries=2)
    for i in range(3):
        store.put_plan(
            f"k{i}",
            model="m",
            capability="tts",
            policy="local_first",
            plan_json=f'{{"i": {i}}}',
            source="planner",
            ttl_seconds=600,
        )
    # Oldest evicted.
    assert store.get_plan("k0") is None
    assert store.get_plan("k1") == {"i": 1}
    assert store.get_plan("k2") == {"i": 2}


def test_memory_store_expires_entries():
    from octomil.runtime.planner.store import MemoryRuntimePlannerStore

    store = MemoryRuntimePlannerStore()
    store.put_plan(
        "k",
        model="m",
        capability="tts",
        policy="local_first",
        plan_json='{"a": 1}',
        source="planner",
        ttl_seconds=-1,  # already expired
    )
    assert store.get_plan("k") is None


def test_memory_store_round_trips_benchmarks():
    from octomil.runtime.planner.store import MemoryRuntimePlannerStore

    store = MemoryRuntimePlannerStore()
    store.put_benchmark(
        "k",
        model="m",
        capability="tts",
        engine="sherpa-onnx",
        tokens_per_second=42.0,
    )
    bm = store.get_benchmark("k")
    assert bm is not None
    assert bm["tokens_per_second"] == 42.0
    assert bm["model"] == "m"


def test_memory_store_is_thread_safe():
    """Smoke test: concurrent put/get from multiple threads doesn't
    raise. The internal RLock + OrderedDict semantics are sufficient
    so this is a no-exception assertion plus a final consistency
    check on hit counters."""
    import threading

    from octomil.runtime.planner.store import MemoryRuntimePlannerStore

    store = MemoryRuntimePlannerStore(max_entries=64)
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        try:
            for j in range(50):
                key = f"k{idx}-{j}"
                store.put_plan(
                    key,
                    model="m",
                    capability="tts",
                    policy="local_first",
                    plan_json=f'{{"i": {j}}}',
                    source="t",
                    ttl_seconds=600,
                )
                store.get_plan(key)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    stats = store.stats()
    assert stats["plan_entries"] <= 64  # bounded


# ---------------------------------------------------------------------------
# Legacy alias
# ---------------------------------------------------------------------------


def test_legacy_RuntimePlannerStore_alias_returns_chosen_backend(tmp_path, monkeypatch):
    """Existing call sites do ``RuntimePlannerStore()`` and expect a
    working store back. The shim must pick the right backend
    transparently."""
    monkeypatch.setenv("OCTOMIL_RUNTIME_PLANNER_DB", str(tmp_path / "planner.sqlite3"))
    monkeypatch.delenv("OCTOMIL_RUNTIME_PLANNER_CACHE", raising=False)
    from octomil.runtime.planner.store import RuntimePlannerStore, SQLiteRuntimePlannerStore

    store = RuntimePlannerStore()
    assert isinstance(store, SQLiteRuntimePlannerStore)
    store.close()


def test_legacy_make_cache_key_static_still_works():
    from octomil.runtime.planner.store import RuntimePlannerStore

    key1 = RuntimePlannerStore._make_cache_key(model="kokoro", capability="tts", policy="local_first")
    key2 = RuntimePlannerStore._make_cache_key(model="kokoro", capability="tts", policy="local_first")
    assert key1 == key2
    assert len(key1) == 32


# ---------------------------------------------------------------------------
# Planner-resolve doesn't crash when sqlite3 is missing
# ---------------------------------------------------------------------------


def test_resolve_planner_selection_does_not_crash_when_sqlite_missing(monkeypatch):
    """The end-to-end claim: even when ``_sqlite3`` is unavailable,
    ``_resolve_planner_selection`` returns ``None`` (planner unable
    to resolve) rather than crashing the caller. The kernel's local
    fallback path then takes over."""
    import octomil.runtime.planner.store as store_module
    from octomil.execution.planner_resolution import _resolve_planner_selection

    store_module._warned_about_missing_sqlite = False

    with patch.object(store_module, "_try_import_sqlite3", return_value=None):
        # Network unavailable in test; planner resolve will return None
        # via the HTTP path. The point is it does NOT raise from
        # store construction.
        result = _resolve_planner_selection(
            model="kokoro-en-v0_19",
            capability="tts",
            policy_preset="local_first",
        )

    # None is the legitimate "planner couldn't resolve" return shape.
    # The pin here is that we got a return value, not a propagating
    # ImportError or sqlite3 attribute error.
    assert result is None or hasattr(result, "candidates")


def test_planner_bootstrap_failure_logs_warning_once(monkeypatch, caplog):
    """Bootstrap failures (import errors, planner module won't load)
    should surface at WARNING — they're configuration-actionable,
    not transient HTTP misses. And the warning fires only once
    per process to avoid log floods."""
    from octomil.execution import planner_resolution as pr

    pr._PLANNER_BOOTSTRAP_WARNED = False

    def boom_planner_resolve(*args, **kwargs):
        raise ImportError("planner module unavailable in this build")

    with patch("octomil.runtime.planner.planner.RuntimePlanner", side_effect=boom_planner_resolve):
        with caplog.at_level(logging.WARNING, logger="octomil.execution.planner_resolution"):
            pr._resolve_planner_selection("m", "tts", "local_first")
            pr._resolve_planner_selection("m", "tts", "local_first")
            pr._resolve_planner_selection("m", "tts", "local_first")

    warnings = [r for r in caplog.records if "runtime planner unavailable" in r.message]
    assert len(warnings) == 1, [r.message for r in warnings]


def test_planner_http_miss_stays_at_debug(monkeypatch, caplog):
    """HTTP misses (auth, 5xx, network) stay at DEBUG so production
    logs aren't flooded. The per-request HTTP client logs its own
    structured error."""
    from octomil.execution import planner_resolution as pr

    pr._PLANNER_BOOTSTRAP_WARNED = False

    class _BoomingPlanner:
        def resolve(self, **kwargs):
            raise RuntimeError("http 503 transient")

    with patch("octomil.runtime.planner.planner.RuntimePlanner", lambda: _BoomingPlanner()):
        with caplog.at_level(logging.WARNING, logger="octomil.execution.planner_resolution"):
            result = pr._resolve_planner_selection("m", "tts", "local_first")

    assert result is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], [r.message for r in warnings]


# ---------------------------------------------------------------------------
# Cache key includes context fields (so planner sessions don't collide)
# ---------------------------------------------------------------------------


def test_cache_key_includes_context_fields():
    """The cache key must distinguish requests across capability,
    model, policy preset, org id, key type, API base, and SDK
    version. Otherwise a planner result for org A leaks to org B."""
    from octomil.runtime.planner.store import _make_cache_key

    base = dict(
        capability="tts",
        model="kokoro-en-v0_19",
        policy="local_first",
        org_id_hash="org-a-hash",
        key_type="server",
        api_base="https://api.octomil.com/v1",
        sdk_version="4.10.1",
        device_profile="darwin-arm64",
    )
    key_a = _make_cache_key(**base)
    key_b = _make_cache_key(**{**base, "org_id_hash": "org-b-hash"})
    key_c = _make_cache_key(**{**base, "policy": "cloud_only"})
    key_d = _make_cache_key(**{**base, "api_base": "https://staging.octomil.com/v1"})
    keys = {key_a, key_b, key_c, key_d}
    assert len(keys) == 4, "every distinguishing field must change the key"


@pytest.mark.parametrize(
    "missing_field",
    ["capability", "model", "policy", "org_id_hash", "api_base", "sdk_version"],
)
def test_cache_key_changes_when_distinguishing_field_changes(missing_field):
    """Each component contributes to the key; mutating any one
    produces a different hash."""
    from octomil.runtime.planner.store import _make_cache_key

    base = dict(
        capability="tts",
        model="kokoro",
        policy="local_first",
        org_id_hash="org-a",
        api_base="https://api.octomil.com/v1",
        sdk_version="4.10.1",
    )
    a = _make_cache_key(**base)
    b = _make_cache_key(**{**base, missing_field: base[missing_field] + "X"})
    assert a != b
