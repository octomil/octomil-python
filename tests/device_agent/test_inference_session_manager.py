"""Tests for InferenceSessionManager refcounting."""

from __future__ import annotations

import threading

import pytest

from octomil.device_agent.inference_session_manager import (
    InferenceSessionManager,
    SessionHandle,
)


@pytest.fixture
def mgr():
    return InferenceSessionManager()


class TestAcquireRelease:
    def test_acquire_creates_handle(self, mgr: InferenceSessionManager) -> None:
        handle = mgr.acquire("m1", "v1")
        assert handle.model_id == "m1"
        assert handle.version == "v1"
        assert handle.session_id

    def test_refcount_increments(self, mgr: InferenceSessionManager) -> None:
        mgr.acquire("m1", "v1")
        mgr.acquire("m1", "v1")
        assert mgr.get_refcount("m1", "v1") == 2

    def test_release_decrements(self, mgr: InferenceSessionManager) -> None:
        h1 = mgr.acquire("m1", "v1")
        h2 = mgr.acquire("m1", "v1")
        mgr.release(h1)
        assert mgr.get_refcount("m1", "v1") == 1
        mgr.release(h2)
        assert mgr.get_refcount("m1", "v1") == 0

    def test_double_release_safe(self, mgr: InferenceSessionManager) -> None:
        handle = mgr.acquire("m1", "v1")
        mgr.release(handle)
        mgr.release(handle)  # should not raise
        assert mgr.get_refcount("m1", "v1") == 0

    def test_refcount_zero_for_unknown(self, mgr: InferenceSessionManager) -> None:
        assert mgr.get_refcount("unknown", "v1") == 0


class TestMultipleVersions:
    def test_independent_refcounts(self, mgr: InferenceSessionManager) -> None:
        mgr.acquire("m1", "v1")
        mgr.acquire("m1", "v2")
        assert mgr.get_refcount("m1", "v1") == 1
        assert mgr.get_refcount("m1", "v2") == 1

    def test_independent_models(self, mgr: InferenceSessionManager) -> None:
        mgr.acquire("m1", "v1")
        mgr.acquire("m2", "v1")
        assert mgr.get_refcount("m1", "v1") == 1
        assert mgr.get_refcount("m2", "v1") == 1


class TestActiveSessions:
    def test_active_sessions_list(self, mgr: InferenceSessionManager) -> None:
        h1 = mgr.acquire("m1", "v1")
        h2 = mgr.acquire("m1", "v2")
        sessions = mgr.get_active_sessions()
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert h1.session_id in ids
        assert h2.session_id in ids

    def test_active_sessions_after_release(self, mgr: InferenceSessionManager) -> None:
        h1 = mgr.acquire("m1", "v1")
        mgr.release(h1)
        assert len(mgr.get_active_sessions()) == 0


class TestThreadSafety:
    def test_concurrent_acquire_release(self, mgr: InferenceSessionManager) -> None:
        handles: list[SessionHandle] = []
        lock = threading.Lock()

        def acquire_and_store() -> None:
            h = mgr.acquire("m1", "v1")
            with lock:
                handles.append(h)

        threads = [threading.Thread(target=acquire_and_store) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mgr.get_refcount("m1", "v1") == 20
        assert len(handles) == 20

        # Release all
        release_threads = [threading.Thread(target=mgr.release, args=(h,)) for h in handles]
        for t in release_threads:
            t.start()
        for t in release_threads:
            t.join()

        assert mgr.get_refcount("m1", "v1") == 0
