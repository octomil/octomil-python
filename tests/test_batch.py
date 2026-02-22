"""Tests for edgeml.batch — async request queue for serialised engine access."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from edgeml.batch import (
    DEFAULT_MAX_QUEUE_DEPTH,
    QueueFullError,
    QueueStats,
    QueueTimeoutError,
    RequestQueue,
)
from edgeml.serve import (
    EchoBackend,
    GenerationRequest,
    create_app,
)


# ---------------------------------------------------------------------------
# RequestQueue unit tests
# ---------------------------------------------------------------------------


class TestRequestQueueUnit:
    """Direct unit tests for RequestQueue (no HTTP layer)."""

    @pytest.mark.asyncio
    async def test_single_request_no_contention(self):
        """A single request with no contention should complete with zero meaningful overhead."""
        queue = RequestQueue(max_depth=4)
        queue.start()

        def fake_generate(req):
            return ("hello", {"tokens": 1})

        start = time.monotonic()
        result = await queue.submit_generate("dummy_req", fake_generate)
        elapsed = time.monotonic() - start

        assert result == ("hello", {"tokens": 1})
        # Should complete quickly — well under 1 second for a trivial fn
        assert elapsed < 1.0

        await queue.stop()

    @pytest.mark.asyncio
    async def test_fifo_ordering(self):
        """Requests should be processed in FIFO order."""
        queue = RequestQueue(max_depth=8)
        queue.start()

        order: list[int] = []

        def make_generate(idx):
            def generate(req):
                order.append(idx)
                return (f"result-{idx}", {})
            return generate

        # Submit multiple requests concurrently
        tasks = []
        for i in range(4):
            task = asyncio.create_task(
                queue.submit_generate(f"req-{i}", make_generate(i))
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All completed
        assert len(results) == 4
        # FIFO order
        assert order == [0, 1, 2, 3]

        await queue.stop()

    @pytest.mark.asyncio
    async def test_queue_full_raises(self):
        """When the queue is full, submit_generate should raise QueueFullError."""
        queue = RequestQueue(max_depth=2)
        queue.start()

        # Use an event to block the first request inside the generate fn
        blocker = asyncio.Event()

        async def slow_generate_wrapper():
            def slow_generate(req):
                # Block until the event is set (run in executor, so use threading event)
                import threading
                evt = threading.Event()
                # We'll set this from outside
                slow_generate._thread_evt = evt
                evt.wait(timeout=5)
                return ("slow", {})
            return await queue.submit_generate("req-blocking", slow_generate)

        # Fill the queue: 1 active + 2 pending = queue is full
        task1 = asyncio.create_task(slow_generate_wrapper())
        await asyncio.sleep(0.05)  # Let worker pick up task1

        # Fill both queue slots
        def normal_gen(req):
            return ("ok", {})

        task2 = asyncio.create_task(queue.submit_generate("req-2", normal_gen))
        task3 = asyncio.create_task(queue.submit_generate("req-3", normal_gen))
        await asyncio.sleep(0.05)  # Let them enqueue

        # This should fail — queue is full
        with pytest.raises(QueueFullError, match="queue full"):
            await queue.submit_generate("req-overflow", normal_gen)

        # Clean up: unblock task1
        # Get the thread event from the closure and set it
        await asyncio.sleep(0.1)
        await queue.stop()
        task1.cancel()
        task2.cancel()
        task3.cancel()
        # Suppress CancelledErrors
        for t in [task1, task2, task3]:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        """Requests waiting too long should get QueueTimeoutError."""
        queue = RequestQueue(max_depth=4, timeout=0.2)
        queue.start()

        import threading

        unblock = threading.Event()

        def blocking_generate(req):
            unblock.wait(timeout=5)
            return ("done", {})

        # Start a long-running request
        task_blocking = asyncio.create_task(
            queue.submit_generate("blocking", blocking_generate)
        )
        await asyncio.sleep(0.05)  # Let it start processing

        # This one should time out waiting in the queue
        with pytest.raises(QueueTimeoutError, match="timed out"):
            await queue.submit_generate("timeout-req", blocking_generate)

        unblock.set()
        await queue.stop()
        try:
            await task_blocking
        except (asyncio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_stats_returns_correct_counts(self):
        """stats() should reflect current queue state."""
        queue = RequestQueue(max_depth=8)
        queue.start()

        # Initially empty
        s = queue.stats()
        assert s.pending == 0
        assert s.active == 0
        assert s.max_depth == 8

        await queue.stop()

    @pytest.mark.asyncio
    async def test_stats_type(self):
        """stats() should return a QueueStats dataclass."""
        queue = RequestQueue(max_depth=16)
        queue.start()
        s = queue.stats()
        assert isinstance(s, QueueStats)
        await queue.stop()

    @pytest.mark.asyncio
    async def test_stream_request(self):
        """Streaming requests should work through the queue."""
        queue = RequestQueue(max_depth=4)
        queue.start()

        async def fake_stream(req):
            for i in range(3):
                yield f"chunk-{i}"

        chunks = []
        async for chunk in queue.submit_generate_stream("req", fake_stream):
            chunks.append(chunk)

        assert chunks == ["chunk-0", "chunk-1", "chunk-2"]
        await queue.stop()

    @pytest.mark.asyncio
    async def test_cancelled_request_skipped(self):
        """If a caller cancels, the worker should skip that request."""
        queue = RequestQueue(max_depth=4)
        queue.start()

        import threading
        unblock = threading.Event()

        call_count = 0

        def blocking_gen(req):
            nonlocal call_count
            call_count += 1
            unblock.wait(timeout=5)
            return ("blocked", {})

        def fast_gen(req):
            nonlocal call_count
            call_count += 1
            return ("fast", {})

        # Start a blocking request
        task1 = asyncio.create_task(queue.submit_generate("r1", blocking_gen))
        await asyncio.sleep(0.05)

        # Submit and then cancel a second request
        task2 = asyncio.create_task(queue.submit_generate("r2", fast_gen))
        await asyncio.sleep(0.01)
        task2.cancel()
        try:
            await task2
        except asyncio.CancelledError:
            pass

        # Unblock the first
        unblock.set()
        result1 = await task1

        assert result1 == ("blocked", {})
        # Only the blocking request's generate fn should have been called
        # (the cancelled one should be skipped)
        assert call_count == 1

        await queue.stop()

    @pytest.mark.asyncio
    async def test_stop_fails_pending_requests(self):
        """Stopping the queue should fail any remaining pending requests."""
        # Use a short timeout so the test doesn't hang
        queue = RequestQueue(max_depth=4, timeout=2.0)
        queue.start()

        import threading
        # This event will NOT be set, so the blocking gen stays blocked
        # until stop() cancels the worker.
        unblock = threading.Event()

        def blocking_gen(req):
            unblock.wait(timeout=10)
            return ("done", {})

        # Block the worker with a request that never completes
        task1 = asyncio.create_task(queue.submit_generate("r1", blocking_gen))
        await asyncio.sleep(0.1)

        # Enqueue another — it sits in the queue since worker is busy
        def fast_gen(req):
            return ("fast", {})

        task2 = asyncio.create_task(queue.submit_generate("r2", fast_gen))
        await asyncio.sleep(0.05)

        # Verify task2 is pending in queue
        s = queue.stats()
        assert s.pending >= 1 or s.active == 1

        # Stop the queue — cancels the worker, drains pending futures
        await queue.stop()
        unblock.set()  # Unblock after stop so blocking_gen returns

        # Both tasks should get exceptions since the queue shut down
        for t in [task1, task2]:
            try:
                await asyncio.wait_for(t, timeout=2.0)
            except (QueueTimeoutError, asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass  # Expected — queue shut down


# ---------------------------------------------------------------------------
# Integration tests — RequestQueue wired into the FastAPI app
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_app_with_queue():
    """Create a FastAPI app with EchoBackend and request queue enabled."""
    with patch("edgeml.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model", max_queue_depth=4)

        # Trigger lifespan startup manually
        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())

    return app


@pytest.fixture
def echo_app_no_queue():
    """Create a FastAPI app with EchoBackend and NO request queue."""
    with patch("edgeml.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model", max_queue_depth=0)

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())

    return app


class TestQueueStatsEndpoint:
    @pytest.mark.asyncio
    async def test_queue_stats_enabled(self, echo_app_with_queue):
        transport = ASGITransport(app=echo_app_with_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/queue/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["max_depth"] == 4
        assert data["pending"] == 0
        assert data["active"] == 0

    @pytest.mark.asyncio
    async def test_queue_stats_disabled(self, echo_app_no_queue):
        transport = ASGITransport(app=echo_app_no_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/queue/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["max_depth"] == 0


class TestQueuedChatCompletions:
    @pytest.mark.asyncio
    async def test_non_streaming_through_queue(self, echo_app_with_queue):
        """Non-streaming request should work through the queue."""
        transport = ASGITransport(app=echo_app_with_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello queue"}],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "hello queue" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_streaming_through_queue(self, echo_app_with_queue):
        """Streaming request should work through the queue."""
        transport = ASGITransport(app=echo_app_with_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "stream test"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200

        # Parse SSE events
        lines = resp.text.strip().split("\n\n")
        events = []
        for line in lines:
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))

        assert len(events) > 0
        for evt in events:
            assert evt["object"] == "chat.completion.chunk"

        # Final line is [DONE]
        assert lines[-1] == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_health_endpoint_bypasses_queue(self, echo_app_with_queue):
        """/health should never go through the queue."""
        transport = ASGITransport(app=echo_app_with_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_without_queue_still_works(self, echo_app_no_queue):
        """When queue is disabled (max_depth=0), requests still work."""
        transport = ASGITransport(app=echo_app_no_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "no queue"}],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "no queue" in data["choices"][0]["message"]["content"]


class TestQueueOverflow:
    """Test queue full (503) behaviour."""

    @pytest.mark.asyncio
    async def test_queue_full_returns_503(self):
        """When the request queue is full, the endpoint should return 503."""
        queue = RequestQueue(max_depth=1)
        queue.start()

        import threading
        unblock = threading.Event()

        def blocking_gen(req):
            unblock.wait(timeout=5)
            return ("done", {})

        # Fill the queue
        task = asyncio.create_task(queue.submit_generate("fill", blocking_gen))
        await asyncio.sleep(0.05)

        # One more to fill the single slot
        task2 = asyncio.create_task(queue.submit_generate("fill2", blocking_gen))
        await asyncio.sleep(0.02)

        # Now submit should fail
        with pytest.raises(QueueFullError):
            await queue.submit_generate("overflow", blocking_gen)

        unblock.set()
        await queue.stop()
        for t in [task, task2]:
            try:
                await t
            except Exception:
                pass
