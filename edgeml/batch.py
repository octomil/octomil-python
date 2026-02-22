"""Async request queue for serialised engine access.

Provides a ``RequestQueue`` that accepts generation requests, queues them
when the engine is busy, and processes them FIFO.  Each request's result
is delivered via an ``asyncio.Future`` so callers (including SSE streams)
can ``await`` independently.

This is *not* true continuous batching (multiple sequences generating
tokens in parallel within a single engine call).  That requires deep
engine integration we don't have yet.  This is a practical first step
that serialises access and prevents request failures under concurrent
load.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_QUEUE_DEPTH = 32
DEFAULT_QUEUE_TIMEOUT_S = 60.0


class QueueFullError(Exception):
    """Raised when the request queue has reached its maximum depth."""


class QueueTimeoutError(Exception):
    """Raised when a request waited too long in the queue."""


class _RequestType(enum.Enum):
    GENERATE = "generate"
    GENERATE_STREAM = "generate_stream"


@dataclass
class _QueuedRequest:
    """Internal bookkeeping for a queued request."""

    request: Any  # GenerationRequest
    request_type: _RequestType
    future: asyncio.Future[Any]
    enqueued_at: float = field(default_factory=time.monotonic)
    cancelled: bool = False


@dataclass
class QueueStats:
    """Snapshot of queue state for the stats endpoint."""

    pending: int
    active: int
    max_depth: int


class RequestQueue:
    """Async request queue that serialises access to an inference backend.

    Parameters
    ----------
    max_depth:
        Maximum number of pending requests.  When reached, new requests
        get a ``QueueFullError``.
    timeout:
        Maximum seconds a request may wait in the queue before getting
        a ``QueueTimeoutError``.
    """

    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_QUEUE_DEPTH,
        timeout: float = DEFAULT_QUEUE_TIMEOUT_S,
    ) -> None:
        self._max_depth = max_depth
        self._timeout = timeout
        self._queue: Optional[asyncio.Queue[_QueuedRequest]] = None
        self._active: int = 0
        self._started = False
        self._worker_task: Optional[asyncio.Task[None]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the queue as started.

        The background worker is lazily created on the first submit call
        so that it runs on the correct event loop (important for test
        environments where lifespan and request handling may use different
        loops).
        """
        self._started = True

    def _ensure_worker(self) -> None:
        """Create the queue and background worker task if not already running.

        Both the ``asyncio.Queue`` and the worker ``Task`` are created
        lazily here so they bind to the event loop that is actually
        processing requests (important in test environments where
        lifespan may run on a throwaway loop).
        """
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._max_depth)
        if self._worker_task is None or self._worker_task.done():
            loop = asyncio.get_event_loop()
            self._worker_task = loop.create_task(self._worker())

    async def stop(self) -> None:
        """Cancel the worker and drain remaining requests."""
        self._started = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        # Fail any remaining queued requests
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if not item.future.done():
                        item.future.set_exception(
                            QueueTimeoutError("Queue shutting down")
                        )
                except asyncio.QueueEmpty:
                    break

    # ------------------------------------------------------------------
    # Public API — submit requests
    # ------------------------------------------------------------------

    async def submit_generate(
        self,
        request: Any,
        generate_fn: Callable[..., Any],
    ) -> Any:
        """Queue a non-streaming generate request and await its result.

        Returns the ``(text, metrics)`` tuple from ``generate_fn(request)``.

        Raises
        ------
        QueueFullError
            If the queue is at capacity.
        QueueTimeoutError
            If the request waited longer than ``self._timeout``.
        asyncio.CancelledError
            If the caller cancels (e.g. client disconnect).
        """
        self._ensure_worker()
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()
        entry = _QueuedRequest(
            request=request,
            request_type=_RequestType.GENERATE,
            future=future,
        )
        # Attach the callable so the worker can invoke it
        entry._generate_fn = generate_fn  # type: ignore[attr-defined]

        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            raise QueueFullError(
                f"Request queue full ({self._max_depth} pending). Try again later."
            )

        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            entry.cancelled = True
            raise QueueTimeoutError(
                f"Request timed out after {self._timeout}s waiting in queue."
            )
        except asyncio.CancelledError:
            entry.cancelled = True
            raise

    async def submit_generate_stream(
        self,
        request: Any,
        generate_stream_fn: Callable[..., AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """Queue a streaming generate request and return an async iterator.

        The returned iterator yields ``GenerationChunk`` objects once
        the request reaches the front of the queue and begins generating.

        Raises
        ------
        QueueFullError
            If the queue is at capacity.
        QueueTimeoutError
            If the request waited longer than ``self._timeout``.
        """
        self._ensure_worker()
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()
        entry = _QueuedRequest(
            request=request,
            request_type=_RequestType.GENERATE_STREAM,
            future=future,
        )
        entry._generate_stream_fn = generate_stream_fn  # type: ignore[attr-defined]

        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            raise QueueFullError(
                f"Request queue full ({self._max_depth} pending). Try again later."
            )

        # Wait for our turn — the worker will resolve the future with
        # a "go" signal (the async generator), or with an exception.
        try:
            result = await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            entry.cancelled = True
            raise QueueTimeoutError(
                f"Request timed out after {self._timeout}s waiting in queue."
            )
        except asyncio.CancelledError:
            entry.cancelled = True
            raise

        # result is an _StreamHandle — yield from it
        async for chunk in result:
            yield chunk

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> QueueStats:
        """Return a snapshot of queue state."""
        return QueueStats(
            pending=self._queue.qsize() if self._queue is not None else 0,
            active=self._active,
            max_depth=self._max_depth,
        )

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    async def _worker(self) -> None:
        """Drain the queue, processing one request at a time."""
        while self._started:
            try:
                entry = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Skip cancelled requests
            if entry.cancelled or entry.future.done():
                continue

            # Check timeout *before* we start work
            waited = time.monotonic() - entry.enqueued_at
            if waited > self._timeout:
                if not entry.future.done():
                    entry.future.set_exception(
                        QueueTimeoutError(
                            f"Request timed out after {waited:.1f}s in queue."
                        )
                    )
                continue

            self._active = 1
            try:
                if entry.request_type == _RequestType.GENERATE:
                    await self._process_generate(entry)
                else:
                    await self._process_generate_stream(entry)
            except Exception as exc:
                if not entry.future.done():
                    entry.future.set_exception(exc)
            finally:
                self._active = 0

    async def _process_generate(self, entry: _QueuedRequest) -> None:
        """Execute a non-streaming generate request."""
        generate_fn = entry._generate_fn  # type: ignore[attr-defined]
        loop = asyncio.get_event_loop()
        try:
            # generate() may be sync (blocking) — run in executor
            result = await loop.run_in_executor(None, generate_fn, entry.request)
            if not entry.future.done():
                entry.future.set_result(result)
        except Exception as exc:
            if not entry.future.done():
                entry.future.set_exception(exc)

    async def _process_generate_stream(self, entry: _QueuedRequest) -> None:
        """Execute a streaming generate request.

        We resolve the future with an async generator that the caller
        iterates.  The worker waits for the stream to complete before
        moving to the next request.
        """
        generate_stream_fn = entry._generate_stream_fn  # type: ignore[attr-defined]

        # Create a queue to pipe chunks from the stream to the caller
        chunk_queue: asyncio.Queue[Any] = asyncio.Queue()
        _sentinel = object()

        async def _chunk_iter() -> AsyncIterator[Any]:
            """Async iterator the caller consumes."""
            while True:
                item = await chunk_queue.get()
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

        # Resolve the future so the caller starts iterating
        if not entry.future.done():
            entry.future.set_result(_chunk_iter())

        # Pump the real stream into chunk_queue
        try:
            async for chunk in generate_stream_fn(entry.request):
                if entry.cancelled:
                    break
                await chunk_queue.put(chunk)
        except Exception as exc:
            await chunk_queue.put(exc)
        finally:
            await chunk_queue.put(_sentinel)
