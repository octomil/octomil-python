"""Inference loop — serves requests using the active model version.

Reads only local state (active_model_pointer). Never mutates model
versions or triggers downloads. Uses InferenceSessionManager for
refcounted version pinning.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..inference_session_manager import InferenceSessionManager
from ..model_registry import DeviceModelRegistry
from ..telemetry.telemetry_store import TelemetryStore

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """A request to run inference on a model."""

    model_id: str
    prompt: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    result_future: threading.Event = field(default_factory=threading.Event)
    result: Optional[dict[str, Any]] = None
    error: Optional[Exception] = None


class InferenceLoop:
    """Main inference serving loop.

    Acquires a session handle at the start of each request, pinning the
    model version. Releases on completion. The loop itself runs in a
    background thread polling for work, or can be driven externally via
    process_request().
    """

    def __init__(
        self,
        session_manager: InferenceSessionManager,
        model_registry: DeviceModelRegistry,
        telemetry_store: TelemetryStore,
        *,
        inference_fn: Optional[Callable[..., dict[str, Any]]] = None,
    ) -> None:
        self._session_manager = session_manager
        self._model_registry = model_registry
        self._telemetry_store = telemetry_store
        self._inference_fn = inference_fn
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._request_queue: queue.Queue[InferenceRequest] = queue.Queue()

    def start(self) -> None:
        """Start the inference loop in a background thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="inference-loop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the inference loop to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._running = False
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def process_request(self, request: InferenceRequest) -> dict[str, Any]:
        """Process a single inference request synchronously.

        Acquires a session handle (pinning the model version), runs inference,
        releases the handle, and logs a telemetry event. Can be called from
        any thread. If the background loop is running, the request is queued
        for processing; otherwise it is executed inline.
        """
        if self._running:
            self._request_queue.put(request)
            request.result_future.wait(timeout=300.0)
            if request.error is not None:
                raise request.error
            return request.result or {}
        else:
            return self._execute_request(request)

    def _run(self) -> None:
        """Loop body. Polls the request queue and processes requests."""
        logger.info("Inference loop started")
        while not self._stop_event.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                result = self._execute_request(request)
                request.result = result
            except Exception as exc:
                request.error = exc
            finally:
                request.result_future.set()
        logger.info("Inference loop stopped")

    def _execute_request(self, request: InferenceRequest) -> dict[str, Any]:
        """Execute a single inference request with session pinning and telemetry."""
        # Look up the active model version
        active = self._model_registry.get_active_model(request.model_id)
        if active is None:
            raise ValueError(f"No active model version for model_id={request.model_id}")

        version = active["active_version"]
        model_path = active["path"]

        # Pin the model version via session handle
        handle = self._session_manager.acquire(request.model_id, version)
        start_time = time.monotonic()
        try:
            if self._inference_fn is not None:
                result = self._inference_fn(
                    model_id=request.model_id,
                    version=version,
                    model_path=model_path,
                    prompt=request.prompt,
                    **request.kwargs,
                )
            else:
                # Stub: return a placeholder result when no inference_fn is set
                result = {
                    "model_id": request.model_id,
                    "version": version,
                    "output": f"[stub] inference on {request.model_id}@{version}",
                }

            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            # Log telemetry
            self._telemetry_store.append_auto(
                "serving.request.completed",
                {
                    "latency_ms": round(elapsed_ms, 2),
                    "model_path": model_path,
                    "prompt_length": len(request.prompt),
                },
                model_id=request.model_id,
                model_version=version,
                session_id=handle.session_id,
            )

            return result
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            self._telemetry_store.append_auto(
                "serving.request.completed",
                {
                    "latency_ms": round(elapsed_ms, 2),
                    "error": str(exc),
                },
                model_id=request.model_id,
                model_version=version,
                session_id=handle.session_id,
            )
            raise
        finally:
            self._session_manager.release(handle)
