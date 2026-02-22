"""
Lightweight telemetry reporter for ``edgeml serve``.

Reports inference lifecycle events (generation_started, chunk_produced,
generation_completed, generation_failed) to the EdgeML platform API.

All reporting is best-effort: failures are logged as warnings and never
propagate to the caller.  Events are dispatched on a background thread
so they do not block inference requests.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import queue
import threading
import time
import uuid
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_API_BASE = "https://api.edgeml.io/api/v1"


def _generate_device_id() -> str:
    """Derive a stable device ID from hostname + MAC address."""
    hostname = platform.node()
    try:
        mac = uuid.getnode()
        raw = f"{hostname}:{mac}"
    except Exception:
        raw = hostname
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class TelemetryReporter:
    """Best-effort telemetry reporter for inference events.

    Events are placed onto an internal queue and dispatched by a daemon
    thread so the caller is never blocked.

    Parameters
    ----------
    api_key:
        Bearer token for the EdgeML API.
    api_base:
        Base URL of the EdgeML API (should include ``/api/v1`` suffix).
    org_id:
        Organisation identifier sent with every event.
    device_id:
        Stable device identifier.  When ``None``, one is derived from
        the hostname and MAC address.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = _DEFAULT_API_BASE,
        org_id: str = "default",
        device_id: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.org_id = org_id
        self.device_id = device_id or _generate_device_id()

        self._queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue(maxsize=1024)
        self._worker = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public reporting methods
    # ------------------------------------------------------------------

    def report_generation_started(
        self,
        model_id: str,
        version: str,
        session_id: str,
        modality: str = "text",
        attention_backend: str | None = None,
    ) -> None:
        """Report that a new generation has started."""
        metrics: dict[str, Any] | None = None
        if attention_backend is not None:
            metrics = {"attention_backend": attention_backend}
        self._enqueue(
            event_type="generation_started",
            model_id=model_id,
            version=version,
            session_id=session_id,
            modality=modality,
            metrics=metrics,
        )

    def report_chunk_produced(
        self,
        session_id: str,
        model_id: str,
        version: str,
        chunk_index: int,
        ttfc_ms: float | None = None,
        chunk_latency_ms: float | None = None,
        modality: str = "text",
    ) -> None:
        """Report that a single chunk (token / frame) was produced."""
        metrics: dict[str, Any] = {"chunk_index": chunk_index}
        if ttfc_ms is not None:
            metrics["ttfc_ms"] = ttfc_ms
        if chunk_latency_ms is not None:
            metrics["chunk_latency_ms"] = chunk_latency_ms
        self._enqueue(
            event_type="chunk_produced",
            model_id=model_id,
            version=version,
            session_id=session_id,
            modality=modality,
            metrics=metrics,
        )

    def report_generation_completed(
        self,
        session_id: str,
        model_id: str,
        version: str,
        total_chunks: int,
        total_duration_ms: float,
        ttfc_ms: float,
        throughput: float,
        modality: str = "text",
        attention_backend: str | None = None,
    ) -> None:
        """Report that a generation finished successfully."""
        metrics: dict[str, Any] = {
            "total_chunks": total_chunks,
            "total_duration_ms": total_duration_ms,
            "ttfc_ms": ttfc_ms,
            "throughput": throughput,
        }
        if attention_backend is not None:
            metrics["attention_backend"] = attention_backend
        self._enqueue(
            event_type="generation_completed",
            model_id=model_id,
            version=version,
            session_id=session_id,
            modality=modality,
            metrics=metrics,
        )

    def report_generation_failed(
        self,
        session_id: str,
        model_id: str,
        version: str,
        modality: str = "text",
    ) -> None:
        """Report that a generation failed."""
        self._enqueue(
            event_type="generation_failed",
            model_id=model_id,
            version=version,
            session_id=session_id,
            modality=modality,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Signal the dispatch thread to drain remaining events and exit."""
        self._queue.put(None)  # sentinel
        self._worker.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(
        self,
        *,
        event_type: str,
        model_id: str,
        version: str,
        session_id: str,
        modality: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Build the payload and place it on the queue (non-blocking)."""
        payload: dict[str, Any] = {
            "device_id": self.device_id,
            "model_id": model_id,
            "version": version,
            "modality": modality,
            "session_id": session_id,
            "event_type": event_type,
            "timestamp_ms": int(time.time() * 1000),
            "org_id": self.org_id,
        }
        if metrics:
            payload["metrics"] = metrics
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.debug("Telemetry queue full — dropping event %s", event_type)

    def _dispatch_loop(self) -> None:
        """Background thread: pull payloads from the queue and POST them."""
        client = httpx.Client(timeout=5.0)
        url = f"{self.api_base}/inference/events"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            while True:
                payload = self._queue.get()
                if payload is None:
                    # Drain remaining items before exiting
                    while not self._queue.empty():
                        remaining = self._queue.get_nowait()
                        if remaining is not None:
                            self._send(client, url, headers, remaining)
                    break
                self._send(client, url, headers, payload)
        finally:
            client.close()

    @staticmethod
    def _send(
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> None:
        """POST a single payload — best-effort, never raises."""
        try:
            client.post(url, json=payload, headers=headers)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telemetry event dispatch failed: %s", exc)
