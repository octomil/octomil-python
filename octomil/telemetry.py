"""
Lightweight telemetry reporter for ``octomil serve``.

Reports inference lifecycle events (generation_started, chunk_produced,
generation_completed, generation_failed) to the Octomil platform API
using the v2 OTLP envelope format.

All reporting is best-effort: failures are logged as warnings and never
propagate to the caller.  Events are dispatched on a background thread
so they do not block inference requests.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import queue
import sys
import threading
import time
import uuid
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_API_BASE = "https://api.octomil.com/api/v1"


def _generate_device_id() -> str:
    """Derive a stable device ID from hostname + MAC address."""
    hostname = platform.node()
    try:
        mac = uuid.getnode()
        raw = f"{hostname}:{mac}"
    except Exception:
        raw = hostname
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _compute_load_balance(
    activation_counts: dict[int, int],
    num_experts: int,
) -> float:
    """Compute a load balance score from expert activation counts.

    Returns a value between 0.0 (all tokens routed to one expert)
    and 1.0 (perfectly uniform distribution across experts).

    Uses coefficient of variation normalized against the worst case
    (all tokens to one expert).
    """
    if not activation_counts or num_experts <= 1:
        return 1.0

    total = sum(activation_counts.values())
    if total == 0:
        return 1.0

    # Expected count per expert if perfectly balanced
    expected = total / num_experts

    # Compute variance across all experts (including those with 0 activations)
    counts = [activation_counts.get(i, 0) for i in range(num_experts)]
    variance = sum((c - expected) ** 2 for c in counts) / num_experts
    std_dev = variance**0.5

    # Normalize: cv=0 means perfect balance, cv grows with imbalance
    if expected == 0:
        return 1.0
    cv = std_dev / expected

    # Convert to 0-1 score where 1.0 = perfect balance
    # Maximum CV for N experts (all to one) is sqrt(N-1)
    max_cv = (num_experts - 1) ** 0.5
    if max_cv == 0:
        return 1.0
    score = max(0.0, 1.0 - cv / max_cv)
    return score


def _get_sdk_version() -> str:
    """Return the SDK version, avoiding circular imports."""
    from octomil import __version__

    return __version__


def _v2_url(api_base: str) -> str:
    """Derive the v2 telemetry endpoint URL from the v1 api_base."""
    base = api_base.rstrip("/")
    if base.endswith("/api/v1"):
        return base[: -len("/api/v1")] + "/api/v2/telemetry/events"
    # Fallback: append v2 path
    return base + "/v2/telemetry/events"


class TelemetryReporter:
    """Best-effort telemetry reporter for inference events.

    Events are placed onto an internal queue and dispatched by a daemon
    thread so the caller is never blocked.  All events are sent as a
    v2 OTLP envelope to a single endpoint.

    Parameters
    ----------
    api_key:
        Bearer token for the Octomil API.
    api_base:
        Base URL of the Octomil API (should include ``/api/v1`` suffix).
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
    # Resource envelope
    # ------------------------------------------------------------------

    def _resource(self) -> dict[str, Any]:
        """Build the OTLP resource block."""
        return {
            "sdk": "python",
            "sdk_version": _get_sdk_version(),
            "device_id": self.device_id,
            "platform": sys.platform,
            "org_id": self.org_id,
        }

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
        attributes: dict[str, Any] = {
            "model.id": model_id,
            "model.version": version,
            "inference.session_id": session_id,
            "inference.modality": modality,
        }
        if attention_backend is not None:
            attributes["inference.attention_backend"] = attention_backend
        self._enqueue(name="inference.started", attributes=attributes)

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
        attributes: dict[str, Any] = {
            "model.id": model_id,
            "model.version": version,
            "inference.session_id": session_id,
            "inference.modality": modality,
            "inference.chunk_index": chunk_index,
        }
        if ttfc_ms is not None:
            attributes["inference.ttfc_ms"] = ttfc_ms
        if chunk_latency_ms is not None:
            attributes["inference.chunk_latency_ms"] = chunk_latency_ms
        self._enqueue(name="inference.chunk_produced", attributes=attributes)

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
        early_exit_stats: dict[str, Any] | None = None,
    ) -> None:
        """Report that a generation finished successfully."""
        # Compute TPOT: time per output token
        tpot_ms: float | None = None
        if total_chunks > 1 and total_duration_ms > 0:
            tpot_ms = (total_duration_ms - ttfc_ms) / (total_chunks - 1)

        attributes: dict[str, Any] = {
            "model.id": model_id,
            "model.version": version,
            "inference.session_id": session_id,
            "inference.modality": modality,
            "inference.duration_ms": total_duration_ms,
            "inference.ttft_ms": ttfc_ms,
            "inference.total_tokens": total_chunks,
            "inference.throughput_tps": throughput,
        }
        if tpot_ms is not None:
            attributes["inference.tpot_ms"] = tpot_ms
        if attention_backend is not None:
            attributes["inference.attention_backend"] = attention_backend
        if early_exit_stats is not None:
            attributes["inference.early_exit"] = early_exit_stats
        self._enqueue(name="inference.completed", attributes=attributes)

    def report_early_exit_stats(
        self,
        session_id: str,
        model_id: str,
        version: str,
        total_tokens: int,
        early_exit_tokens: int,
        exit_percentage: float,
        avg_layers_used: float,
        avg_entropy: float,
    ) -> None:
        """Report early exit / adaptive computation depth metrics."""
        attributes: dict[str, Any] = {
            "model.id": model_id,
            "model.version": version,
            "inference.session_id": session_id,
            "inference.modality": "text",
            "inference.early_exit.total_tokens": total_tokens,
            "inference.early_exit.early_exit_tokens": early_exit_tokens,
            "inference.early_exit.exit_percentage": exit_percentage,
            "inference.early_exit.avg_layers_used": avg_layers_used,
            "inference.early_exit.avg_entropy": avg_entropy,
        }
        self._enqueue(name="inference.early_exit_stats", attributes=attributes)

    def report_generation_failed(
        self,
        session_id: str,
        model_id: str,
        version: str,
        modality: str = "text",
    ) -> None:
        """Report that a generation failed."""
        attributes: dict[str, Any] = {
            "model.id": model_id,
            "model.version": version,
            "inference.session_id": session_id,
            "inference.modality": modality,
            "error.type": "generation_failed",
        }
        self._enqueue(name="inference.failed", attributes=attributes)

    def report_moe_routing(
        self,
        session_id: str,
        model_id: str,
        version: str,
        num_experts: int,
        active_experts: int,
        expert_activation_counts: dict[int, int] | None = None,
        load_balance_score: float | None = None,
        expert_memory_mb: float | None = None,
        total_tokens_routed: int = 0,
    ) -> None:
        """Report MoE expert routing telemetry for a generation."""
        attributes: dict[str, Any] = {
            "model.id": model_id,
            "model.version": version,
            "inference.session_id": session_id,
            "inference.modality": "text",
            "inference.moe.num_experts": num_experts,
            "inference.moe.active_experts": active_experts,
            "inference.moe.total_tokens_routed": total_tokens_routed,
        }
        if expert_activation_counts is not None:
            attributes["inference.moe.expert_activation_counts"] = expert_activation_counts
            if load_balance_score is None:
                load_balance_score = _compute_load_balance(
                    expert_activation_counts, num_experts
                )
        if load_balance_score is not None:
            attributes["inference.moe.load_balance_score"] = round(load_balance_score, 4)
        if expert_memory_mb is not None:
            attributes["inference.moe.expert_memory_mb"] = round(expert_memory_mb, 2)

        self._enqueue(name="inference.moe_routing", attributes=attributes)

    def report_prompt_compressed(
        self,
        session_id: str,
        model_id: str,
        version: str,
        original_tokens: int,
        compressed_tokens: int,
        compression_ratio: float,
        strategy: str,
        duration_ms: float,
        modality: str = "text",
    ) -> None:
        """Report that a prompt was compressed before inference."""
        attributes: dict[str, Any] = {
            "model.id": model_id,
            "model.version": version,
            "inference.session_id": session_id,
            "inference.modality": modality,
            "inference.compression.original_tokens": original_tokens,
            "inference.compression.compressed_tokens": compressed_tokens,
            "inference.compression.compression_ratio": compression_ratio,
            "inference.compression.tokens_saved": original_tokens - compressed_tokens,
            "inference.compression.strategy": strategy,
            "inference.compression.duration_ms": duration_ms,
        }
        self._enqueue(name="inference.prompt_compressed", attributes=attributes)

    def report_funnel_event(
        self,
        stage: str,
        success: bool = True,
        device_id: str | None = None,
        model_id: str | None = None,
        rollout_id: str | None = None,
        session_id: str | None = None,
        failure_reason: str | None = None,
        failure_category: str | None = None,
        duration_ms: int | None = None,
        platform: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Report a funnel analytics event. Non-blocking."""
        attributes: dict[str, Any] = {
            "funnel.success": success,
        }
        if device_id is not None:
            attributes["funnel.device_id"] = device_id
        if model_id is not None:
            attributes["model.id"] = model_id
        if rollout_id is not None:
            attributes["funnel.rollout_id"] = rollout_id
        if session_id is not None:
            attributes["inference.session_id"] = session_id
        if failure_reason is not None:
            attributes["error.message"] = failure_reason
        if failure_category is not None:
            attributes["error.type"] = failure_category
        if duration_ms is not None:
            attributes["funnel.duration_ms"] = duration_ms
        if platform is not None:
            attributes["funnel.platform"] = platform
        if metadata is not None:
            attributes["funnel.metadata"] = metadata
        self._enqueue(name=f"funnel.{stage}", attributes=attributes)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Signal dispatch thread to drain remaining events and exit."""
        self._queue.put(None)  # sentinel
        self._worker.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(
        self,
        *,
        name: str,
        attributes: dict[str, Any],
    ) -> None:
        """Build a v2 event and place it on the queue (non-blocking)."""
        event: dict[str, Any] = {
            "name": name,
            "timestamp_ms": int(time.time() * 1000),
            "attributes": attributes,
        }
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.debug("Telemetry queue full — dropping event %s", name)

    def _dispatch_loop(self) -> None:
        """Background thread: pull events from the queue and POST them as OTLP envelopes."""
        client = httpx.Client(timeout=5.0)
        url = _v2_url(self.api_base)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resource = self._resource()
        try:
            while True:
                event = self._queue.get()
                if event is None:
                    # Drain remaining items before exiting
                    remaining_events: list[dict[str, Any]] = []
                    while not self._queue.empty():
                        remaining = self._queue.get_nowait()
                        if remaining is not None:
                            remaining_events.append(remaining)
                    if remaining_events:
                        envelope = {"resource": resource, "events": remaining_events}
                        self._send(client, url, headers, envelope)
                    break
                envelope = {"resource": resource, "events": [event]}
                self._send(client, url, headers, envelope)
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
