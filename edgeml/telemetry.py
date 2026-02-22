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
        early_exit_stats: dict[str, Any] | None = None,
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
        if early_exit_stats is not None:
            metrics["early_exit"] = early_exit_stats
        self._enqueue(
            event_type="generation_completed",
            model_id=model_id,
            version=version,
            session_id=session_id,
            modality=modality,
            metrics=metrics,
        )

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
        """Report early exit / adaptive computation depth metrics.

        Parameters
        ----------
        total_tokens:
            Total tokens generated in this request.
        early_exit_tokens:
            Number of tokens that exited before the final layer.
        exit_percentage:
            Percentage of tokens that exited early (0-100).
        avg_layers_used:
            Average number of transformer layers used per token.
        avg_entropy:
            Average logit entropy at exit points.
        """
        metrics: dict[str, Any] = {
            "total_tokens": total_tokens,
            "early_exit_tokens": early_exit_tokens,
            "exit_percentage": exit_percentage,
            "avg_layers_used": avg_layers_used,
            "avg_entropy": avg_entropy,
        }
        self._enqueue(
            event_type="early_exit_stats",
            model_id=model_id,
            version=version,
            session_id=session_id,
            modality="text",
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
        """Report MoE expert routing telemetry for a generation.

        Parameters
        ----------
        session_id:
            The generation session this routing data belongs to.
        model_id:
            Model identifier.
        version:
            Model version.
        num_experts:
            Total number of experts in the MoE model.
        active_experts:
            Number of experts activated per token.
        expert_activation_counts:
            Map of expert index to number of tokens routed to it.
            Used to compute load balancing statistics.
        load_balance_score:
            Load balance score (0.0 = worst, 1.0 = perfectly balanced).
            If not provided, computed from expert_activation_counts.
        expert_memory_mb:
            Memory used by loaded experts in MB.
        total_tokens_routed:
            Total number of tokens processed through expert routing.
        """
        metrics: dict[str, Any] = {
            "num_experts": num_experts,
            "active_experts": active_experts,
            "total_tokens_routed": total_tokens_routed,
        }
        if expert_activation_counts is not None:
            metrics["expert_activation_counts"] = expert_activation_counts
            if load_balance_score is None:
                load_balance_score = _compute_load_balance(
                    expert_activation_counts, num_experts
                )
        if load_balance_score is not None:
            metrics["load_balance_score"] = round(load_balance_score, 4)
        if expert_memory_mb is not None:
            metrics["expert_memory_mb"] = round(expert_memory_mb, 2)

        self._enqueue(
            event_type="moe_routing",
            model_id=model_id,
            version=version,
            session_id=session_id,
            modality="text",
            metrics=metrics,
        )

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
        metrics: dict[str, Any] = {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compression_ratio,
            "tokens_saved": original_tokens - compressed_tokens,
            "strategy": strategy,
            "compression_duration_ms": duration_ms,
        }
        self._enqueue(
            event_type="prompt_compressed",
            model_id=model_id,
            version=version,
            session_id=session_id,
            modality=modality,
            metrics=metrics,
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
