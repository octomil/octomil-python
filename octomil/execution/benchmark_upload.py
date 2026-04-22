"""Background benchmark telemetry upload.

Extracted from kernel.py -- contains the benchmark upload logic and
payload sanitization for the runtime planner telemetry endpoint.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

from octomil.execution.planner_resolution import _PLANNER_CAPABILITY_MAP

logger = logging.getLogger(__name__)


# Keys that must NEVER appear in benchmark upload payloads.
_BANNED_BENCHMARK_KEYS = frozenset(
    {
        "prompt",
        "input",
        "output",
        "response",
        "audio",
        "audio_data",
        "file",
        "file_path",
        "text",
        "content",
        "messages",
    }
)


def _sanitize_benchmark_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove any banned keys from a benchmark payload. Returns a clean copy."""
    return {k: v for k, v in payload.items() if k not in _BANNED_BENCHMARK_KEYS}


def _upload_benchmark_async(
    *,
    model: str,
    capability: str,
    engine: Optional[str],
    policy_preset: str,
    tokens_per_second: float = 0.0,
    ttft_ms: float = 0.0,
    latency_ms: float = 0.0,
    peak_memory_bytes: Optional[int] = None,
) -> None:
    """Upload benchmark telemetry in a background thread. Best-effort, never blocks.

    Skips upload for private policy or when no server credentials are configured.
    """
    if policy_preset in ("private", "local_only"):
        return

    api_key = os.environ.get("OCTOMIL_SERVER_KEY") or os.environ.get("OCTOMIL_API_KEY")
    if not api_key:
        return

    payload = _sanitize_benchmark_payload(
        {
            "source": "execution_kernel",
            "model": model,
            "capability": _PLANNER_CAPABILITY_MAP.get(capability, capability),
            "engine": engine or "",
            "success": True,
            "tokens_per_second": tokens_per_second,
            "ttft_ms": ttft_ms,
            "latency_ms": latency_ms,
            "peak_memory_bytes": peak_memory_bytes,
        }
    )

    def _upload() -> None:
        try:
            from octomil.runtime.planner.client import RuntimePlannerClient

            base_url = os.environ.get("OCTOMIL_API_BASE") or "https://api.octomil.com"
            client = RuntimePlannerClient(base_url=base_url, api_key=api_key)
            client.upload_benchmark(payload)
        except Exception:
            logger.debug("Background benchmark upload failed", exc_info=True)

    thread = threading.Thread(target=_upload, daemon=True, name="octomil-benchmark-upload")
    thread.start()
