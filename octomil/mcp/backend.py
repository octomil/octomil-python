"""Lazy inference backend wrapper for the MCP server.

Stdout protection is critical: MCP stdio uses stdout as the JSON-RPC
channel.  Any stray stdout output (e.g. from ``click.echo`` during model
selection) corrupts the protocol.  We redirect stdout -> stderr during
model loading and inference.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_NOT_LOADED = object()

_STDOUT_REDIRECT_LOCK = threading.RLock()


@contextmanager
def _stdout_to_stderr_guard() -> Iterator[None]:
    """Thread-safe context manager that redirects stdout to stderr.

    Prevents model-loading output (click.echo, tqdm, etc.) from
    corrupting the MCP JSON-RPC channel.
    """
    with _STDOUT_REDIRECT_LOCK:
        original = sys.stdout
        sys.stdout = sys.stderr
        try:
            yield
        finally:
            sys.stdout = original


class OctomilMCPBackend:
    """Lazy wrapper around ``_detect_backend()`` for MCP tool calls.

    The model is loaded on first ``generate()`` call, not at server
    startup.  This avoids the 5-15s load time blocking MCP handshake.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model_name: str = model or os.environ.get("OCTOMIL_MCP_MODEL") or "qwen-coder-7b"
        self._backend: object = _NOT_LOADED
        self._engine_name: str = "unknown"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._backend is not _NOT_LOADED

    def _ensure_loaded(self) -> Any:
        """Load the backend on first call, with stdout protection."""
        if self._backend is not _NOT_LOADED:
            return self._backend

        with _stdout_to_stderr_guard():
            from octomil.serve import _detect_backend

            logger.info("Loading model: %s", self._model_name)
            backend = _detect_backend(self._model_name)
            backend.load_model(self._model_name)
            self._engine_name = getattr(backend, "name", "unknown")
            self._backend = backend
            logger.info("Model loaded: %s via %s", self._model_name, self._engine_name)

        return self._backend

    def generate(
        self, messages: list[dict[str, str]], max_tokens: int = 2048, temperature: float = 0.3
    ) -> tuple[str, dict[str, Any]]:
        """Run inference and return (text, metrics_dict)."""
        from octomil.serve import GenerationRequest

        backend = self._ensure_loaded()

        with _stdout_to_stderr_guard():
            request = GenerationRequest(
                model=self._model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text, metrics = backend.generate(request)

        metrics_dict: dict[str, Any] = {
            "engine": self._engine_name,
            "model": self._model_name,
            "tokens_per_second": getattr(metrics, "tokens_per_second", 0.0),
            "total_tokens": getattr(metrics, "total_tokens", 0),
            "ttfc_ms": getattr(metrics, "ttfc_ms", 0.0),
        }
        return text, metrics_dict

    def format_metrics(self, metrics: dict[str, Any]) -> str:
        """Format metrics as a compact tag for appending to tool output."""
        model = metrics.get("model", self._model_name)
        engine = metrics.get("engine", self._engine_name)
        tps = metrics.get("tokens_per_second", 0.0)
        total = metrics.get("total_tokens", 0)
        ttfc = metrics.get("ttfc_ms", 0.0)
        return f"[Octomil: {model} via {engine} | {tps:.1f} tok/s | {total} tokens | {ttfc:.0f}ms TTFC]"
