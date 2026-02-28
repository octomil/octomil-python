"""Ollama engine plugin — zero-pip fallback via local Ollama server.

Ollama provides a standalone binary that manages model downloads and
inference without requiring ``pip install``.  This engine sits below
all native engines (MLX, llama.cpp, MNN, etc.) but above the useless
echo backend, giving users a working inference path even when no
Python ML packages are installed.

Priority order:
    mlx-lm (10) > mnn (15) > ... > onnxruntime (30) > **ollama (50)** > echo (999)

Detection:
    Probes ``http://localhost:11434`` (Ollama's default port).
    Falls back to checking ``which ollama`` on PATH.

Model support:
    Resolves Octomil catalog names to Ollama tags via the ``ollama``
    field on ``VariantSpec``.  Also accepts any raw Ollama tag directly.
"""

from __future__ import annotations

import logging
import shutil
import time
from typing import Any, Optional

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Build a catalog of models that have an Ollama tag in the unified catalog.
from ..models.catalog import CATALOG as _UNIFIED_CATALOG

_OLLAMA_CATALOG: dict[str, str] = {}  # octomil name -> ollama tag
for _name, _entry in _UNIFIED_CATALOG.items():
    for _variant in _entry.variants.values():
        if _variant.ollama:
            _OLLAMA_CATALOG[_name] = _variant.ollama
            break


def _is_ollama_reachable(base_url: str = "http://localhost:11434") -> bool:
    """Quick health check against the Ollama HTTP API."""
    try:
        import httpx

        resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _is_ollama_on_path() -> bool:
    """Check if the ``ollama`` binary is on PATH."""
    return shutil.which("ollama") is not None


class OllamaEngine(EnginePlugin):
    """Fallback inference engine proxying to a local Ollama server."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama (local server)"

    @property
    def priority(self) -> int:
        return 50  # After all native engines, before echo (999)

    @property
    def manages_own_download(self) -> bool:
        return True  # Ollama pulls models via its own API

    def detect(self) -> bool:
        return _is_ollama_reachable(self._base_url)

    def detect_info(self) -> str:
        if not _is_ollama_reachable(self._base_url):
            if _is_ollama_on_path():
                return "installed but not running (start with: ollama serve)"
            return ""
        try:
            import httpx

            resp = httpx.get(f"{self._base_url}/api/version", timeout=2.0)
            if resp.status_code == 200:
                version = resp.json().get("version", "unknown")
                return f"v{version}, {self._base_url}"
        except Exception:
            pass
        return self._base_url

    def supports_model(self, model_name: str) -> bool:
        from ..models.catalog import _resolve_alias

        canonical = _resolve_alias(model_name)

        # Catalog model with an Ollama tag
        if canonical in _OLLAMA_CATALOG:
            return True

        # Any name that looks like an Ollama tag (contains : or is lowercase alpha)
        # Ollama resolves its own model names, so we accept broadly
        if ":" in model_name:
            return True

        # HuggingFace repo IDs are NOT supported by Ollama
        if "/" in model_name:
            return False

        # Accept bare names — Ollama can try to resolve them
        return True

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        """Quick benchmark via Ollama /api/chat."""
        if not _is_ollama_reachable(self._base_url):
            return BenchmarkResult(
                engine_name=self.name, error="Ollama not reachable"
            )

        tag = self._resolve_tag(model_name)
        try:
            import httpx

            start = time.monotonic()
            resp = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": tag,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False,
                    "options": {"num_predict": n_tokens},
                },
                timeout=120.0,
            )
            elapsed = time.monotonic() - start
            resp.raise_for_status()

            data = resp.json()
            eval_count = data.get("eval_count", 0)

            # Use wall-clock elapsed for fair comparison with other engines
            if eval_count > 0 and elapsed > 0:
                tps = eval_count / elapsed
            else:
                tps = 0.0

            prompt_eval_ns = data.get("prompt_eval_duration", 0)
            ttft = prompt_eval_ns / 1e6 if prompt_eval_ns else 0.0

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                ttft_ms=ttft,
                metadata={
                    "ollama_tag": tag,
                    "eval_count": eval_count,
                },
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        base_url = kwargs.pop("base_url", self._base_url)
        return OllamaBackend(model_name, base_url=base_url, **kwargs)

    def _resolve_tag(self, model_name: str) -> str:
        """Resolve an Octomil model name to an Ollama tag."""
        from ..models.catalog import _resolve_alias

        canonical = _resolve_alias(model_name)
        return _OLLAMA_CATALOG.get(canonical, model_name)


class OllamaBackend:
    """Inference backend proxying to Ollama's native /api/chat endpoint.

    Uses Ollama's native API (not OpenAI compat) for better metrics.
    Implements the InferenceBackend interface from serve.py.
    """

    name = "ollama"
    attention_backend = "ollama"

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self._base_url = base_url
        self._tag: Optional[str] = None
        self._kwargs = kwargs

    def load_model(self, model_name: str) -> None:
        """Resolve catalog name to Ollama tag and auto-pull if needed."""
        from ..models.catalog import _resolve_alias
        from ..ollama import pull_ollama_model

        canonical = _resolve_alias(model_name)
        self._tag = _OLLAMA_CATALOG.get(canonical, model_name)
        self.model_name = model_name

        # Check if the model is already pulled
        try:
            import httpx

            resp = httpx.post(
                f"{self._base_url}/api/show",
                json={"name": self._tag},
                timeout=10.0,
            )
            if resp.status_code == 200:
                logger.info("Ollama model ready: %s (%s)", model_name, self._tag)
                return
        except Exception:
            pass

        # Auto-pull the model
        logger.info("Pulling Ollama model: %s", self._tag)
        pull_ollama_model(self._tag, base_url=self._base_url)
        logger.info("Ollama model pulled: %s", self._tag)

    def generate(self, request: Any) -> tuple[str, Any]:
        """Synchronous generation via POST /api/chat."""
        import httpx

        if self._tag is None:
            self.load_model(self.model_name)

        messages = request.messages if hasattr(request, "messages") else []
        max_tokens = getattr(request, "max_tokens", 512)
        temperature = getattr(request, "temperature", 0.7)

        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._tag,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data.get("message", {}).get("content", "")

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        prompt_eval_ns = data.get("prompt_eval_duration", 0)
        total_duration_ns = data.get("total_duration", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
        ttfc = prompt_eval_ns / 1e6 if prompt_eval_ns else 0.0

        # Import InferenceMetrics from serve to match interface
        from ..serve import InferenceMetrics

        metrics = InferenceMetrics(
            ttfc_ms=ttfc,
            prompt_tokens=prompt_eval_count,
            total_tokens=eval_count,
            tokens_per_second=tps,
            total_duration_ms=total_duration_ns / 1e6 if total_duration_ns else 0.0,
            attention_backend="ollama",
        )

        return text, metrics

    async def generate_stream(self, request: Any):
        """Async streaming generation via POST /api/chat with stream: true."""
        import json

        import httpx

        if self._tag is None:
            self.load_model(self.model_name)

        messages = request.messages if hasattr(request, "messages") else []
        max_tokens = getattr(request, "max_tokens", 512)
        temperature = getattr(request, "temperature", 0.7)

        # Import GenerationChunk from serve
        from ..serve import GenerationChunk

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json={
                    "model": self._tag,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = data.get("message", {}).get("content", "")
                    done = data.get("done", False)
                    if content:
                        yield GenerationChunk(
                            text=content,
                            token_count=1,
                            finish_reason="stop" if done else None,
                        )
                    if done:
                        break

    def list_models(self) -> list[str]:
        """List models available on the local Ollama instance."""
        try:
            import httpx

            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return [self.model_name] if self.model_name else []
