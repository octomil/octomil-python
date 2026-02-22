"""ONNX Runtime engine plugin — portable inference across CPU, CUDA, CoreML, and more.

ONNX Runtime is the most portable inference engine, supporting:
- CPU (all platforms)
- CUDA (NVIDIA GPUs)
- DirectML (Windows GPUs)
- CoreML (Apple Silicon / ANE)
- NNAPI (Android NPU)
- TensorRT (NVIDIA optimized)
- OpenVINO (Intel)

For LLM inference, uses ``onnxruntime-genai`` when available; falls back
to raw ``onnxruntime`` InferenceSession for non-LLM models.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Optional

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Models known to work with ONNX Runtime — derived from the unified catalog.
from ..models.catalog import CATALOG as _UNIFIED_CATALOG

_ORT_CATALOG = {
    name for name, entry in _UNIFIED_CATALOG.items() if "onnxruntime" in entry.engines
}


def _has_onnxruntime() -> bool:
    """Check if the onnxruntime package is importable."""
    try:
        import onnxruntime  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _has_onnxruntime_genai() -> bool:
    """Check if onnxruntime-genai (LLM support) is importable."""
    try:
        import onnxruntime_genai  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _get_execution_providers() -> list[str]:
    """Return the list of available ONNX Runtime execution providers."""
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        return ort.get_available_providers()
    except (ImportError, AttributeError):
        return []


class ONNXRuntimeEngine(EnginePlugin):
    """Cross-platform engine using ONNX Runtime / ONNX Runtime GenAI."""

    @property
    def name(self) -> str:
        return "onnxruntime"

    @property
    def display_name(self) -> str:
        providers = _get_execution_providers()
        # Pick the most notable accelerator for the display name
        accel = "CPU"
        for provider, label in [
            ("TensorrtExecutionProvider", "TensorRT"),
            ("CUDAExecutionProvider", "CUDA"),
            ("CoreMLExecutionProvider", "CoreML"),
            ("DmlExecutionProvider", "DirectML"),
            ("OpenVINOExecutionProvider", "OpenVINO"),
        ]:
            if provider in providers:
                accel = label
                break
        return f"ONNX Runtime ({accel})"

    @property
    def priority(self) -> int:
        return 30  # After llama.cpp (20), before echo (999)

    def detect(self) -> bool:
        return _has_onnxruntime()

    def detect_info(self) -> str:
        providers = _get_execution_providers()
        if not providers:
            return ""
        parts = [", ".join(providers)]
        if _has_onnxruntime_genai():
            parts.append("GenAI available")
        return "; ".join(parts)

    def supports_model(self, model_name: str) -> bool:
        return (
            model_name in _ORT_CATALOG
            or model_name.endswith(".onnx")
            or "/" in model_name
        )

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        # Try onnxruntime-genai first (LLMs), then raw onnxruntime
        if _has_onnxruntime_genai():
            return self._benchmark_genai(model_name, n_tokens)
        if _has_onnxruntime():
            return self._benchmark_session(model_name, n_tokens)
        return BenchmarkResult(
            engine_name=self.name, error="onnxruntime not available"
        )

    def _benchmark_genai(
        self, model_name: str, n_tokens: int
    ) -> BenchmarkResult:
        try:
            import onnxruntime_genai as og  # type: ignore[import-untyped]

            model_path = self._resolve_model_path(model_name)
            model = og.Model(model_path)
            tokenizer = og.Tokenizer(model)
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=n_tokens + 64)

            prompt = "Hello, how are you?"
            input_ids = tokenizer.encode(prompt)
            params.input_ids = input_ids

            start = time.monotonic()
            first_token_time: Optional[float] = None
            tokens_generated = 0

            generator = og.Generator(model, params)
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                if first_token_time is None:
                    first_token_time = time.monotonic()
                tokens_generated += 1
                if tokens_generated >= n_tokens:
                    break

            elapsed = time.monotonic() - start
            ttft = ((first_token_time or start) - start) * 1000
            tps = tokens_generated / elapsed if elapsed > 0 else 0.0

            del model, tokenizer, generator

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                ttft_ms=ttft,
                metadata={"method": "genai"},
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def _benchmark_session(
        self, model_name: str, n_tokens: int
    ) -> BenchmarkResult:
        try:
            import numpy as np
            import onnxruntime as ort  # type: ignore[import-untyped]

            model_path = self._resolve_model_path(model_name)
            session = ort.InferenceSession(model_path)

            # Build dummy inputs matching the model's expected shapes
            inputs: dict[str, Any] = {}
            for inp in session.get_inputs():
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                if inp.type == "tensor(int64)":
                    inputs[inp.name] = np.ones(shape, dtype=np.int64)
                elif inp.type == "tensor(float16)":
                    inputs[inp.name] = np.ones(shape, dtype=np.float16)
                else:
                    inputs[inp.name] = np.ones(shape, dtype=np.float32)

            # Warm-up run
            session.run(None, inputs)

            start = time.monotonic()
            for _ in range(n_tokens):
                session.run(None, inputs)
            elapsed = time.monotonic() - start

            tps = n_tokens / elapsed if elapsed > 0 else 0.0

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                metadata={"method": "session"},
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        return _ORTBackend(model_name, **kwargs)

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve a model name to a local path or HuggingFace cache path."""
        import os

        # Direct .onnx file
        if model_name.endswith(".onnx") and os.path.isfile(model_name):
            return model_name

        # Check common local directories
        candidates = [
            model_name,
            os.path.expanduser(f"~/.edgeml/models/{model_name}"),
            os.path.expanduser(f"~/.edgeml/models/{model_name}/model.onnx"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        # Try HuggingFace Hub download
        if "/" in model_name:
            try:
                from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

                return snapshot_download(model_name)
            except Exception:
                pass

        # Return as-is and let the caller handle errors
        return model_name


class _ORTBackend:
    """Inference backend using ONNX Runtime / ONNX Runtime GenAI.

    Uses onnxruntime-genai for LLM inference (tokenizer + iterative generation)
    and falls back to raw onnxruntime InferenceSession for non-LLM models.
    """

    name = "onnxruntime"
    attention_backend = "sdpa"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._kwargs = kwargs
        self._model: Any = None
        self._tokenizer: Any = None
        self._session: Any = None
        self._use_genai = False

    def load_model(self, model_name: str) -> None:
        self._model_name = model_name
        engine = ONNXRuntimeEngine()
        model_path = engine._resolve_model_path(model_name)

        # Prefer onnxruntime-genai for LLM models
        if _has_onnxruntime_genai():
            try:
                import onnxruntime_genai as og  # type: ignore[import-untyped]

                self._model = og.Model(model_path)
                self._tokenizer = og.Tokenizer(self._model)
                self._use_genai = True
                logger.info(
                    "Loaded %s with ONNX Runtime GenAI", model_name
                )
                return
            except Exception as exc:
                logger.debug(
                    "GenAI load failed for %s, falling back to session: %s",
                    model_name,
                    exc,
                )

        # Fallback: raw InferenceSession
        if _has_onnxruntime():
            import onnxruntime as ort  # type: ignore[import-untyped]

            self._session = ort.InferenceSession(model_path)
            self._use_genai = False
            logger.info(
                "Loaded %s with ONNX Runtime InferenceSession", model_name
            )
        else:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Install with: pip install 'edgeml-sdk[onnx]'"
            )

    def generate(
        self, request: Any
    ) -> tuple[str, Any]:
        from ..serve import InferenceMetrics

        if self._model is None and self._session is None:
            self.load_model(self._model_name)

        messages = request.messages if hasattr(request, "messages") else []
        prompt = messages[-1]["content"] if messages else ""
        max_tokens = getattr(request, "max_tokens", 512)

        if self._use_genai and self._model is not None:
            return self._generate_genai(prompt, max_tokens)
        if self._session is not None:
            return self._generate_session(prompt, max_tokens)

        return "[ort: no model loaded]", InferenceMetrics()

    def _generate_genai(
        self, prompt: str, max_tokens: int
    ) -> tuple[str, Any]:
        import onnxruntime_genai as og  # type: ignore[import-untyped]

        from ..serve import InferenceMetrics

        params = og.GeneratorParams(self._model)
        params.set_search_options(max_length=max_tokens + 64)

        input_ids = self._tokenizer.encode(prompt)
        params.input_ids = input_ids

        start = time.monotonic()
        first_token_time: Optional[float] = None
        output_tokens: list[int] = []

        generator = og.Generator(self._model, params)
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            if first_token_time is None:
                first_token_time = time.monotonic()
            token = generator.get_next_tokens()[0]
            output_tokens.append(token)
            if len(output_tokens) >= max_tokens:
                break

        elapsed = time.monotonic() - start
        ttft = ((first_token_time or start) - start) * 1000
        text = self._tokenizer.decode(output_tokens)
        tps = len(output_tokens) / elapsed if elapsed > 0 else 0.0

        return text, InferenceMetrics(
            ttfc_ms=ttft,
            total_tokens=len(output_tokens),
            tokens_per_second=tps,
            total_duration_ms=elapsed * 1000,
            attention_backend="sdpa",  # ORT GenAI uses scaled dot-product attention
        )

    def _generate_session(
        self, prompt: str, max_tokens: int
    ) -> tuple[str, Any]:
        import numpy as np

        from ..serve import InferenceMetrics

        # Basic session inference — not true LLM generation
        inputs: dict[str, Any] = {}
        for inp in self._session.get_inputs():
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            if inp.type == "tensor(int64)":
                inputs[inp.name] = np.ones(shape, dtype=np.int64)
            elif inp.type == "tensor(float16)":
                inputs[inp.name] = np.ones(shape, dtype=np.float16)
            else:
                inputs[inp.name] = np.ones(shape, dtype=np.float32)

        start = time.monotonic()
        outputs = self._session.run(None, inputs)
        elapsed = time.monotonic() - start

        # Return a string representation of the output
        text = str(outputs[0]) if outputs else ""
        return text, InferenceMetrics(
            total_tokens=1,
            tokens_per_second=1 / elapsed if elapsed > 0 else 0.0,
            total_duration_ms=elapsed * 1000,
            attention_backend="sdpa",  # ORT session uses SDPA when available
        )

    async def generate_stream(
        self,
        request: Any,
    ) -> AsyncIterator[Any]:
        import asyncio

        from ..serve import GenerationChunk

        if self._model is None and self._session is None:
            self.load_model(self._model_name)

        messages = request.messages if hasattr(request, "messages") else []
        prompt = messages[-1]["content"] if messages else ""
        max_tokens = getattr(request, "max_tokens", 512)

        if self._use_genai and self._model is not None:
            import onnxruntime_genai as og  # type: ignore[import-untyped]

            params = og.GeneratorParams(self._model)
            params.set_search_options(max_length=max_tokens + 64)

            input_ids = self._tokenizer.encode(prompt)
            params.input_ids = input_ids

            generator = og.Generator(self._model, params)
            loop = asyncio.get_event_loop()

            tokens_generated = 0
            start = time.monotonic()

            def _next_token() -> Optional[int]:
                if generator.is_done():
                    return None
                generator.compute_logits()
                generator.generate_next_token()
                return generator.get_next_tokens()[0]

            while True:
                token = await loop.run_in_executor(None, _next_token)
                if token is None:
                    yield GenerationChunk(text="", finish_reason="stop")
                    break
                tokens_generated += 1
                elapsed = time.monotonic() - start
                tps = tokens_generated / elapsed if elapsed > 0 else 0.0
                text = self._tokenizer.decode([token])
                yield GenerationChunk(
                    text=text,
                    token_count=tokens_generated,
                    tokens_per_second=tps,
                )
                if tokens_generated >= max_tokens:
                    yield GenerationChunk(
                        text="", finish_reason="length"
                    )
                    break
        else:
            # Non-LLM fallback: single output chunk
            text, _metrics = self.generate(request)
            yield GenerationChunk(
                text=text,
                token_count=1,
                finish_reason="stop",
            )

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []
