"""MLC-LLM engine plugin — universal GPU inference via Apache TVM.

MLC-LLM (Machine Learning Compilation for LLMs) compiles models using
Apache TVM for high-performance inference on NVIDIA, AMD, Apple, and
mobile GPUs. Provides an OpenAI-compatible API out of the box.

Key features:
- GPU-accelerated on all major platforms (CUDA, Metal, Vulkan, OpenCL)
- Pre-compiled models on HuggingFace (mlc-ai org)
- OpenAI-compatible chat completions API
- Quantized models (q4f16, q4f32, q0f16, q0f32)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Models known to work with MLC-LLM — derived from the unified catalog.
from ..models.catalog import CATALOG as _UNIFIED_CATALOG

_MLC_CATALOG = {
    name for name, entry in _UNIFIED_CATALOG.items() if "mlc-llm" in entry.engines
}

# MLC-LLM quantization format suffixes
_MLC_QUANT_SUFFIXES = frozenset(
    {
        "q4f16_1",
        "q4f32_1",
        "q0f16",
        "q0f32",
        "q4f16_0",
        "q4f32_0",
        "q3f16_1",
        "q8f16_1",
    }
)


def _has_mlc_llm() -> bool:
    """Check if the mlc_llm package is importable."""
    try:
        import mlc_llm  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _get_mlc_version() -> str:
    """Return mlc_llm version string, or empty if unavailable."""
    try:
        import mlc_llm  # type: ignore[import-untyped]

        return getattr(mlc_llm, "__version__", "unknown")
    except ImportError:
        return ""


def _detect_gpu() -> Optional[str]:
    """Detect available GPU backend for MLC-LLM.

    Returns a string like 'cuda', 'metal', 'vulkan', 'opencl',
    or None if no GPU is detected.
    """
    import platform

    system = platform.system()

    # macOS: Metal is always available on Apple Silicon
    if system == "Darwin" and platform.machine() == "arm64":
        return "metal"

    # Try CUDA
    try:
        import tvm  # type: ignore[import-untyped]

        if tvm.cuda(0).exist:
            return "cuda"
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Try checking for NVIDIA GPU via subprocess as fallback
    try:
        import shutil
        import subprocess

        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return "cuda"
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Vulkan (Linux/Windows)
    if system in ("Linux", "Windows"):
        try:
            import tvm  # type: ignore[import-untyped]

            if tvm.vulkan(0).exist:
                return "vulkan"
        except (ImportError, AttributeError, RuntimeError):
            pass

    return None


class MLCEngine(EnginePlugin):
    """MLC-LLM inference engine — TVM-compiled GPU inference."""

    @property
    def name(self) -> str:
        return "mlc-llm"

    @property
    def display_name(self) -> str:
        gpu = _detect_gpu()
        accel = gpu.upper() if gpu else "GPU"
        return f"MLC-LLM ({accel})"

    @property
    def priority(self) -> int:
        return 18  # After mnn (15), before llama.cpp (20) — GPU runtime

    def detect(self) -> bool:
        """Check if mlc_llm is importable and a GPU is available."""
        if not _has_mlc_llm():
            return False
        return _detect_gpu() is not None

    def detect_info(self) -> str:
        version = _get_mlc_version()
        if not version:
            return ""
        gpu = _detect_gpu() or "no GPU"
        return f"mlc_llm {version}; {gpu}"

    def supports_model(self, model_name: str) -> bool:
        """Check if this engine can serve the given model.

        Supports:
        - Catalog names (gemma-2b, phi-mini, etc.)
        - HuggingFace repo IDs (org/model-MLC)
        - Anything ending in -MLC or containing mlc-ai/
        """
        return (
            model_name in _MLC_CATALOG
            or model_name.endswith("-MLC")
            or "mlc-ai/" in model_name
            or "/" in model_name
        )

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        """Run a quick inference benchmark using MLC-LLM engine."""
        if not _has_mlc_llm():
            return BenchmarkResult(engine_name=self.name, error="mlc_llm not available")

        gpu = _detect_gpu()
        if gpu is None:
            return BenchmarkResult(
                engine_name=self.name, error="No GPU detected for MLC-LLM"
            )

        try:
            from mlc_llm import MLCEngine as _MLCEngine  # type: ignore[import-untyped]

            repo_id = self._resolve_repo_id(model_name)
            device = gpu if gpu != "metal" else "auto"
            engine = _MLCEngine(model=repo_id, device=device)

            prompt = "Hello, how are you?"

            start = time.monotonic()
            first_token_time: Optional[float] = None
            tokens_generated = 0

            for chunk in engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=n_tokens,
                temperature=0.7,
                stream=True,
            ):
                if first_token_time is None:
                    first_token_time = time.monotonic()
                content = chunk.choices[0].delta.content
                if content:
                    tokens_generated += 1

            elapsed = time.monotonic() - start
            ttft = ((first_token_time or start) - start) * 1000
            tps = tokens_generated / elapsed if elapsed > 0 else 0.0

            # Clean up
            del engine

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                ttft_ms=ttft,
                metadata={
                    "method": "chat.completions",
                    "device": gpu,
                    "model_repo": repo_id,
                },
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        """Create an MLCBackend for serving."""
        return MLCBackend(model_name, **kwargs)

    def _resolve_repo_id(self, model_name: str) -> str:
        """Resolve a model name to a HuggingFace repo ID for MLC-LLM."""
        # Already a full repo ID
        if "/" in model_name:
            return model_name

        # Check catalog
        from ..models.catalog import CATALOG, _resolve_alias

        canonical = _resolve_alias(model_name)
        entry = CATALOG.get(canonical)
        if entry and "mlc-llm" in entry.engines:
            # Get the default quant variant's mlc repo
            variant = entry.variants.get(entry.default_quant)
            if variant and variant.mlc:
                return variant.mlc

        # Fallback: pass through as-is and let MLC-LLM handle resolution
        return model_name


class MLCBackend:
    """Inference backend using MLC-LLM for GPU-accelerated generation.

    Implements the InferenceBackend interface with generate() and
    generate_stream() methods. Uses the OpenAI-compatible chat
    completions API provided by mlc_llm.MLCEngine.
    """

    name = "mlc-llm"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._kwargs = kwargs
        self._engine: Any = None
        self._device: Optional[str] = None

    def load_model(self, model_name: str) -> None:
        """Load the model into the MLC-LLM engine."""
        from mlc_llm import MLCEngine as _MLCEngine  # type: ignore[import-untyped]

        self._model_name = model_name
        self._device = _detect_gpu() or "auto"

        # Resolve to repo ID
        engine_plugin = MLCEngine()
        repo_id = engine_plugin._resolve_repo_id(model_name)

        device = self._device if self._device != "metal" else "auto"
        logger.info("Loading MLC-LLM model: %s (device=%s)", repo_id, device)
        self._engine = _MLCEngine(model=repo_id, device=device)
        logger.info("MLC-LLM model loaded: %s", model_name)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response (non-streaming)."""
        if self._engine is None:
            self.load_model(self._model_name)

        response = self._engine.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=False,
            **kwargs,
        )

        return response.choices[0].message.content or ""

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Generate tokens in a streaming fashion, yielding each token."""
        if self._engine is None:
            self.load_model(self._model_name)

        for chunk in self._engine.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
            **kwargs,
        ):
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def unload(self) -> None:
        """Unload the model and free resources."""
        if self._engine is not None:
            logger.info("Unloading MLC-LLM model: %s", self._model_name)
            del self._engine
            self._engine = None

    def list_models(self) -> list[str]:
        """Return list of loaded model names."""
        return [self._model_name] if self._model_name else []
