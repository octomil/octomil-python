"""ExecuTorch engine plugin — Meta's on-device inference runtime.

ExecuTorch is the successor to PyTorch Mobile, purpose-built for edge inference.
Supports CoreML delegate (iOS), XNNPACK delegate (CPU), Vulkan delegate
(Android GPU), and Qualcomm QNN delegate via .pte exported models.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Any, Optional

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Delegate selection based on platform
_DELEGATES = {
    "coreml": {"platforms": {"Darwin"}, "description": "CoreML (Apple Neural Engine)"},
    "xnnpack": {
        "platforms": {"Darwin", "Linux", "Windows"},
        "description": "XNNPACK (CPU)",
    },
    "vulkan": {"platforms": {"Linux", "Windows"}, "description": "Vulkan (GPU)"},
    "qnn": {"platforms": {"Linux"}, "description": "Qualcomm QNN (NPU)"},
}

# Models known to have ExecuTorch exports — derived from the unified catalog.
from ..models.catalog import CATALOG as _UNIFIED_CATALOG

_ET_CATALOG = {
    name for name, entry in _UNIFIED_CATALOG.items() if "executorch" in entry.engines
}


def _has_executorch() -> bool:
    """Check if executorch Python package is available."""
    try:
        import executorch  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _get_best_delegate() -> str:
    """Select the best available delegate for the current platform."""
    sys = platform.system()

    if sys == "Darwin":
        # Prefer CoreML on macOS/iOS for ANE acceleration
        return "coreml"
    if sys == "Linux":
        # Check for Qualcomm hardware first, then Vulkan, then CPU
        try:
            import os

            if os.path.exists("/dev/kgsl-3d0"):  # Qualcomm GPU device
                return "qnn"
        except OSError:
            pass
        return "xnnpack"
    return "xnnpack"


class ExecuTorchEngine(EnginePlugin):
    """ExecuTorch inference engine — CoreML/XNNPACK/Vulkan/QNN delegates."""

    def __init__(self, delegate: Optional[str] = None) -> None:
        self._delegate = delegate or _get_best_delegate()

    @property
    def name(self) -> str:
        return "executorch"

    @property
    def display_name(self) -> str:
        delegate_info = _DELEGATES.get(self._delegate, {})
        desc = (
            delegate_info.get("description", self._delegate)
            if delegate_info
            else self._delegate
        )
        return f"ExecuTorch ({desc})"

    @property
    def priority(self) -> int:
        return 25  # After mlx-lm (10), mnn (15), llama.cpp (20)

    @property
    def delegate(self) -> str:
        """Active delegate name."""
        return self._delegate

    def detect(self) -> bool:
        return _has_executorch()

    def detect_info(self) -> str:
        if not _has_executorch():
            return ""
        parts = [f"delegate: {self._delegate}"]

        try:
            import executorch  # type: ignore[import-untyped]

            if hasattr(executorch, "__version__"):
                parts.append(f"v{executorch.__version__}")
        except (ImportError, AttributeError):
            pass

        return ", ".join(parts)

    def supports_model(self, model_name: str) -> bool:
        from ..models.catalog import _resolve_alias

        canonical = _resolve_alias(model_name)
        return (
            canonical in _ET_CATALOG
            or model_name.endswith(".pte")
            or "/" in model_name
        )

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        try:
            from executorch.runtime import Runtime  # type: ignore[import-untyped]

            runtime = Runtime.get()

            # Load the .pte model
            model_path = self._resolve_model_path(model_name)
            program = runtime.load_program(model_path)
            method = program.load_method("forward")

            # Run a simple benchmark
            import torch

            # Create dummy input (token IDs)
            input_ids = torch.randint(0, 32000, (1, 64), dtype=torch.long)

            start = time.monotonic()
            tokens_generated = 0
            for _ in range(n_tokens):
                _output = method.execute([input_ids])
                tokens_generated += 1
            elapsed = time.monotonic() - start

            tps = tokens_generated / elapsed if elapsed > 0 else 0.0

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=tps,
                metadata={"delegate": self._delegate},
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        delegate = kwargs.pop("delegate", self._delegate)
        return _ExecuTorchBackend(model_name, delegate=delegate, **kwargs)

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve a model name to a .pte file path."""
        if model_name.endswith(".pte"):
            return model_name

        # Check common export locations
        import os

        candidates = [
            f"{model_name}.pte",
            os.path.expanduser(f"~/.octomil/models/{model_name}.pte"),
            os.path.expanduser(f"~/.octomil/models/{model_name}/{self._delegate}.pte"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f"No .pte model found for '{model_name}'. "
            f"Export with: python -m executorch.examples.models.llama.export "
            f"--model {model_name} --delegate {self._delegate}"
        )


class _ExecuTorchBackend:
    """Thin wrapper around ExecuTorch runtime for the InferenceBackend interface."""

    def __init__(
        self, model_name: str, delegate: str = "xnnpack", **kwargs: Any
    ) -> None:
        self.model_name = model_name
        self._delegate = delegate
        self._method: Any = None
        self._kwargs = kwargs

    def load_model(self, model_name: str) -> None:
        from executorch.runtime import Runtime  # type: ignore[import-untyped]

        engine = ExecuTorchEngine(delegate=self._delegate)
        model_path = engine._resolve_model_path(model_name)

        runtime = Runtime.get()
        program = runtime.load_program(model_path)
        self._method = program.load_method("forward")
        self.model_name = model_name

    def generate(self, request: Any) -> tuple[str, Any]:
        if self._method is None:
            self.load_model(self.model_name)

        import torch

        prompt = request.messages[-1]["content"] if request.messages else ""
        max_tokens = getattr(request, "max_tokens", 512)

        # Simple tokenization placeholder — real impl would use a tokenizer
        input_ids = torch.randint(0, 32000, (1, len(prompt.split())), dtype=torch.long)

        start = time.monotonic()
        tokens_generated = 0
        for _ in range(max_tokens):
            output = self._method.execute([input_ids])
            tokens_generated += 1
            # In real impl: sample next token, append, check for EOS
            if output is None:
                break
        elapsed = time.monotonic() - start

        tps = tokens_generated / elapsed if elapsed > 0 else 0.0

        from dataclasses import dataclass

        @dataclass
        class Metrics:
            total_tokens: int = tokens_generated
            tokens_per_second: float = tps
            ttfc_ms: float = 0.0

        return f"[ExecuTorch output: {tokens_generated} tokens]", Metrics()

    def list_models(self) -> list[str]:
        return [self.model_name] if self.model_name else []
