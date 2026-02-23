"""Samsung ONE (On-device Neural Engine) plugin — Samsung NPU runtime for Galaxy/Exynos.

Samsung ONE is Samsung's open-source on-device inference framework targeting
Exynos NPU, GPU, and CPU backends.  It uses the ``onert`` Python API with
models packaged as ``.nnpackage`` archives (Circle/TFLite flatbuffers + manifest).

Key characteristics:
- NPU-accelerated inference on Samsung Galaxy devices (Exynos chipsets)
- CPU and GPU fallback when NPU is unavailable
- Model format: ``.nnpackage`` (zip of Circle model + MANIFEST)
- Python API: ``onert.infer.session(path, backends)``
- On-device training via ``onert.experimental.train.TrainSession``

Reference: https://github.com/Samsung/ONE
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)


def _has_onert() -> bool:
    """Check if the Samsung ONE ``onert`` Python package is importable."""
    try:
        import onert  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _get_onert_version() -> Optional[str]:
    """Return the onert package version if available."""
    try:
        import onert  # type: ignore[import-untyped]

        return getattr(onert, "__version__", None)
    except ImportError:
        return None


@dataclass
class SamsungOneMetrics:
    """Inference metrics returned alongside generated text."""

    total_tokens: int = 0
    tokens_per_second: float = 0.0
    ttfc_ms: float = 0.0


def _select_backend() -> str:
    """Select the best available Samsung ONE backend for the current system.

    Prefers NPU when running on an Exynos device, falls back to CPU.
    """
    # On Linux/Android with Exynos, the NPU device node is typically present
    npu_indicators = [
        "/dev/mali0",  # ARM Mali GPU (Exynos)
        "/sys/class/npu",  # Samsung NPU sysfs class
        "/dev/vertex0",  # Samsung NPU vertex device
    ]
    for indicator in npu_indicators:
        if os.path.exists(indicator):
            return "npu"
    return "cpu"


class SamsungOneEngine(EnginePlugin):
    """Samsung ONE inference engine — Exynos NPU/GPU/CPU accelerated.

    Uses the ``onert`` Python API to load ``.nnpackage`` models and run
    inference on Samsung hardware.  When an Exynos NPU is detected the
    engine automatically selects the ``npu`` backend; otherwise it falls
    back to ``cpu``.
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        self._backend = backend or _select_backend()

    @property
    def name(self) -> str:
        return "samsung-one"

    @property
    def display_name(self) -> str:
        backend_labels = {
            "npu": "Exynos NPU",
            "gpu": "Mali GPU",
            "cpu": "CPU",
        }
        label = backend_labels.get(self._backend, self._backend.upper())
        return f"Samsung ONE ({label})"

    @property
    def priority(self) -> int:
        # NPU is the fastest accelerator on Samsung devices — high priority
        return 18

    @property
    def backend(self) -> str:
        """Active backend name (npu, gpu, or cpu)."""
        return self._backend

    def detect(self) -> bool:
        return _has_onert()

    def detect_info(self) -> str:
        if not _has_onert():
            return ""
        parts = [f"backend: {self._backend}"]
        version = _get_onert_version()
        if version:
            parts.append(f"v{version}")
        return ", ".join(parts)

    def supports_model(self, model_name: str) -> bool:
        """Check if the model is a Samsung ONE nnpackage.

        Supports:
        - Paths ending in ``.nnpackage``
        - Directories that look like an nnpackage (contain metadata/MANIFEST)
        - Paths to ``.circle`` or ``.tflite`` model files
        """
        if model_name.endswith(".nnpackage"):
            return True
        if model_name.endswith(".circle") or model_name.endswith(".tflite"):
            return True
        # Check if it's a directory with nnpackage structure
        if os.path.isdir(model_name):
            manifest = os.path.join(model_name, "metadata", "MANIFEST")
            return os.path.isfile(manifest)
        return False

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        """Run a quick inference benchmark using the onert session API.

        Loads the model, creates dummy inputs matching the model's expected
        tensor shapes, and measures inference throughput over ``n_tokens``
        iterations (warmup + measured runs).
        """
        if not _has_onert():
            return BenchmarkResult(
                engine_name=self.name,
                error="onert package not available",
            )

        try:
            import numpy as np
            from onert import infer  # type: ignore[import-untyped]

            model_path = self._resolve_model_path(model_name)
            sess = infer.session(model_path, self._backend)

            # Build dummy inputs matching the model's expected shapes
            input_infos = sess.get_inputs_tensorinfo()
            dummy_inputs = []
            for info in input_infos:
                shape = tuple(info.dims[: info.rank])
                dummy_inputs.append(
                    np.random.rand(*shape).astype(info.dtype)
                )

            # Warmup (3 runs)
            for _ in range(3):
                sess.infer(dummy_inputs)

            # Measured runs
            start = time.monotonic()
            for _ in range(n_tokens):
                sess.infer(dummy_inputs)
            elapsed = time.monotonic() - start

            inferences_per_second = n_tokens / elapsed if elapsed > 0 else 0.0

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=inferences_per_second,
                metadata={
                    "backend": self._backend,
                    "method": "onert_session",
                    "iterations": n_tokens,
                },
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        if not _has_onert():
            raise RuntimeError(
                "Samsung ONE onert package is required for serving. "
                "Install from https://github.com/Samsung/ONE or via the "
                "Samsung ONE SDK."
            )
        backend = kwargs.pop("backend", self._backend)
        return _SamsungOneBackend(model_name, backend=backend, **kwargs)

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve a model name to a local nnpackage path."""
        # Direct nnpackage path
        if os.path.exists(model_name):
            return model_name

        # Check common local directories
        candidates = [
            os.path.expanduser(f"~/.octomil/models/{model_name}"),
            os.path.expanduser(f"~/.octomil/models/{model_name}.nnpackage"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"No nnpackage found for '{model_name}'. "
            f"Samsung ONE models must be in .nnpackage format. "
            f"Convert with: one-import-onnx -i model.onnx -o model.circle && "
            f"one-pack -i model.circle -o model.nnpackage"
        )


class _SamsungOneBackend:
    """Inference backend using Samsung ONE onert session.

    Provides the InferenceBackend interface (load_model, generate,
    list_models) on top of the onert Python API.
    """

    name = "samsung-one"

    def __init__(
        self, model_name: str, backend: str = "cpu", **kwargs: Any
    ) -> None:
        self.model_name = model_name
        self._backend = backend
        self._session: Any = None
        self._kwargs = kwargs

    def load_model(self, model_name: str) -> None:
        from onert import infer  # type: ignore[import-untyped]

        engine = SamsungOneEngine(backend=self._backend)
        model_path = engine._resolve_model_path(model_name)
        self._session = infer.session(model_path, self._backend)
        self.model_name = model_name
        logger.info(
            "Loaded %s with Samsung ONE (backend=%s)", model_name, self._backend
        )

    def generate(self, request: Any) -> tuple[str, Any]:
        """Run a single inference pass and return (text, metrics).

        The *request* parameter is currently unused — Samsung ONE models
        operate on fixed tensor shapes defined by the model.  It is
        accepted as a placeholder for future input wiring (e.g. mapping
        request fields to model input tensors).
        """
        import numpy as np

        if self._session is None:
            self.load_model(self.model_name)

        # Build inputs from the model's expected tensor info
        input_infos = self._session.get_inputs_tensorinfo()
        inputs = []
        for info in input_infos:
            shape = tuple(info.dims[: info.rank])
            inputs.append(np.ones(shape, dtype=info.dtype))

        start = time.monotonic()
        outputs = self._session.infer(inputs)
        elapsed = time.monotonic() - start

        text = str(outputs[0]) if outputs else ""
        return text, SamsungOneMetrics(
            total_tokens=1,
            tokens_per_second=1 / elapsed if elapsed > 0 else 0.0,
            ttfc_ms=elapsed * 1000,
        )

    def list_models(self) -> list[str]:
        return [self.model_name] if self.model_name else []
