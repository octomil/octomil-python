"""Engine router for unified model deployment.

``deploy()`` auto-detects and loads models from any supported engine
(ONNX, PyTorch, TFLite, CoreML, GGUF, MLX) and wraps them with
TrackedModel instrumentation for timing and metrics.

``deploy_remote()`` orchestrates cross-platform deployment via the
EdgeML registry, handling optimization and rollouts.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional

from .local import ModelRunResult, TrackedModel


class Engine(str, Enum):
    AUTO = "auto"
    ONNX = "onnx"
    TORCH = "torch"
    TFLITE = "tflite"
    COREML = "coreml"
    GGUF = "gguf"
    MLX = "mlx"


_EXT_MAP = {
    ".onnx": Engine.ONNX,
    ".pt": Engine.TORCH,
    ".pth": Engine.TORCH,
    ".tflite": Engine.TFLITE,
    ".mlmodelc": Engine.COREML,
    ".gguf": Engine.GGUF,
    ".bin": Engine.MLX,
    ".safetensors": Engine.MLX,
}

# Which engines can potentially load which file formats.
_FORMAT_ENGINES: dict[str, list[Engine]] = {
    ".onnx": [Engine.ONNX, Engine.TORCH],
    ".pt": [Engine.TORCH],
    ".pth": [Engine.TORCH],
    ".tflite": [Engine.TFLITE],
    ".mlmodelc": [Engine.COREML],
    ".gguf": [Engine.GGUF],
    ".bin": [Engine.MLX],
    ".safetensors": [Engine.MLX],
}

_IOS_DEVICES = frozenset({"iphone_15_pro", "iphone_14", "ipad_pro"})


class DeployedModel:
    """A loaded model wrapped with TrackedModel instrumentation."""

    def __init__(
        self,
        name: str,
        engine: Engine,
        predict_fn: Any,
        stream_fn: Any,
        tracked: TrackedModel,
        raw_model: Any,
        benchmark_results: Optional[dict[str, float]] = None,
    ) -> None:
        self.name = name
        self.engine = engine
        self._predict_fn = predict_fn
        self._stream_fn = stream_fn
        self._tracked = tracked
        self._raw_model = raw_model
        self.benchmark_results = benchmark_results

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference, instrumented with timing."""
        return self._tracked.run(self._predict_fn, *args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Generator:
        """Streaming inference (generator), instrumented with timing."""
        if self._stream_fn is None:
            raise NotImplementedError(
                f"Engine {self.engine.value} does not support streaming"
            )
        return self._tracked.stream(self._stream_fn, *args, **kwargs)

    @property
    def last_result(self) -> Optional[ModelRunResult]:
        return self._tracked.last_result

    def metrics(self) -> List[ModelRunResult]:
        return self._tracked.metrics()

    @property
    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine": self.engine.value,
            "format": self.engine.value,
        }

    def benchmark(self, input_data: Any, n_runs: int = 5) -> dict[str, Any]:
        """Benchmark the current engine with real input data.

        Args:
            input_data: Input matching what ``predict()`` expects.
            n_runs: Number of inference runs (default 5).

        Returns:
            Dict with engine, median_ms, min_ms, max_ms, runs.
        """
        latencies: list[float] = []
        for _ in range(n_runs):
            start = time.monotonic()
            self._predict_fn(input_data)
            latencies.append((time.monotonic() - start) * 1000)
        latencies.sort()
        return {
            "engine": self.engine.value,
            "median_ms": latencies[len(latencies) // 2],
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "runs": n_runs,
        }


# ---------------------------------------------------------------------------
# Engine loaders — all third-party imports are lazy so the SDK never
# requires any engine package at import time.
# ---------------------------------------------------------------------------


def _load_onnx(path: Path) -> tuple:
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "ONNX Runtime not installed. Run: pip install onnxruntime"
        )
    session = ort.InferenceSession(str(path))
    input_name = session.get_inputs()[0].name

    def predict(input_data: Any) -> Any:
        return session.run(None, {input_name: input_data})

    return session, predict, None


def _load_torch(path: Path) -> tuple:
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not installed. Run: pip install torch")
    model = torch.load(str(path), weights_only=False, map_location="cpu")
    if callable(model):
        model.eval()

    def predict(input_data: Any) -> Any:
        with torch.no_grad():
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data)
            return model(input_data)

    return model, predict, None


def _load_tflite(path: Path) -> tuple:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            raise ImportError(
                "TFLite not installed. Run: pip install tflite-runtime"
            )
    interp = tflite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    def predict(input_data: Any) -> Any:
        import numpy as np

        input_data = np.array(input_data, dtype=inp["dtype"])
        if input_data.shape != tuple(inp["shape"]):
            input_data = input_data.reshape(inp["shape"])
        interp.set_tensor(inp["index"], input_data)
        interp.invoke()
        return interp.get_tensor(out["index"]).copy()

    return interp, predict, None


def _load_coreml(path: Path) -> tuple:
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "CoreML Tools not installed. Run: pip install coremltools"
        )
    model = ct.models.MLModel(str(path))

    def predict(input_dict: Any) -> Any:
        return model.predict(input_dict)

    return model, predict, None


def _load_gguf(path: Path) -> tuple:
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. Run: pip install llama-cpp-python"
        )
    llm = Llama(model_path=str(path), verbose=False)

    def predict(prompt: str, **kwargs: Any) -> Any:
        return llm(prompt, **kwargs)

    def stream(prompt: str, **kwargs: Any) -> Generator:
        for chunk in llm(prompt, stream=True, **kwargs):
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    return llm, predict, stream


def _load_mlx(path: Path) -> tuple:
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_lm import load as mlx_load, generate as mlx_generate, stream_generate
    except ImportError:
        raise ImportError(
            "MLX not installed. Run: pip install mlx mlx-lm"
        )
    model, tokenizer = mlx_load(str(path))

    def predict(prompt: str, **kwargs: Any) -> Any:
        return mlx_generate(model, tokenizer, prompt=prompt, **kwargs)

    def stream(prompt: str, **kwargs: Any) -> Generator:
        for text in stream_generate(model, tokenizer, prompt=prompt, **kwargs):
            yield text

    return (model, tokenizer), predict, stream


_LOADERS: dict[Engine, Callable] = {
    Engine.ONNX: _load_onnx,
    Engine.TORCH: _load_torch,
    Engine.TFLITE: _load_tflite,
    Engine.COREML: _load_coreml,
    Engine.GGUF: _load_gguf,
    Engine.MLX: _load_mlx,
}


# ---------------------------------------------------------------------------
# Multi-engine benchmarking
# ---------------------------------------------------------------------------


def _benchmark_engine(
    engine: Engine, path: Path, n_runs: int = 3
) -> Optional[tuple[Any, Any, Any, float]]:
    """Try loading a model with *engine* and benchmark it.

    Returns ``(raw_model, predict_fn, stream_fn, median_ms)`` on success,
    or ``None`` if the engine is unavailable or loading fails.
    """
    loader = _LOADERS.get(engine)
    if loader is None:
        return None
    try:
        raw_model, predict_fn, stream_fn = loader(path)
    except (ImportError, Exception):
        return None

    # Attempt a timed run — many models need real input so this is best-effort.
    try:
        latencies: list[float] = []
        for _ in range(n_runs):
            start = time.monotonic()
            predict_fn(None)
            latencies.append((time.monotonic() - start) * 1000)
        latencies.sort()
        median = latencies[len(latencies) // 2]
        return raw_model, predict_fn, stream_fn, median
    except Exception:
        # Can't benchmark without real input — return with inf so the first
        # engine that actually loads wins.
        return raw_model, predict_fn, stream_fn, float("inf")


# ---------------------------------------------------------------------------
# deploy()
# ---------------------------------------------------------------------------


def deploy(
    model: str,
    engine: str | Engine = "auto",
    name: Optional[str] = None,
    version: Optional[str] = None,
    benchmark: bool = False,
    engines: Optional[list[str | Engine]] = None,
) -> DeployedModel:
    """Load a model file and wrap it with instrumented inference.

    Auto-detects the engine from the file extension, or specify explicitly.

    Args:
        model: Path to the model file.
        engine: Engine to use. ``"auto"`` detects from extension.
        name: Human-readable model name (defaults to file stem).
        version: Optional version tag for metrics.
        benchmark: When ``True``, try all compatible engines (or those in
            *engines*), benchmark each, and pick the fastest.
        engines: Explicit list of engines to benchmark. Only used when
            *benchmark* is ``True``.

    Returns:
        A :class:`DeployedModel` with ``.predict()`` and ``.stream()`` methods.
    """
    path = Path(model)

    # Resolve name
    if name is None:
        name = path.stem

    # ---- Benchmark mode: try multiple engines, pick fastest ---------------
    if benchmark:
        ext = path.suffix.lower()

        # Determine candidate engines
        if engines is not None:
            candidates = [Engine(e) if isinstance(e, str) else e for e in engines]
        else:
            candidates = _FORMAT_ENGINES.get(ext, [])
            if not candidates:
                supported = ", ".join(sorted(_FORMAT_ENGINES.keys()))
                raise ValueError(
                    f"No engines known for '{ext}'. Supported: {supported}"
                )

        results: dict[str, float] = {}
        best: Optional[tuple[Engine, Any, Any, Any, float]] = None

        for eng in candidates:
            outcome = _benchmark_engine(eng, path)
            if outcome is None:
                continue
            raw_model, predict_fn, stream_fn, median = outcome
            results[eng.value] = median
            if best is None or median < best[4]:
                best = (eng, raw_model, predict_fn, stream_fn, median)

        if best is None:
            tried = ", ".join(e.value for e in candidates)
            raise RuntimeError(
                f"No engine could load '{path.name}'. Tried: {tried}"
            )

        chosen_engine, raw_model, predict_fn, stream_fn, _ = best
        tracked = TrackedModel(name, format=chosen_engine.value, version=version)
        return DeployedModel(
            name=name,
            engine=chosen_engine,
            predict_fn=predict_fn,
            stream_fn=stream_fn,
            tracked=tracked,
            raw_model=raw_model,
            benchmark_results=results,
        )

    # ---- Single-engine mode (default) -------------------------------------
    if engine == "auto" or engine is Engine.AUTO:
        ext = path.suffix.lower()
        resolved = _EXT_MAP.get(ext)
        if resolved is None:
            supported = ", ".join(sorted(_EXT_MAP.keys()))
            raise ValueError(
                f"Cannot auto-detect engine for '{ext}'. Supported: {supported}"
            )
        engine = resolved
    elif isinstance(engine, str):
        engine = Engine(engine)

    loader = _LOADERS[engine]
    raw_model, predict_fn, stream_fn = loader(path)
    tracked = TrackedModel(name, format=engine.value, version=version)

    return DeployedModel(
        name=name,
        engine=engine,
        predict_fn=predict_fn,
        stream_fn=stream_fn,
        tracked=tracked,
        raw_model=raw_model,
    )


# ---------------------------------------------------------------------------
# Cross-platform deployment
# ---------------------------------------------------------------------------


class Deployment:
    """Represents a cross-platform model deployment via the EdgeML registry."""

    def __init__(
        self,
        model_id: str,
        version: str,
        targets: list[str],
        registry: Any,
    ) -> None:
        self.model_id = model_id
        self.version = version
        self.targets = targets
        self._registry = registry
        self.status: dict[str, dict[str, Any]] = {}
        self._optimization: Optional[dict[str, Any]] = None

    def advance(self, percentage: int) -> None:
        """Advance rollout to the given percentage across all targets."""
        for _target, info in self.status.items():
            rollout_id = info.get("rollout_id")
            if rollout_id is not None:
                self._registry.deploy_version(
                    model_id=self.model_id,
                    version=self.version,
                    rollout_percentage=percentage,
                    target_percentage=percentage,
                )
                info["rollout"] = percentage

    def pause(self) -> None:
        """Pause all active rollouts (sets target to current percentage)."""
        for _target, info in self.status.items():
            current = info.get("rollout", 0)
            rollout_id = info.get("rollout_id")
            if rollout_id is not None:
                self._registry.deploy_version(
                    model_id=self.model_id,
                    version=self.version,
                    rollout_percentage=current,
                    target_percentage=current,
                )

    def __repr__(self) -> str:
        targets = list(self.status.keys()) or self.targets
        return f"Deployment(model={self.model_id!r}, version={self.version!r}, targets={targets})"


def deploy_remote(
    model: str,
    version: str,
    targets: list[str],
    optimize: bool = True,
    rollout: int = 10,
    target_rollout: int = 100,
    increment_step: int = 10,
    accuracy_threshold: float = 0.95,
    size_budget_mb: Optional[float] = None,
) -> Deployment:
    """Deploy a model to multiple target devices via the EdgeML platform.

    Requires :func:`edgeml.connect` to have been called first.

    Args:
        model: Model name or model_id in the registry.
        version: Version string to deploy.
        targets: Device profiles, e.g. ``["iphone_15_pro", "pixel_8"]``.
        optimize: Run the optimization pipeline (pruning, quantization, conversion).
        rollout: Initial rollout percentage.
        target_rollout: Target rollout percentage.
        increment_step: Rollout increment step size.
        accuracy_threshold: Minimum accuracy retention for optimization (0.0–1.0).
        size_budget_mb: Maximum model size constraint in MB.

    Returns:
        A :class:`Deployment` with per-target status and rollout controls.

    Raises:
        RuntimeError: If ``edgeml.connect()`` has not been called.
    """
    from .local import _connection

    if _connection is None:
        raise RuntimeError("Call edgeml.connect(api_key=...) before deploy_remote()")

    from .registry import ModelRegistry

    registry = ModelRegistry(
        auth_token_provider=lambda: _connection.api_key,
        org_id=_connection.org_id,
        api_base=_connection.api_base,
    )

    deployment = Deployment(model, version, targets, registry)

    # Check compatibility before doing anything expensive
    compat = registry.check_compatibility(model_id=model, target_devices=targets)
    deployment._compatibility = compat
    incompatible = compat.get("incompatible_devices", [])
    if incompatible:
        raise RuntimeError(
            f"Model {model!r} is incompatible with devices: {incompatible}. "
            f"Recommendations: {compat.get('recommendations', 'N/A')}"
        )

    # Optimize if requested
    if optimize:
        deployment._optimization = registry.optimize(
            model_id=model,
            target_devices=targets,
            accuracy_threshold=accuracy_threshold,
            size_budget_mb=size_budget_mb,
        )

    # Create rollout
    rollout_result = registry.create_rollout(
        model_id=model,
        version=version,
        rollout_percentage=rollout,
        target_percentage=target_rollout,
        increment_step=increment_step,
        start_immediately=True,
    )

    # Build per-target status
    for target in targets:
        fmt = "coreml" if target in _IOS_DEVICES else "tflite"
        info: dict[str, Any] = {
            "format": fmt,
            "rollout": rollout,
            "rollout_id": rollout_result.get("id"),
            "optimized": optimize,
        }
        if deployment._optimization is not None:
            info["size_mb"] = deployment._optimization.get("optimized_size_mb")
            info["compression_ratio"] = deployment._optimization.get("compression_ratio")
        deployment.status[target] = info

    return deployment
