"""Engine router for unified model deployment.

``deploy()`` auto-detects and loads models from any supported engine
(ONNX, PyTorch, TFLite, CoreML, GGUF, MLX) and wraps them with
TrackedModel instrumentation for timing and metrics.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Generator, List, Optional

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
    ) -> None:
        self.name = name
        self.engine = engine
        self._predict_fn = predict_fn
        self._stream_fn = stream_fn
        self._tracked = tracked
        self._raw_model = raw_model

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


_LOADERS = {
    Engine.ONNX: _load_onnx,
    Engine.TORCH: _load_torch,
    Engine.TFLITE: _load_tflite,
    Engine.COREML: _load_coreml,
    Engine.GGUF: _load_gguf,
    Engine.MLX: _load_mlx,
}


def deploy(
    model: str,
    engine: str | Engine = "auto",
    name: Optional[str] = None,
    version: Optional[str] = None,
) -> DeployedModel:
    """Load a model file and wrap it with instrumented inference.

    Auto-detects the engine from the file extension, or specify explicitly.

    Args:
        model: Path to the model file.
        engine: Engine to use. ``"auto"`` detects from extension.
        name: Human-readable model name (defaults to file stem).
        version: Optional version tag for metrics.

    Returns:
        A :class:`DeployedModel` with ``.predict()`` and ``.stream()`` methods.
    """
    path = Path(model)

    # Resolve engine
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

    # Resolve name
    if name is None:
        name = path.stem

    # Load
    loader = _LOADERS[engine]
    raw_model, predict_fn, stream_fn = loader(path)

    # Wrap with TrackedModel
    tracked = TrackedModel(name, format=engine.value, version=version)

    return DeployedModel(
        name=name,
        engine=engine,
        predict_fn=predict_fn,
        stream_fn=stream_fn,
        tracked=tracked,
        raw_model=raw_model,
    )
