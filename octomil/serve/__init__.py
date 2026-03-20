"""OpenAI-compatible local inference server with multi-engine auto-benchmark.

Auto-detects available engines, benchmarks each, and picks the fastest:
  1. mlx-lm (Apple Silicon -- quantized HuggingFace models)
  2. llama-cpp-python (cross-platform -- GGUF models)
  3. MNN-LLM (Metal/Vulkan/OpenCL/CUDA -- via engine plugin)
  4. ONNX Runtime (CPU/CUDA/DirectML/TensorRT -- via engine plugin)
  5. Echo (fallback -- for testing the API layer)

Usage::

    from octomil.serve import run_server
    run_server("gemma-2b", port=8080)

    # Override engine selection:
    run_server("gemma-2b", port=8080, engine="llama.cpp")

    # Then:
    # curl localhost:8080/v1/chat/completions \\
    #   -d '{"model":"gemma-2b","messages":[{"role":"user","content":"Hi"}]}'
"""

from .app import create_app, run_server
from .backends.echo import EchoBackend
from .backends.llamacpp import LlamaCppBackend
from .backends.mlx import _DRAFT_MODEL_MAP, MLXBackend
from .config import MoEConfig, MultiModelServerState, ServerState
from .detection import _detect_backend, _get_cache_manager, _log_startup_error
from .grammar_helpers import _inject_json_system_prompt, _resolve_grammar
from .instrumentation import InstrumentedBackend, unwrap_backend
from .models import (
    _GGUF_MODELS,
    _MLX_MODELS,
    ChatCompletionBody,
    ChatMessage,
    resolve_model_name,
)
from .multi_model import (
    _MultiModelStateAdapter,
    create_multi_model_app,
    run_multi_model_server,
)
from .streaming import _queued_stream_response, _stream_response
from .types import (
    GenerationChunk,
    GenerationRequest,
    InferenceBackend,
    InferenceMetrics,
    StreamableState,
)
from .verbose_events import RuntimeEvent, VerboseEventEmitter

__all__ = [
    "ChatCompletionBody",
    "ChatMessage",
    "EchoBackend",
    "GenerationChunk",
    "GenerationRequest",
    "InferenceBackend",
    "InferenceMetrics",
    "InstrumentedBackend",
    "LlamaCppBackend",
    "MLXBackend",
    "MoEConfig",
    "MultiModelServerState",
    "ServerState",
    "StreamableState",
    "_DRAFT_MODEL_MAP",
    "_GGUF_MODELS",
    "_MLX_MODELS",
    "_MultiModelStateAdapter",
    "_detect_backend",
    "_get_cache_manager",
    "_inject_json_system_prompt",
    "_log_startup_error",
    "_queued_stream_response",
    "_resolve_grammar",
    "_stream_response",
    "create_app",
    "create_multi_model_app",
    "resolve_model_name",
    "run_multi_model_server",
    "run_server",
    "unwrap_backend",
    "RuntimeEvent",
    "VerboseEventEmitter",
]
