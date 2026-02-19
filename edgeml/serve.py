"""
OpenAI-compatible local inference server with multi-engine auto-benchmark.

Auto-detects available engines, benchmarks each, and picks the fastest:
  1. mlx-lm (Apple Silicon — quantized HuggingFace models)
  2. llama-cpp-python (cross-platform — GGUF models)
  3. MNN-LLM (Metal/Vulkan/OpenCL/CUDA — via engine plugin)
  4. ONNX Runtime (CPU/CUDA/DirectML/TensorRT — via engine plugin)
  5. Echo (fallback — for testing the API layer)

Usage::

    from edgeml.serve import run_server
    run_server("gemma-2b", port=8080)

    # Override engine selection:
    run_server("gemma-2b", port=8080, engine="llama.cpp")

    # Then:
    # curl localhost:8080/v1/chat/completions \\
    #   -d '{"model":"gemma-2b","messages":[{"role":"user","content":"Hi"}]}'
"""

from __future__ import annotations

import json
import logging
import platform
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for OpenAI-compatible API
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionBody(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    response_format: Optional[dict[str, Any]] = None
    grammar: Optional[str] = None


# ---------------------------------------------------------------------------
# Model catalog — short names → HuggingFace repo IDs
# ---------------------------------------------------------------------------

# MLX-community quantized models (Apple Silicon)
_MLX_MODELS: dict[str, str] = {
    "gemma-1b": "mlx-community/gemma-3-1b-it-4bit",
    "gemma-4b": "mlx-community/gemma-3-4b-it-4bit",
    "gemma-12b": "mlx-community/gemma-3-12b-it-4bit",
    "gemma-27b": "mlx-community/gemma-3-27b-it-4bit",
    "llama-1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "llama-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "phi-4": "mlx-community/phi-4-4bit",
    "phi-mini": "mlx-community/Phi-3.5-mini-instruct-4bit",
    "qwen-1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "qwen-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "qwen-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mistral-7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "smollm-360m": "mlx-community/SmolLM-360M-Instruct-4bit",
}

# GGUF models for llama.cpp (cross-platform)
_GGUF_MODELS: dict[str, tuple[str, str]] = {
    # (repo_id, filename)
    "gemma-1b": ("bartowski/google_gemma-3-1b-it-GGUF", "google_gemma-3-1b-it-Q4_K_M.gguf"),
    "gemma-4b": ("bartowski/google_gemma-3-4b-it-GGUF", "google_gemma-3-4b-it-Q4_K_M.gguf"),
    "llama-1b": (
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    ),
    "llama-3b": (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
    "llama-8b": (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    "phi-mini": (
        "bartowski/Phi-3.5-mini-instruct-GGUF",
        "Phi-3.5-mini-instruct-Q4_K_M.gguf",
    ),
    "qwen-1.5b": (
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ),
    "qwen-3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "mistral-7b": (
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    ),
    "smollm-360m": (
        "bartowski/SmolLM2-360M-Instruct-GGUF",
        "SmolLM2-360M-Instruct-Q4_K_M.gguf",
    ),
}


def resolve_model_name(name: str, backend: str) -> str:
    """Resolve a short model name to a HuggingFace repo ID.

    If the name contains '/' it's assumed to be a full repo ID already.
    """
    if "/" in name:
        return name

    if backend == "mlx":
        if name in _MLX_MODELS:
            return _MLX_MODELS[name]
        raise ValueError(
            f"Unknown model '{name}' for mlx backend. "
            f"Available: {', '.join(sorted(_MLX_MODELS))}\n"
            f"Or pass a full HuggingFace repo ID (e.g. mlx-community/gemma-3-1b-it-4bit)"
        )

    if backend == "gguf":
        if name in _GGUF_MODELS:
            return name  # resolved at download time
        raise ValueError(
            f"Unknown model '{name}' for llama.cpp backend. "
            f"Available: {', '.join(sorted(_GGUF_MODELS))}\n"
            f"Or pass a path to a local .gguf file"
        )

    return name


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


@dataclass
class GenerationRequest:
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    grammar: Optional[str] = None
    json_mode: bool = False


@dataclass
class GenerationChunk:
    text: str
    token_count: int = 0
    tokens_per_second: float = 0.0
    finish_reason: Optional[str] = None


@dataclass
class InferenceMetrics:
    """Metrics for a single inference request."""

    ttfc_ms: float = 0.0
    prompt_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    total_duration_ms: float = 0.0
    cache_hit: bool = False


class InferenceBackend:
    """Base class for inference backends."""

    name: str = "base"

    def load_model(self, model_name: str) -> None:
        raise NotImplementedError

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        raise NotImplementedError

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        raise NotImplementedError
        yield  # pragma: no cover — makes this an async generator

    def list_models(self) -> list[str]:
        raise NotImplementedError


class MLXBackend(InferenceBackend):
    """Apple Silicon backend using mlx-lm.

    Loads quantized models from HuggingFace (auto-downloads + caches).
    Uses the tokenizer's built-in chat template for proper formatting.
    Supports KV cache persistence for prefix reuse across requests.
    """

    name = "mlx-lm"

    def __init__(
        self,
        cache_size_mb: int = 2048,
        cache_enabled: bool = True,
    ) -> None:
        from .cache import KVCacheManager

        self._model: Any = None
        self._tokenizer: Any = None
        self._model_name: str = ""
        self._repo_id: str = ""
        self._cache_enabled = cache_enabled
        self._kv_cache: KVCacheManager = KVCacheManager(max_cache_size_mb=cache_size_mb)

    def load_model(self, model_name: str) -> None:
        import mlx_lm  # type: ignore[import-untyped]

        self._model_name = model_name
        self._repo_id = resolve_model_name(model_name, "mlx")

        logger.info("Loading %s (%s) with mlx-lm...", model_name, self._repo_id)
        self._model, self._tokenizer = mlx_lm.load(self._repo_id)
        logger.info("Model loaded: %s", self._repo_id)

        # Collect special token strings for output filtering
        self._stop_strings: set[str] = set()
        if hasattr(self._tokenizer, "eos_token") and self._tokenizer.eos_token:
            self._stop_strings.add(self._tokenizer.eos_token)
        # Common chat model turn markers
        for marker in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>", "</s>"):
            self._stop_strings.add(marker)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Format messages using the model's chat template."""
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without a chat template
            parts: list[str] = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            return "\n".join(parts)

    def _make_sampler(self, temperature: float, top_p: float) -> Any:
        """Create an mlx-lm sampler with the given temperature and top_p."""
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-untyped]

        return make_sampler(temp=temperature, top_p=top_p)

    def _is_stop_token(self, text: str) -> bool:
        """Check if the token text is a stop/EOS marker."""
        stripped = text.strip()
        return stripped in self._stop_strings

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        import mlx_lm  # type: ignore[import-untyped]

        prompt = self._apply_chat_template(request.messages)
        prompt_token_ids = self._tokenizer.encode(prompt)
        prompt_tokens = len(prompt_token_ids)
        sampler = self._make_sampler(request.temperature, request.top_p)
        start = time.monotonic()
        first_token_time: Optional[float] = None
        tokens: list[str] = []
        final_tps: float = 0.0
        cache_hit = False

        # Check KV cache for a matching prefix
        extra_kwargs: dict[str, Any] = {}
        if self._cache_enabled:
            cached = self._kv_cache.get_cached_prefix(prompt_token_ids)
            if cached is not None:
                extra_kwargs["prompt_cache"] = cached.kv_state
                cache_hit = True
                logger.debug(
                    "MLX cache hit: reusing %d/%d prompt tokens",
                    cached.prefix_length,
                    prompt_tokens,
                )

        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
            **extra_kwargs,
        ):
            if first_token_time is None:
                first_token_time = time.monotonic()
            final_tps = response.generation_tps
            if response.finish_reason or self._is_stop_token(response.text):
                # Capture prompt_cache from the final response for storage
                if (
                    self._cache_enabled
                    and hasattr(response, "prompt_cache")
                    and response.prompt_cache is not None
                ):
                    self._kv_cache.store_prefix(prompt_token_ids, response.prompt_cache)
                break
            tokens.append(response.text)

        elapsed = time.monotonic() - start
        ttfc = ((first_token_time or start) - start) * 1000
        text = "".join(tokens)

        return text, InferenceMetrics(
            ttfc_ms=ttfc,
            prompt_tokens=prompt_tokens,
            total_tokens=len(tokens),
            tokens_per_second=final_tps,
            total_duration_ms=elapsed * 1000,
            cache_hit=cache_hit,
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio

        import mlx_lm  # type: ignore[import-untyped]

        prompt = self._apply_chat_template(request.messages)
        prompt_token_ids = self._tokenizer.encode(prompt)
        sampler = self._make_sampler(request.temperature, request.top_p)

        # Check KV cache for a matching prefix
        extra_kwargs: dict[str, Any] = {}
        if self._cache_enabled:
            cached = self._kv_cache.get_cached_prefix(prompt_token_ids)
            if cached is not None:
                extra_kwargs["prompt_cache"] = cached.kv_state

        # mlx_lm.stream_generate is a sync generator that yields
        # GenerationResponse objects as tokens are produced.
        # We run it in a thread to avoid blocking the event loop.
        gen = mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
            **extra_kwargs,
        )

        loop = asyncio.get_event_loop()

        def _next_token() -> Any:
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            response = await loop.run_in_executor(None, _next_token)
            if response is None:
                break
            if response.finish_reason or self._is_stop_token(response.text):
                # Store prompt cache from the final response
                if (
                    self._cache_enabled
                    and hasattr(response, "prompt_cache")
                    and response.prompt_cache is not None
                ):
                    self._kv_cache.store_prefix(prompt_token_ids, response.prompt_cache)
                yield GenerationChunk(
                    text="",
                    finish_reason="stop",
                )
                break
            yield GenerationChunk(
                text=response.text,
                token_count=response.generation_tokens,
                tokens_per_second=response.generation_tps,
            )

    @property
    def kv_cache(self) -> Any:
        """Expose cache manager for stats endpoint."""
        return self._kv_cache

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


class LlamaCppBackend(InferenceBackend):
    """Cross-platform backend using llama-cpp-python.

    Loads GGUF models from HuggingFace via from_pretrained (auto-downloads).
    Chat templates are handled by llama.cpp internally.
    Supports built-in LlamaCache for automatic KV prefix reuse.
    """

    name = "llama.cpp"

    def __init__(
        self,
        cache_size_mb: int = 2048,
        cache_enabled: bool = True,
    ) -> None:
        self._llm: Any = None
        self._model_name: str = ""
        self._cache_size_mb = cache_size_mb
        self._cache_enabled = cache_enabled
        self._llama_cache: Any = None

    def _attach_cache(self) -> None:
        """Attach a LlamaCache to the loaded model if caching is enabled."""
        if not self._cache_enabled or self._llm is None:
            return
        try:
            from llama_cpp import LlamaCache  # type: ignore[import-untyped]

            capacity_bytes = self._cache_size_mb * 1024 * 1024
            self._llama_cache = LlamaCache(capacity_bytes=capacity_bytes)
            self._llm.set_cache(self._llama_cache)
            logger.info("LlamaCache enabled: %d MB capacity", self._cache_size_mb)
        except (ImportError, AttributeError, TypeError) as exc:
            logger.debug("LlamaCache not available: %s", exc)

    def load_model(self, model_name: str) -> None:
        from llama_cpp import Llama  # type: ignore[import-untyped]

        self._model_name = model_name

        # Local GGUF file path
        if model_name.endswith(".gguf"):
            logger.info("Loading local GGUF: %s", model_name)
            self._llm = Llama(
                model_path=model_name,
                n_ctx=4096,
                n_gpu_layers=-1,  # offload all layers to GPU
                verbose=False,
            )
            self._attach_cache()
            return

        # Full HuggingFace repo ID (user/repo)
        if "/" in model_name:
            logger.info("Loading from HuggingFace: %s", model_name)
            self._llm = Llama.from_pretrained(
                repo_id=model_name,
                filename="*Q4_K_M.gguf",
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False,
            )
            self._attach_cache()
            return

        # Short name → catalog lookup
        if model_name not in _GGUF_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {', '.join(sorted(_GGUF_MODELS))}\n"
                f"Or pass a local .gguf path or HuggingFace repo ID."
            )

        repo_id, filename = _GGUF_MODELS[model_name]
        logger.info("Loading %s from %s (%s)...", model_name, repo_id, filename)
        self._llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        self._attach_cache()
        logger.info("Model loaded: %s", model_name)

    def _grammar_arg(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the grammar kwarg for create_chat_completion, if any."""
        if not request.grammar:
            return {}
        try:
            from llama_cpp import LlamaGrammar  # type: ignore[import-untyped]

            return {"grammar": LlamaGrammar.from_string(request.grammar)}
        except Exception as exc:
            logger.warning("Failed to compile GBNF grammar: %s", exc)
            return {}

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        start = time.monotonic()
        grammar_kw = self._grammar_arg(request)
        result = self._llm.create_chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            **grammar_kw,
        )
        elapsed = time.monotonic() - start
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return text, InferenceMetrics(
            prompt_tokens=prompt_tokens,
            total_tokens=completion_tokens,
            tokens_per_second=completion_tokens / elapsed if elapsed > 0 else 0,
            total_duration_ms=elapsed * 1000,
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio

        loop = asyncio.get_event_loop()

        grammar_kw = self._grammar_arg(request)
        # create_chat_completion with stream=True returns a sync iterator
        stream = self._llm.create_chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
            **grammar_kw,
        )

        def _next_chunk() -> Any:
            try:
                return next(stream)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, _next_chunk)
            if chunk is None:
                break
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            finish = chunk["choices"][0].get("finish_reason")
            if content or finish:
                yield GenerationChunk(text=content, finish_reason=finish)

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


class EchoBackend(InferenceBackend):
    """Fallback backend that echoes input — useful for testing the API layer."""

    name = "echo"

    def __init__(self) -> None:
        self._model_name: str = ""

    def load_model(self, model_name: str) -> None:
        self._model_name = model_name
        logger.warning(
            "No inference backend available. Using echo backend for '%s'. "
            "Install mlx-lm (Apple Silicon) or llama-cpp-python for real inference: "
            "pip install 'edgeml-sdk[mlx]' or pip install 'edgeml-sdk[llama]'",
            model_name,
        )

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        last_msg = request.messages[-1]["content"] if request.messages else ""
        text = f"[echo:{self._model_name}] {last_msg}"
        return text, InferenceMetrics(total_tokens=len(text.split()))

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio

        last_msg = request.messages[-1]["content"] if request.messages else ""
        words = f"[echo:{self._model_name}] {last_msg}".split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield GenerationChunk(
                text=word + ("" if is_last else " "),
                finish_reason="stop" if is_last else None,
            )
            await asyncio.sleep(0.02)

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


def _detect_backend(
    model_name: str,
    *,
    cache_size_mb: int = 2048,
    cache_enabled: bool = True,
    engine_override: Optional[str] = None,
) -> InferenceBackend:
    """Auto-detect engines, benchmark each, and return the fastest backend.

    Uses the engine registry plugin system. Each registered engine is:
    1. Detected (is the library installed? does it support this model?)
    2. Benchmarked (quick 32-token generation to measure tok/s)
    3. Ranked (highest tok/s wins)

    If engine_override is set, skip benchmarking and use that engine directly.
    """
    from .engines import get_registry

    registry = get_registry()

    backend_kwargs = {
        "cache_size_mb": cache_size_mb,
        "cache_enabled": cache_enabled,
    }

    if engine_override:
        engine = registry.get_engine(engine_override)
        if engine is None:
            available = [e.name for e in registry.engines]
            raise ValueError(
                f"Unknown engine '{engine_override}'. "
                f"Available: {', '.join(available)}"
            )
        return engine.create_backend(model_name, **backend_kwargs)

    # Detect all available engines for this model
    detections = registry.detect_all(model_name)
    available_engines = [d.engine for d in detections if d.available]

    for d in detections:
        if d.available:
            logger.info("Engine detected: %s (%s)", d.engine.name, d.info)
        else:
            logger.debug("Engine unavailable: %s", d.engine.name)

    # No real engines → echo fallback
    real_engines = [e for e in available_engines if e.name != "echo"]
    if not real_engines:
        echo = EchoBackend()
        echo.load_model(model_name)
        return echo

    # Benchmark real engines and pick fastest
    ranked = registry.benchmark_all(model_name, n_tokens=32, engines=real_engines)
    best = registry.select_best(ranked)
    if best is None:
        echo = EchoBackend()
        echo.load_model(model_name)
        return echo

    return best.engine.create_backend(model_name, **backend_kwargs)




def _log_startup_error(model_name: str, exc: Exception) -> None:
    """Print a human-readable startup error instead of a raw traceback."""
    err_type = type(exc).__name__
    err_msg = str(exc)

    # HuggingFace auth / repo errors
    if "RepositoryNotFoundError" in err_type or "401" in err_msg or "403" in err_msg:
        logger.error(
            "Failed to load model '%s': HuggingFace authentication required.\n"
            "  Fix: Run `huggingface-cli login` or set the HF_TOKEN env var.\n"
            "  Get a token at https://huggingface.co/settings/tokens",
            model_name,
        )
    elif "404" in err_msg or "not found" in err_msg.lower():
        logger.error(
            "Failed to load model '%s': model not found on HuggingFace.\n"
            "  Check the model name and try a full repo ID:\n"
            "    edgeml serve mlx-community/gemma-3-1b-it-4bit\n"
            "  List available short names:\n"
            "    edgeml serve --help",
            model_name,
        )
    elif "Unknown model" in err_msg:
        logger.error("Failed to load model: %s", err_msg)
    else:
        logger.error(
            "Failed to start server for model '%s': %s\n"
            "  If this is a HuggingFace model, try: huggingface-cli login",
            model_name,
            exc,
        )


def _get_cache_manager(backend: InferenceBackend) -> Any:
    """Extract the KVCacheManager from a backend, if available."""
    if isinstance(backend, MLXBackend):
        return backend.kv_cache
    return None


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    """Shared mutable state for the serve app."""

    backend: Optional[InferenceBackend] = None
    model_name: str = ""
    engine_name: str = ""
    start_time: float = field(default_factory=time.time)
    request_count: int = 0
    api_key: Optional[str] = None
    api_base: str = "https://api.edgeml.io/api/v1"
    default_json_mode: bool = False
    cache_size_mb: int = 2048
    cache_enabled: bool = True
    engine_override: Optional[str] = None


def _resolve_grammar(
    body: ChatCompletionBody, default_json_mode: bool = False
) -> tuple[Optional[str], bool]:
    """Determine the GBNF grammar string and json_mode flag from a request.

    Returns (grammar_string_or_None, is_json_mode).
    """
    from .grammar import json_mode_grammar, json_schema_to_gbnf

    # Explicit grammar takes precedence
    if body.grammar:
        return body.grammar, False

    rf = body.response_format
    if rf is None and default_json_mode:
        rf = {"type": "json_object"}

    if rf is None:
        return None, False

    fmt_type = rf.get("type")
    if fmt_type == "json_object":
        return json_mode_grammar(), True
    if fmt_type == "json_schema":
        schema = rf.get("json_schema") or rf.get("schema")
        if schema:
            # The OpenAI API wraps the actual schema under a "schema" key
            # inside json_schema. Handle both nesting levels.
            actual_schema = schema.get("schema", schema)
            return json_schema_to_gbnf(actual_schema), True
        return json_mode_grammar(), True

    return None, False


def _inject_json_system_prompt(
    messages: list[dict[str, str]],
    schema: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    """Prepend a JSON-mode system prompt if one isn't already present."""
    from .grammar import json_system_prompt

    # Don't double-inject
    if messages and messages[0].get("role") == "system":
        existing = messages[0].get("content", "")
        if "JSON" in existing or "json" in existing:
            return messages

    prompt = json_system_prompt(schema)
    return [{"role": "system", "content": prompt}] + list(messages)


def create_app(
    model_name: str,
    *,
    api_key: Optional[str] = None,
    api_base: str = "https://api.edgeml.io/api/v1",
    json_mode: bool = False,
    cache_size_mb: int = 2048,
    cache_enabled: bool = True,
    engine: Optional[str] = None,
) -> Any:
    """Create a FastAPI app with OpenAI-compatible endpoints.

    Parameters
    ----------
    json_mode:
        When ``True``, all requests default to ``response_format={"type": "json_object"}``
        unless the caller explicitly sets a different ``response_format``.
    engine:
        Force a specific engine (e.g. ``"mlx-lm"``, ``"llama.cpp"``).
        When ``None``, auto-benchmarks all available engines and picks fastest.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    state = ServerState(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        default_json_mode=json_mode,
        cache_size_mb=cache_size_mb,
        cache_enabled=cache_enabled,
        engine_override=engine,
    )

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        try:
            state.backend = _detect_backend(
                model_name,
                cache_size_mb=state.cache_size_mb,
                cache_enabled=state.cache_enabled,
                engine_override=state.engine_override,
            )
        except Exception as exc:
            _log_startup_error(model_name, exc)
            raise
        state.engine_name = state.backend.name if state.backend else "none"
        state.start_time = time.time()
        yield

    app = FastAPI(title="EdgeML Serve", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        models = state.backend.list_models() if state.backend else []
        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": int(state.start_time),
                    "owned_by": "edgeml",
                }
                for m in models
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionBody) -> Any:
        if state.backend is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        grammar_str, is_json = _resolve_grammar(body, state.default_json_mode)

        messages = [{"role": m.role, "content": m.content} for m in body.messages]

        # For backends without native grammar support (MLX, echo),
        # inject a system prompt nudging JSON output when json_mode is on.
        uses_grammar_natively = isinstance(state.backend, LlamaCppBackend)
        schema_for_prompt: Optional[dict[str, Any]] = None
        if is_json and not uses_grammar_natively:
            rf = body.response_format or {}
            if rf.get("type") == "json_schema":
                raw = rf.get("json_schema") or rf.get("schema")
                schema_for_prompt = raw.get("schema", raw) if raw else None
            messages = _inject_json_system_prompt(messages, schema_for_prompt)

        gen_req = GenerationRequest(
            model=body.model or state.model_name,
            messages=messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stream=body.stream,
            grammar=grammar_str if uses_grammar_natively else None,
            json_mode=is_json,
        )

        state.request_count += 1
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if gen_req.stream:
            return StreamingResponse(
                _stream_response(state, gen_req, req_id),
                media_type="text/event-stream",
            )

        text, metrics = state.backend.generate(gen_req)

        # JSON validation + retry for non-grammar backends
        if is_json and not uses_grammar_natively:
            from .grammar import extract_json, validate_json_output

            if not validate_json_output(text):
                extracted = extract_json(text)
                if extracted is not None:
                    text = json.dumps(extracted)
                else:
                    # Retry once with a stronger system prompt
                    retry_messages = _inject_json_system_prompt(
                        messages, schema_for_prompt
                    )
                    retry_req = GenerationRequest(
                        model=gen_req.model,
                        messages=retry_messages,
                        max_tokens=gen_req.max_tokens,
                        temperature=max(gen_req.temperature - 0.2, 0.0),
                        top_p=gen_req.top_p,
                        stream=False,
                        json_mode=True,
                    )
                    text, metrics = state.backend.generate(retry_req)
                    # Best-effort extraction on retry
                    if not validate_json_output(text):
                        extracted = extract_json(text)
                        if extracted is not None:
                            text = json.dumps(extracted)

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": gen_req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.total_tokens,
                "total_tokens": metrics.prompt_tokens + metrics.total_tokens,
                "cache_hit": metrics.cache_hit,
            },
        }

    @app.get("/v1/cache/stats")
    async def cache_stats() -> dict[str, Any]:
        """Return KV cache statistics."""
        if state.backend is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        cache_mgr = _get_cache_manager(state.backend)
        if cache_mgr is None:
            return {
                "enabled": False,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "entries": 0,
                "memory_mb": 0.0,
                "max_memory_mb": state.cache_size_mb,
            }

        stats = cache_mgr.stats()
        return {
            "enabled": True,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": round(stats.hit_rate, 4),
            "entries": stats.entries,
            "memory_mb": round(stats.memory_mb, 2),
            "max_memory_mb": state.cache_size_mb,
        }

    @app.get("/v1/engines")
    async def list_engines() -> dict[str, Any]:
        """List detected engines and their benchmark results."""
        from .engines import get_registry

        registry = get_registry()
        detections = registry.detect_all(state.model_name)

        engines_list = []
        for d in detections:
            entry: dict[str, Any] = {
                "name": d.engine.name,
                "display_name": d.engine.display_name,
                "available": d.available,
                "info": d.info,
                "active": (
                    state.backend is not None and state.backend.name == d.engine.name
                ),
            }
            engines_list.append(entry)

        return {
            "active_engine": state.engine_name,
            "engines": engines_list,
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        cache_info: dict[str, Any] = {"enabled": state.cache_enabled}
        cache_mgr = (
            _get_cache_manager(state.backend) if state.backend is not None else None
        )
        if cache_mgr is not None:
            cs = cache_mgr.stats()
            cache_info["entries"] = cs.entries
            cache_info["hit_rate"] = round(cs.hit_rate, 4)

        return {
            "status": "ok",
            "model": state.model_name,
            "engine": state.engine_name,
            "backend": state.backend.name if state.backend else "none",
            "requests_served": state.request_count,
            "uptime_seconds": int(time.time() - state.start_time),
            "cache": cache_info,
        }

    return app


async def _stream_response(
    state: ServerState,
    request: GenerationRequest,
    req_id: str,
) -> AsyncIterator[str]:
    """Yield SSE chunks in OpenAI streaming format."""
    assert state.backend is not None

    async for chunk in state.backend.generate_stream(request):
        data = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.text} if chunk.text else {},
                    "finish_reason": chunk.finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


def run_server(
    model_name: str,
    *,
    port: int = 8080,
    host: str = "0.0.0.0",
    api_key: Optional[str] = None,
    api_base: str = "https://api.edgeml.io/api/v1",
    json_mode: bool = False,
    cache_size_mb: int = 2048,
    cache_enabled: bool = True,
    engine: Optional[str] = None,
) -> None:
    """Start the inference server (blocking).

    Parameters
    ----------
    json_mode:
        When ``True``, all requests default to ``response_format={"type": "json_object"}``.
    engine:
        Force a specific engine (e.g. ``"mlx-lm"``, ``"llama.cpp"``).
        When ``None``, auto-benchmarks and picks fastest.
    """
    import uvicorn

    app = create_app(
        model_name,
        api_key=api_key,
        api_base=api_base,
        json_mode=json_mode,
        cache_size_mb=cache_size_mb,
        cache_enabled=cache_enabled,
        engine=engine,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
