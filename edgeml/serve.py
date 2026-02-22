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
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

if TYPE_CHECKING:
    from .telemetry import TelemetryReporter

from pydantic import BaseModel, Field

from .models.catalog import CATALOG as _UNIFIED_CATALOG
from .models.resolver import ModelResolutionError as _NewResolutionError
from .models.resolver import resolve as _resolve_new

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
# Model catalog — backwards-compatible dicts derived from unified catalog
# ---------------------------------------------------------------------------

_MLX_MODELS: dict[str, str] = {}
_GGUF_MODELS: dict[str, tuple[str, str]] = {}

for _name, _entry in _UNIFIED_CATALOG.items():
    _default_variant = _entry.variants.get(_entry.default_quant)
    if _default_variant is None:
        continue
    if _default_variant.mlx:
        _MLX_MODELS[_name] = _default_variant.mlx
    if _default_variant.gguf:
        _GGUF_MODELS[_name] = (
            _default_variant.gguf.repo,
            _default_variant.gguf.filename,
        )


def resolve_model_name(name: str, backend: str) -> str:
    """Resolve a short model name (with optional :variant) to a HuggingFace repo ID.

    Uses the unified model catalog for structured resolution. Supports
    Ollama-style ``model:variant`` syntax (e.g. ``gemma-3b:4bit``).

    Full repo paths (containing ``/``) and local file paths pass through
    unchanged.
    """
    # Local file path
    if name.endswith((".gguf", ".pte", ".mnn")):
        return name

    # Full repo path — pass through
    if "/" in name:
        return name

    # Map backend names to engine names for the resolver
    engine_map = {"mlx": "mlx-lm", "gguf": "llama.cpp"}
    engine = engine_map.get(backend, backend)

    try:
        resolved = _resolve_new(name, engine=engine)
    except _NewResolutionError as exc:
        raise ValueError(str(exc)) from exc

    if backend == "mlx":
        if resolved.mlx_repo:
            return resolved.mlx_repo
        if resolved.hf_repo:
            return resolved.hf_repo
        raise ValueError(
            f"No MLX source found for '{name}'. "
            f"Pass a full HuggingFace repo ID (e.g. mlx-community/gemma-3-1b-it-4bit)"
        )

    if backend == "gguf":
        # For GGUF, return the short name — LlamaCppBackend resolves via _GGUF_MODELS
        family = resolved.family
        if family and family in _GGUF_MODELS:
            return family
        # If the resolver found a GGUF artifact, return the family name
        if resolved.is_gguf and family:
            return family
        raise ValueError(
            f"No GGUF source found for '{name}'. "
            f"Pass a path to a local .gguf file or a HuggingFace repo ID."
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

        # Try the new resolver for model:variant syntax and catalog lookup
        try:
            resolved = _resolve_new(model_name, engine="llama.cpp")
            if resolved.is_gguf and resolved.filename:
                repo_id = resolved.hf_repo
                filename = resolved.filename
                logger.info(
                    "Loading %s from %s (%s)...",
                    model_name,
                    repo_id,
                    filename,
                )
                self._llm = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=filename,
                    n_ctx=4096,
                    n_gpu_layers=-1,
                    verbose=False,
                )
                self._attach_cache()
                return
        except _NewResolutionError:
            pass

        # Fallback: short name → legacy catalog lookup
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
                f"Unknown engine '{engine_override}'. Available: {', '.join(available)}"
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
    whisper_backend: Any = None  # _WhisperBackend instance (speech-to-text)
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
    reporter: Optional["TelemetryReporter"] = None
    max_queue_depth: int = 32
    request_queue: Any = None  # RequestQueue instance


@dataclass
class MultiModelServerState:
    """Shared mutable state for multi-model serving with routing."""

    backends: dict[str, InferenceBackend] = field(default_factory=dict)
    model_names: list[str] = field(default_factory=list)
    router: Any = None  # QueryRouter instance
    start_time: float = field(default_factory=time.time)
    request_count: int = 0
    routed_counts: dict[str, int] = field(default_factory=dict)
    fallback_counts: int = 0
    api_key: Optional[str] = None
    api_base: str = "https://api.edgeml.io/api/v1"
    default_json_mode: bool = False
    cache_size_mb: int = 2048
    cache_enabled: bool = True
    engine_override: Optional[str] = None
    reporter: Optional["TelemetryReporter"] = None
    route_strategy: str = "complexity"


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
    max_queue_depth: int = 32,
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
    max_queue_depth:
        Maximum number of pending requests in the queue.  When full, new
        requests get a 503 response.  Set to 0 to disable queueing.
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
        max_queue_depth=max_queue_depth,
    )

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        # Check if this is a whisper (speech-to-text) model
        from .engines.whisper_engine import is_whisper_model

        if is_whisper_model(model_name):
            from .engines.whisper_engine import WhisperCppEngine

            whisper_engine = WhisperCppEngine()
            whisper_backend = whisper_engine.create_backend(model_name)
            try:
                whisper_backend.load_model(model_name)
            except Exception as exc:
                _log_startup_error(model_name, exc)
                raise
            state.whisper_backend = whisper_backend
            state.engine_name = "whisper.cpp"
        else:
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

        # Initialise request queue
        if state.max_queue_depth > 0:
            from .batch import RequestQueue

            state.request_queue = RequestQueue(max_depth=state.max_queue_depth)
            state.request_queue.start()
            logger.info("Request queue enabled (max_depth=%d)", state.max_queue_depth)

        # Create telemetry reporter when an API key is configured
        if state.api_key:
            try:
                from .telemetry import TelemetryReporter as _TR

                state.reporter = _TR(
                    api_key=state.api_key,
                    api_base=state.api_base,
                    org_id="default",
                )
                logger.info("Telemetry enabled — reporting to %s", state.api_base)
            except Exception as exc:
                logger.warning("Failed to initialise telemetry: %s", exc)

        yield

        # Graceful shutdown: stop request queue
        if state.request_queue is not None:
            await state.request_queue.stop()

        # Graceful shutdown: drain pending telemetry events
        if state.reporter is not None:
            state.reporter.close()

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

        # --- Tier-0: deterministic routing (arithmetic, etc.) ---
        # Check the last user message for a query that can be answered
        # without invoking any model.
        if body.messages and not body.stream:
            last_user_msg = ""
            for msg in reversed(body.messages):
                if msg.role == "user":
                    last_user_msg = msg.content
                    break
            if last_user_msg:
                from .routing import check_deterministic

                det = check_deterministic(last_user_msg)
                if det is not None:
                    state.request_count += 1
                    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    return {
                        "id": req_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": body.model or state.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": det.answer,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "deterministic": True,
                            "deterministic_method": det.method,
                        },
                    }

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
        session_id = uuid.uuid4().hex
        model_version = "latest"
        _reporter = state.reporter

        # Report generation_started (best-effort)
        if _reporter is not None:
            try:
                _reporter.report_generation_started(
                    model_id=gen_req.model,
                    version=model_version,
                    session_id=session_id,
                )
            except Exception:
                pass

        # --- Queue-aware dispatch ---
        _queue = state.request_queue

        if gen_req.stream:
            if _queue is not None:
                # Stream through the request queue
                return StreamingResponse(
                    _queued_stream_response(state, gen_req, req_id, session_id, _queue),
                    media_type="text/event-stream",
                )
            return StreamingResponse(
                _stream_response(state, gen_req, req_id, session_id),
                media_type="text/event-stream",
            )

        gen_start = time.monotonic()
        try:
            if _queue is not None:
                from .batch import QueueFullError, QueueTimeoutError

                try:
                    text, metrics = await _queue.submit_generate(
                        gen_req, state.backend.generate
                    )
                except QueueFullError:
                    raise HTTPException(
                        status_code=503,
                        detail="Server busy — request queue full. Try again later.",
                    )
                except QueueTimeoutError:
                    raise HTTPException(
                        status_code=504,
                        detail="Request timed out waiting in queue.",
                    )
            else:
                text, metrics = state.backend.generate(gen_req)
        except HTTPException:
            raise
        except Exception:
            if _reporter is not None:
                try:
                    _reporter.report_generation_failed(
                        session_id=session_id,
                        model_id=gen_req.model,
                        version=model_version,
                    )
                except Exception:
                    pass
            raise
        gen_elapsed_ms = (time.monotonic() - gen_start) * 1000

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

        # Report generation_completed (best-effort)
        if _reporter is not None:
            try:
                total_tokens = metrics.total_tokens
                throughput = (
                    total_tokens / (gen_elapsed_ms / 1000)
                    if gen_elapsed_ms > 0
                    else 0.0
                )
                _reporter.report_generation_completed(
                    session_id=session_id,
                    model_id=gen_req.model,
                    version=model_version,
                    total_chunks=total_tokens,
                    total_duration_ms=gen_elapsed_ms,
                    ttfc_ms=metrics.ttfc_ms,
                    throughput=throughput,
                )
            except Exception:
                pass

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

    @app.get("/v1/queue/stats")
    async def queue_stats() -> dict[str, Any]:
        """Return request queue statistics."""
        if state.request_queue is None:
            return {
                "pending": 0,
                "active": 0,
                "max_depth": 0,
                "enabled": False,
            }
        qs = state.request_queue.stats()
        return {
            "pending": qs.pending,
            "active": qs.active,
            "max_depth": qs.max_depth,
            "enabled": True,
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

        backend_name = "none"
        if state.whisper_backend is not None:
            backend_name = state.whisper_backend.name
        elif state.backend is not None:
            backend_name = state.backend.name

        return {
            "status": "ok",
            "model": state.model_name,
            "engine": state.engine_name,
            "backend": backend_name,
            "requests_served": state.request_count,
            "uptime_seconds": int(time.time() - state.start_time),
            "cache": cache_info,
        }

    @app.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        file: Any = None,
        model: str = "",
    ) -> dict[str, Any]:
        """OpenAI Whisper API-compatible audio transcription endpoint.

        Accepts a multipart file upload and returns transcribed text
        with segment-level timestamps.
        """
        if state.whisper_backend is None:
            raise HTTPException(
                status_code=503,
                detail="No whisper model loaded. Start server with a whisper model: "
                "edgeml serve whisper-base",
            )

        if file is None:
            raise HTTPException(status_code=400, detail="No audio file provided")

        # Save uploaded file to a temp location
        import tempfile

        filename = getattr(file, "filename", "audio.wav") or "audio.wav"
        suffix = os.path.splitext(filename)[1] or ".wav"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            content = await file.read()
            os.write(fd, content)
            os.close(fd)

            state.request_count += 1
            result = state.whisper_backend.transcribe(tmp_path)
            return result
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return app


async def _stream_response(
    state: ServerState,
    request: GenerationRequest,
    req_id: str,
    session_id: str = "",
) -> AsyncIterator[str]:
    """Yield SSE chunks in OpenAI streaming format."""
    assert state.backend is not None

    _reporter = state.reporter
    model_version = "latest"
    chunk_index = 0
    stream_start = time.monotonic()
    first_chunk_time: Optional[float] = None
    prev_chunk_time = stream_start
    failed = False

    try:
        async for chunk in state.backend.generate_stream(request):
            now = time.monotonic()
            chunk_latency_ms = (now - prev_chunk_time) * 1000
            prev_chunk_time = now

            if first_chunk_time is None and chunk.text:
                first_chunk_time = now

            # Report each chunk (best-effort)
            if _reporter is not None and chunk.text:
                try:
                    ttfc = (
                        (first_chunk_time - stream_start) * 1000
                        if first_chunk_time is not None and chunk_index == 0
                        else None
                    )
                    _reporter.report_chunk_produced(
                        session_id=session_id,
                        model_id=request.model,
                        version=model_version,
                        chunk_index=chunk_index,
                        ttfc_ms=ttfc,
                        chunk_latency_ms=chunk_latency_ms,
                    )
                except Exception:
                    pass
                chunk_index += 1

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
    except Exception:
        failed = True
        if _reporter is not None:
            try:
                _reporter.report_generation_failed(
                    session_id=session_id,
                    model_id=request.model,
                    version=model_version,
                )
            except Exception:
                pass
        raise

    yield "data: [DONE]\n\n"

    # Report generation_completed (best-effort)
    if _reporter is not None and not failed:
        try:
            total_duration_ms = (time.monotonic() - stream_start) * 1000
            ttfc_ms = (
                (first_chunk_time - stream_start) * 1000
                if first_chunk_time is not None
                else 0.0
            )
            throughput = (
                chunk_index / (total_duration_ms / 1000)
                if total_duration_ms > 0
                else 0.0
            )
            _reporter.report_generation_completed(
                session_id=session_id,
                model_id=request.model,
                version=model_version,
                total_chunks=chunk_index,
                total_duration_ms=total_duration_ms,
                ttfc_ms=ttfc_ms,
                throughput=throughput,
            )
        except Exception:
            pass


async def _queued_stream_response(
    state: Any,
    request: GenerationRequest,
    req_id: str,
    session_id: str,
    queue: Any,
) -> AsyncIterator[str]:
    """Yield SSE chunks after waiting in the request queue.

    Submits a streaming request to the queue.  Once the request reaches
    the front, chunks are forwarded as SSE events.  Queue errors (full,
    timeout) are surfaced as SSE error events so the client gets useful
    feedback even on a streaming connection.
    """
    from .batch import QueueFullError, QueueTimeoutError

    assert state.backend is not None

    try:
        chunk_iter = queue.submit_generate_stream(
            request, state.backend.generate_stream
        )
        # Build SSE events from the chunk iterator (same format as _stream_response)
        chunk_index = 0
        async for chunk in chunk_iter:
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
            chunk_index += 1

        yield "data: [DONE]\n\n"
    except QueueFullError:
        error_data = {
            "error": {
                "message": "Server busy — request queue full. Try again later.",
                "type": "server_error",
                "code": 503,
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    except QueueTimeoutError:
        error_data = {
            "error": {
                "message": "Request timed out waiting in queue.",
                "type": "server_error",
                "code": 504,
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"


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
    max_queue_depth: int = 32,
) -> None:
    """Start the inference server (blocking).

    Parameters
    ----------
    json_mode:
        When ``True``, all requests default to ``response_format={"type": "json_object"}``.
    engine:
        Force a specific engine (e.g. ``"mlx-lm"``, ``"llama.cpp"``).
        When ``None``, auto-benchmarks and picks fastest.
    max_queue_depth:
        Maximum number of pending requests in the queue (default 32).
        Set to 0 to disable queueing.
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
        max_queue_depth=max_queue_depth,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Multi-model serving with query routing
# ---------------------------------------------------------------------------


def create_multi_model_app(
    model_names: list[str],
    *,
    api_key: Optional[str] = None,
    api_base: str = "https://api.edgeml.io/api/v1",
    json_mode: bool = False,
    cache_size_mb: int = 2048,
    cache_enabled: bool = True,
    engine: Optional[str] = None,
    route_strategy: str = "complexity",
) -> Any:
    """Create a FastAPI app that loads multiple models and routes queries.

    Parameters
    ----------
    model_names:
        Ordered list of model names, smallest to largest.
    route_strategy:
        Routing strategy (currently only ``"complexity"``).
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    from .decomposer import ResultMerger, SubTaskResult
    from .routing import (
        DecomposedRoutingDecision,
        QueryRouter,
        RoutingDecision,
        assign_tiers,
    )

    state = MultiModelServerState(
        model_names=model_names,
        api_key=api_key,
        api_base=api_base,
        default_json_mode=json_mode,
        cache_size_mb=cache_size_mb,
        cache_enabled=cache_enabled,
        engine_override=engine,
        route_strategy=route_strategy,
    )

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        # Load all models
        for name in model_names:
            try:
                backend = _detect_backend(
                    name,
                    cache_size_mb=state.cache_size_mb,
                    cache_enabled=state.cache_enabled,
                    engine_override=state.engine_override,
                )
                state.backends[name] = backend
                state.routed_counts[name] = 0
                logger.info("Loaded model: %s (engine: %s)", name, backend.name)
            except Exception as exc:
                _log_startup_error(name, exc)
                raise

        # Build router with auto-assigned tiers
        model_infos = assign_tiers(model_names)
        state.router = QueryRouter(
            model_infos,
            strategy=state.route_strategy,
        )

        state.start_time = time.time()

        # Create telemetry reporter
        if state.api_key:
            try:
                from .telemetry import TelemetryReporter as _TR

                state.reporter = _TR(
                    api_key=state.api_key,
                    api_base=state.api_base,
                    org_id="default",
                )
            except Exception as exc:
                logger.warning("Failed to initialise telemetry: %s", exc)

        yield

        if state.reporter is not None:
            state.reporter.close()

    app = FastAPI(
        title="EdgeML Serve (Multi-Model)", version="1.0.0", lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        all_models: list[str] = []
        for backend in state.backends.values():
            all_models.extend(backend.list_models())
        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": int(state.start_time),
                    "owned_by": "edgeml",
                }
                for m in all_models
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionBody) -> Any:
        if not state.backends:
            raise HTTPException(status_code=503, detail="No models loaded")

        messages = [{"role": m.role, "content": m.content} for m in body.messages]

        # Attempt query decomposition for non-streaming requests
        assert state.router is not None
        if not body.stream:
            decomp_decision = state.router.route_decomposed(messages)
            if isinstance(decomp_decision, DecomposedRoutingDecision):
                return await _handle_decomposed(state, body, messages, decomp_decision)

        # Standard single-task routing
        decision: RoutingDecision = state.router.route(messages)

        routed_model = decision.model_name
        fallback_chain = decision.fallback_chain

        # Try the routed model, then fallback chain
        last_error: Optional[Exception] = None
        tried_models: list[str] = [routed_model] + fallback_chain
        used_fallback = False

        for model_name in tried_models:
            backend = state.backends.get(model_name)
            if backend is None:
                continue

            grammar_str, is_json = _resolve_grammar(body, state.default_json_mode)

            req_messages = list(messages)
            uses_grammar_natively = isinstance(backend, LlamaCppBackend)
            schema_for_prompt: Optional[dict[str, Any]] = None
            if is_json and not uses_grammar_natively:
                rf = body.response_format or {}
                if rf.get("type") == "json_schema":
                    raw = rf.get("json_schema") or rf.get("schema")
                    schema_for_prompt = raw.get("schema", raw) if raw else None
                req_messages = _inject_json_system_prompt(
                    req_messages, schema_for_prompt
                )

            gen_req = GenerationRequest(
                model=model_name,
                messages=req_messages,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                stream=body.stream,
                grammar=grammar_str if uses_grammar_natively else None,
                json_mode=is_json,
            )

            state.request_count += 1
            state.routed_counts[model_name] = state.routed_counts.get(model_name, 0) + 1
            req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            session_id = uuid.uuid4().hex
            model_version = "latest"
            _reporter = state.reporter

            # Report generation_started
            if _reporter is not None:
                try:
                    _reporter.report_generation_started(
                        model_id=model_name,
                        version=model_version,
                        session_id=session_id,
                    )
                except Exception:
                    pass

            if gen_req.stream:
                headers = {
                    "X-EdgeML-Routed-Model": model_name,
                    "X-EdgeML-Complexity": str(decision.complexity_score),
                    "X-EdgeML-Tier": decision.tier,
                }
                if used_fallback:
                    headers["X-EdgeML-Fallback"] = "true"

                return StreamingResponse(
                    _stream_response(
                        _MultiModelStateAdapter(state, backend, model_name),
                        gen_req,
                        req_id,
                        session_id,
                    ),
                    media_type="text/event-stream",
                    headers=headers,
                )

            gen_start = time.monotonic()
            try:
                text, metrics = backend.generate(gen_req)
            except Exception as exc:
                last_error = exc
                used_fallback = True
                state.fallback_counts += 1
                logger.warning("Model %s failed, trying fallback: %s", model_name, exc)
                if _reporter is not None:
                    try:
                        _reporter.report_generation_failed(
                            session_id=session_id,
                            model_id=model_name,
                            version=model_version,
                        )
                    except Exception:
                        pass
                continue
            gen_elapsed_ms = (time.monotonic() - gen_start) * 1000

            # JSON validation + retry for non-grammar backends
            if is_json and not uses_grammar_natively:
                from .grammar import extract_json, validate_json_output

                if not validate_json_output(text):
                    extracted = extract_json(text)
                    if extracted is not None:
                        text = json.dumps(extracted)
                    else:
                        retry_messages = _inject_json_system_prompt(
                            messages, schema_for_prompt
                        )
                        retry_req = GenerationRequest(
                            model=model_name,
                            messages=retry_messages,
                            max_tokens=gen_req.max_tokens,
                            temperature=max(gen_req.temperature - 0.2, 0.0),
                            top_p=gen_req.top_p,
                            stream=False,
                            json_mode=True,
                        )
                        text, metrics = backend.generate(retry_req)
                        if not validate_json_output(text):
                            extracted = extract_json(text)
                            if extracted is not None:
                                text = json.dumps(extracted)

            # Report generation_completed
            if _reporter is not None:
                try:
                    total_tokens = metrics.total_tokens
                    throughput = (
                        total_tokens / (gen_elapsed_ms / 1000)
                        if gen_elapsed_ms > 0
                        else 0.0
                    )
                    _reporter.report_generation_completed(
                        session_id=session_id,
                        model_id=model_name,
                        version=model_version,
                        total_chunks=total_tokens,
                        total_duration_ms=gen_elapsed_ms,
                        ttfc_ms=metrics.ttfc_ms,
                        throughput=throughput,
                    )
                except Exception:
                    pass

            response_data = {
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
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

            resp = JSONResponse(content=response_data)
            resp.headers["X-EdgeML-Routed-Model"] = model_name
            resp.headers["X-EdgeML-Complexity"] = str(decision.complexity_score)
            resp.headers["X-EdgeML-Tier"] = decision.tier
            if used_fallback:
                resp.headers["X-EdgeML-Fallback"] = "true"
            return resp

        # All models failed
        raise HTTPException(
            status_code=503,
            detail=f"All models failed. Last error: {last_error}",
        )

    @app.get("/v1/routing/stats")
    async def routing_stats() -> dict[str, Any]:
        """Return routing statistics."""
        return {
            "total_requests": state.request_count,
            "routed_counts": dict(state.routed_counts),
            "fallback_count": state.fallback_counts,
            "models": state.model_names,
            "strategy": state.route_strategy,
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        models_status = {}
        for name, backend in state.backends.items():
            models_status[name] = {
                "engine": backend.name,
                "requests": state.routed_counts.get(name, 0),
            }
        return {
            "status": "ok",
            "mode": "multi-model",
            "models": models_status,
            "strategy": state.route_strategy,
            "requests_served": state.request_count,
            "fallback_count": state.fallback_counts,
            "uptime_seconds": int(time.time() - state.start_time),
        }

    async def _handle_decomposed(
        mm_state: MultiModelServerState,
        body: ChatCompletionBody,
        messages: list[dict[str, str]],
        decomp: DecomposedRoutingDecision,
    ) -> Any:
        """Execute a decomposed multi-task query with parallel dispatch.

        Independent sub-tasks are executed concurrently via asyncio.gather.
        Dependent sub-tasks wait for their prerequisites before executing.
        """
        import asyncio

        decomposer_merger = ResultMerger()
        sub_task_results: dict[int, SubTaskResult] = {}

        async def _execute_subtask(
            task_idx: int,
        ) -> SubTaskResult:
            task = decomp.tasks[task_idx]
            decision = decomp.sub_decisions[task_idx]
            model_name = decision.model_name
            backend = mm_state.backends.get(model_name)

            if backend is None:
                # Use first available backend as fallback
                model_name = next(iter(mm_state.backends))
                backend = mm_state.backends[model_name]

            sub_messages: list[dict[str, str]] = []
            # Carry over system prompt
            for msg in messages:
                if msg.get("role") == "system":
                    sub_messages.append(msg)
                    break
            sub_messages.append({"role": "user", "content": task.text})

            grammar_str, is_json = _resolve_grammar(body, mm_state.default_json_mode)
            uses_grammar_natively = isinstance(backend, LlamaCppBackend)

            req_messages = list(sub_messages)
            if is_json and not uses_grammar_natively:
                rf = body.response_format or {}
                schema_for_prompt = None
                if rf.get("type") == "json_schema":
                    raw = rf.get("json_schema") or rf.get("schema")
                    schema_for_prompt = raw.get("schema", raw) if raw else None
                req_messages = _inject_json_system_prompt(
                    req_messages, schema_for_prompt
                )

            gen_req = GenerationRequest(
                model=model_name,
                messages=req_messages,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                stream=False,
                grammar=grammar_str if uses_grammar_natively else None,
                json_mode=is_json,
            )

            mm_state.request_count += 1
            mm_state.routed_counts[model_name] = (
                mm_state.routed_counts.get(model_name, 0) + 1
            )

            try:
                text, _metrics = backend.generate(gen_req)
            except Exception as exc:
                logger.warning(
                    "Sub-task %d failed on model %s: %s",
                    task_idx,
                    model_name,
                    exc,
                )
                text = f"[Error processing sub-task {task_idx + 1}]"

            return SubTaskResult(
                task=task,
                response=text,
                model_used=model_name,
                tier=decision.tier,
            )

        # Build execution order respecting dependencies
        # Phase 1: execute independent tasks in parallel
        independent = [i for i, t in enumerate(decomp.tasks) if not t.depends_on]
        dependent = [i for i, t in enumerate(decomp.tasks) if t.depends_on]

        # Execute independent tasks concurrently
        if independent:
            results = await asyncio.gather(*[_execute_subtask(i) for i in independent])
            for result in results:
                sub_task_results[result.task.index] = result

        # Execute dependent tasks sequentially (in index order)
        for idx in sorted(dependent):
            # Wait for prerequisites (they should already be done)
            result = await _execute_subtask(idx)
            sub_task_results[result.task.index] = result

        # Merge results
        ordered_results = [sub_task_results[i] for i in range(len(decomp.tasks))]
        merged_text = decomposer_merger.merge(ordered_results)

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        response_data = {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": ordered_results[0].model_used,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": merged_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        resp = JSONResponse(content=response_data)
        resp.headers["X-EdgeML-Decomposed"] = "true"
        resp.headers["X-EdgeML-Subtasks"] = str(len(decomp.tasks))
        resp.headers["X-EdgeML-Routed-Model"] = ordered_results[0].model_used
        resp.headers["X-EdgeML-Tier"] = ordered_results[0].tier
        return resp

    return app


class _MultiModelStateAdapter:
    """Adapts MultiModelServerState for ``_stream_response`` compatibility.

    ``_stream_response`` expects a ``ServerState``-like object with
    ``.backend`` and ``.reporter``.  This adapter provides those.
    """

    def __init__(
        self,
        state: MultiModelServerState,
        backend: InferenceBackend,
        model_name: str,
    ) -> None:
        self.backend = backend
        self.reporter = state.reporter
        self.model_name = model_name


def run_multi_model_server(
    model_names: list[str],
    *,
    port: int = 8080,
    host: str = "0.0.0.0",
    api_key: Optional[str] = None,
    api_base: str = "https://api.edgeml.io/api/v1",
    json_mode: bool = False,
    cache_size_mb: int = 2048,
    cache_enabled: bool = True,
    engine: Optional[str] = None,
    route_strategy: str = "complexity",
) -> None:
    """Start a multi-model inference server with query routing (blocking).

    Parameters
    ----------
    model_names:
        Ordered list of model names, smallest to largest.
    route_strategy:
        Routing strategy (``"complexity"`` is the only one currently).
    """
    import uvicorn

    app = create_multi_model_app(
        model_names,
        api_key=api_key,
        api_base=api_base,
        json_mode=json_mode,
        cache_size_mb=cache_size_mb,
        cache_enabled=cache_enabled,
        engine=engine,
        route_strategy=route_strategy,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
