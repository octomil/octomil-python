"""
OpenAI-compatible local inference server.

Dispatches to the best available backend:
  1. mlx-lm (Apple Silicon — quantized HuggingFace models)
  2. llama-cpp-python (cross-platform — GGUF models)
  3. Echo (fallback — for testing the API layer)

Usage::

    from edgeml.serve import run_server
    run_server("gemma-2b", port=8080)

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
    "smollm-360m": "mlx-community/SmolLM2-360M-Instruct-4bit",
}

# GGUF models for llama.cpp (cross-platform)
_GGUF_MODELS: dict[str, tuple[str, str]] = {
    # (repo_id, filename)
    "gemma-1b": ("bartowski/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf"),
    "gemma-4b": ("bartowski/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
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
    """

    name = "mlx-lm"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_name: str = ""
        self._repo_id: str = ""

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
        prompt_tokens = len(self._tokenizer.encode(prompt))
        sampler = self._make_sampler(request.temperature, request.top_p)
        start = time.monotonic()
        first_token_time: Optional[float] = None
        tokens: list[str] = []
        final_tps: float = 0.0

        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
        ):
            if first_token_time is None:
                first_token_time = time.monotonic()
            final_tps = response.generation_tps
            if response.finish_reason or self._is_stop_token(response.text):
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
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio

        import mlx_lm  # type: ignore[import-untyped]

        prompt = self._apply_chat_template(request.messages)
        sampler = self._make_sampler(request.temperature, request.top_p)

        # mlx_lm.stream_generate is a sync generator that yields
        # GenerationResponse objects as tokens are produced.
        # We run it in a thread to avoid blocking the event loop.
        gen = mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
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

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


class LlamaCppBackend(InferenceBackend):
    """Cross-platform backend using llama-cpp-python.

    Loads GGUF models from HuggingFace via from_pretrained (auto-downloads).
    Chat templates are handled by llama.cpp internally.
    """

    name = "llama.cpp"

    def __init__(self) -> None:
        self._llm: Any = None
        self._model_name: str = ""

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
        logger.info("Model loaded: %s", model_name)

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        start = time.monotonic()
        result = self._llm.create_chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
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

        # create_chat_completion with stream=True returns a sync iterator
        stream = self._llm.create_chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
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


def _detect_backend(model_name: str) -> InferenceBackend:
    """Pick the best available backend and load the model.

    Priority: mlx-lm (Apple Silicon) → llama.cpp → echo (fallback).
    Models are downloaded from HuggingFace automatically on first use.
    """

    # Apple Silicon → mlx-lm (best perf for quantized models)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_lm  # type: ignore[import-untyped] # noqa: F401

            # Check if model is in MLX catalog or is a full repo ID
            if model_name in _MLX_MODELS or "/" in model_name:
                mlx_backend: InferenceBackend = MLXBackend()
                mlx_backend.load_model(model_name)
                return mlx_backend
            else:
                logger.debug(
                    "Model '%s' not in MLX catalog, trying llama.cpp", model_name
                )
        except ImportError:
            logger.debug("mlx-lm not installed")
        except Exception as exc:
            logger.warning("mlx-lm failed to load '%s': %s", model_name, exc)

    # llama.cpp — cross-platform, GGUF files
    try:
        import llama_cpp  # type: ignore[import-untyped] # noqa: F401

        if (
            model_name in _GGUF_MODELS
            or model_name.endswith(".gguf")
            or "/" in model_name
        ):
            cpp_backend: InferenceBackend = LlamaCppBackend()
            cpp_backend.load_model(model_name)
            return cpp_backend
        else:
            logger.debug(
                "Model '%s' not in GGUF catalog, falling back to echo", model_name
            )
    except ImportError:
        logger.debug("llama-cpp-python not installed")
    except Exception as exc:
        logger.warning("llama.cpp failed to load '%s': %s", model_name, exc)

    # Echo fallback
    echo_backend: InferenceBackend = EchoBackend()
    echo_backend.load_model(model_name)
    return echo_backend


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    """Shared mutable state for the serve app."""

    backend: Optional[InferenceBackend] = None
    model_name: str = ""
    start_time: float = field(default_factory=time.time)
    request_count: int = 0
    api_key: Optional[str] = None
    api_base: str = "https://api.edgeml.io/api/v1"


def create_app(
    model_name: str,
    *,
    api_key: Optional[str] = None,
    api_base: str = "https://api.edgeml.io/api/v1",
) -> Any:
    """Create a FastAPI app with OpenAI-compatible endpoints."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    state = ServerState(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
    )

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        state.backend = _detect_backend(model_name)
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

        gen_req = GenerationRequest(
            model=body.model or state.model_name,
            messages=[{"role": m.role, "content": m.content} for m in body.messages],
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stream=body.stream,
        )

        state.request_count += 1
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if gen_req.stream:
            return StreamingResponse(
                _stream_response(state, gen_req, req_id),
                media_type="text/event-stream",
            )

        text, metrics = state.backend.generate(gen_req)
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
            },
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": state.model_name,
            "backend": state.backend.name if state.backend else "none",
            "requests_served": state.request_count,
            "uptime_seconds": int(time.time() - state.start_time),
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
) -> None:
    """Start the inference server (blocking)."""
    import uvicorn

    app = create_app(model_name, api_key=api_key, api_base=api_base)
    uvicorn.run(app, host=host, port=port, log_level="info")
