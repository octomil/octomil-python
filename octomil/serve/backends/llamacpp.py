"""Cross-platform backend using llama-cpp-python."""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from ...errors import OctomilError, OctomilErrorCode
from ...models.resolver import ModelResolutionError as _NewResolutionError
from ...models.resolver import resolve as _resolve_new
from ..models import _GGUF_MODELS
from ..types import GenerationChunk, GenerationRequest, InferenceBackend, InferenceMetrics

logger = logging.getLogger(__name__)


class LlamaCppBackend(InferenceBackend):
    """Cross-platform backend using llama-cpp-python.

    Loads GGUF models from HuggingFace via from_pretrained (auto-downloads).
    Chat templates are handled by llama.cpp internally.
    Supports built-in LlamaCache for automatic KV prefix reuse.
    Flash attention is enabled by default (``flash_attn=True``) for better
    performance on long sequences.
    """

    name = "llama.cpp"
    attention_backend = "flash_attention"

    def __init__(
        self,
        cache_size_mb: int = 2048,
        cache_enabled: bool = True,
        verbose_emitter: Any = None,
    ) -> None:
        super().__init__()
        self._llm: Any = None
        self._model_name: str = ""
        self._cache_size_mb = cache_size_mb
        self._cache_enabled = cache_enabled
        self._llama_cache: Any = None
        self._verbose = verbose_emitter

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

    def _emit_load_completed(self, load_start: float) -> None:
        """Emit model.load_completed verbose event if emitter is active."""
        if self._verbose:
            load_elapsed_ms = (time.monotonic() - load_start) * 1000
            self._verbose.emit(
                "model.load_completed",
                model=self._model_name,
                engine="llama.cpp",
                load_time_ms=round(load_elapsed_ms, 1),
                cache_enabled=self._cache_enabled,
            )

    def load_model(self, model_name: str) -> None:
        from llama_cpp import Llama  # type: ignore[import-untyped]

        self._model_name = model_name
        if self._verbose:
            self._verbose.emit("model.load_started", model=model_name, engine="llama.cpp")
        load_start = time.monotonic()

        # Local GGUF file path
        if model_name.endswith(".gguf"):
            logger.info("Loading local GGUF: %s", model_name)
            self._llm = Llama(
                model_path=model_name,
                n_ctx=4096,
                n_batch=256,
                n_gpu_layers=-1,  # offload all layers to GPU
                flash_attn=True,  # fused attention kernels for better perf on long sequences
                verbose=False,
            )
            self._attach_cache()
            self._emit_load_completed(load_start)
            return

        # Full HuggingFace repo ID (user/repo)
        if "/" in model_name:
            logger.info("Loading from HuggingFace: %s", model_name)
            # Pick a sensible GGUF filename pattern.  For repos that are
            # explicitly GGUF collections (name contains "GGUF"), try the
            # most common quantization patterns in preference order.
            filename_patterns = ["*Q4_K_M.gguf", "*Q4_K_S.gguf", "*q4_k_m.gguf", "*.gguf"]
            last_err: Exception | None = None
            for pattern in filename_patterns:
                try:
                    self._llm = Llama.from_pretrained(
                        repo_id=model_name,
                        filename=pattern,
                        n_ctx=4096,
                        n_batch=256,
                        n_gpu_layers=-1,
                        flash_attn=True,
                        verbose=False,
                    )
                    self._attach_cache()
                    self._emit_load_completed(load_start)
                    return
                except ValueError as exc:
                    last_err = exc
                    continue
            if last_err is not None:
                raise last_err
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
                    n_batch=256,
                    n_gpu_layers=-1,
                    flash_attn=True,
                    verbose=False,
                )
                self._attach_cache()
                self._emit_load_completed(load_start)
                return
        except _NewResolutionError:
            pass

        # Fallback: short name -> legacy catalog lookup
        if model_name not in _GGUF_MODELS:
            raise OctomilError(
                code=OctomilErrorCode.MODEL_NOT_FOUND,
                message=f"Unknown model '{model_name}'. "
                f"Available: {', '.join(sorted(_GGUF_MODELS))}\n"
                f"Or pass a local .gguf path or HuggingFace repo ID.",
            )

        repo_id, filename = _GGUF_MODELS[model_name]
        logger.info("Loading %s from %s (%s)...", model_name, repo_id, filename)
        self._llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=4096,
            n_batch=256,
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=False,
        )
        self._attach_cache()
        self._emit_load_completed(load_start)
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
        if self._verbose:
            self._verbose.emit(
                "inference.pre_generation",
                engine="llama.cpp",
                model=self._model_name,
                max_tokens=request.max_tokens,
            )
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
        tps = completion_tokens / elapsed if elapsed > 0 else 0

        if self._verbose:
            self._verbose.emit(
                "inference.completed",
                engine="llama.cpp",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_per_second=round(tps, 1),
                duration_ms=round(elapsed * 1000, 1),
            )

        return text, InferenceMetrics(
            prompt_tokens=prompt_tokens,
            total_tokens=completion_tokens,
            tokens_per_second=tps,
            total_duration_ms=elapsed * 1000,
            attention_backend="flash_attention",
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio

        if self._verbose:
            self._verbose.emit(
                "inference.pre_generation",
                engine="llama.cpp",
                model=self._model_name,
                max_tokens=request.max_tokens,
                stream=True,
            )

        loop = asyncio.get_event_loop()
        start = time.monotonic()
        tokens_generated = 0

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

        try:
            while True:
                chunk = await loop.run_in_executor(self._executor, _next_chunk)
                if chunk is None:
                    break
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                finish = chunk["choices"][0].get("finish_reason")
                if content or finish:
                    tokens_generated += 1
                    yield GenerationChunk(text=content, finish_reason=finish)
        finally:
            if self._verbose:
                elapsed = time.monotonic() - start
                tps = tokens_generated / elapsed if elapsed > 0 else 0
                self._verbose.emit(
                    "inference.stream_completed",
                    engine="llama.cpp",
                    tokens_generated=tokens_generated,
                    tokens_per_second=round(tps, 1),
                    duration_ms=round(elapsed * 1000, 1),
                )

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []
