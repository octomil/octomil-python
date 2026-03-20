"""Apple Silicon backend using mlx-lm."""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Optional

from ..models import resolve_model_name
from ..types import GenerationChunk, GenerationRequest, InferenceBackend, InferenceMetrics

logger = logging.getLogger(__name__)


_DRAFT_MODEL_MAP: dict[str, str] = {
    "phi-4": "REDACTED",
    "llama-8b": "REDACTED",
    "llama-3b": "REDACTED",
    "qwen-7b": "REDACTED",
    "qwen-3b": "REDACTED",
    "gemma-4b": "REDACTED",
    "gemma-12b": "REDACTED",
    "gemma-27b": "REDACTED",
}


class MLXBackend(InferenceBackend):
    """Apple Silicon backend using mlx-lm.

    Loads quantized models from HuggingFace (auto-downloads + caches).
    Uses the tokenizer's built-in chat template for proper formatting.
    Supports KV cache persistence for prefix reuse across requests.

    MLX uses Metal fused attention automatically on Apple Silicon -- no
    explicit configuration needed.  Reported as ``metal_fused``.
    """

    name = "mlx-lm"
    attention_backend = "metal_fused"

    def __init__(
        self,
        cache_size_mb: int = 2048,
        cache_enabled: bool = True,
        verbose_emitter: Any = None,
    ) -> None:
        super().__init__()
        self._model: Any = None
        self._tokenizer: Any = None
        self._draft_model: Any = None
        self._model_name: str = ""
        self._repo_id: str = ""
        self._cache_enabled = cache_enabled
        self._verbose = verbose_emitter  # VerboseEventEmitter or None
        # Multi-entry KV cache pool -- LRU eviction by both entry count and size.
        from ...cache import KVCacheManager

        self._cache_mgr = KVCacheManager(max_cache_size_mb=cache_size_mb, max_entries=4)
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._last_timings: dict[str, Any] = {}

    def load_model(self, model_name: str) -> None:
        import mlx_lm  # type: ignore[import-untyped]

        self._model_name = model_name
        self._repo_id = resolve_model_name(model_name, "mlx")

        if self._verbose:
            self._verbose.emit("model.load_started", model=model_name, repo_id=self._repo_id, engine="mlx-lm")

        load_start = time.time()
        logger.info("Loading %s (%s) with mlx-lm...", model_name, self._repo_id)
        self._model, self._tokenizer, *_ = mlx_lm.load(self._repo_id)
        load_elapsed_ms = (time.time() - load_start) * 1000
        logger.info("Model loaded: %s", self._repo_id)

        # Set Metal buffer cache limit to prevent unbounded growth
        # during long inference sessions.  2GB is enough for ~14B models.
        try:
            import mlx.core as mx  # type: ignore[import-untyped]

            mx.set_cache_limit(2 * 1024 * 1024 * 1024)  # 2 GB
        except Exception:
            pass

        # Load draft model for speculative decoding (if available)
        draft_repo = _DRAFT_MODEL_MAP.get(model_name)
        if draft_repo:
            logger.info("Loading draft model for speculative decoding: %s", draft_repo)
            try:
                self._draft_model, *_ = mlx_lm.load(draft_repo)
            except Exception:
                logger.debug("Draft model load failed (non-fatal)", exc_info=True)

        # Collect special token strings for output filtering
        self._stop_strings: set[str] = set()
        if hasattr(self._tokenizer, "eos_token") and self._tokenizer.eos_token:
            self._stop_strings.add(self._tokenizer.eos_token)
        # Common chat model turn markers
        for marker in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>", "</s>"):
            self._stop_strings.add(marker)

        # Warmup: prefill a short prompt to compile Metal shaders and prime
        # the KV cache so the first real request doesn't pay cold-start cost.
        self.warmup()

        if self._verbose:
            # Collect memory info if available
            mem_info: dict[str, Any] = {}
            try:
                import mlx.core as mx  # type: ignore[import-untyped]

                mem_info["active_memory_mb"] = round(mx.metal.get_active_memory() / (1024 * 1024), 1)
                mem_info["peak_memory_mb"] = round(mx.metal.get_peak_memory() / (1024 * 1024), 1)
            except Exception:
                pass
            self._verbose.emit(
                "model.load_completed",
                model=model_name,
                repo_id=self._repo_id,
                engine="mlx-lm",
                load_duration_ms=round(load_elapsed_ms, 1),
                has_draft_model=self._draft_model is not None,
                cache_enabled=self._cache_enabled,
                **mem_info,
            )

    def warmup(self) -> None:
        """Run a single prefill+decode to compile Metal shaders and prime KV cache."""
        import mlx_lm  # type: ignore[import-untyped]
        from mlx_lm.models.cache import make_prompt_cache  # type: ignore[import-untyped]

        warmup_start = time.perf_counter()
        prompt = self._apply_chat_template([{"role": "user", "content": "hi"}])
        token_ids = self._tokenizer.encode(prompt)
        cache = make_prompt_cache(self._model)
        sampler = self._make_sampler(0.0, 1.0)

        # Generate 1 token -- forces Metal shader compilation and prefill
        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=token_ids,
            max_tokens=1,
            sampler=sampler,
            prompt_cache=cache,
        ):
            break  # only need the first token

        # Store in cache so first real request gets a partial hit
        self._store_cache(token_ids, cache)

        elapsed = (time.perf_counter() - warmup_start) * 1000
        logger.info("MLX warmup complete: %.0fms (Metal shaders compiled, KV cache primed)", elapsed)

    def _apply_chat_template(self, messages: list[dict[str, str]], *, enable_thinking: bool | None = None) -> str:
        """Format messages using the model's chat template."""
        try:
            kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            result: str = self._tokenizer.apply_chat_template(messages, **kwargs)
            return result
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

    def _fetch_or_create_cache(self, prompt_token_ids: list[int]) -> tuple[list[Any], list[int], bool]:
        """Fetch a reusable prompt cache or create a fresh one.

        Returns (cache, remaining_tokens, cache_hit).  Looks up the longest
        matching prefix in the ``KVCacheManager`` pool, trims the MLX KV
        tensors to the common prefix, and returns only the remaining tokens
        so ``stream_generate`` only processes the delta.
        """
        from mlx_lm.models.cache import (  # type: ignore[import-untyped]
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )

        if self._cache_enabled:
            hit = self._cache_mgr.get_cached_prefix(prompt_token_ids)
            if hit is not None:
                cache = hit.kv_state
                common_len = hit.prefix_length

                # Trim cache to the reusable prefix.
                # For exact match (common_len == len(prompt)), trim to
                # common_len - 1 and re-feed the last token, because
                # stream_generate requires a non-empty prompt.
                trim_target = common_len
                if common_len == len(prompt_token_ids):
                    trim_target = common_len - 1

                total_cached = cache[0].offset
                num_to_trim = total_cached - trim_target
                if num_to_trim > 0:
                    if can_trim_prompt_cache(cache):
                        trim_prompt_cache(cache, num_to_trim)
                    else:
                        cache = make_prompt_cache(self._model)
                        self._cache_misses += 1
                        return cache, prompt_token_ids, False

                remaining = prompt_token_ids[trim_target:]
                logger.debug(
                    "MLX cache hit: reusing %d/%d prompt tokens (%d to process)",
                    trim_target,
                    len(prompt_token_ids),
                    len(remaining),
                )
                self._cache_hits += 1
                return cache, remaining, True

        # No usable cache -- create fresh
        self._cache_misses += 1
        return make_prompt_cache(self._model), prompt_token_ids, False

    def _store_cache(self, prompt_token_ids: list[int], cache: list[Any]) -> None:
        """Store the prompt cache in the pool for potential reuse."""
        if self._cache_enabled:
            self._cache_mgr.store_prefix(prompt_token_ids, cache)

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        import mlx_lm  # type: ignore[import-untyped]

        prompt = self._apply_chat_template(request.messages, enable_thinking=request.enable_thinking)
        prompt_token_ids = self._tokenizer.encode(prompt)
        prompt_tokens = len(prompt_token_ids)
        sampler = self._make_sampler(request.temperature, request.top_p)
        start = time.monotonic()
        first_token_time: Optional[float] = None
        tokens: list[str] = []
        final_tps: float = 0.0

        # Fetch reusable prompt cache or create fresh
        cache, remaining_tokens, cache_hit = self._fetch_or_create_cache(prompt_token_ids)

        extra_kwargs: dict[str, Any] = {
            "prompt_cache": cache,
            "prefill_step_size": 4096,
        }
        if self._draft_model is not None:
            extra_kwargs["draft_model"] = self._draft_model

        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=remaining_tokens,
            max_tokens=request.max_tokens,
            sampler=sampler,
            **extra_kwargs,
        ):
            if first_token_time is None:
                first_token_time = time.monotonic()
            final_tps = response.generation_tps
            if response.finish_reason or self._is_stop_token(response.text):
                break
            tokens.append(response.text)

        # Store cache for next request (cache was updated in-place)
        self._store_cache(prompt_token_ids, cache)

        elapsed = time.monotonic() - start
        ttfc = ((first_token_time or start) - start) * 1000
        text = "".join(tokens)

        return (
            text,
            InferenceMetrics(
                ttfc_ms=ttfc,
                prompt_tokens=prompt_tokens,
                total_tokens=len(tokens),
                tokens_per_second=final_tps,
                total_duration_ms=elapsed * 1000,
                cache_hit=cache_hit,
                attention_backend="metal_fused",  # MLX uses Metal fused attention automatically
            ),
        )

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio
        import threading

        import mlx_lm  # type: ignore[import-untyped]

        t_start = time.perf_counter()

        prompt = self._apply_chat_template(request.messages, enable_thinking=request.enable_thinking)
        t_template = time.perf_counter()

        prompt_token_ids = self._tokenizer.encode(prompt)
        t_tokenize = time.perf_counter()

        sampler = self._make_sampler(request.temperature, request.top_p)
        t_sampler = time.perf_counter()

        # Fetch reusable prompt cache or create fresh
        cache, remaining_tokens, _cache_hit = self._fetch_or_create_cache(prompt_token_ids)
        t_cache = time.perf_counter()

        extra_kwargs: dict[str, Any] = {
            "prompt_cache": cache,
            "prefill_step_size": 4096,
        }
        if self._draft_model is not None:
            extra_kwargs["draft_model"] = self._draft_model

        # mlx_lm.stream_generate is a sync generator -- push chunks from
        # a background thread into an asyncio.Queue for non-blocking consumption.
        # asyncio.Queue is NOT thread-safe, so use call_soon_threadsafe for puts.
        queue: asyncio.Queue[Optional[Any]] = asyncio.Queue()
        cancelled = threading.Event()
        done = threading.Event()
        loop = asyncio.get_event_loop()
        t_first_token: Optional[float] = None

        def _produce() -> None:
            nonlocal t_first_token
            try:
                for response in mlx_lm.stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=remaining_tokens,
                    max_tokens=request.max_tokens,
                    sampler=sampler,
                    **extra_kwargs,
                ):
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    if cancelled.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, response)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
                done.set()

        t_pre_submit = time.perf_counter()
        self._executor.submit(_produce)
        t_post_submit = time.perf_counter()

        # Log pre-generation timing breakdown
        timings = {
            "chat_template_ms": (t_template - t_start) * 1000,
            "tokenize_ms": (t_tokenize - t_template) * 1000,
            "sampler_ms": (t_sampler - t_tokenize) * 1000,
            "cache_lookup_ms": (t_cache - t_sampler) * 1000,
            "queue_setup_ms": (t_pre_submit - t_cache) * 1000,
            "executor_submit_ms": (t_post_submit - t_pre_submit) * 1000,
            "total_pre_gen_ms": (t_post_submit - t_start) * 1000,
            "cache_hit": _cache_hit,
            "prompt_tokens": len(prompt_token_ids),
            "remaining_tokens": len(remaining_tokens),
        }
        logger.info("MLX pre-generation timings: %s", timings)

        # Store latest timings for the debug endpoint
        self._last_timings = timings

        if self._verbose:
            self._verbose.emit("inference.pre_generation", **timings)

        try:
            while True:
                response = await queue.get()
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
        finally:
            cancelled.set()
            t_done_wait_start = time.perf_counter()
            # Wait for producer to fully exit before reusing the model/cache.
            # Without this, Metal command buffers from the dying producer
            # collide with the next request's Metal operations -> SIGABRT.
            done.wait(timeout=5.0)
            t_done_wait_end = time.perf_counter()
            # Store cache for next request. The done.wait() above guarantees
            # the producer has fully stopped and all Metal ops are complete,
            # so the cache is in a consistent (if partially filled) state.
            self._store_cache(prompt_token_ids, cache)
            t_store = time.perf_counter()

            # Update timings with post-generation metrics
            self._last_timings["done_wait_ms"] = (t_done_wait_end - t_done_wait_start) * 1000
            self._last_timings["cache_store_ms"] = (t_store - t_done_wait_end) * 1000
            if t_first_token is not None:
                self._last_timings["prefill_ms"] = (t_first_token - t_post_submit) * 1000
                self._last_timings["total_ttft_ms"] = (t_first_token - t_start) * 1000
            logger.info("MLX full timings: %s", self._last_timings)

            if self._verbose:
                mem_info: dict[str, Any] = {}
                try:
                    import mlx.core as mx  # type: ignore[import-untyped]

                    mem_info["active_memory_mb"] = round(mx.metal.get_active_memory() / (1024 * 1024), 1)
                    mem_info["peak_memory_mb"] = round(mx.metal.get_peak_memory() / (1024 * 1024), 1)
                except Exception:
                    pass
                self._verbose.emit("inference.stream_completed", **self._last_timings, **mem_info)

    @property
    def kv_cache(self) -> Any:
        """Expose the KVCacheManager for the stats endpoint."""
        return self._cache_mgr

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []
