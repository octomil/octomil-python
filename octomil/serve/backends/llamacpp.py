"""Cross-platform backend using llama-cpp-python.

DEPRECATED for product `chat.completion`: the v0.1.2 hard-cutover
moved local-GGUF chat to ``octomil.runtime.native.chat_backend
.NativeChatBackend``, which speaks ``octomil-runtime`` directly.
``LlamaCppEngine.create_backend`` no longer vends this class.

This module is kept INTERNAL for:
  - Non-product dev utilities that need the Python ``llama_cpp``
    surface (e.g., the cutover gate's Python-comparison harness in
    ``tests/test_chat_cutover_gate.py``).
  - Any future grammar / JSON-mode / streaming flows that haven't
    been ported to the native runtime yet — those would need a
    separate planner route, not silent fallback from native.

Do NOT add new product call sites.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, AsyncIterator

from ...errors import OctomilError, OctomilErrorCode
from ...models.resolver import ModelResolutionError as _NewResolutionError
from ...models.resolver import resolve as _resolve_new
from ..models import _GGUF_MODELS
from ..types import GenerationChunk, GenerationRequest, InferenceBackend, InferenceMetrics

logger = logging.getLogger(__name__)


from ..types import BackendCapabilities  # noqa: E402


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
    # Cutover follow-up #71: legacy llama_cpp.Llama path supports
    # GBNF grammar internally (via LlamaGrammar.from_string) and
    # JSON mode via the same machinery. Streaming via
    # generate_stream is supported. Tools / function-calling are
    # NOT in this backend's surface.
    capabilities = BackendCapabilities(
        grammar_supported=True,
        json_mode_supported=True,
        streaming_supported=True,
        tools_supported=False,
        attention_backend="flash_attention",
    )

    def __init__(
        self,
        cache_size_mb: int = 2048,
        cache_enabled: bool = True,
        model_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._llm: Any = None
        self._model_name: str = ""
        self._cache_size_mb = cache_size_mb
        self._cache_enabled = cache_enabled
        self._llama_cache: Any = None
        # Optional caller-supplied model directory. PrepareManager passes
        # this when the planner has materialized the artifact under
        # ``<cache>/artifacts/<artifact_id>/``. We look for a ``.gguf``
        # file inside (sentinel ``artifact`` first, then any *.gguf) and
        # load it via ``llama_cpp.Llama(model_path=...)`` so the prepared
        # bytes drive inference instead of huggingface_hub downloading
        # the GGUF anew.
        self._injected_model_dir: str | None = model_dir

    def _resolve_local_gguf_file(self) -> str | None:
        """Return a path to a GGUF file inside the injected ``model_dir``.

        Resolution order:

        1. PrepareManager's single-file sentinel ``<dir>/artifact``. The
           planner emits this when ``required_files`` is empty; the
           sentinel has no extension so a naive ``.gguf`` glob would
           miss it. We return the sentinel verbatim — llama_cpp.Llama
           opens the file as GGUF regardless of suffix because it
           inspects the magic bytes.
        2. Any ``*.gguf`` at the top level of the directory. Covers the
           multi-file artifact / legacy ``required_files=['name.gguf']``
           case.

        Returns ``None`` when no dir is injected, the dir is missing,
        or no candidate file is found — the existing HF/repo path runs
        unchanged.
        """
        if not self._injected_model_dir:
            return None
        model_dir = self._injected_model_dir
        if not os.path.isdir(model_dir):
            return None
        sentinel = os.path.join(model_dir, "artifact")
        if os.path.isfile(sentinel):
            return sentinel
        for entry in sorted(os.listdir(model_dir)):
            if entry.lower().endswith(".gguf"):
                return os.path.join(model_dir, entry)
        return None

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

        # PrepareManager-materialized artifact takes priority over every
        # other lookup. Resolve a single .gguf file inside the injected
        # directory and load via Llama(model_path=...) so the prepared
        # bytes are used. If the directory has no candidate GGUF (e.g.
        # malformed manifest) we fall through to the existing HF/repo
        # resolution path so users still get an error message that
        # points at the recovery action.
        prepared_path = self._resolve_local_gguf_file()
        if prepared_path:
            logger.info("Loading prepared GGUF: %s", prepared_path)
            self._llm = Llama(
                model_path=prepared_path,
                n_ctx=4096,
                n_batch=256,
                n_gpu_layers=-1,
                flash_attn=True,
                verbose=False,
            )
            self._attach_cache()
            return

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
        tps = completion_tokens / elapsed if elapsed > 0 else 0

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
            chunk = await loop.run_in_executor(self._executor, _next_chunk)
            if chunk is None:
                break
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            finish = chunk["choices"][0].get("finish_reason")
            if content or finish:
                yield GenerationChunk(text=content, finish_reason=finish)

    def get_verbose_metadata(
        self,
        event_name: str,
        *,
        request: GenerationRequest | None = None,
        metrics: InferenceMetrics | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        if event_name == "backend.load_completed":
            meta["cache_enabled"] = self._cache_enabled
            meta["cache_size_mb"] = self._cache_size_mb
        elif event_name == "backend.generate_completed" and metrics is not None:
            meta["attention_backend"] = metrics.attention_backend
        return meta

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []
