"""FastAPI app factory and server runner for single-model serving."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

from ..errors import OctomilError, OctomilErrorCode
from .config import CloudConfig, MoEConfig, ServerState
from .detection import _detect_backend, _get_cache_manager, _log_startup_error
from .grammar_helpers import (
    _inject_json_system_prompt,
    _reject_explicit_grammar_on_non_grammar_backend,
    _resolve_grammar,
)
from .instrumentation import unwrap_backend
from .models import ChatCompletionBody
from .streaming import _queued_stream_response, _stream_response
from .types import GenerationRequest

if TYPE_CHECKING:
    from ..early_exit import EarlyExitConfig

logger = logging.getLogger(__name__)


def _is_sherpa_tts_model(model_name: str) -> bool:
    """Lazy wrapper around the sherpa engine's TTS-model check.

    Imported as a free function so the dispatch site stays a one-liner and
    we avoid hauling sherpa-onnx into the import path of non-TTS servers.
    """
    from ..runtime.engines.sherpa import is_sherpa_tts_model

    return is_sherpa_tts_model(model_name)


def _create_cloud_backend_from_profile(profile: Any, model: str) -> Any:
    """Build a CloudInferenceBackend from a config CloudProfile.

    Used only for config-driven cloud routing — not for explicit --cloud
    or catalog :cloud paths.
    """
    from .backends.cloud import CloudInferenceBackend

    api_key = os.environ.get(profile.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Cloud routing requires {profile.api_key_env} to be set.")

    from ..execution.kernel import _openai_base_url

    backend = CloudInferenceBackend(
        base_url=_openai_base_url(profile),
        api_key=api_key,
        model=model,
    )
    backend.load_model(model)
    return backend


def _backend_for_locality(state: ServerState, locality: str) -> Any:
    """Return the backend matching the given locality."""
    if locality == "cloud":
        return state.cloud_backend
    return state.backend


def _active_backend_for_metadata(state: ServerState) -> Any:
    """Return the request-primary backend for metadata endpoints."""
    if state.engine_name == "cloud" and state.cloud_backend is not None:
        return state.cloud_backend
    return state.backend or state.cloud_backend


async def _generate_with_backend(backend: Any, request: GenerationRequest) -> tuple[str, Any]:
    """Call generate on a backend, preferring async when available."""
    generate_async = getattr(backend, "generate_async", None)
    if generate_async is not None:
        return await generate_async(request)
    return backend.generate(request)


async def _generate_with_routing(
    state: ServerState,
    request: GenerationRequest,
    decision: Any,
) -> tuple[str, Any, str, bool]:
    """Dispatch generation through kernel routing with optional fallback.

    Returns (text, metrics, used_locality, fallback_used).
    """
    primary = _backend_for_locality(state, decision.primary_locality)
    if primary is None:
        raise OctomilError(
            code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
            message=f"No backend available for {decision.primary_locality} routing.",
        )

    try:
        text, metrics = await _generate_with_backend(primary, request)
        return text, metrics, decision.primary_locality, False
    except Exception as primary_exc:
        if decision.fallback_locality is None:
            raise

        fallback = _backend_for_locality(state, decision.fallback_locality)
        if fallback is None:
            raise

        logger.warning(
            "Primary %s backend failed (%s), falling back to %s",
            decision.primary_locality,
            primary_exc,
            decision.fallback_locality,
        )
        text, metrics = await _generate_with_backend(fallback, request)
        return text, metrics, decision.fallback_locality, True


async def _kernel_driven_startup(state: ServerState, model_name: str) -> None:
    """Initialise backends based on the kernel routing policy.

    Called during lifespan when ``config_set`` was provided.
    """
    kernel = state.kernel
    assert kernel is not None

    from .._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy
    from ..config.local import CAPABILITY_CHAT
    from ..execution.kernel import _cloud_available, _resolve_routing_policy

    defaults = kernel._resolve(CAPABILITY_CHAT, model=model_name)
    routing_policy = _resolve_routing_policy(defaults)
    cloud_available = _cloud_available(defaults)

    # Cloud-only means cloud-only. Do not probe local engines or trigger local
    # model downloads before we know the policy permits local execution.
    _local_ok = False
    if routing_policy.mode != ContractRoutingPolicy.CLOUD_ONLY:
        try:
            state.backend = _detect_backend(
                model_name,
                cache_size_mb=state.cache_size_mb,
                cache_enabled=state.cache_enabled,
                engine_override=state.engine_override,
                verbose_emitter=state.verbose_emitter,
            )
            _local_ok = state.backend is not None
            state.engine_name = state.backend.name if state.backend else "none"
        except Exception as exc:
            logger.info("Local backend detection failed (%s); continuing with cloud if policy allows it.", exc)

    # Resolve routing to determine what we actually need
    try:
        decision = kernel.resolve_chat_routing(
            model=model_name,
            local_available=_local_ok,
            cloud_available=cloud_available,
        )
    except RuntimeError:
        # Routing resolution can fail when policy demands a resource that
        # is unavailable (e.g. private with no local).  Raise to fail startup.
        raise

    state.routing_policy_preset = decision.policy_preset

    # Construct cloud backend when routing needs it
    needs_cloud = decision.primary_locality == "cloud" or decision.fallback_locality == "cloud"
    if needs_cloud and decision.cloud_profile is not None:
        try:
            state.cloud_backend = _create_cloud_backend_from_profile(decision.cloud_profile, model_name)
            logger.info("Cloud backend ready (profile=%s)", decision.cloud_profile.name)
        except Exception as exc:
            if decision.primary_locality == "cloud" and decision.fallback_locality is None:
                raise RuntimeError(f"Cloud backend required by policy but could not be initialised: {exc}") from exc
            logger.warning("Failed to create cloud backend from profile; cloud fallback disabled: %s", exc)

    if decision.primary_locality == "cloud" and state.cloud_backend is not None:
        state.engine_name = "cloud"

    # If the policy has no local fallback, clear any local backend we probed.
    if decision.primary_locality == "cloud" and decision.fallback_locality is None:
        if state.backend is not None and state.cloud_backend is not None:
            logger.info("Cloud-primary policy has no local fallback; clearing local backend.")
            state.backend = None

    # If local failed and policy requires it with no fallback, fail startup
    if not _local_ok and decision.primary_locality == "on_device" and decision.fallback_locality is None:
        raise RuntimeError(
            f"Local backend required by policy '{decision.policy_preset}' but "
            f"no engine could load model '{model_name}'."
        )


def create_app(
    model_name: str,
    *,
    api_key: Optional[str] = None,
    api_base: str = "https://api.octomil.com/api/v1",
    json_mode: bool = False,
    cache_size_mb: int = 2048,
    cache_enabled: bool = True,
    engine: Optional[str] = None,
    max_queue_depth: int = 32,
    moe_config: Optional[MoEConfig] = None,
    compress_context: bool = False,
    compression_strategy: str = "token_pruning",
    compression_ratio: float = 0.5,
    compression_max_turns: int = 4,
    compression_threshold: int = 256,
    tool_use: bool = False,
    early_exit_config: Optional["EarlyExitConfig"] = None,
    verbose: bool = False,
    cloud_config: Optional[CloudConfig] = None,
    config_set: Any = None,
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
    moe_config:
        Configuration for Mixture of Experts model features.
        When ``None``, uses defaults (auto-detection enabled).
    compress_context:
        Enable prompt compression.  Long prompts are compressed before
        inference to reduce context window usage and speed up prefill.
    compression_strategy:
        Compression strategy: ``"token_pruning"`` or ``"sliding_window"``.
    compression_ratio:
        Target compression ratio (0.0--1.0) for token pruning.
    compression_max_turns:
        Number of recent turns to keep verbatim (sliding_window strategy).
    compression_threshold:
        Minimum estimated token count before compression activates.
    tool_use:
        When ``True``, pre-load coding agent tool schemas and expose
        them at ``/v1/tool-schemas``.  Tools include ``read_file``,
        ``write_file``, ``edit_file``, ``run_command``, ``search_files``.
    early_exit_config:
        Configuration for early exit / adaptive computation depth.
        When ``None`` or not enabled, early exit monitoring is disabled.
    verbose:
        When ``True``, emit rich low-level runtime events for debugging.
        Events are logged locally and posted to the server (if api_key set).
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    from ..errors import OctomilError, OctomilErrorCode  # noqa: F811

    # Detect MoE model from catalog
    from ..models.catalog import get_model, get_moe_metadata
    from ..models.catalog import is_moe_model as _is_moe

    _moe_detected = _is_moe(model_name)
    _moe_meta = get_moe_metadata(model_name)

    # Detect reasoning model from catalog capabilities
    # Strip quant suffix (e.g. "qwen3-4b:q2_k" -> "qwen3-4b") for catalog lookup
    _catalog_lookup = model_name.split(":")[0] if ":" in model_name else model_name
    _catalog_entry = get_model(_catalog_lookup)
    _is_reasoning = bool(_catalog_entry and "reasoning" in _catalog_entry.capabilities)

    # Build compressor if enabled
    _compressor = None
    if compress_context:
        from ..compression import CompressionConfig, PromptCompressor

        _compressor = PromptCompressor(
            CompressionConfig(
                enabled=True,
                strategy=compression_strategy,
                target_ratio=compression_ratio,
                max_turns_verbatim=compression_max_turns,
                token_threshold=compression_threshold,
            )
        )

    # --- Kernel construction (config-driven routing) ---
    _kernel = None
    if config_set is not None:
        from ..execution.kernel import ExecutionKernel

        _kernel = ExecutionKernel(config_set=config_set)

    state = ServerState(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        default_json_mode=json_mode,
        cache_size_mb=cache_size_mb,
        cache_enabled=cache_enabled,
        engine_override=engine,
        max_queue_depth=max_queue_depth,
        moe_config=moe_config or MoEConfig(),
        is_moe_model=_moe_detected,
        moe_metadata=_moe_meta,
        compressor=_compressor,
        early_exit_config=early_exit_config,
        tool_use=tool_use,
        is_reasoning_model=_is_reasoning,
        verbose_runtime_logs=verbose,
        cloud_config=cloud_config,
        kernel=_kernel,
        config_set=config_set,
    )

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        # Initialise verbose runtime event emitter early (before backend
        # detection) so model load events are captured.
        if state.verbose_runtime_logs:
            from ..telemetry import _generate_device_id
            from .verbose_events import VerboseEventEmitter

            state.verbose_emitter = VerboseEventEmitter(
                api_key=state.api_key,
                api_base=state.api_base,
                device_id=_generate_device_id(),
                model_name=model_name,
            )
            logger.info("Verbose runtime logging enabled")

        if state.is_moe_model and state.moe_metadata and state.moe_config.enabled:
            _m = state.moe_metadata
            logger.info(
                "MoE model detected: %s (%d experts, %d active per token, %s total params, %s active params)",
                model_name,
                _m.num_experts,
                _m.active_experts,
                _m.total_params,
                _m.active_params,
            )

        # Check if this is a whisper (speech-to-text) model
        from ..runtime.engines.whisper import is_whisper_model

        if is_whisper_model(model_name):
            from ..runtime.engines.whisper import WhisperCppEngine

            whisper_engine = WhisperCppEngine()
            whisper_backend = whisper_engine.create_backend(model_name)
            try:
                whisper_backend.load_model(model_name)
            except Exception as exc:
                _log_startup_error(model_name, exc)
                raise
            state.whisper_backend = whisper_backend
            state.engine_name = "whisper.cpp"
        elif _is_sherpa_tts_model(model_name):
            # ``_SherpaTtsBackend._resolve_model_dir`` requires an
            # injected prepared ``model_dir=`` (the PR D cutover
            # removed the legacy ``~/.octomil/models/sherpa`` and env
            # fallback). Route TTS startup through ``kernel.prepare``
            # so PrepareManager materializes the artifact, then
            # construct + load the backend ONCE against that
            # ``artifact_dir``.
            #
            # We deliberately call ``prepare`` instead of ``warmup``:
            # warmup also loads + caches a backend in
            # ``kernel._warmed_backends``, but the lifespan owns the
            # canonical backend on ``state.sherpa_tts_backend``.
            # Going through warmup would double model load time and
            # resident memory while the kernel-cached copy sat
            # unused for the lifetime of the server.
            from ..execution.kernel import ExecutionKernel
            from ..runtime.engines.sherpa import SherpaTtsEngine

            tts_kernel = state.kernel or ExecutionKernel(config_set=state.config_set)
            try:
                prepare_outcome = await asyncio.to_thread(tts_kernel.prepare, model=model_name, capability="tts")
                artifact_dir = str(prepare_outcome.artifact_dir)
                sherpa_backend = SherpaTtsEngine().create_backend(
                    model_name,
                    model_dir=artifact_dir,
                )
                sherpa_backend.load_model(model_name)
            except Exception as exc:
                _log_startup_error(model_name, exc)
                raise
            state.sherpa_tts_backend = sherpa_backend
            state.kernel = tts_kernel
            state.engine_name = "sherpa-onnx"
        elif state.kernel is not None:
            # --- Config-driven startup: use kernel routing to decide backends ---
            await _kernel_driven_startup(state, model_name)
        elif state.cloud_config is not None and state.engine_override is None:
            # Cloud-only mode: use CloudInferenceBackend directly
            from .backends.cloud import CloudInferenceBackend

            cloud_backend = CloudInferenceBackend(
                base_url=state.cloud_config.base_url,
                api_key=state.cloud_config.api_key,
                model=state.cloud_config.model,
            )
            cloud_backend.load_model(state.cloud_config.model)
            state.backend = cloud_backend
            state.engine_name = "cloud"
            state.model_name = state.cloud_config.model
        else:
            try:
                state.backend = _detect_backend(
                    model_name,
                    cache_size_mb=state.cache_size_mb,
                    cache_enabled=state.cache_enabled,
                    engine_override=state.engine_override,
                    verbose_emitter=state.verbose_emitter,
                )
            except Exception as exc:
                _log_startup_error(model_name, exc)
                raise
            state.engine_name = state.backend.name if state.backend else "none"
        state.start_time = time.time()

        if state.verbose_emitter is not None:
            state.verbose_emitter.emit(
                "server.started",
                model=model_name,
                engine=state.engine_name,
                cache_enabled=state.cache_enabled,
                cache_size_mb=state.cache_size_mb,
            )

        # Initialise early exit monitor
        if state.early_exit_config is not None and state.early_exit_config.enabled:
            from ..early_exit import EarlyExitMonitor as _EEM

            state.early_exit_monitor = _EEM(config=state.early_exit_config)
            logger.info(
                "Early exit enabled (threshold=%.2f, min_layers_frac=%.2f%s)",
                state.early_exit_config.effective_threshold,
                state.early_exit_config.effective_min_layers_fraction,
                f", preset={state.early_exit_config.preset.value}" if state.early_exit_config.preset else "",
            )

        # Initialise request queue
        if state.max_queue_depth > 0:
            from ..batch import RequestQueue

            state.request_queue = RequestQueue(max_depth=state.max_queue_depth)
            state.request_queue.start()
            logger.info("Request queue enabled (max_depth=%d)", state.max_queue_depth)

        # Create telemetry reporter when an API key is configured
        if state.api_key:
            try:
                from ..telemetry import TelemetryReporter as _TR

                state.reporter = _TR(
                    api_key=state.api_key,
                    api_base=state.api_base,
                    org_id="default",
                )
                logger.info("Telemetry enabled -- reporting to %s", state.api_base)
            except Exception as exc:
                logger.warning("Failed to initialise telemetry: %s", exc)

        yield

        # Graceful shutdown: stop verbose emitter
        if state.verbose_emitter is not None:
            state.verbose_emitter.close()

        # Graceful shutdown: stop request queue
        if state.request_queue is not None:
            await state.request_queue.stop()

        # Graceful shutdown: drain pending telemetry events
        if state.reporter is not None:
            state.reporter.close()

        # Graceful shutdown: close all warmed embeddings runtimes
        # via the factory's cache reset. Doing this once at lifespan
        # exit (after uvicorn has drained the request queue) avoids
        # the close-during-embed race that would arise from closing
        # per-request when a model_id swap happens.
        try:
            from ..runtime.native.embeddings_runtime import reset_runtime_cache

            reset_runtime_cache()
        except Exception:  # noqa: BLE001
            logger.warning("native embeddings runtime cache reset failed", exc_info=True)
        state.embeddings_backend = None

    app = FastAPI(title="Octomil Serve", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(OctomilError)
    async def octomil_error_handler(request: Request, exc: OctomilError) -> JSONResponse:
        from .._generated.error_code import ERROR_CLASSIFICATION, RetryClass

        # Cutover follow-up #70: SHARED status map across single-
        # and multi-model handlers. See octomil/errors.py.
        from ..errors import octomil_error_to_http_status

        classification = ERROR_CLASSIFICATION.get(exc.code)
        status_code = octomil_error_to_http_status(exc.code)
        return JSONResponse(
            status_code=status_code,
            content={
                "code": exc.code.value,
                "message": str(exc),
                "retryable": classification.retry_class != RetryClass.NEVER if classification else False,
                "category": classification.category.value if classification else "unknown",
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def agent_context_middleware(request, call_next):  # type: ignore[no-untyped-def]
        """Log and store agent context from the X-Octomil-Agent-Context header.

        Coding agents (Aider, Goose, OpenCode) can send this header to
        identify themselves and provide context for smarter routing decisions.
        """
        agent_ctx = request.headers.get("X-Octomil-Agent-Context")
        if agent_ctx:
            logger.info("Agent context: %s", agent_ctx)
            # Store on request state for downstream use
            request.state.agent_context = agent_ctx
        else:
            request.state.agent_context = None
        response = await call_next(request)
        return response

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        active_backend = _active_backend_for_metadata(state)
        models = active_backend.list_models() if active_backend else []
        data = []
        for m in models:
            entry: dict[str, Any] = {
                "id": m,
                "object": "model",
                "created": int(state.start_time),
                "owned_by": "octomil",
            }
            if state.is_moe_model and state.moe_metadata:
                entry["architecture"] = "moe"
                entry["moe"] = {
                    "num_experts": state.moe_metadata.num_experts,
                    "active_experts": state.moe_metadata.active_experts,
                    "expert_size": state.moe_metadata.expert_size,
                    "total_params": state.moe_metadata.total_params,
                    "active_params": state.moe_metadata.active_params,
                }
            else:
                entry["architecture"] = "dense"
            data.append(entry)
        return {
            "object": "list",
            "data": data,
        }

    @app.get("/v1/tool-schemas")
    async def tool_schemas() -> dict[str, Any]:
        """Return pre-loaded coding agent tool schemas.

        Only populated when the server is started with ``--tool-use``.
        Coding agents (Aider, Goose, OpenCode) can discover tools here
        and use them for structured output enforcement.
        """
        if not state.tool_use:
            return {"tools": [], "enabled": False}

        from ..tool_schemas import get_tool_use_tools

        return {"tools": get_tool_use_tools(), "enabled": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionBody) -> Any:
        if state.backend is None and state.cloud_backend is None:
            raise OctomilError(code=OctomilErrorCode.MODEL_LOAD_FAILED, message="Model not loaded")

        # --- Kernel routing decision (when config-driven) ---
        _routing_decision = None
        if state.kernel is not None:
            _routing_decision = state.kernel.resolve_chat_routing(
                model=state.model_name,
                local_available=state.backend is not None,
                cloud_available=state.cloud_backend is not None,
            )

        # --- Tier-0: deterministic routing (arithmetic, etc.) ---
        # Check the last user message for a query that can be answered
        # without invoking any model.
        if body.messages and not body.stream:
            last_user_msg = ""
            for msg in reversed(body.messages):
                if msg.role == "user":
                    last_user_msg = msg.content if isinstance(msg.content, str) else ""
                    break
            if last_user_msg:
                from ..routing import check_deterministic

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

        messages: list[dict[str, Any]] = [{"role": m.role, "content": m.content or ""} for m in body.messages]

        # --- Prompt compression ---
        compression_stats = None
        if state.compressor is not None:
            messages, compression_stats = state.compressor.compress(messages)

        # For backends without native grammar support (MLX, echo),
        # inject a system prompt nudging JSON output when json_mode is on.
        # When kernel routing is active, use the primary backend for this check.
        _primary_backend = state.backend
        if _routing_decision is not None:
            _primary_backend = _backend_for_locality(state, _routing_decision.primary_locality) or state.backend
        # Cutover follow-up #71: capability query replaces the
        # legacy isinstance(_, LlamaCppBackend) check. The post-
        # cutover NativeChatBackend declares grammar_supported=False;
        # legacy LlamaCppBackend declares True. The serve layer
        # routes accordingly.
        # Cutover follow-up #71 (R5 Codex): use resolve_backend_capabilities
        # so duck-typed backends (e.g. _ORTBackend, _OllamaBackend) without
        # an explicit `capabilities` class attr fall through to conservative
        # defaults rather than AttributeError'ing the chat handler.
        from .types import resolve_backend_capabilities  # noqa: PLC0415

        uses_grammar_natively = (
            _primary_backend is not None
            and resolve_backend_capabilities(unwrap_backend(_primary_backend)).grammar_supported
        )
        # Cutover follow-up #71 (R1 Codex): reject explicit caller GBNF
        # against non-grammar backends instead of silently stripping it.
        if _primary_backend is not None:
            _reject_explicit_grammar_on_non_grammar_backend(
                backend_name=unwrap_backend(_primary_backend).name,
                grammar_str=grammar_str,
                is_json=is_json,
                uses_grammar_natively=uses_grammar_natively,
            )
        schema_for_prompt: Optional[dict[str, Any]] = None
        if is_json and not uses_grammar_natively:
            rf = body.response_format or {}
            if rf.get("type") == "json_schema":
                raw = rf.get("json_schema") or rf.get("schema")
                schema_for_prompt = raw.get("schema", raw) if raw else None
            messages = _inject_json_system_prompt(messages, schema_for_prompt)

        # Extract enable_thinking from the request body (Qwen3 / OpenClaw convention)
        _enable_thinking: Optional[bool] = getattr(body, "enable_thinking", None)

        gen_req = GenerationRequest(
            model=body.model or state.model_name,
            messages=messages,
            max_tokens=body.max_tokens or 512,
            temperature=body.temperature,
            top_p=body.top_p,
            stream=body.stream,
            grammar=grammar_str if uses_grammar_natively else None,
            # Cutover follow-up #71 (R1 Codex): only forward json_mode=True
            # to backends that handle JSON natively via grammar. For
            # non-grammar backends we already injected a system prompt
            # (line above); forwarding json_mode=True caused
            # NativeChatBackend's gate to raise UNSUPPORTED_MODALITY,
            # turning every JSON-mode request into a 422.
            json_mode=is_json and uses_grammar_natively,
            enable_thinking=_enable_thinking,
        )

        state.request_count += 1
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        session_id = uuid.uuid4().hex
        model_version = "latest"
        _reporter = state.reporter

        # Report inference_started (best-effort)
        if _reporter is not None:
            try:
                _reporter.report_inference_started(
                    model_id=gen_req.model,
                    version=model_version,
                    session_id=session_id,
                )
            except Exception:
                pass

        # Report compression telemetry (best-effort)
        if _reporter is not None and compression_stats is not None:
            try:
                _reporter.report_prompt_compressed(
                    session_id=session_id,
                    model_id=gen_req.model,
                    version=model_version,
                    original_tokens=compression_stats.original_tokens,
                    compressed_tokens=compression_stats.compressed_tokens,
                    compression_ratio=compression_stats.compression_ratio,
                    strategy=compression_stats.strategy,
                    duration_ms=compression_stats.duration_ms,
                )
            except Exception:
                pass

        # --- Queue-aware dispatch ---
        _queue = state.request_queue

        if gen_req.stream:
            _is_reasoning_stream = state.is_reasoning_model
            # Select the routed backend for streaming (no streaming fallback in this PR)
            _stream_backend = None
            if _routing_decision is not None:
                _stream_backend = _backend_for_locality(state, _routing_decision.primary_locality)
            if _queue is not None:
                # Stream through the request queue
                return StreamingResponse(
                    _queued_stream_response(
                        state,
                        gen_req,
                        req_id,
                        session_id,
                        _queue,
                        is_reasoning=_is_reasoning_stream,
                        backend=_stream_backend,
                    ),
                    media_type="text/event-stream",
                )
            return StreamingResponse(
                _stream_response(
                    state,
                    gen_req,
                    req_id,
                    session_id,
                    is_reasoning=_is_reasoning_stream,
                    backend=_stream_backend,
                ),
                media_type="text/event-stream",
            )

        _verbose = state.verbose_emitter
        if _verbose is not None:
            _verbose.emit(
                "inference.request_received",
                request_id=req_id,
                model=gen_req.model,
                prompt_messages=len(gen_req.messages),
                max_tokens=gen_req.max_tokens,
                temperature=gen_req.temperature,
                stream=gen_req.stream,
                json_mode=gen_req.json_mode,
                engine=state.engine_name,
            )

        gen_start = time.monotonic()
        try:
            if _routing_decision is not None and _queue is None:
                # Config-driven routing: dispatch with fallback
                text, metrics, _used_locality, _fallback_used = await _generate_with_routing(
                    state, gen_req, _routing_decision
                )
            elif _queue is not None:
                from ..batch import QueueFullError, QueueTimeoutError

                # Queue-aware dispatch.  When routing is active, wrap the
                # generate callable so the queue uses the routed backend.
                if _routing_decision is not None:

                    def _routed_generate_sync(req: GenerationRequest) -> tuple[str, Any]:
                        """Sync wrapper for queue — uses sync .generate() with fallback."""
                        primary = _backend_for_locality(state, _routing_decision.primary_locality)
                        if primary is None:
                            raise OctomilError(
                                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                                message=f"No backend for {_routing_decision.primary_locality}.",
                            )
                        try:
                            return primary.generate(req)
                        except Exception:
                            if _routing_decision.fallback_locality is None:
                                raise
                            fb = _backend_for_locality(state, _routing_decision.fallback_locality)
                            if fb is None:
                                raise
                            return fb.generate(req)

                    try:
                        text, metrics = await _queue.submit_generate(gen_req, _routed_generate_sync)
                    except QueueFullError:
                        raise OctomilError(
                            code=OctomilErrorCode.RATE_LIMITED,
                            message="Server busy -- request queue full. Try again later.",
                        )
                    except QueueTimeoutError:
                        raise OctomilError(
                            code=OctomilErrorCode.REQUEST_TIMEOUT,
                            message="Request timed out waiting in queue.",
                        )
                else:
                    assert state.backend is not None
                    try:
                        text, metrics = await _queue.submit_generate(gen_req, state.backend.generate)
                    except QueueFullError:
                        raise OctomilError(
                            code=OctomilErrorCode.RATE_LIMITED,
                            message="Server busy -- request queue full. Try again later.",
                        )
                    except QueueTimeoutError:
                        raise OctomilError(
                            code=OctomilErrorCode.REQUEST_TIMEOUT,
                            message="Request timed out waiting in queue.",
                        )
            else:
                assert state.backend is not None
                text, metrics = state.backend.generate(gen_req)
        except OctomilError:
            raise
        except Exception:
            if _reporter is not None:
                try:
                    _reporter.report_inference_failed(
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
            from ..grammar import extract_json, validate_json_output

            if not validate_json_output(text):
                extracted = extract_json(text)
                if extracted is not None:
                    text = json.dumps(extracted)
                else:
                    # Retry once with a stronger system prompt
                    retry_messages = _inject_json_system_prompt(messages, schema_for_prompt)
                    # Cutover follow-up #71 (R2 Codex): the retry block
                    # only fires in the `is_json and not uses_grammar_natively`
                    # branch — by construction the non-grammar fallback. The
                    # system prompt is doing the JSON constraining; forwarding
                    # `json_mode=True` to a non-grammar backend (e.g., native)
                    # would 422 with UNSUPPORTED_MODALITY, so the retry would
                    # always fail instead of producing a corrected response.
                    retry_req = GenerationRequest(
                        model=gen_req.model,
                        messages=retry_messages,
                        max_tokens=gen_req.max_tokens,
                        temperature=max(gen_req.temperature - 0.2, 0.0),
                        top_p=gen_req.top_p,
                        stream=False,
                        json_mode=False,
                    )
                    if _routing_decision is not None:
                        text, metrics, _, _ = await _generate_with_routing(state, retry_req, _routing_decision)
                    else:
                        assert state.backend is not None
                        text, metrics = state.backend.generate(retry_req)
                    # Best-effort extraction on retry
                    if not validate_json_output(text):
                        extracted = extract_json(text)
                        if extracted is not None:
                            text = json.dumps(extracted)

        # Early exit monitoring (best-effort)
        _ee_monitor = state.early_exit_monitor
        ee_metrics_dict: Optional[dict[str, Any]] = None
        if _ee_monitor is not None and _ee_monitor.config.enabled:
            try:
                total_layers = _ee_monitor.config.total_layers or 32
                ee_req_metrics = _ee_monitor.simulate_token_exits(
                    token_count=metrics.total_tokens,
                    total_layers=total_layers,
                )
                _ee_monitor.record_request(ee_req_metrics)
                metrics.early_exit_tokens = ee_req_metrics.early_exit_tokens
                metrics.avg_layers_used = ee_req_metrics.avg_layers_used
                ee_metrics_dict = ee_req_metrics.to_dict()
            except Exception:
                pass

        # Report inference_completed (best-effort)
        if _reporter is not None:
            try:
                total_tokens = metrics.total_tokens
                throughput = total_tokens / (gen_elapsed_ms / 1000) if gen_elapsed_ms > 0 else 0.0
                _reporter.report_inference_completed(
                    session_id=session_id,
                    model_id=gen_req.model,
                    version=model_version,
                    total_chunks=total_tokens,
                    total_duration_ms=gen_elapsed_ms,
                    ttfc_ms=metrics.ttfc_ms,
                    throughput=throughput,
                    attention_backend=metrics.attention_backend,
                    early_exit_stats=ee_metrics_dict,
                )
            except Exception:
                pass

        # Verbose runtime event (after generation)
        if _verbose is not None:
            total_tokens = metrics.total_tokens
            throughput = total_tokens / (gen_elapsed_ms / 1000) if gen_elapsed_ms > 0 else 0.0
            _verbose.emit(
                "inference.completed",
                request_id=req_id,
                model=gen_req.model,
                engine=state.engine_name,
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=total_tokens,
                ttfc_ms=round(metrics.ttfc_ms, 2),
                total_duration_ms=round(gen_elapsed_ms, 2),
                tokens_per_second=round(metrics.tokens_per_second, 2)
                if metrics.tokens_per_second
                else round(throughput, 2),
                cache_hit=metrics.cache_hit,
                attention_backend=metrics.attention_backend,
            )

        usage: dict[str, Any] = {
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.total_tokens,
            "total_tokens": metrics.prompt_tokens + metrics.total_tokens,
            "cache_hit": metrics.cache_hit,
        }

        # Include compression stats when compression was applied
        if compression_stats is not None and compression_stats.strategy != "none":
            usage["compression"] = {
                "original_tokens": compression_stats.original_tokens,
                "compressed_tokens": compression_stats.compressed_tokens,
                "ratio": round(compression_stats.compression_ratio, 4),
                "strategy": compression_stats.strategy,
                "duration_ms": round(compression_stats.duration_ms, 2),
            }
        if ee_metrics_dict is not None:
            usage["early_exit"] = ee_metrics_dict

        response_msg: dict[str, Any] = {"role": "assistant", "content": text}
        if state.is_reasoning_model:
            from .thinking import strip_thinking

            content, reasoning = strip_thinking(text)
            response_msg["content"] = content
            if reasoning:
                response_msg["reasoning_content"] = reasoning

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": gen_req.model,
            "choices": [
                {
                    "index": 0,
                    "message": response_msg,
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    @app.post("/v1/embeddings")
    async def create_embeddings(body: dict[str, Any]) -> dict[str, Any]:
        """OpenAI-compatible embeddings endpoint.

        Routes through ``ModelRuntimeRegistry`` so request resolution
        uses the same prepared-artifact lookup as the SDK kernel —
        no direct backend construction in the route, no env-var-only
        path, no model-id mismatch with a stale cached backend.

        Request body::

            {"model": "nomic-embed-text-v1.5",
             "input": "single string" | ["batch", "of", "strings"]}

        Response (OpenAI shape)::

            {"object": "list",
             "data": [{"object": "embedding", "embedding": [...], "index": N}],
             "model": "...",
             "usage": {"prompt_tokens": N, "total_tokens": N}}

        Errors propagate as bounded :class:`OctomilError`. Chat-only
        and unprepared models reject ``UNSUPPORTED_MODALITY`` /
        ``MODEL_NOT_FOUND`` from the registry / backend.

        The synchronous ``backend.embed(...)`` runs in a worker
        thread (``asyncio.to_thread``) so a multi-second embedding
        does not stall the asyncio event loop / chat traffic.
        """
        from ..runtime.core.registry import ModelRuntimeRegistry
        from ..runtime.native.embeddings_runtime import (
            NativeEmbeddingsRuntime,
            is_embedding_model,
        )

        model_id = body.get("model") or state.model_name
        raw_input = body.get("input")
        if raw_input is None:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="`input` is required (string or list of strings)",
            )
        if isinstance(raw_input, str):
            inputs: list[str] = [raw_input]
        elif isinstance(raw_input, list) and all(isinstance(x, str) for x in raw_input):
            inputs = raw_input
        else:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="`input` must be a string or list of strings",
            )
        if not inputs:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="`input` must contain at least one string",
            )

        if not is_embedding_model(model_id):
            raise OctomilError(
                code=OctomilErrorCode.UNSUPPORTED_MODALITY,
                message=(
                    f"Model {model_id!r} is not an embedding-capable family. "
                    f"Supported families: nomic-embed, bge-, e5-mistral/base/large/small, "
                    f"gte-, mxbai-embed, snowflake-arctic-embed, all-minilm, jina-embed. "
                    f"HuggingFace 'org/' prefixes (e.g. BAAI/bge-base-en-v1.5) are accepted."
                ),
            )

        runtime = ModelRuntimeRegistry.shared().resolve(model_id)
        if not isinstance(runtime, NativeEmbeddingsRuntime):
            # Either the registry returned None (no prepared artifact)
            # or it returned a chat runtime for a model id that prefix-
            # matched an embedding family by accident.
            raise OctomilError(
                code=OctomilErrorCode.MODEL_NOT_FOUND,
                message=(
                    f"No prepared embedding artifact for {model_id!r}. "
                    f"Run `octomil prepare {model_id}` (or set "
                    f"OCTOMIL_NATIVE_EMBEDDINGS_MODEL_DIR for dev) before "
                    f"calling /v1/embeddings."
                ),
            )

        # Track the most-recently-used runtime on state for the
        # lifespan teardown. We deliberately do NOT close the previous
        # runtime here — closing while a concurrent request is mid-
        # ``embed()`` (in a worker thread) would free a backend that's
        # actively serving an inference. The factory's ``_runtime_cache``
        # owns runtime lifetime; the lifespan iterates the cache at
        # shutdown via ``reset_runtime_cache()``.
        state.embeddings_backend = runtime

        state.request_count += 1
        result = await asyncio.to_thread(runtime.embed, inputs)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vector, "index": idx}
                for idx, vector in enumerate(result.embeddings)
            ],
            "model": result.model or model_id,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        }

    @app.get("/v1/cache/stats")
    async def cache_stats() -> dict[str, Any]:
        """Return KV cache statistics."""
        if state.backend is None:
            raise OctomilError(code=OctomilErrorCode.MODEL_LOAD_FAILED, message="Model not loaded")

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

    @app.get("/v1/debug/timings")
    async def debug_timings() -> dict[str, Any]:
        """Return the latest MLX pre-generation timing breakdown."""
        if state.backend is None:
            raise OctomilError(code=OctomilErrorCode.MODEL_LOAD_FAILED, message="Model not loaded")
        if hasattr(state.backend, "_last_timings"):
            return state.backend._last_timings
        return {"error": "No timings available (not an MLX backend or no requests yet)"}

    @app.get("/v1/debug/runtime-events")
    async def runtime_events() -> dict[str, Any]:
        """Return recent verbose runtime events (only populated with -v)."""
        if state.verbose_emitter is None:
            return {"enabled": False, "events": []}
        return {
            "enabled": True,
            "events": state.verbose_emitter.recent_events(),
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

    @app.get("/v1/early-exit/stats")
    async def early_exit_stats() -> dict[str, Any]:
        """Return early exit / adaptive computation depth statistics."""
        if state.early_exit_monitor is None:
            return {
                "enabled": False,
                "config": {},
                "stats": {
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_early_exit_tokens": 0,
                    "exit_percentage": 0.0,
                    "avg_layers_used": 0.0,
                    "avg_entropy": 0.0,
                },
            }
        return state.early_exit_monitor.get_stats_dict()

    @app.get("/v1/engines")
    async def list_engines() -> dict[str, Any]:
        """List detected engines and their benchmark results."""
        from ..runtime.engines import get_registry

        registry = get_registry()
        detections = registry.detect_all(state.model_name)

        engines_list = []
        for d in detections:
            entry: dict[str, Any] = {
                "name": d.engine.name,
                "display_name": d.engine.display_name,
                "available": d.available,
                "info": d.info,
                "active": (state.backend is not None and state.backend.name == d.engine.name),
            }
            engines_list.append(entry)

        return {
            "active_engine": state.engine_name,
            "engines": engines_list,
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        cache_info: dict[str, Any] = {"enabled": state.cache_enabled}
        cache_mgr = _get_cache_manager(state.backend) if state.backend is not None else None
        if cache_mgr is not None:
            cs = cache_mgr.stats()
            cache_info["entries"] = cs.entries
            cache_info["hit_rate"] = round(cs.hit_rate, 4)

        backend_name = "none"
        if state.whisper_backend is not None:
            backend_name = state.whisper_backend.name
        elif state.engine_name == "cloud" and state.cloud_backend is not None:
            backend_name = state.cloud_backend.name
        elif state.backend is not None:
            backend_name = state.backend.name
        elif state.cloud_backend is not None:
            backend_name = state.cloud_backend.name

        health_data: dict[str, Any] = {
            "status": "ok",
            "model": state.model_name,
            "engine": state.engine_name,
            "backend": backend_name,
            "requests_served": state.request_count,
            "uptime_seconds": int(time.time() - state.start_time),
            "cache": cache_info,
        }
        if state.early_exit_monitor is not None:
            ee_stats = state.early_exit_monitor.stats
            health_data["early_exit"] = {
                "enabled": True,
                "exit_percentage": round(ee_stats.exit_percentage, 2),
                "avg_layers_used": round(ee_stats.avg_layers_used, 2),
            }
        return health_data

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
            raise OctomilError(
                code=OctomilErrorCode.MODEL_LOAD_FAILED,
                message="No whisper model loaded. Start server with a whisper model: octomil serve whisper-base",
            )

        if file is None:
            raise OctomilError(code=OctomilErrorCode.INVALID_INPUT, message="No audio file provided")

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
            result: dict[str, Any] = state.whisper_backend.transcribe(tmp_path)
            return result
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @app.post("/v1/audio/speech/stream")
    async def synthesize_speech_stream(body: dict[str, Any]) -> Any:
        """Streaming TTS endpoint.

        Returns ``application/octet-stream`` with raw PCM int16 LE chunks.
        Metadata is in response headers; clients should NOT depend on
        HTTP trailers (many proxies drop them). Completion is signalled
        by clean EOF.

        Body shape::

            {
              "model": "kokoro-82m",
              "input": "text to synthesize",
              "voice": "af_bella",            # native voice (back-compat)
              "speaker": "madam_ambrose",    # logical speaker (preferred)
              "speed": 1.0,
              "response_format": "pcm_s16le"  # only pcm_s16le today
            }

        Response headers::

            Content-Type: application/octet-stream
            X-Octomil-Sample-Rate: 24000
            X-Octomil-Channels: 1
            X-Octomil-Sample-Format: pcm_s16le
            X-Octomil-Streaming-Capability-Mode: sentence_chunk
            X-Octomil-Model: kokoro-82m
            X-Octomil-Voice: af_bella
            X-Octomil-Speaker: madam_ambrose   # only when resolved
            X-Octomil-Speaker-Source: planner_profile

        Reference-audio file paths are NEVER exposed in headers, even
        when the request resolved to a few-shot voice-cloning profile.
        """
        if state.sherpa_tts_backend is None:
            raise OctomilError(
                code=OctomilErrorCode.MODEL_LOAD_FAILED,
                message="No TTS model loaded. Start server with a sherpa-onnx TTS model: octomil serve kokoro-82m",
            )

        text = (body.get("input") or "").strip()
        if not text:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="`input` must be a non-empty string.",
            )

        voice = body.get("voice")
        speaker = body.get("speaker")
        speed_raw = body.get("speed", 1.0)
        try:
            speed = float(speed_raw)
        except (TypeError, ValueError) as exc:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"`speed` must be a number, got {speed_raw!r}",
            ) from exc

        from octomil.audio.streaming import SAMPLE_FORMAT_PCM_S16LE, SUPPORTED_STREAM_FORMATS

        response_format = (body.get("response_format") or SAMPLE_FORMAT_PCM_S16LE).lower()
        if response_format not in SUPPORTED_STREAM_FORMATS:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"unsupported_stream_format: '{response_format}'. "
                    f"Use one of: {', '.join(SUPPORTED_STREAM_FORMATS)}."
                ),
            )
        # The serve layer is intentionally narrower than the SDK
        # streaming API: a single non-streaming engine wraps the
        # process today, so wav-streaming would just be the local
        # final-chunk shape. Keep the wire simple and refuse anything
        # but pcm_s16le here; the SDK still exposes both formats.
        if response_format != SAMPLE_FORMAT_PCM_S16LE:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"server_streaming_format: only '{SAMPLE_FORMAT_PCM_S16LE}' is supported on "
                    f"the streaming endpoint; use /v1/audio/speech for full WAV."
                ),
            )

        backend = state.sherpa_tts_backend
        # The contract is: a streaming backend implements
        # synthesize_stream. supports_streaming as a bool flag was
        # removed in the cutover — capability is advertised via
        # streaming_capability(text) instead.
        if not callable(getattr(backend, "synthesize_stream", None)):
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="local_tts_streaming_unavailable: backend does not implement synthesize_stream.",
            )

        # Pre-validate the speaker / voice synchronously: if we let an
        # unsupported request surface mid-stream, the binary client has
        # already received 200 application/octet-stream and lost the
        # ability to render a structured 4xx. The serve layer runs the
        # same speaker resolver the SDK kernel uses against an empty
        # planner selection (the in-process server doesn't carry app
        # context yet) — that's enough to honor speaker= as an alias
        # for native voices on the loaded engine. OctomilError raised
        # here bubbles up through the route's exception handler as JSON.
        from octomil.execution.tts_speaker_resolver import resolve_tts_speaker

        resolved_speaker = resolve_tts_speaker(
            speaker=speaker,
            voice=voice,
            selection=None,
            is_app_ref=False,
        )
        validate_voice = getattr(backend, "validate_voice", None)
        if callable(validate_voice):
            _sid_unused, resolved_voice = validate_voice(resolved_speaker.native_voice)
        else:
            resolved_voice = resolved_speaker.native_voice or getattr(backend, "_default_voice", "") or ""

        # Honest streaming-mode header: advertise what the backend
        # actually claims for THIS input, not a static "realtime" lie.
        capability_fn = getattr(backend, "streaming_capability", None)
        if callable(capability_fn):
            advertised = capability_fn(text)
            advertised_mode = advertised.mode.value
        else:
            advertised_mode = "final_chunk"

        state.request_count += 1
        sample_rate = int(getattr(backend, "_sample_rate", 24000) or 24000)
        model_name = getattr(backend, "_model_name", "") or ""

        async def chunk_iter() -> Any:
            extra = {"speaker_profile": resolved_speaker} if getattr(backend, "accepts_speaker_profile", False) else {}
            inner = backend.synthesize_stream(text, resolved_speaker.native_voice, speed, **extra)
            try:
                async for raw in inner:
                    pcm: bytes = raw["pcm_s16le"]
                    if pcm:
                        yield pcm
            finally:
                close = getattr(inner, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        pass

        # Speaker / source headers are only set when there's something
        # meaningful to publish — avoids a meaningless
        # ``X-Octomil-Speaker:`` empty-value header for callers using
        # ``voice=`` only. Reference-audio paths are NEVER published.
        headers = {
            "X-Octomil-Sample-Rate": str(sample_rate),
            "X-Octomil-Channels": "1",
            "X-Octomil-Sample-Format": SAMPLE_FORMAT_PCM_S16LE,
            # Replaces the legacy X-Octomil-Streaming-Mode header.
            # Values: final_chunk | sentence_chunk | progressive.
            "X-Octomil-Streaming-Capability-Mode": advertised_mode,
            "X-Octomil-Model": str(model_name),
            "X-Octomil-Voice": str(resolved_voice),
            "X-Octomil-Speaker-Source": resolved_speaker.source,
        }
        if resolved_speaker.speaker:
            headers["X-Octomil-Speaker"] = resolved_speaker.speaker

        return StreamingResponse(
            chunk_iter(),
            media_type="application/octet-stream",
            headers=headers,
        )

    @app.post("/v1/audio/speech")
    async def synthesize_speech(body: dict[str, Any]) -> Any:
        """OpenAI ``audio.speech.create``-compatible synthesis endpoint.

        Body shape (subset of OpenAI + ``speaker``):
            {
              "model": "kokoro-82m",
              "input": "text to synthesize",
              "voice": "af_bella",            # native voice (back-compat)
              "speaker": "madam_ambrose",    # logical speaker (preferred)
              "response_format": "wav",       # only wav today
              "speed": 1.0
            }

        Returns raw audio bytes with the correct ``Content-Type``.
        Resolved speaker / source are surfaced in ``X-Octomil-Speaker``
        / ``X-Octomil-Speaker-Source`` when applicable; reference-audio
        file paths are NEVER published.
        """
        from fastapi.responses import Response

        if state.sherpa_tts_backend is None:
            raise OctomilError(
                code=OctomilErrorCode.MODEL_LOAD_FAILED,
                message="No TTS model loaded. Start server with a sherpa-onnx TTS model: octomil serve kokoro-82m",
            )

        text = (body.get("input") or "").strip()
        if not text:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="`input` must be a non-empty string.",
            )

        voice = body.get("voice")
        speaker = body.get("speaker")
        speed_raw = body.get("speed", 1.0)
        try:
            speed = float(speed_raw)
        except (TypeError, ValueError) as exc:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"`speed` must be a number, got {speed_raw!r}",
            ) from exc

        response_format = (body.get("response_format") or "wav").lower()
        if response_format != "wav":
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"Unsupported response_format '{response_format}'. Only 'wav' is supported by the local sherpa-onnx engine.",
            )

        # Same resolver the streaming route uses — speaker= takes
        # precedence; voice= remains for back-compat.
        from octomil.execution.tts_speaker_resolver import resolve_tts_speaker

        resolved_speaker = resolve_tts_speaker(
            speaker=speaker,
            voice=voice,
            selection=None,
            is_app_ref=False,
        )
        backend = state.sherpa_tts_backend
        extra = {"speaker_profile": resolved_speaker} if getattr(backend, "accepts_speaker_profile", False) else {}

        state.request_count += 1
        result: dict[str, Any] = backend.synthesize(
            text,
            voice=resolved_speaker.native_voice,
            speed=speed,
            **extra,
        )
        headers = {
            "X-Octomil-Sample-Rate": str(result["sample_rate"]),
            "X-Octomil-Duration-Ms": str(result["duration_ms"]),
            "X-Octomil-Voice": str(result.get("voice") or ""),
            "X-Octomil-Model": str(result.get("model") or ""),
            "X-Octomil-Speaker-Source": resolved_speaker.source,
        }
        if resolved_speaker.speaker:
            headers["X-Octomil-Speaker"] = resolved_speaker.speaker
        return Response(
            content=result["audio_bytes"],
            media_type=result["content_type"],
            headers=headers,
        )

    # Anthropic Messages API translation layer
    from ..serve_anthropic import register_anthropic_routes

    register_anthropic_routes(app, state)

    return app


def run_server(
    model_name: str,
    *,
    port: int = 8080,
    host: str = "0.0.0.0",
    api_key: Optional[str] = None,
    api_base: str = "https://api.octomil.com/api/v1",
    json_mode: bool = False,
    cache_size_mb: int = 2048,
    cache_enabled: bool = True,
    engine: Optional[str] = None,
    max_queue_depth: int = 32,
    moe_config: Optional[MoEConfig] = None,
    compress_context: bool = False,
    compression_strategy: str = "token_pruning",
    compression_ratio: float = 0.5,
    compression_max_turns: int = 4,
    compression_threshold: int = 256,
    tool_use: bool = False,
    early_exit_config: Optional["EarlyExitConfig"] = None,
    verbose: bool = False,
    cloud_config: Optional[CloudConfig] = None,
    config_set: Any = None,
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
    moe_config:
        Configuration for Mixture of Experts model features.
        When ``None``, uses defaults (auto-detection enabled).
    compress_context:
        Enable prompt compression before inference.
    compression_strategy:
        Compression strategy: ``"token_pruning"`` or ``"sliding_window"``.
    compression_ratio:
        Target compression ratio (0.0--1.0) for token pruning.
    compression_max_turns:
        Number of recent turns to keep verbatim (sliding_window).
    compression_threshold:
        Minimum token count before compression activates.
    tool_use:
        Pre-load coding agent tool schemas for structured output.
    early_exit_config:
        Configuration for early exit / adaptive computation depth.
    verbose:
        When ``True``, emit verbose runtime events for debugging.
    cloud_config:
        Cloud provider configuration. When set, serves via cloud backend.
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
        moe_config=moe_config,
        compress_context=compress_context,
        compression_strategy=compression_strategy,
        compression_ratio=compression_ratio,
        compression_max_turns=compression_max_turns,
        compression_threshold=compression_threshold,
        tool_use=tool_use,
        early_exit_config=early_exit_config,
        verbose=verbose,
        cloud_config=cloud_config,
        config_set=config_set,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
