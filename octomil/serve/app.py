"""FastAPI app factory and server runner for single-model serving."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

from ..errors import OctomilError
from .backends.llamacpp import LlamaCppBackend
from .config import CloudConfig, MoEConfig, ServerState
from .detection import _detect_backend, _get_cache_manager, _log_startup_error
from .grammar_helpers import _inject_json_system_prompt, _resolve_grammar
from .models import ChatCompletionBody
from .streaming import _queued_stream_response, _stream_response
from .types import GenerationRequest

if TYPE_CHECKING:
    from ..early_exit import EarlyExitConfig

logger = logging.getLogger(__name__)


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

    app = FastAPI(title="Octomil Serve", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(OctomilError)
    async def octomil_error_handler(request: Request, exc: OctomilError) -> JSONResponse:
        from .._generated.error_code import ERROR_CLASSIFICATION, RetryClass

        classification = ERROR_CLASSIFICATION.get(exc.code)
        status_map = {
            OctomilErrorCode.INVALID_INPUT: 400,
            OctomilErrorCode.AUTHENTICATION_FAILED: 401,
            OctomilErrorCode.INVALID_API_KEY: 401,
            OctomilErrorCode.FORBIDDEN: 403,
            OctomilErrorCode.MODEL_NOT_FOUND: 404,
            OctomilErrorCode.RATE_LIMITED: 429,
            OctomilErrorCode.MODEL_LOAD_FAILED: 503,
            OctomilErrorCode.RUNTIME_UNAVAILABLE: 503,
            OctomilErrorCode.INFERENCE_FAILED: 503,
            OctomilErrorCode.SERVER_ERROR: 500,
            OctomilErrorCode.REQUEST_TIMEOUT: 504,
        }
        status_code = status_map.get(exc.code, 500)
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
        models = state.backend.list_models() if state.backend else []
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
        if state.backend is None:
            raise OctomilError(code=OctomilErrorCode.MODEL_LOAD_FAILED, message="Model not loaded")

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
        uses_grammar_natively = isinstance(state.backend, LlamaCppBackend)
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
            json_mode=is_json,
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
            if _queue is not None:
                from ..batch import QueueFullError, QueueTimeoutError

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
        elif state.backend is not None:
            backend_name = state.backend.name

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
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
