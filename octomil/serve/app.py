"""FastAPI app factory and server runner for single-model serving."""

from __future__ import annotations

import asyncio
import base64
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


def _embedding_values_to_json(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


async def _run_native_audio_route(call: Any, *, context: str) -> Any:
    """Run a sync native audio call and bound any raw native status leak."""
    try:
        return await asyncio.to_thread(call)
    except OctomilError:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "NativeRuntimeError" and hasattr(exc, "status"):
            from ..runtime.native.error_mapping import map_oct_status

            raise map_oct_status(
                int(getattr(exc, "status")),
                str(getattr(exc, "last_error", "") or ""),
                message=context,
                default_unsupported_code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
            ) from exc
        raise


def _parse_positive_int_field(body: dict[str, Any], field: str, default: int) -> int:
    raw = body.get(field, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=f"`{field}` must be a positive integer, got {raw!r}",
        ) from exc
    if value <= 0:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=f"`{field}` must be a positive integer, got {value!r}",
        )
    return value


def _decode_pcm_f32_body(body: dict[str, Any]) -> Any:
    """Return mono PCM-f32 request audio as list-like samples or raw bytes.

    JSON callers can pass either ``audio`` / ``audio_pcm_f32`` as a list
    of floats, or ``audio_pcm_f32_base64`` / ``audio_base64`` as base64
    encoded float32-LE bytes. Native backends own final validation.
    """
    for field in ("audio", "audio_pcm_f32"):
        if field in body:
            audio = body[field]
            if isinstance(audio, list):
                return audio
            if isinstance(audio, str):
                try:
                    return base64.b64decode(audio, validate=True)
                except Exception as exc:  # noqa: BLE001
                    raise OctomilError(
                        code=OctomilErrorCode.INVALID_INPUT,
                        message=f"`{field}` must be base64-encoded PCM-f32 bytes when provided as a string.",
                    ) from exc
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=f"`{field}` must be a float list or base64 PCM-f32 string.",
            )

    for field in ("audio_pcm_f32_base64", "audio_base64"):
        if field in body:
            raw = body[field]
            if not isinstance(raw, str):
                raise OctomilError(
                    code=OctomilErrorCode.INVALID_INPUT,
                    message=f"`{field}` must be a base64 PCM-f32 string.",
                )
            try:
                return base64.b64decode(raw, validate=True)
            except Exception as exc:  # noqa: BLE001
                raise OctomilError(
                    code=OctomilErrorCode.INVALID_INPUT,
                    message=f"`{field}` must be base64-encoded PCM-f32 bytes.",
                ) from exc

    raise OctomilError(
        code=OctomilErrorCode.INVALID_INPUT,
        message="Missing audio. Provide `audio` as a float list or `audio_pcm_f32_base64` as float32-LE bytes.",
    )


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

        # Check if this is a whisper (speech-to-text) model. v0.1.5
        # PR-2B cutover: the product path uses the NATIVE
        # `audio.transcription` backend (octomil-runtime + whisper.cpp
        # via cffi). The legacy pywhispercpp engine is no longer
        # registered for production use and lives under
        # `_legacy_pywhisper.py` for benchmarking only.
        from ..runtime.engines.whisper import is_whisper_model

        if is_whisper_model(model_name):
            from .stt_serve_adapter import NativeSttServeAdapter

            whisper_backend = NativeSttServeAdapter()
            try:
                whisper_backend.load_model(model_name)
            except Exception as exc:
                _log_startup_error(model_name, exc)
                raise
            state.whisper_backend = whisper_backend
            state.engine_name = "native-whisper-cpp"
        elif _is_sherpa_tts_model(model_name):
            # Native cutover: batch and stream TTS both route through
            # octomil-runtime. The runtime advertises these capabilities
            # only when OCTOMIL_SHERPA_TTS_MODEL points at the canonical
            # piper-amy ONNX with required sidecars and digest gates.
            from ..runtime.native.tts_batch_backend import NativeTtsBatchBackend

            try:
                native_batch_backend = NativeTtsBatchBackend()
                native_batch_backend.load_model(model_name)
            except Exception as exc:
                logger.info(
                    "native audio.tts.batch backend unavailable at startup (%s); "
                    "/v1/audio/speech will return RUNTIME_UNAVAILABLE on request",
                    exc,
                )
                state.native_tts_batch_backend = None
            else:
                state.native_tts_batch_backend = native_batch_backend
                state.engine_name = native_batch_backend.name

            # Same runtime/artifact gate, separate capability/session.
            try:
                from ..runtime.native.tts_stream_backend import NativeTtsStreamBackend

                native_stream_backend = NativeTtsStreamBackend()
                native_stream_backend.load_model(model_name)
                state.native_tts_stream_backend = native_stream_backend
                if not state.engine_name:
                    state.engine_name = native_stream_backend.name
            except Exception as exc:  # noqa: BLE001
                # The native runtime might not advertise tts.stream
                # (sidecars missing, dylib unavailable, etc.). That
                # is non-fatal at startup. The stream route handler
                # raises RUNTIME_UNAVAILABLE on first hit.
                logger.info(
                    "native audio.tts.stream backend unavailable at startup (%s); "
                    "/v1/audio/speech/stream will return RUNTIME_UNAVAILABLE on request",
                    exc,
                )
                state.native_tts_stream_backend = None
            if state.native_tts_batch_backend is None and state.native_tts_stream_backend is None:
                state.engine_name = "native-sherpa-tts-unavailable"
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
        for attr in (
            "native_vad_backend",
            "native_speaker_embedding_backend",
            "native_diarization_backend",
            "native_tts_batch_backend",
            "native_tts_stream_backend",
        ):
            backend = getattr(state, attr, None)
            if backend is None:
                continue
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.warning("%s.close failed during shutdown", attr, exc_info=True)
            setattr(state, attr, None)

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
        json_mode_config = None
        if is_json:
            from .json_mode import resolve_json_mode_config

            rf_for_config = body.response_format or {"type": "json_object"}
            json_mode_config = resolve_json_mode_config(rf_for_config)

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
            schema_for_prompt = json_mode_config.schema if json_mode_config is not None else None
            messages = _inject_json_system_prompt(messages, schema_for_prompt)

        # Extract enable_thinking from the request body (Qwen3 / OpenClaw convention)
        _enable_thinking: Optional[bool] = getattr(body, "enable_thinking", None)
        _max_tokens = body.max_completion_tokens or body.max_tokens or 512

        gen_req = GenerationRequest(
            model=body.model or state.model_name,
            messages=messages,
            max_tokens=_max_tokens,
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
        json_validation_status: Optional[str] = None
        if is_json:
            assert json_mode_config is not None
            from .json_mode import coerce_json_mode_output

            try:
                validated = coerce_json_mode_output(text, json_mode_config)
                text = validated.text
                json_validation_status = validated.status
            except OctomilError:
                if uses_grammar_natively:
                    raise
                # Retry once with a stronger system prompt
                retry_messages = _inject_json_system_prompt(messages, schema_for_prompt, force=True)
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
                validated = coerce_json_mode_output(text, json_mode_config)
                text = validated.text
                json_validation_status = validated.status

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
        if json_validation_status is not None:
            usage["json_validation"] = {
                "mode": "json_object",
                "status": json_validation_status,
            }

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

    def _native_vad_backend() -> Any:
        if state.native_vad_backend is None:
            from ..audio import vad as vad_module

            backend = vad_module.NativeVadBackend()
            backend.open()
            state.native_vad_backend = backend
        return state.native_vad_backend

    def _native_speaker_embedding_backend(model: str) -> Any:
        if state.native_speaker_embedding_backend is None:
            from ..audio import speaker_embedding as speaker_module

            backend = speaker_module.NativeSpeakerEmbeddingBackend()
            backend.load_model(model)
            state.native_speaker_embedding_backend = backend
        else:
            # NativeSpeakerEmbeddingBackend validates unsupported model
            # names before its idempotent loaded fast path, so this keeps
            # per-request model rejects bounded without reloading.
            state.native_speaker_embedding_backend.load_model(model)
        return state.native_speaker_embedding_backend

    def _native_diarization_backend() -> Any:
        if state.native_diarization_backend is None:
            from ..audio import diarization as diarization_module

            backend = diarization_module.NativeDiarizationBackend()
            backend.open()
            state.native_diarization_backend = backend
        return state.native_diarization_backend

    @app.post("/v1/audio/vad")
    async def detect_voice_activity(body: dict[str, Any]) -> dict[str, Any]:
        """Native-only voice activity detection endpoint."""
        audio = _decode_pcm_f32_body(body)
        sample_rate_hz = _parse_positive_int_field(body, "sample_rate_hz", 16000)
        deadline_ms = _parse_positive_int_field(body, "deadline_ms", 300_000)

        def _run() -> list[Any]:
            backend = _native_vad_backend()
            with backend.open_session(sample_rate_hz=sample_rate_hz) as session:
                session.feed_chunk(audio, sample_rate_hz=sample_rate_hz)
                return list(
                    session.poll_transitions(
                        deadline_ms=deadline_ms,
                        drain_until_completed=True,
                    )
                )

        transitions = await _run_native_audio_route(_run, context="native audio.vad route failed")
        state.request_count += 1
        return {
            "object": "audio.vad",
            "model": "silero-vad",
            "sample_rate_hz": sample_rate_hz,
            "transitions": [
                {
                    "kind": transition.kind,
                    "timestamp_ms": int(transition.timestamp_ms),
                    "confidence": float(transition.confidence),
                }
                for transition in transitions
            ],
        }

    @app.post("/v1/audio/speaker_embeddings")
    async def create_speaker_embedding(body: dict[str, Any]) -> dict[str, Any]:
        """Native-only speaker embedding endpoint."""
        audio = _decode_pcm_f32_body(body)
        sample_rate_hz = _parse_positive_int_field(body, "sample_rate_hz", 16000)
        deadline_ms = _parse_positive_int_field(body, "deadline_ms", 300_000)
        model = str(body.get("model") or "sherpa-eres2netv2-base")

        def _run() -> list[float]:
            backend = _native_speaker_embedding_backend(model)
            return _embedding_values_to_json(
                backend.embed(
                    audio,
                    sample_rate_hz=sample_rate_hz,
                    deadline_ms=deadline_ms,
                )
            )

        embedding = await _run_native_audio_route(
            _run,
            context="native audio.speaker.embedding route failed",
        )
        state.request_count += 1
        return {
            "object": "audio.speaker.embedding",
            "model": model,
            "sample_rate_hz": sample_rate_hz,
            "embedding": embedding,
            "dimensions": len(embedding),
        }

    @app.post("/v1/audio/diarizations")
    async def diarize_audio(body: dict[str, Any]) -> dict[str, Any]:
        """Native-only speaker diarization endpoint."""
        audio = _decode_pcm_f32_body(body)
        sample_rate_hz = _parse_positive_int_field(body, "sample_rate_hz", 16000)
        deadline_ms = _parse_positive_int_field(body, "deadline_ms", 300_000)

        def _run() -> list[Any]:
            backend = _native_diarization_backend()
            return list(
                backend.diarize(
                    audio,
                    sample_rate_hz=sample_rate_hz,
                    deadline_ms=deadline_ms,
                )
            )

        segments = await _run_native_audio_route(
            _run,
            context="native audio.diarization route failed",
        )
        state.request_count += 1
        return {
            "object": "audio.diarization",
            "sample_rate_hz": sample_rate_hz,
            "segments": [
                {
                    "start_ms": int(segment.start_ms),
                    "end_ms": int(segment.end_ms),
                    "speaker_id": int(segment.speaker_id),
                    "speaker_label": str(segment.speaker_label or ""),
                }
                for segment in segments
            ],
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
        """Streaming TTS endpoint — live-conditional native cutover.

        HARD-CUT to ``NativeTtsStreamBackend`` (octomil-runtime
        ``audio.tts.stream`` capability). No silent fallback to the
        Python sherpa stream path.

        v0.1.9 progressive flip: worker-thread Generate delivers chunks
        during synthesis. ``X-Octomil-Streaming-Honesty`` carries
        ``progressive_during_synthesis``. Proof artifact (first_audio_ratio
        =0.5909, gate < 0.75, gate_pass=true):
        /tmp/v019-progressive-proof-20260508T185426Z.json

        Returns ``application/octet-stream`` with raw PCM int16 LE
        chunks. Metadata is in response headers; clients should NOT
        depend on HTTP trailers (many proxies drop them). Completion
        is signalled by clean EOF.

        Body shape::

            {
              "model": "kokoro-82m",
              "input": "text to synthesize",
              "voice": "0",                    # numeric sid (sherpa ABI)
              "speed": 1.0,
              "response_format": "pcm_s16le"  # only pcm_s16le today
            }

        Voice validation runs synchronously *before* HTTP 200 — an
        unsupported voice raises OctomilError(INVALID_INPUT) which
        bubbles up through the route's exception handler as a
        structured 4xx, rather than after the consumer has already
        attached to the streaming response.

        Reference-audio file paths are NEVER exposed in headers.
        """
        if state.native_tts_stream_backend is None:
            # Hard-cut: the python-sherpa stream path is no longer
            # reachable on this route. If the native backend failed
            # to load at startup (no dylib advertising
            # audio.tts.stream), surface RUNTIME_UNAVAILABLE rather
            # than fall back to the Python sherpa engine.
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native audio.tts.stream backend not loaded. "
                    "Set OCTOMIL_RUNTIME_DYLIB + OCTOMIL_SHERPA_TTS_MODEL "
                    "(canonical-pinned ONNX with sibling tokens.txt + "
                    "espeak-ng-data/) and restart the server."
                ),
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
        # v0.1.8 Lane C: the runtime adapter ignores ``speed`` (sherpa
        # adapter wires it as 1.0 internally). Surface that honestly:
        # accept the parameter for API back-compat but ignore it
        # rather than silently mis-rendering. Speeds other than 1.0
        # are documented to be a no-op until the runtime adapter
        # threads the parameter through.
        del speed  # explicit no-op marker; see honesty caveat above.

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
        if response_format != SAMPLE_FORMAT_PCM_S16LE:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"server_streaming_format: only '{SAMPLE_FORMAT_PCM_S16LE}' is supported on "
                    f"the streaming endpoint; use /v1/audio/speech for full WAV."
                ),
            )

        native_backend = state.native_tts_stream_backend

        # Pre-stream voice validation BEFORE FastAPI emits HTTP 200.
        # If voice is invalid, the OctomilError raised here is caught
        # by the route's exception handler and rendered as a typed 4xx
        # — clients have NOT seen the streaming response yet, so they
        # can render the structured error properly. This mirrors the
        # batch route's discipline (and the SDK kernel's pre-stream
        # validation hook).
        from octomil.execution.tts_speaker_resolver import resolve_tts_speaker

        resolved_speaker = resolve_tts_speaker(
            speaker=speaker,
            voice=voice,
            selection=None,
            is_app_ref=False,
        )
        # Native ABI speaker_id is a numeric string. Reject anything
        # else BEFORE opening a session; the runtime would reject too,
        # but doing it Python-side keeps the HTTP-200 boundary clean.
        resolved_voice_str = native_backend.validate_voice(resolved_speaker.native_voice)

        state.request_count += 1
        # Sherpa-onnx VITS models are 22050 Hz mono; the native
        # backend reads sample_rate off each chunk and the runtime
        # is consistent across chunks. We surface the first observed
        # rate via the per-chunk fields. Falls back to a documented
        # 22050 default for the header until the first chunk arrives.
        # (Headers must be set before StreamingResponse begins.)
        # Codex R1 P1 fix: read the actual SR from the loaded sherpa
        # engine (which learned it from the VITS model at startup)
        # rather than hardcoding 22050. Kokoro is 24 kHz, piper-amy
        # is 22050 Hz — different models advertise different rates.
        # The native backend's per-chunk sample_rate matches; this
        # header serves clients that read headers BEFORE the first
        # chunk arrives. Falls back to 24000 (Kokoro default) if
        # sherpa hasn't surfaced an SR yet.
        observed_sample_rate = 22050
        model_name = getattr(state.native_tts_batch_backend, "_model_name", "") or state.model_name or ""

        # Codex R1 P1 fix: open the session SYNCHRONOUSLY here so
        # that an out-of-range numeric sid (e.g. voice="999" against
        # a single-speaker piper-amy model) raises BEFORE FastAPI
        # commits HTTP 200. ``synthesize_with_chunks`` does the
        # session_open + send_text synchronously and returns a lazy
        # iterator over the chunk drain — so OctomilError(INVALID_INPUT)
        # for sid range / send_text / NaN-text lands here as a 4xx
        # body, not as a mid-stream failure after status/headers
        # have been committed. The drain itself runs on a worker
        # thread to keep the event loop responsive while progressive
        # synthesis drains native events.
        chunk_iterator = native_backend.synthesize_with_chunks(
            text,
            voice_id=resolved_voice_str,
        )

        # Convert PCM-f32 native chunks to pcm_s16le for the wire.
        def _f32_to_pcm_s16le(pcm_f32: Any) -> bytes:
            import numpy as _np

            arr = _np.asarray(pcm_f32, dtype=_np.float32)
            clipped = _np.clip(arr, -1.0, 1.0)
            i16 = (clipped * 32767.0).astype(_np.int16)
            return bytes(i16.tobytes())

        async def chunk_iter() -> Any:
            import asyncio as _asyncio
            import queue as _queue

            q: _queue.Queue[Any] = _queue.Queue(maxsize=64)
            _SENTINEL = object()
            _ERROR = object()

            def _producer() -> None:
                try:
                    for chunk in chunk_iterator:
                        q.put((chunk.pcm_f32, chunk.is_final))
                    q.put(_SENTINEL)
                except Exception as exc:  # noqa: BLE001
                    q.put((_ERROR, exc))

            loop = _asyncio.get_running_loop()
            fut = loop.run_in_executor(None, _producer)
            try:
                while True:
                    item = await loop.run_in_executor(None, q.get)
                    if item is _SENTINEL:
                        break
                    if isinstance(item, tuple) and item[0] is _ERROR:
                        raise item[1]
                    pcm_f32, _is_final = item
                    pcm_s16 = _f32_to_pcm_s16le(pcm_f32)
                    if pcm_s16:
                        yield pcm_s16
            finally:
                try:
                    await fut
                except Exception:  # noqa: BLE001
                    pass

        # v0.1.9 progressive flip: advertise sentence_chunk in the
        # capability header (matches contract enum vocabulary), AND surface
        # progressive_during_synthesis in the honesty header — proven by
        # first_audio_ratio=0.5909 < 0.75 gate (proof_artifact). Clients
        # can use either header.
        headers = {
            "X-Octomil-Sample-Rate": str(observed_sample_rate),
            "X-Octomil-Channels": "1",
            "X-Octomil-Sample-Format": SAMPLE_FORMAT_PCM_S16LE,
            "X-Octomil-Streaming-Capability-Mode": "sentence_chunk",
            "X-Octomil-Streaming-Honesty": "progressive_during_synthesis",
            "X-Octomil-Backend": native_backend.name,
            "X-Octomil-Model": str(model_name),
            "X-Octomil-Voice": str(resolved_voice_str),
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

        if state.native_tts_batch_backend is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    "native audio.tts.batch backend not loaded. Set OCTOMIL_RUNTIME_DYLIB + "
                    "OCTOMIL_SHERPA_TTS_MODEL to the canonical piper-amy ONNX bundle and restart the server."
                ),
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
                message=f"Unsupported response_format '{response_format}'. Only 'wav' is supported by native audio.tts.batch.",
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
        backend = state.native_tts_batch_backend

        state.request_count += 1
        result: dict[str, Any] = backend.synthesize(
            text,
            voice=resolved_speaker.native_voice,
            speed=speed,
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
