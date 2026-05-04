"""Multi-model serving with query routing."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

from ..errors import OctomilError
from .config import MultiModelServerState
from .detection import _detect_backend, _log_startup_error
from .grammar_helpers import (
    _inject_json_system_prompt,
    _reject_explicit_grammar_on_non_grammar_backend,
    _resolve_grammar,
)
from .instrumentation import unwrap_backend
from .models import ChatCompletionBody
from .streaming import _stream_response
from .types import GenerationRequest, InferenceBackend, resolve_backend_capabilities

logger = logging.getLogger(__name__)


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
        self.backend: Optional[InferenceBackend] = backend
        self.reporter = state.reporter
        self.model_name = model_name


def create_multi_model_app(
    model_names: list[str],
    *,
    api_key: Optional[str] = None,
    api_base: str = "https://api.octomil.com/api/v1",
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

    from fastapi import FastAPI, Request  # noqa: F811
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    from ..decomposer import ResultMerger, SubTaskResult
    from ..errors import OctomilError, OctomilErrorCode  # noqa: F811
    from ..routing import (
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
        from ..models.catalog import get_model as _get_catalog_model

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

                # Detect reasoning models (strip quant suffix for catalog lookup)
                _catalog_lookup = name.split(":")[0] if ":" in name else name
                _entry = _get_catalog_model(_catalog_lookup)
                if _entry and "reasoning" in _entry.capabilities:
                    state.reasoning_models.add(name)
                    logger.info("Model %s detected as reasoning model", name)
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
                from ..telemetry import TelemetryReporter as _TR

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

    app = FastAPI(title="Octomil Serve (Multi-Model)", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(OctomilError)
    async def octomil_error_handler(request: Request, exc: OctomilError) -> JSONResponse:  # type: ignore[misc]
        from .._generated.error_code import ERROR_CLASSIFICATION, RetryClass

        # Cutover follow-up #70 (Codex R1 B2): SHARED status map
        # across single- and multi-model handlers. Pre-fix this
        # block had its own dict missing the v0.1.2 cutover codes
        # — `octomil serve --auto-route` would surface
        # UNSUPPORTED_MODALITY as 503 (INFERENCE_FAILED) instead of
        # 422. Calling the shared helper guarantees both handlers
        # stay aligned on the bounded SDK taxonomy.
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
                    "owned_by": "octomil",
                }
                for m in all_models
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionBody) -> Any:
        if not state.backends:
            raise OctomilError(code=OctomilErrorCode.MODEL_LOAD_FAILED, message="No models loaded")

        messages: list[dict[str, Any]] = [{"role": m.role, "content": m.content or ""} for m in body.messages]

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
            # Cutover follow-up #71: query the declared capability
            # rather than backend type identity. Pre-cutover this
            # was `isinstance(_, LlamaCppBackend)`; post-cutover the
            # type became `NativeChatBackend` (which does NOT
            # support grammar) and the isinstance check silently
            # evaluated False — semantically correct here, but the
            # type-marker shape is brittle as more backends ship.
            uses_grammar_natively = resolve_backend_capabilities(unwrap_backend(backend)).grammar_supported
            # Cutover follow-up #71 (R1 Codex): reject explicit caller GBNF
            # against non-grammar backends instead of silently stripping it.
            _reject_explicit_grammar_on_non_grammar_backend(
                backend_name=unwrap_backend(backend).name,
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
                req_messages = _inject_json_system_prompt(req_messages, schema_for_prompt)

            gen_req = GenerationRequest(
                model=model_name,
                messages=req_messages,
                max_tokens=body.max_tokens or 512,
                temperature=body.temperature,
                top_p=body.top_p,
                stream=body.stream,
                grammar=grammar_str if uses_grammar_natively else None,
                # Cutover follow-up #71 (R1 Codex): only forward json_mode=True
                # to grammar-capable backends; for non-grammar backends the
                # system-prompt fallback above handles JSON nudging and
                # forwarding the flag would cause NativeChatBackend to 422.
                json_mode=is_json and uses_grammar_natively,
            )

            state.request_count += 1
            state.routed_counts[model_name] = state.routed_counts.get(model_name, 0) + 1
            req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            session_id = uuid.uuid4().hex
            model_version = "latest"
            _reporter = state.reporter

            # Report inference_started
            if _reporter is not None:
                try:
                    _reporter.report_inference_started(
                        model_id=model_name,
                        version=model_version,
                        session_id=session_id,
                    )
                except Exception:
                    pass

            if gen_req.stream:
                headers = {
                    "X-Octomil-Routed-Model": model_name,
                    "X-Octomil-Complexity": str(decision.complexity_score),
                    "X-Octomil-Tier": decision.tier,
                }
                if used_fallback:
                    headers["X-Octomil-Fallback"] = "true"

                return StreamingResponse(
                    _stream_response(
                        _MultiModelStateAdapter(state, backend, model_name),
                        gen_req,
                        req_id,
                        session_id,
                        is_reasoning=model_name in state.reasoning_models,
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
                        _reporter.report_inference_failed(
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
                from ..grammar import extract_json, validate_json_output

                if not validate_json_output(text):
                    extracted = extract_json(text)
                    if extracted is not None:
                        text = json.dumps(extracted)
                    else:
                        retry_messages = _inject_json_system_prompt(messages, schema_for_prompt)
                        # Cutover follow-up #71 (R2 Codex): retry block only fires
                        # in the non-grammar fallback branch; system prompt is doing
                        # the JSON constraining. Forwarding json_mode=True would 422
                        # native with UNSUPPORTED_MODALITY, breaking the retry.
                        retry_req = GenerationRequest(
                            model=model_name,
                            messages=retry_messages,
                            max_tokens=gen_req.max_tokens,
                            temperature=max(gen_req.temperature - 0.2, 0.0),
                            top_p=gen_req.top_p,
                            stream=False,
                            json_mode=False,
                        )
                        text, metrics = backend.generate(retry_req)
                        if not validate_json_output(text):
                            extracted = extract_json(text)
                            if extracted is not None:
                                text = json.dumps(extracted)

            # Report inference_completed
            if _reporter is not None:
                try:
                    total_tokens = metrics.total_tokens
                    throughput = total_tokens / (gen_elapsed_ms / 1000) if gen_elapsed_ms > 0 else 0.0
                    _reporter.report_inference_completed(
                        session_id=session_id,
                        model_id=model_name,
                        version=model_version,
                        total_chunks=total_tokens,
                        total_duration_ms=gen_elapsed_ms,
                        ttfc_ms=metrics.ttfc_ms,
                        throughput=throughput,
                        attention_backend=metrics.attention_backend,
                    )
                except Exception:
                    pass

            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if model_name in state.reasoning_models:
                from .thinking import strip_thinking

                content, reasoning = strip_thinking(text)
                msg["content"] = content
                if reasoning:
                    msg["reasoning_content"] = reasoning

            response_data = {
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": msg,
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
            resp.headers["X-Octomil-Routed-Model"] = model_name
            resp.headers["X-Octomil-Complexity"] = str(decision.complexity_score)
            resp.headers["X-Octomil-Tier"] = decision.tier
            if used_fallback:
                resp.headers["X-Octomil-Fallback"] = "true"
            return resp

        # All models failed.
        # Cutover follow-up #70 (Codex R1 B2): if the last failure
        # was an OctomilError, propagate ITS typed code instead of
        # blanket INFERENCE_FAILED. A grammar/json_mode/streaming
        # request that every backend rejects with
        # UNSUPPORTED_MODALITY should still surface as 422, not 503.
        # The "all-models" wrapper preserves the bounded SDK
        # taxonomy from the underlying rejection.
        if isinstance(last_error, OctomilError):
            raise OctomilError(
                code=last_error.code,
                message=f"All models failed. Last error: {last_error}",
            )
        raise OctomilError(
            code=OctomilErrorCode.INFERENCE_FAILED,
            message=f"All models failed. Last error: {last_error}",
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
        messages: list[dict[str, Any]],
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
            # Cutover follow-up #71: query the declared capability
            # rather than backend type identity. Pre-cutover this
            # was `isinstance(_, LlamaCppBackend)`; post-cutover the
            # type became `NativeChatBackend` (which does NOT
            # support grammar) and the isinstance check silently
            # evaluated False — semantically correct here, but the
            # type-marker shape is brittle as more backends ship.
            uses_grammar_natively = resolve_backend_capabilities(unwrap_backend(backend)).grammar_supported
            # Cutover follow-up #71 (R1 Codex): reject explicit caller GBNF
            # against non-grammar backends instead of silently stripping it.
            _reject_explicit_grammar_on_non_grammar_backend(
                backend_name=unwrap_backend(backend).name,
                grammar_str=grammar_str,
                is_json=is_json,
                uses_grammar_natively=uses_grammar_natively,
            )

            req_messages = list(sub_messages)
            if is_json and not uses_grammar_natively:
                rf = body.response_format or {}
                schema_for_prompt = None
                if rf.get("type") == "json_schema":
                    raw = rf.get("json_schema") or rf.get("schema")
                    schema_for_prompt = raw.get("schema", raw) if raw else None
                req_messages = _inject_json_system_prompt(req_messages, schema_for_prompt)

            gen_req = GenerationRequest(
                model=model_name,
                messages=req_messages,
                max_tokens=body.max_tokens or 512,
                temperature=body.temperature,
                top_p=body.top_p,
                stream=False,
                grammar=grammar_str if uses_grammar_natively else None,
                # Cutover follow-up #71 (R1 Codex): only forward json_mode=True
                # to grammar-capable backends; for non-grammar backends the
                # system-prompt fallback above handles JSON nudging and
                # forwarding the flag would cause NativeChatBackend to 422.
                json_mode=is_json and uses_grammar_natively,
            )

            mm_state.request_count += 1
            mm_state.routed_counts[model_name] = mm_state.routed_counts.get(model_name, 0) + 1

            try:
                text, _metrics = backend.generate(gen_req)
            except OctomilError:
                # Cutover follow-up #70 (Codex R2 B1): bounded SDK
                # errors (UNSUPPORTED_MODALITY for grammar / json_mode
                # / streaming etc.) MUST propagate, not be swallowed
                # into a 200-OK sub-task response. Pre-fix the
                # decomposed `--auto-route` path on multi-model
                # would return a successful JSONResponse with the
                # text "[Error processing sub-task N]" — completely
                # bypassing the 4xx mapping the cutover wired in.
                # Re-raise so the FastAPI exception handler maps
                # via the shared status_map.
                raise
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
        resp.headers["X-Octomil-Decomposed"] = "true"
        resp.headers["X-Octomil-Subtasks"] = str(len(decomp.tasks))
        resp.headers["X-Octomil-Routed-Model"] = ordered_results[0].model_used
        resp.headers["X-Octomil-Tier"] = ordered_results[0].tier
        return resp

    return app


def run_multi_model_server(
    model_names: list[str],
    *,
    port: int = 8080,
    host: str = "0.0.0.0",
    api_key: Optional[str] = None,
    api_base: str = "https://api.octomil.com/api/v1",
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
