"""OctomilResponses — developer-facing Response API (Layer 2).

**Tier: Core Contract (MUST)**
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Optional, Union

from octomil._generated.message_role import MessageRole
from octomil.execution.planner_resolution import (
    _is_synthetic_cloud_fallback as _is_synthetic_cloud_fallback_impl,
)
from octomil.execution.planner_resolution import (
    _runtime_model_for_selection,
)
from octomil.execution.route_metadata_mapper import (
    RouteMetadata,
    _route_metadata_from_selection,
)
from octomil.model_ref import ModelRef, _ModelRefCapability, _ModelRefId
from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.cloud_runtime import CloudModelRuntime
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.registry import ModelRuntimeRegistry
from octomil.runtime.core.router import LOCALITY_CLOUD, LOCALITY_ON_DEVICE, RouterModelRuntime
from octomil.runtime.core.types import (
    GenerationConfig,
    RuntimeCapabilities,
    RuntimeContentPart,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeToolCall,
    RuntimeToolDef,
    RuntimeUsage,
)
from octomil.runtime.core.types import (
    RuntimeResponse as _RuntimeResponse,
)
from octomil.runtime.routing.attempt_runner import (
    AttemptLoopResult,
    AttemptStage,
    AttemptStatus,
    CandidateAttemptRunner,
    FallbackTrigger,
    RouteAttempt,
    RuntimeChecker,
)
from octomil.runtime.routing.evaluators import (
    EvaluatorRegistry,
    RegistryBackedEvaluator,
)
from octomil.runtime.routing.route_event import (
    CandidateAttemptSummary,
)
from octomil.runtime.routing.route_event import (
    build_route_event as _build_route_decision_event,
)
from octomil.runtime.routing.route_event import (
    emit_route_event as _emit_route_decision_event,
)

from .types import (
    AssistantInput,
    AudioContent,
    DoneEvent,
    FileContent,
    ImageContent,
    InputItem,
    JsonSchemaFormat,
    OutputItem,
    Response,
    ResponseFormat,
    ResponseRequest,
    ResponseStreamEvent,
    ResponseToolCall,
    ResponseUsage,
    SystemInput,
    TextContent,
    TextDeltaEvent,
    TextOutput,
    ToolCallDeltaEvent,
    ToolCallOutput,
    ToolResultInput,
    UserInput,
    system_input,
    text_input,
)

if TYPE_CHECKING:
    from octomil.manifest.catalog_service import ModelCatalogService
    from octomil.runtime.planner.schemas import RuntimeSelection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NoRuntimeAvailableError(RuntimeError):
    """Raised when neither the planner nor the runtime registry can route the
    inference request — and we want to tell the caller something they can act on.

    The legacy message ("No ModelRuntime registered for model: <id>") was a
    dead-end: it surfaced an internal registry concept and gave no guidance on
    how to recover. In practice this fires for two distinct, both-actionable
    reasons:

    1. **No local engine installed.** The user installed bare ``octomil`` but no
       ``[mlx]`` / ``[litert]`` / ``[onnxruntime]`` extra. The planner can't
       prepare a local candidate because there's no engine to run on, and the
       legacy resolver doesn't know how to route an ``@app/<slug>/<cap>`` alias
       to cloud either.
    2. **Cloud reachable but planner returned no candidates.** Auth missing or
       expired, app slug doesn't exist, or planner-side bug. The fallthrough
       can't recover because aliases can't be resolved without the planner.

    The message names both branches and the concrete fix for each. The error
    subclasses :class:`RuntimeError` so existing ``except RuntimeError`` catches
    keep working.
    """

    def __init__(self, *, model_id: str, planner_enabled: bool) -> None:
        self.model_id = model_id
        self.planner_enabled = planner_enabled
        self.is_alias = self._looks_like_alias(model_id)
        super().__init__(self._format_message())

    @staticmethod
    def _looks_like_alias(model_id: str) -> bool:
        # ``@app/<slug>/<capability>`` and ``@capability/<cap>`` are the
        # planner-resolved forms. The legacy registry resolver can't handle
        # either; the planner is the only path.
        return model_id.startswith("@app/") or model_id.startswith("@capability/")

    def _format_message(self) -> str:
        lines = [
            f"No runtime available for model {self.model_id!r}.",
            "",
        ]
        if self.is_alias:
            lines += [
                f"  ``{self.model_id}`` is a planner-resolved alias. Aliases need either",
                "  a local engine installed (the planner can prepare an on-device candidate)",
                "  or a server-side cloud route (the planner emits a cloud candidate).",
                "",
            ]
        if not self.planner_enabled:
            lines += [
                "  Planner is DISABLED on this client (planner_enabled=False or",
                "  OCTOMIL_DISABLE_PLANNER=1). Aliases can't resolve without it.",
                "",
            ]
        lines += [
            "Likely fixes:",
            "  • Install a local engine extra (names match pyproject.toml):",
            "      pip install 'octomil[mlx]'     # macOS Apple Silicon (mlx-lm)",
            "      pip install 'octomil[onnx]'    # cross-platform via ONNX Runtime",
            "      pip install 'octomil[llama]'   # cross-platform GGUF (llama-cpp-python)",
            "      pip install 'octomil[whisper]' # speech-to-text",
            "      pip install 'octomil[tts]'     # speech synthesis (sherpa-onnx)",
            "  • Verify cloud auth: OCTOMIL_SERVER_KEY (oct_app_live_… or oct_org_live_…) is set",
            "    and your org has at least one cloud provider connection configured at",
            "    /dashboard/settings/providers.",
        ]
        if self.is_alias:
            lines += [
                f"  • Verify the alias exists: open /dashboard/apps and confirm '{self._slug_or_id()}'",
                "    has at least one capability slot.",
            ]
        return "\n".join(lines)

    def _slug_or_id(self) -> str:
        if self.model_id.startswith("@app/"):
            # ``@app/<slug>/<capability>`` → return ``<slug>``.
            parts = self.model_id.split("/", 2)
            return parts[1] if len(parts) >= 2 else self.model_id
        return self.model_id


# ---------------------------------------------------------------------------
# Attempt runner resolution result
# ---------------------------------------------------------------------------


@dataclass
class _AttemptRunnerResult:
    """Result from _resolve_via_attempt_runner when a plan is available."""

    response: _RuntimeResponse
    locality: str  # "on_device" | "cloud"
    is_fallback: bool
    attempt_loop: AttemptLoopResult
    route: RouteMetadata
    selection: Any | None = None


def _policy_name_for_telemetry(
    selection: Any | None,
    routing_policy: Optional[RoutingPolicy],
) -> str | None:
    """Resolve the policy string recorded on route telemetry."""
    if selection is not None:
        app_resolution = getattr(selection, "app_resolution", None)
        if app_resolution is not None and getattr(app_resolution, "routing_policy", None):
            return app_resolution.routing_policy
        resolution = getattr(selection, "resolution", None)
        if resolution is not None and getattr(resolution, "routing_policy", None):
            return resolution.routing_policy
    if routing_policy is not None:
        return routing_policy.mode.value
    return None


def _attempt_summaries_for_telemetry(
    route: RouteMetadata | None,
    attempt_loop: AttemptLoopResult | None,
) -> list[CandidateAttemptSummary]:
    """Convert attempt-runner state into canonical route telemetry summaries."""
    if attempt_loop is not None and attempt_loop.attempts:
        summaries: list[CandidateAttemptSummary] = []
        for attempt in attempt_loop.attempts:
            summaries.append(
                CandidateAttemptSummary(
                    locality=attempt.locality,
                    engine=attempt.engine or "",
                    success=attempt.status == AttemptStatus.SELECTED,
                    failure_reason="" if attempt.status == AttemptStatus.SELECTED else attempt.reason_code,
                )
            )
        return summaries

    execution = route.execution if route is not None else None
    if execution is None:
        return []
    return [
        CandidateAttemptSummary(
            locality=execution.locality,
            engine=execution.engine or "",
            success=True,
        )
    ]


def _emit_route_decision_if_needed(
    telemetry: Optional[object],
    *,
    response: Response,
    model_id: str,
    route: RouteMetadata | None,
    selection: Any | None = None,
    attempt_loop: AttemptLoopResult | None = None,
    routing_policy: Optional[RoutingPolicy] = None,
) -> None:
    """Emit canonical route telemetry for a completed response, best-effort."""
    if telemetry is None or route is None or route.execution is None:
        return

    app_resolution = getattr(selection, "app_resolution", None) if selection is not None else None
    resolution = getattr(selection, "resolution", None) if selection is not None else None
    raw_fallback_trigger: Any = attempt_loop.fallback_trigger if attempt_loop is not None else None
    if raw_fallback_trigger is None and route.fallback.trigger:
        raw_fallback_trigger = route.fallback.trigger

    if isinstance(raw_fallback_trigger, dict):
        fallback_trigger_code = raw_fallback_trigger.get("code")
        fallback_trigger_stage = raw_fallback_trigger.get("stage")
    else:
        fallback_trigger_code = getattr(raw_fallback_trigger, "code", None)
        fallback_trigger_stage = getattr(raw_fallback_trigger, "stage", None)

    try:
        event = _build_route_decision_event(
            request_id=response.id,
            app_id=getattr(app_resolution, "app_id", None),
            app_slug=getattr(app_resolution, "app_slug", None),
            deployment_id=getattr(resolution, "deployment_id", None),
            experiment_id=getattr(resolution, "experiment_id", None),
            variant_id=(
                getattr(app_resolution, "selected_model_variant_id", None) or getattr(resolution, "variant_id", None)
            ),
            capability=route.model.requested.capability,
            policy=_policy_name_for_telemetry(selection, routing_policy),
            planner_source=route.planner.source,
            model_ref=model_id,
            model_ref_kind=route.model.requested.kind,
            selected_locality=route.execution.locality,
            final_mode=route.execution.mode,
            fallback_used=route.fallback.used,
            fallback_trigger_code=fallback_trigger_code,
            fallback_trigger_stage=fallback_trigger_stage,
            candidate_attempts=_attempt_summaries_for_telemetry(route, attempt_loop),
            engine=route.execution.engine,
            artifact_id=route.artifact.id if route.artifact is not None else None,
            cache_status=route.artifact.cache.status if route.artifact is not None else None,
            total_tokens=response.usage.total_tokens if response.usage is not None else None,
        )
        _emit_route_decision_event(event, telemetry)
    except Exception:
        logger.debug("Failed to emit route decision telemetry", exc_info=True)


# ---------------------------------------------------------------------------
# Planner plan resolution (non-fatal)
# ---------------------------------------------------------------------------


def _try_resolve_planner_selection(model_id: str) -> Optional[RuntimeSelection]:
    """Attempt to resolve a planner selection for the given model.

    Returns None if the planner is unavailable or fails.
    Never raises.
    """
    import os

    if os.environ.get("OCTOMIL_RUNTIME_PLANNER_CACHE") == "0":
        return None
    try:
        from octomil.runtime.planner.planner import RuntimePlanner

        planner = RuntimePlanner()
        return planner.resolve(
            model=model_id,
            capability="responses",
            routing_policy="local_first",
        )
    except Exception:
        logger.debug("Planner selection failed in OctomilResponses", exc_info=True)
        return None


def _selection_to_candidate_dicts(selection: Any) -> list[dict[str, Any]]:
    """Convert a RuntimeSelection into candidate dicts for the attempt runner."""
    from dataclasses import asdict as _asdict

    candidates = getattr(selection, "candidates", None)
    if candidates:
        result: list[dict[str, Any]] = []
        for c in candidates:
            d: dict[str, Any] = {
                "locality": c.locality,
                "priority": c.priority,
                "confidence": c.confidence,
                "reason": c.reason,
            }
            if c.engine:
                d["engine"] = c.engine
            if c.artifact:
                d["artifact"] = _asdict(c.artifact)
            if c.gates:
                d["gates"] = [_asdict(g) for g in c.gates]
            result.append(d)
        return result

    # Single selection without structured candidates
    d2: dict[str, Any] = {
        "locality": selection.locality,
        "priority": 0,
        "confidence": 1.0,
        "reason": getattr(selection, "reason", "") or "planner selection",
    }
    if getattr(selection, "engine", None):
        d2["engine"] = selection.engine
    if getattr(selection, "artifact", None):
        d2["artifact"] = _asdict(selection.artifact)
    return [d2]


# ---------------------------------------------------------------------------
# Locality helpers
# ---------------------------------------------------------------------------


def _determine_locality(
    runtime: ModelRuntime,
    model_id: str,
    routing_policy: Optional[RoutingPolicy] = None,
) -> tuple[str, bool]:
    """Return (locality, is_fallback) for a resolved runtime.

    locality: "on_device" | "cloud"
    is_fallback: True when RouterModelRuntime fell back from local to cloud.
    """
    if isinstance(runtime, RouterModelRuntime):
        try:
            return runtime.resolve_locality(routing_policy)
        except RuntimeError:
            return LOCALITY_CLOUD, False
    if isinstance(runtime, CloudModelRuntime):
        return LOCALITY_CLOUD, False
    if isinstance(runtime, InferenceBackendAdapter):
        return LOCALITY_ON_DEVICE, False
    return LOCALITY_ON_DEVICE, False


def _locality_for_candidate(candidate: dict[str, Any]) -> str:
    """Map candidate locality to internal constant."""
    if candidate.get("locality") == "cloud":
        return LOCALITY_CLOUD
    return LOCALITY_ON_DEVICE


def _is_synthetic_cloud_fallback(selection: Any) -> bool:
    """True when the offline planner merely reported local engine absence.

    Delegates to ``octomil.execution.planner_resolution._is_synthetic_cloud_fallback``.
    """
    return _is_synthetic_cloud_fallback_impl(selection)


def _output_quality_evaluator_for_request(
    runtime_request: RuntimeRequest,
) -> RegistryBackedEvaluator:
    """Build the default SDK-local output quality evaluator for a request."""
    json_schema: dict[str, Any] | None = None
    if runtime_request.json_schema and runtime_request.json_schema != "{}":
        try:
            parsed = json.loads(runtime_request.json_schema)
            if isinstance(parsed, dict):
                json_schema = parsed
        except ValueError:
            json_schema = None

    return RegistryBackedEvaluator(EvaluatorRegistry.with_defaults(json_schema=json_schema))


class _PlannerRuntimeAvailabilityChecker(RuntimeChecker):
    """Reject planner local candidates whose concrete engine is unavailable."""

    _unsupported_engines = frozenset({"ollama", "echo"})

    def check(self, *, engine: str | None, locality: str) -> tuple[bool, str | None]:
        if locality != "local" or not engine:
            return True, None

        try:
            from octomil.runtime.engines import get_registry as get_engine_registry
            from octomil.runtime.planner.planner import _canonical_engine_id

            canonical_engine = _canonical_engine_id(str(engine)) or str(engine)
            if canonical_engine in self._unsupported_engines:
                return False, "engine_not_supported"
            runtime_engine = get_engine_registry().get_engine(canonical_engine)
            if runtime_engine is None or runtime_engine.name in self._unsupported_engines:
                return False, "engine_not_registered"
            if not runtime_engine.detect():
                return False, "engine_not_available"
            return True, None
        except Exception:
            logger.debug("Planner runtime availability check failed", exc_info=True)
            return False, "engine_detection_failed"


# ---------------------------------------------------------------------------
# OctomilResponses
# ---------------------------------------------------------------------------


class OctomilResponses:
    """Developer-facing Response API (Layer 2).

    Provides create() and stream() methods that resolve a ModelRuntime,
    build structured RuntimeRequest messages, and return structured responses.

    Resolution order (3-step):
      1. ModelCatalogService (if configured)
      2. Custom runtime_resolver callback (if provided)
      3. ModelRuntimeRegistry (global fallback)

    When a planner plan is available, the CandidateAttemptRunner is used to
    evaluate candidates with fallback semantics before invoking inference.
    """

    def __init__(
        self,
        runtime_resolver: Optional[Callable[[str], Optional[ModelRuntime]]] = None,
        catalog: Optional[ModelCatalogService] = None,
        telemetry_reporter: Optional[object] = None,
        routing_policies: Optional[dict[str, RoutingPolicy]] = None,
        model_deployment_map: Optional[dict[str, str]] = None,
        default_routing_policy: Optional[RoutingPolicy] = None,
        planner_enabled: bool = True,
    ) -> None:
        self._runtime_resolver = runtime_resolver
        self._catalog = catalog
        self._response_cache: dict[str, Response] = {}
        self._telemetry = telemetry_reporter
        self._routing_policies = routing_policies or {}
        self._model_deployment_map = model_deployment_map or {}
        self._default_routing_policy = default_routing_policy
        self._planner_enabled = planner_enabled

    def _resolve_routing_policy(
        self,
        model: Union[str, ModelRef],
        metadata: Optional[dict[str, str]],
    ) -> Optional[RoutingPolicy]:
        """Resolve routing policy: per-request metadata > deployment map > model map > default."""
        explicit = RoutingPolicy.from_metadata(metadata)
        if explicit is not None:
            return explicit
        if metadata and self._routing_policies:
            dep_id = metadata.get("deployment_id")
            if dep_id and dep_id in self._routing_policies:
                return self._routing_policies[dep_id]
        if self._model_deployment_map and self._routing_policies:
            model_id = _model_id_str(model)
            dep_id = self._model_deployment_map.get(model_id)
            if dep_id and dep_id in self._routing_policies:
                return self._routing_policies[dep_id]
        return self._default_routing_policy

    # ------------------------------------------------------------------
    # Attempt runner integration
    # ------------------------------------------------------------------

    async def _resolve_via_attempt_runner(
        self,
        model_id: str,
        runtime_request: RuntimeRequest,
        *,
        streaming: bool = False,
    ) -> Optional[_AttemptRunnerResult]:
        """Try planner-driven candidate selection with CandidateAttemptRunner.

        Returns an _AttemptRunnerResult if a plan is available and inference
        succeeds via the attempt loop. Returns None if no plan is available.
        """
        if not self._planner_enabled:
            return None

        selection = _try_resolve_planner_selection(model_id)
        if selection is None:
            return None
        if _is_synthetic_cloud_fallback(selection):
            return None

        candidates = _selection_to_candidate_dicts(selection)
        if not candidates:
            return None

        fallback_allowed = getattr(selection, "fallback_allowed", True)
        runner = CandidateAttemptRunner(
            fallback_allowed=fallback_allowed,
            streaming=streaming,
            output_quality_evaluator=_output_quality_evaluator_for_request(runtime_request),
        )
        runtime_model_id = _runtime_model_for_selection(selection, model_id)

        async def _execute_candidate(candidate: dict[str, Any]) -> _RuntimeResponse:
            locality = candidate.get("locality", "local")
            runtime = self._resolve_runtime_for_candidate(runtime_model_id, candidate)
            if locality == "cloud":
                if isinstance(runtime, RouterModelRuntime):
                    return await runtime.run(runtime_request, policy=RoutingPolicy.cloud_only())
                return await runtime.run(runtime_request)
            else:
                if isinstance(runtime, RouterModelRuntime):
                    return await runtime.run(runtime_request, policy=RoutingPolicy.local_only())
                return await runtime.run(runtime_request)

        attempt_loop = await runner.run_with_inference(
            candidates,
            execute_candidate=_execute_candidate,
            runtime_checker=_PlannerRuntimeAvailabilityChecker(),
        )

        if not attempt_loop.succeeded:
            if attempt_loop.error is not None:
                raise attempt_loop.error
            last_attempt = attempt_loop.attempts[-1] if attempt_loop.attempts else None
            reason = last_attempt.reason_message if last_attempt is not None else "No planner candidate was runnable"
            raise RuntimeError(reason)

        response = attempt_loop.value
        if not isinstance(response, _RuntimeResponse):
            return None

        selected = attempt_loop.selected_attempt
        locality = LOCALITY_ON_DEVICE
        if selected is not None and selected.locality == "cloud":
            locality = LOCALITY_CLOUD

        route = _route_metadata_from_selection(
            selection,
            locality,
            attempt_loop.fallback_used,
            model_name=model_id,
            capability="chat",
            attempt_loop=attempt_loop,
        )

        return _AttemptRunnerResult(
            response=response,
            locality=locality,
            is_fallback=attempt_loop.fallback_used,
            attempt_loop=attempt_loop,
            route=route,
            selection=selection,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create(self, request: ResponseRequest) -> Response:
        """Create a response, using planner-driven routing when available."""
        model_id = _model_id_str(request.model)
        effective_request = self._apply_previous_response(request)
        runtime_request = self._build_runtime_request(effective_request)

        # Try planner-driven attempt runner first
        runner_result = await self._resolve_via_attempt_runner(model_id, runtime_request, streaming=False)
        if runner_result is not None:
            if runner_result.is_fallback and self._telemetry is not None:
                try:
                    self._telemetry.report_fallback_cloud(  # type: ignore[attr-defined]
                        model_id=model_id,
                        fallback_reason="local_unavailable",
                    )
                except Exception:
                    pass
            response = self._build_response(
                request.model,
                runner_result.response,
                locality=runner_result.locality,
                route=runner_result.route,
            )
            _emit_route_decision_if_needed(
                self._telemetry,
                response=response,
                model_id=model_id,
                route=runner_result.route,
                selection=runner_result.selection,
                attempt_loop=runner_result.attempt_loop,
            )
            self._response_cache[response.id] = response
            return response

        # Fallback: direct runtime call (no plan available)
        runtime = self._resolve_runtime(request.model)
        routing_policy = self._resolve_routing_policy(request.model, request.metadata)
        locality, is_fallback = _determine_locality(runtime, model_id, routing_policy)

        if is_fallback and self._telemetry is not None:
            try:
                self._telemetry.report_fallback_cloud(  # type: ignore[attr-defined]
                    model_id=model_id,
                    fallback_reason="local_unavailable",
                )
            except Exception:
                pass

        route = _route_metadata_from_selection(
            None,
            locality,
            is_fallback,
            model_name=model_id,
            capability="chat",
        )

        if isinstance(runtime, RouterModelRuntime) and routing_policy is not None:
            runtime_response = await runtime.run(runtime_request, policy=routing_policy)
        else:
            runtime_response = await runtime.run(runtime_request)
        response = self._build_response(request.model, runtime_response, locality=locality, route=route)
        _emit_route_decision_if_needed(
            self._telemetry,
            response=response,
            model_id=model_id,
            route=route,
            routing_policy=routing_policy,
        )
        self._response_cache[response.id] = response
        return response

    async def stream(self, request: ResponseRequest) -> AsyncIterator[ResponseStreamEvent]:
        """Stream a response, using planner-driven routing when available."""
        model_id = _model_id_str(request.model)
        runtime_request = self._build_runtime_request(request)

        # Try planner-driven attempt runner for streaming
        if self._planner_enabled:
            selection = _try_resolve_planner_selection(model_id)
            if selection is not None and not _is_synthetic_cloud_fallback(selection):
                candidates = _selection_to_candidate_dicts(selection)
                if candidates:
                    fallback_allowed = getattr(selection, "fallback_allowed", True)
                    runner = CandidateAttemptRunner(
                        fallback_allowed=fallback_allowed,
                        streaming=True,
                        output_quality_evaluator=_output_quality_evaluator_for_request(runtime_request),
                    )
                    async for event in self._stream_with_attempt_runner(
                        request, model_id, runtime_request, candidates, runner, selection
                    ):
                        yield event
                    return

        # Fallback: direct runtime streaming (no plan available)
        runtime = self._resolve_runtime(request.model)
        routing_policy = self._resolve_routing_policy(request.model, request.metadata)
        locality, is_fallback = _determine_locality(runtime, model_id, routing_policy)

        if is_fallback and self._telemetry is not None:
            try:
                self._telemetry.report_fallback_cloud(  # type: ignore[attr-defined]
                    model_id=model_id,
                    fallback_reason="local_unavailable",
                )
            except Exception:
                pass

        route = _route_metadata_from_selection(
            None,
            locality,
            is_fallback,
            model_name=model_id,
            capability="chat",
        )

        response_id = _generate_id()
        text_parts: list[str] = []
        tool_call_buffers: dict[int, _ToolCallBuffer] = {}
        last_usage: Optional[RuntimeUsage] = None

        if isinstance(runtime, RouterModelRuntime) and routing_policy is not None:
            stream_iter = runtime.stream(runtime_request, policy=routing_policy)
        else:
            stream_iter = runtime.stream(runtime_request)
        async for chunk in stream_iter:
            if chunk.text is not None:
                text_parts.append(chunk.text)
                yield TextDeltaEvent(delta=chunk.text)
            if chunk.tool_call_delta is not None:
                delta = chunk.tool_call_delta
                buffer = tool_call_buffers.setdefault(delta.index, _ToolCallBuffer())
                if delta.id is not None:
                    buffer.id = delta.id
                if delta.name is not None:
                    buffer.name = delta.name
                if delta.arguments_delta is not None:
                    buffer.arguments += delta.arguments_delta
                yield ToolCallDeltaEvent(
                    index=delta.index,
                    id=delta.id,
                    name=delta.name,
                    arguments_delta=delta.arguments_delta,
                )
            if chunk.usage is not None:
                last_usage = chunk.usage

        output: list[OutputItem] = []
        full_text = "".join(text_parts)
        if full_text:
            output.append(TextOutput(text=full_text))
        for idx in sorted(tool_call_buffers):
            buf = tool_call_buffers[idx]
            output.append(
                ToolCallOutput(
                    tool_call=ResponseToolCall(
                        id=buf.id or _generate_id(),
                        name=buf.name or "",
                        arguments=buf.arguments,
                    )
                )
            )

        finish_reason = "tool_calls" if tool_call_buffers else "stop"
        usage = (
            ResponseUsage(
                prompt_tokens=last_usage.prompt_tokens,
                completion_tokens=last_usage.completion_tokens,
                total_tokens=last_usage.total_tokens,
            )
            if last_usage
            else None
        )

        response = Response(
            id=response_id,
            model=request.model,
            output=output,
            finish_reason=finish_reason,
            usage=usage,
            locality=locality,
            route=route,
        )
        _emit_route_decision_if_needed(
            self._telemetry,
            response=response,
            model_id=model_id,
            route=route,
            routing_policy=routing_policy,
        )
        yield DoneEvent(response=response)

    # ------------------------------------------------------------------
    # Streaming with attempt runner (first-token fallback)
    # ------------------------------------------------------------------

    async def _stream_with_attempt_runner(
        self,
        request: ResponseRequest,
        model_id: str,
        runtime_request: RuntimeRequest,
        candidates: list[dict[str, Any]],
        runner: CandidateAttemptRunner,
        selection: Any,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """Stream with CandidateAttemptRunner fallback semantics.

        - If streaming fails BEFORE first token: fall back to next candidate
        - If streaming fails AFTER first token: raise (do not fall back)
        """
        response_id = _generate_id()
        text_parts: list[str] = []
        tool_call_buffers: dict[int, _ToolCallBuffer] = {}
        last_usage: Optional[RuntimeUsage] = None
        selected_locality = LOCALITY_ON_DEVICE
        is_fallback = False
        last_error: Optional[Exception] = None
        runtime_model_id = _runtime_model_for_selection(selection, model_id)
        stream_attempts: list[RouteAttempt] = []
        selected_attempt: RouteAttempt | None = None
        fallback_trigger: FallbackTrigger | None = None
        from_attempt: int | None = None

        for idx, candidate in enumerate(candidates):
            first_token_emitted = False
            readiness = CandidateAttemptRunner(
                fallback_allowed=False,
                streaming=True,
            ).run([candidate], runtime_checker=_PlannerRuntimeAvailabilityChecker())
            ready_attempt = readiness.attempts[0] if readiness.attempts else None
            if ready_attempt is not None:
                ready_attempt.index = idx
            if not readiness.succeeded:
                if ready_attempt is not None:
                    stream_attempts.append(ready_attempt)
                    if fallback_trigger is None:
                        fallback_trigger = FallbackTrigger(
                            code=ready_attempt.reason_code,
                            stage=ready_attempt.stage.value,
                            message=ready_attempt.reason_message,
                        )
                        from_attempt = idx
                if (
                    runner.should_fallback_after_inference_error(first_token_emitted=False)
                    and idx < len(candidates) - 1
                ):
                    continue
                raise RuntimeError(ready_attempt.reason_message if ready_attempt else "No runtime available")

            try:
                runtime = self._resolve_runtime_for_candidate(runtime_model_id, candidate)
                policy = (
                    RoutingPolicy.cloud_only() if candidate.get("locality") == "cloud" else RoutingPolicy.local_only()
                )
                if isinstance(runtime, RouterModelRuntime):
                    stream_iter = runtime.stream(runtime_request, policy=policy)
                else:
                    stream_iter = runtime.stream(runtime_request)

                async for chunk in stream_iter:
                    if chunk.text is not None:
                        first_token_emitted = True
                        text_parts.append(chunk.text)
                        yield TextDeltaEvent(delta=chunk.text)
                    if chunk.tool_call_delta is not None:
                        first_token_emitted = True
                        delta = chunk.tool_call_delta
                        buffer = tool_call_buffers.setdefault(delta.index, _ToolCallBuffer())
                        if delta.id is not None:
                            buffer.id = delta.id
                        if delta.name is not None:
                            buffer.name = delta.name
                        if delta.arguments_delta is not None:
                            buffer.arguments += delta.arguments_delta
                        yield ToolCallDeltaEvent(
                            index=delta.index,
                            id=delta.id,
                            name=delta.name,
                            arguments_delta=delta.arguments_delta,
                        )
                    if chunk.usage is not None:
                        last_usage = chunk.usage

                assert ready_attempt is not None
                quality_failure = runner._evaluate_output_quality_gates(
                    candidate,
                    "".join(text_parts),
                    ready_attempt,
                    idx,
                    first_token_emitted=first_token_emitted,
                )
                if quality_failure is not None and not first_token_emitted:
                    stream_attempts.append(quality_failure)
                    if (
                        runner.should_fallback_after_inference_error(first_token_emitted=False)
                        and idx < len(candidates) - 1
                    ):
                        if fallback_trigger is None:
                            fallback_trigger = FallbackTrigger(
                                code=quality_failure.reason_code,
                                stage=AttemptStage.OUTPUT_QUALITY.value,
                                message=quality_failure.reason_message,
                                gate_code=quality_failure.reason_code.replace("quality_gate_", ""),
                                gate_class="output_quality",
                                evaluation_phase="post_inference",
                                candidate_index=idx,
                            )
                            from_attempt = idx
                        text_parts.clear()
                        tool_call_buffers.clear()
                        last_usage = None
                        continue
                    raise RuntimeError(quality_failure.reason_message)

                selected_locality = _locality_for_candidate(candidate)
                is_fallback = idx > 0
                selected_attempt = ready_attempt
                if selected_attempt is not None:
                    selected_attempt.reason_message = "stream completed"
                    stream_attempts.append(selected_attempt)
                break

            except Exception as exc:
                last_error = exc
                reason_code = (
                    "inference_error_after_first_token" if first_token_emitted else "inference_error_before_first_token"
                )
                failed_attempt = RouteAttempt(
                    index=idx,
                    locality=candidate.get("locality", "local"),
                    mode=CandidateAttemptRunner._mode_for_candidate(candidate),
                    engine=candidate.get("engine"),
                    artifact=ready_attempt.artifact if ready_attempt is not None else None,
                    status=AttemptStatus.FAILED,
                    stage=AttemptStage.INFERENCE,
                    gate_results=ready_attempt.gate_results if ready_attempt is not None else [],
                    reason_code=reason_code,
                    reason_message=str(exc) or reason_code,
                )
                stream_attempts.append(failed_attempt)
                if (
                    runner.should_fallback_after_inference_error(first_token_emitted=first_token_emitted)
                    and idx < len(candidates) - 1
                ):
                    if fallback_trigger is None:
                        fallback_trigger = FallbackTrigger(
                            code=reason_code,
                            stage=AttemptStage.INFERENCE.value,
                            message=failed_attempt.reason_message,
                        )
                        from_attempt = idx
                    text_parts.clear()
                    tool_call_buffers.clear()
                    last_usage = None
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("No runtime available")

        attempt_loop_result = AttemptLoopResult(
            selected_attempt=selected_attempt,
            attempts=stream_attempts,
            fallback_used=is_fallback,
            fallback_trigger=fallback_trigger if is_fallback else None,
            from_attempt=from_attempt if is_fallback else None,
            to_attempt=selected_attempt.index if is_fallback and selected_attempt is not None else None,
        )

        route = _route_metadata_from_selection(
            selection,
            selected_locality,
            is_fallback,
            model_name=model_id,
            capability="chat",
            attempt_loop=attempt_loop_result,
        )

        if is_fallback and self._telemetry is not None:
            try:
                self._telemetry.report_fallback_cloud(  # type: ignore[attr-defined]
                    model_id=model_id,
                    fallback_reason="local_unavailable",
                )
            except Exception:
                pass

        output: list[OutputItem] = []
        full_text = "".join(text_parts)
        if full_text:
            output.append(TextOutput(text=full_text))
        for idx_buf in sorted(tool_call_buffers):
            buf = tool_call_buffers[idx_buf]
            output.append(
                ToolCallOutput(
                    tool_call=ResponseToolCall(
                        id=buf.id or _generate_id(),
                        name=buf.name or "",
                        arguments=buf.arguments,
                    )
                )
            )

        finish_reason = "tool_calls" if tool_call_buffers else "stop"
        usage = (
            ResponseUsage(
                prompt_tokens=last_usage.prompt_tokens,
                completion_tokens=last_usage.completion_tokens,
                total_tokens=last_usage.total_tokens,
            )
            if last_usage
            else None
        )

        response = Response(
            id=response_id,
            model=request.model,
            output=output,
            finish_reason=finish_reason,
            usage=usage,
            locality=selected_locality,
            route=route,
        )
        _emit_route_decision_if_needed(
            self._telemetry,
            response=response,
            model_id=model_id,
            route=route,
            selection=selection,
            attempt_loop=attempt_loop_result,
        )
        yield DoneEvent(response=response)

    # ------------------------------------------------------------------
    # Internal resolution and building
    # ------------------------------------------------------------------

    def _resolve_runtime(self, model: Union[str, ModelRef]) -> ModelRuntime:
        """3-step resolution: catalog -> custom resolver -> registry."""
        if self._catalog is not None:
            if isinstance(model, (_ModelRefId, _ModelRefCapability)):
                runtime = self._catalog.runtime_for_ref(model)
            else:
                runtime = self._catalog.runtime_for_ref(_ModelRefId(model_id=model))
            if runtime is not None:
                return runtime

        model_id: str
        if isinstance(model, _ModelRefId):
            model_id = model.model_id
        elif isinstance(model, _ModelRefCapability):
            model_id = model.capability.value
        else:
            model_id = model

        if self._runtime_resolver is not None:
            runtime = self._runtime_resolver(model_id)
            if runtime is not None:
                return runtime

        runtime = ModelRuntimeRegistry.shared().resolve(model_id)
        if runtime is not None:
            return runtime

        raise NoRuntimeAvailableError(
            model_id=model_id,
            planner_enabled=self._planner_enabled,
        )

    def _resolve_runtime_for_candidate(self, model_id: str, candidate: dict[str, Any]) -> ModelRuntime:
        """Resolve the concrete runtime for a planner candidate.

        For planner-selected local candidates with an explicit engine, execute
        that engine or fail the attempt. Falling back to an unrelated local
        runtime would make route metadata lie about what actually ran.
        """
        locality = candidate.get("locality", "local")
        engine_id = candidate.get("engine")
        if locality != "local" or not engine_id:
            return self._resolve_runtime(model_id)

        try:
            from octomil.runtime.core.engine_bridge import _infer_tool_call_tier
            from octomil.runtime.engines import get_registry as get_engine_registry
            from octomil.runtime.planner.planner import _canonical_engine_id

            canonical_engine = _canonical_engine_id(str(engine_id)) or str(engine_id)
            if canonical_engine in _PlannerRuntimeAvailabilityChecker._unsupported_engines:
                raise RuntimeError(f"Planner-selected engine '{canonical_engine}' is not supported")
            engine_registry = get_engine_registry()
            engine = engine_registry.get_engine(canonical_engine)
            if (
                engine is None
                or not engine.detect()
                or engine.name in _PlannerRuntimeAvailabilityChecker._unsupported_engines
            ):
                raise RuntimeError(f"Planner-selected engine '{canonical_engine}' is not available")

            backend = engine.create_backend(model_id)
            return InferenceBackendAdapter(
                backend=backend,
                model_name=model_id,
                capabilities=RuntimeCapabilities(
                    supports_streaming=True,
                    tool_call_tier=_infer_tool_call_tier(model_id),
                ),
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Planner-selected engine '{engine_id}' failed to initialize: {exc}") from exc

    def _apply_previous_response(self, request: ResponseRequest) -> ResponseRequest:
        """Prepend previous response output as assistant context."""
        if not request.previous_response_id:
            return request
        prev = self._response_cache.get(request.previous_response_id)
        if prev is None:
            return request

        assistant_text = "".join(item.text for item in prev.output if isinstance(item, TextOutput))
        assistant_item = AssistantInput(content=[TextContent(text=assistant_text)] if assistant_text else None)

        input_items: list[InputItem]
        if isinstance(request.input, str):
            input_items = [text_input(request.input)]
        else:
            input_items = list(request.input)

        return ResponseRequest(
            model=request.model,
            input=[assistant_item] + input_items,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            stream=request.stream,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            metadata=request.metadata,
            instructions=request.instructions,
        )

    def _build_runtime_request(self, request: ResponseRequest) -> RuntimeRequest:
        """Build a RuntimeRequest from a ResponseRequest."""
        input_items: list[InputItem]
        if isinstance(request.input, str):
            input_items = [text_input(request.input)]
        else:
            input_items = list(request.input)

        if request.instructions:
            input_items = [system_input(request.instructions)] + input_items

        messages = _input_items_to_messages(input_items)

        tool_defs: Optional[list[RuntimeToolDef]] = None
        if request.tools:
            tool_defs = []
            for t in request.tools:
                fn = t.get("function", t)
                schema = fn.get("input_schema") or fn.get("parameters")
                tool_defs.append(
                    RuntimeToolDef(
                        name=fn.get("name", ""),
                        description=fn.get("description", ""),
                        parameters_schema=json.dumps(schema) if schema else None,
                    )
                )

        json_schema: Optional[str] = None
        if isinstance(request.response_format, JsonSchemaFormat):
            json_schema = request.response_format.schema
        elif request.response_format == ResponseFormat.JSON_OBJECT:
            json_schema = "{}"

        return RuntimeRequest(
            messages=messages,
            generation_config=GenerationConfig(
                max_tokens=request.max_output_tokens or 512,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 1.0,
                stop=request.stop,
            ),
            tool_definitions=tool_defs,
            json_schema=json_schema,
            model=request.model,
        )

    def _build_response(
        self,
        model: str,
        runtime_response: _RuntimeResponse,
        locality: Optional[str] = None,
        route: Optional[RouteMetadata] = None,
    ) -> Response:
        """Build a Response from a RuntimeResponse."""
        output: list[OutputItem] = []
        if runtime_response.text:
            output.append(TextOutput(text=runtime_response.text))
        if runtime_response.tool_calls:
            for call in runtime_response.tool_calls:
                output.append(
                    ToolCallOutput(
                        tool_call=ResponseToolCall(
                            id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                        )
                    )
                )

        finish_reason = "tool_calls" if runtime_response.tool_calls else runtime_response.finish_reason
        usage = (
            ResponseUsage(
                prompt_tokens=runtime_response.usage.prompt_tokens,
                completion_tokens=runtime_response.usage.completion_tokens,
                total_tokens=runtime_response.usage.total_tokens,
            )
            if runtime_response.usage
            else None
        )

        return Response(
            id=_generate_id(),
            model=model,
            output=output,
            finish_reason=finish_reason,
            usage=usage,
            locality=locality,
            route=route,
        )


# -- Layer 2 -> Layer 1 bridge --


def _input_items_to_messages(input_items: list[InputItem]) -> list[RuntimeMessage]:
    """Convert Layer 2 InputItems to Layer 1 RuntimeMessages."""
    messages: list[RuntimeMessage] = []
    for item in input_items:
        if isinstance(item, SystemInput):
            messages.append(
                RuntimeMessage(
                    role=MessageRole.SYSTEM,
                    parts=[RuntimeContentPart.text_part(item.content)],
                )
            )
        elif isinstance(item, UserInput):
            parts = [_resolve_content_part(p) for p in item.content]
            messages.append(RuntimeMessage(role=MessageRole.USER, parts=parts))
        elif isinstance(item, AssistantInput):
            asst_parts: list[RuntimeContentPart] = []
            if item.content:
                for p in item.content:
                    if isinstance(p, TextContent):
                        asst_parts.append(RuntimeContentPart.text_part(p.text))
            rt_tool_calls: Optional[list[RuntimeToolCall]] = None
            if item.tool_calls:
                rt_tool_calls = [
                    RuntimeToolCall(id=call.id, name=call.name, arguments=call.arguments) for call in item.tool_calls
                ]
            if not asst_parts:
                asst_parts = [RuntimeContentPart.text_part("")]
            messages.append(
                RuntimeMessage(
                    role=MessageRole.ASSISTANT,
                    parts=asst_parts,
                    tool_calls=rt_tool_calls,
                )
            )
        elif isinstance(item, ToolResultInput):
            messages.append(
                RuntimeMessage(
                    role=MessageRole.TOOL,
                    parts=[RuntimeContentPart.text_part(item.content)],
                    tool_call_id=item.tool_call_id,
                )
            )
    return messages


def _resolve_content_part(part: object) -> RuntimeContentPart:
    """Convert a Layer 2 ContentPart to a Layer 1 RuntimeContentPart."""
    if isinstance(part, TextContent):
        return RuntimeContentPart.text_part(part.text)
    if isinstance(part, ImageContent):
        if part.data:
            raw = base64.b64decode(part.data)
            return RuntimeContentPart.image_part(raw, part.media_type or "image/png")
        logger.warning("ImageContent without data")
        return RuntimeContentPart.text_part("[image: unresolved]")
    if isinstance(part, AudioContent):
        raw = base64.b64decode(part.data)
        return RuntimeContentPart.audio_part(raw, part.media_type)
    if isinstance(part, FileContent):
        mt = part.media_type.lower()
        raw = base64.b64decode(part.data)
        if mt.startswith("image/"):
            return RuntimeContentPart.image_part(raw, part.media_type)
        if mt.startswith("audio/"):
            return RuntimeContentPart.audio_part(raw, part.media_type)
        if mt.startswith("video/"):
            return RuntimeContentPart.video_part(raw, part.media_type)
        raise ValueError(
            f"Cannot resolve FileContent with mediaType '{part.media_type}' "
            f"to a runtime content part. Supported prefixes: image/*, audio/*, video/*"
        )
    raise TypeError(f"Unknown content part type: {type(part)}")


# -- Helpers --


def _model_id_str(model: Union[str, ModelRef]) -> str:
    """Normalize a model ref to a plain string ID."""
    if isinstance(model, _ModelRefId):
        return model.model_id
    if isinstance(model, _ModelRefCapability):
        return model.capability.value
    return str(model)


def _generate_id() -> str:
    return f"resp_{uuid.uuid4().hex[:16]}"


class _ToolCallBuffer:
    def __init__(self) -> None:
        self.id: Optional[str] = None
        self.name: Optional[str] = None
        self.arguments: str = ""
