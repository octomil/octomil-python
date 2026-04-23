"""Dispatch — cloud/local inference dispatch and planner integration.

Handles runtime resolution (catalog -> custom -> registry), planner-driven
candidate selection, and the CandidateAttemptRunner integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from octomil.execution.planner_resolution import (
    _is_synthetic_cloud_fallback as _is_synthetic_cloud_fallback_impl,
)
from octomil.execution.route_metadata_mapper import (
    RouteMetadata,
    _route_metadata_from_selection,
)
from octomil.model_ref import ModelRef, _ModelRefCapability, _ModelRefId
from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.registry import ModelRuntimeRegistry
from octomil.runtime.core.router import (
    LOCALITY_CLOUD,
    LOCALITY_ON_DEVICE,
    RouterModelRuntime,
)
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeRequest,
)
from octomil.runtime.core.types import (
    RuntimeResponse as _RuntimeResponse,
)
from octomil.runtime.routing.attempt_runner import (
    AttemptLoopResult,
    CandidateAttemptRunner,
)

if TYPE_CHECKING:
    from octomil.manifest.catalog_service import ModelCatalogService
    from octomil.runtime.planner.schemas import RuntimeSelection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attempt runner resolution result
# ---------------------------------------------------------------------------


@dataclass
class AttemptRunnerResult:
    """Result from resolve_via_attempt_runner when a plan is available."""

    response: _RuntimeResponse
    locality: str  # "on_device" | "cloud"
    is_fallback: bool
    attempt_loop: AttemptLoopResult
    route: RouteMetadata


# ---------------------------------------------------------------------------
# Planner plan resolution (non-fatal)
# ---------------------------------------------------------------------------


def try_resolve_planner_selection(model_id: str) -> Optional[RuntimeSelection]:
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


def selection_to_candidate_dicts(selection: Any) -> list[dict[str, Any]]:
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


def _runtime_model_for_selection(selection: Any, requested_model: str) -> str:
    """Return the concrete model a planner selection resolved to."""
    app_resolution = getattr(selection, "app_resolution", None)
    selected_model = getattr(app_resolution, "selected_model", None)
    if selected_model:
        return str(selected_model)
    return requested_model


def is_synthetic_cloud_fallback(selection: Any) -> bool:
    """True when the offline planner merely reported local engine absence.

    Delegates to ``octomil.execution.planner_resolution._is_synthetic_cloud_fallback``.
    """
    return _is_synthetic_cloud_fallback_impl(selection)


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------


def resolve_runtime(
    model: Union[str, ModelRef],
    *,
    catalog: Optional[ModelCatalogService] = None,
    runtime_resolver: Optional[Callable[[str], Optional[ModelRuntime]]] = None,
) -> ModelRuntime:
    """3-step resolution: catalog -> custom resolver -> registry."""
    if catalog is not None:
        if isinstance(model, (_ModelRefId, _ModelRefCapability)):
            runtime = catalog.runtime_for_ref(model)
        else:
            runtime = catalog.runtime_for_ref(_ModelRefId(model_id=model))
        if runtime is not None:
            return runtime

    model_id: str
    if isinstance(model, _ModelRefId):
        model_id = model.model_id
    elif isinstance(model, _ModelRefCapability):
        model_id = model.capability.value
    else:
        model_id = model

    if runtime_resolver is not None:
        runtime = runtime_resolver(model_id)
        if runtime is not None:
            return runtime

    runtime = ModelRuntimeRegistry.shared().resolve(model_id)
    if runtime is not None:
        return runtime

    raise RuntimeError(f"No ModelRuntime registered for model: {model_id}")


def resolve_runtime_for_candidate(
    model_id: str,
    candidate: dict[str, Any],
    *,
    catalog: Optional[ModelCatalogService] = None,
    runtime_resolver: Optional[Callable[[str], Optional[ModelRuntime]]] = None,
) -> ModelRuntime:
    """Resolve the concrete runtime for a planner candidate.

    For planner-selected local candidates with an explicit engine, execute
    that engine or fail the attempt. Falling back to an unrelated local
    runtime would make route metadata lie about what actually ran.
    """
    locality = candidate.get("locality", "local")
    engine_id = candidate.get("engine")
    if locality != "local" or not engine_id:
        return resolve_runtime(model_id, catalog=catalog, runtime_resolver=runtime_resolver)

    try:
        from octomil.runtime.core.engine_bridge import _infer_tool_call_tier
        from octomil.runtime.engines import get_registry as get_engine_registry
        from octomil.runtime.planner.planner import _canonical_engine_id

        canonical_engine = _canonical_engine_id(str(engine_id)) or str(engine_id)
        engine_registry = get_engine_registry()
        engine = engine_registry.get_engine(canonical_engine)
        if engine is None or not engine.detect() or engine.name == "echo":
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


# ---------------------------------------------------------------------------
# Attempt runner integration
# ---------------------------------------------------------------------------


async def resolve_via_attempt_runner(
    model_id: str,
    runtime_request: RuntimeRequest,
    *,
    planner_enabled: bool = True,
    streaming: bool = False,
    catalog: Optional[ModelCatalogService] = None,
    runtime_resolver: Optional[Callable[[str], Optional[ModelRuntime]]] = None,
) -> Optional[AttemptRunnerResult]:
    """Try planner-driven candidate selection with CandidateAttemptRunner.

    Returns an AttemptRunnerResult if a plan is available and inference
    succeeds via the attempt loop. Returns None if no plan is available.
    """
    if not planner_enabled:
        return None

    selection = try_resolve_planner_selection(model_id)
    if selection is None:
        return None
    if is_synthetic_cloud_fallback(selection):
        return None

    candidates = selection_to_candidate_dicts(selection)
    if not candidates:
        return None

    fallback_allowed = getattr(selection, "fallback_allowed", True)
    runner = CandidateAttemptRunner(
        fallback_allowed=fallback_allowed,
        streaming=streaming,
    )
    runtime_model_id = _runtime_model_for_selection(selection, model_id)

    async def _execute_candidate(
        candidate: dict[str, Any],
    ) -> _RuntimeResponse:
        locality = candidate.get("locality", "local")
        runtime = resolve_runtime_for_candidate(
            runtime_model_id,
            candidate,
            catalog=catalog,
            runtime_resolver=runtime_resolver,
        )
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
    )

    if not attempt_loop.succeeded:
        if attempt_loop.error is not None:
            raise attempt_loop.error
        return None

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

    return AttemptRunnerResult(
        response=response,
        locality=locality,
        is_fallback=attempt_loop.fallback_used,
        attempt_loop=attempt_loop,
        route=route,
    )
