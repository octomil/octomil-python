"""Shared execution kernel -- the single execution path for all Octomil surfaces.

Encapsulates:
  - model resolution
  - serving-policy evaluation
  - planner-driven runtime selection
  - local runtime resolution
  - cloud runtime resolution
  - fallback decisions
  - structured response generation
  - post-execution benchmark upload
  - telemetry

Does NOT require an HTTP server process.

Extracted sub-modules (prefer importing from them directly):
  - route_metadata_mapper: RouteMetadata types + builder
  - planner_resolution: planner call wrappers + candidate helpers
  - cloud_dispatch: cloud URL/key helpers
  - benchmark_upload: background benchmark telemetry upload
  - attempt_execution: locality decision logic + policy resolution
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from octomil._generated.message_role import MessageRole
from octomil.config.local import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_TRANSCRIPTION,
    CAPABILITY_TTS,
    CloudProfile,
    InlinePolicy,
    LoadedConfigSet,
    RequestOverrides,
    ResolvedExecutionDefaults,
    load_standalone_config,
    resolve_capability_defaults,
)
from octomil.errors import OctomilError, OctomilErrorCode

# --- Re-exports from extracted modules (backward compatibility) ---
from octomil.execution.attempt_execution import (  # noqa: F401
    _cloud_available,
    _inline_to_routing_policy,
    _resolve_localities,
    _resolve_routing_policy,
    _select_locality_for_capability,
)
from octomil.execution.benchmark_upload import (  # noqa: F401
    _sanitize_benchmark_payload,
    _upload_benchmark_async,
)
from octomil.execution.cloud_dispatch import (  # noqa: F401
    _cloud_api_key,
    _openai_base_url,
    _platform_api_base_url,
)
from octomil.execution.planner_resolution import (  # noqa: F401
    _PLANNER_CAPABILITY_MAP,
    _candidate_fallback_allowed,
    _candidate_to_selection,
    _is_synthetic_cloud_fallback,
    _resolve_planner_selection,
    _routing_policy_for_candidate,
    _runtime_model_for_selection,
    _selection_candidate_dicts,
)
from octomil.execution.route_metadata_mapper import (  # noqa: F401
    ArtifactCache,
    FallbackInfo,
    PlannerInfo,
    RouteArtifact,
    RouteExecution,
    RouteMetadata,
    RouteModel,
    RouteModelRequested,
    RouteModelResolved,
    RouteReason,
    _route_metadata_from_selection,
)
from octomil.runtime.core.router import LOCALITY_CLOUD, LOCALITY_ON_DEVICE, RouterModelRuntime
from octomil.runtime.core.types import (
    GenerationConfig,
    RuntimeContentPart,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeResponse,
)

logger = logging.getLogger(__name__)

# Capabilities whose adapters actually consume the prepared ``artifact_dir``
# today. A capability only enters this set once its dispatch path threads
# the ``model_dir`` from PrepareOutcome into the backend. Anything below
# is a false-success risk: prepare downloads bytes the next inference call
# ignores, then cold-starts through the engine's own lookup.
#
# Wiring history:
#   - tts             (PR 4 + 6 + 7)  — SherpaTtsEngine.create_backend(model_dir)
#   - transcription   (PR 10a)        — _WhisperBackend honors injected model_dir
#                                       and skips pywhispercpp's HF download path.
# Wiring backlog (intentionally NOT in the supported set):
#   - chat (kernel)   — MLXBackend and LlamaCppBackend accept ``model_dir``
#                       (PR 10c backend threading), but the capability is
#                       still gated because:
#                         (a) the public ``client.responses.create`` facade
#                             goes through ``OctomilResponses``, which does
#                             not thread ``model_dir`` into its engine
#                             ``create_backend`` calls — flipping ``chat``
#                             on the kernel would mean ``client.prepare``
#                             succeeds but the next ``client.responses.
#                             create`` cold-loads anyway;
#                         (b) PrepareManager materializes only single-file
#                             artifacts (``<dir>/artifact`` sentinel) and
#                             has no snapshot/manifest support, so MLX
#                             loads from a prepared dir don't work for
#                             real model shapes that mlx_lm requires
#                             (config.json + tokenizer + safetensors).
#                       ``chat`` and ``responses`` flip to wired once
#                       OctomilResponses goes through the kernel (or
#                       threads model_dir itself) AND PrepareManager grows
#                       snapshot materialization proven by an MLX e2e test.
#   - responses        — same gate as chat (dispatches through chat).
#   - embedding        — thread model_dir into the local embeddings backend.
_PREPAREABLE_CAPABILITIES = frozenset({CAPABILITY_TTS, CAPABILITY_TRANSCRIPTION})


# ---------------------------------------------------------------------------
# Result types (ExecutionResult, StreamChunk, ChatRoutingDecision stay here)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Unified result from any execution kernel call."""

    id: str = ""
    model: str = ""
    capability: str = ""
    locality: str = ""  # "on_device" | "cloud"
    fallback_used: bool = False
    output_text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    # Embeddings-specific
    embeddings: Optional[list[list[float]]] = None
    dimensions: Optional[int] = None
    # Transcription-specific
    segments: Optional[list[dict[str, Any]]] = None
    # Raw data for --json
    raw: Optional[dict[str, Any]] = None
    # Route metadata from planner
    route: Optional[RouteMetadata] = None


@dataclass
class StreamChunk:
    """A single chunk during streaming."""

    delta: str = ""
    done: bool = False
    result: Optional[ExecutionResult] = None


@dataclass
class ChatRoutingDecision:
    """Routing decision for a chat request — used by serve to dispatch to the right backend.

    Does not create runtimes or execute inference.  Tells the caller which
    locality to try first, whether fallback is available, and which cloud
    profile to use.
    """

    model: str
    primary_locality: str  # "on_device" | "cloud"
    fallback_locality: Optional[str] = None  # "on_device" | "cloud" | None
    cloud_profile: Optional[CloudProfile] = None
    policy_preset: Optional[str] = None
    inline_policy: Optional[InlinePolicy] = None


def _internal_locality_for_attempt(attempt: Any) -> str:
    if attempt is not None and getattr(attempt, "locality", None) == "cloud":
        return LOCALITY_CLOUD
    return LOCALITY_ON_DEVICE


# ---------------------------------------------------------------------------
# Execution Kernel
# ---------------------------------------------------------------------------


class ExecutionKernel:
    """Shared execution kernel for all Octomil surfaces.

    Usage::

        kernel = ExecutionKernel()
        result = await kernel.create_response("Hello!", model="gemma3-1b")
    """

    def __init__(
        self,
        *,
        config_set: Optional[LoadedConfigSet] = None,
        start_dir: Optional[Path] = None,
        prepare_manager: Optional[Any] = None,
    ) -> None:
        self._config_set = config_set or load_standalone_config(start_dir)
        # Optional caller-supplied PrepareManager. Tests and embedded callers
        # inject one to control where artifacts land and stub the downloader;
        # production code lazily constructs a default-rooted manager on first
        # prepare call.
        self._prepare_manager: Optional[Any] = prepare_manager

    @property
    def config_set(self) -> LoadedConfigSet:
        return self._config_set

    def resolve_chat_defaults(
        self,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> ResolvedExecutionDefaults:
        """Resolve chat defaults without executing inference."""
        return self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)

    # ------------------------------------------------------------------
    # Chat routing decision (for serve integration)
    # ------------------------------------------------------------------

    def resolve_chat_routing(
        self,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        local_available: bool,
        cloud_available: Optional[bool] = None,
    ) -> ChatRoutingDecision:
        """Resolve routing without executing inference.

        Returns a ``ChatRoutingDecision`` that tells the caller which
        backend locality to use as primary and which (if any) as fallback.
        Serve uses this to dispatch to its own ``InferenceBackend`` instances.
        """
        defaults = self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_CHAT)

        routing_policy = _resolve_routing_policy(defaults)

        # Determine cloud availability from config when caller does not know
        if cloud_available is None:
            cloud_available = _cloud_available(defaults)

        primary, fallback = _resolve_localities(
            routing_policy,
            local_available=local_available,
            cloud_available=cloud_available,
        )

        return ChatRoutingDecision(
            model=effective_model,
            primary_locality=primary,
            fallback_locality=fallback,
            cloud_profile=defaults.cloud_profile,
            policy_preset=defaults.policy_preset,
            inline_policy=defaults.inline_policy,
        )

    # ------------------------------------------------------------------
    # Responses
    # ------------------------------------------------------------------

    async def create_response(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """Create a one-shot response. The primary execution entrypoint."""
        defaults = self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_CHAT)

        policy_preset = defaults.policy_preset or "local_first"

        # Planner-driven routing
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_CHAT)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_CHAT, policy_preset)

        # PR B: app-ref refusal — see _enforce_app_ref_routing_policy.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )

        routing_policy = _resolve_routing_policy(defaults)
        # PR B: explicit local-only / private policy forces a
        # cloud_only-rejected candidate list so a planner outage can't
        # leak the request to cloud even if cloud creds are present.
        if _is_local_only_policy(policy):
            routing_policy = routing_policy.with_cloud_disabled()
        candidates = _selection_candidate_dicts(selection, routing_policy)
        fallback_allowed = _candidate_fallback_allowed(selection, routing_policy)
        # PR 10c: drop *individually* unpreparable local sdk_runtime
        # candidates (synthetic plan: missing digest/url, traversal,
        # etc.) so the runner doesn't burn an attempt cold-loading
        # through the engine's own download path. Other local
        # candidates in the same plan survive.
        runner_to_original = _runner_dict_to_planner_candidate(selection, candidates)
        candidates = _filter_unpreparable_local_candidates(
            self, selection, candidates, fallback_allowed=fallback_allowed
        )
        if _is_local_only_policy(policy):
            # Drop any cloud candidate the planner emitted. The
            # explicit local_only / private policy forbids cloud
            # locality regardless of what the plan recommends.
            candidates = [c for c in candidates if c.get("locality") != "cloud"]
        # PR B: when local-only forcing emptied the candidate list
        # (planner gave only cloud, no local) and the caller insisted
        # on local, surface an actionable error rather than a generic
        # "no runtime available".
        if _is_local_only_policy(policy) and not candidates:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"local_runtime_unavailable: caller passed policy={policy!r} but "
                    f"no local candidate is available for {effective_model!r}. "
                    "Install the relevant local engine extras or relax the policy."
                ),
            )

        gen_config = GenerationConfig(
            max_tokens=max_output_tokens or 2048,
            temperature=temperature if temperature is not None else 0.7,
        )
        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.text_part(prompt)],
                ),
            ],
            generation_config=gen_config,
        )

        from octomil.runtime.routing import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=fallback_allowed, streaming=False)

        # Cache prepared dirs by artifact_id within this call so a
        # local-then-local retry (rare but possible under fallback)
        # doesn't re-download. Prepare runs INSIDE the local attempt:
        # cloud-first plans never touch local prepare.
        prepared_dirs: dict[tuple, Optional[str]] = {}

        async def _execute_candidate(candidate: dict[str, Any]) -> RuntimeResponse:
            candidate_selection = _candidate_to_selection(selection, candidate)
            local_dir: Optional[str] = None
            if candidate.get("locality") == "local":
                planner_candidate = runner_to_original.get(id(candidate))
                local_dir = self._prepare_local_chat_artifact_cached(planner_candidate, prepared_dirs)
            router = await self._build_router(
                effective_model,
                CAPABILITY_CHAT,
                defaults,
                planner_selection=candidate_selection,
                prepared_model_dir=local_dir,
                cloud_execution_model=_execution_model_for_cloud_dispatch(
                    requested_model=model,
                    effective_model=effective_model,
                    planner_model=planner_model,
                    app=app,
                ),
            )
            return await router.run(request, policy=_routing_policy_for_candidate(candidate))

        t0 = time.monotonic()
        attempt_loop = await runner.run_with_inference(
            candidates,
            execute_candidate=_execute_candidate,
        )
        if not attempt_loop.succeeded:
            if attempt_loop.error is not None:
                raise attempt_loop.error
            raise RuntimeError("No runtime available")

        response = attempt_loop.value
        if not isinstance(response, RuntimeResponse):
            raise RuntimeError("Runtime returned an invalid response.")
        latency_ms = (time.monotonic() - t0) * 1000

        locality = _internal_locality_for_attempt(attempt_loop.selected_attempt)
        is_fallback = attempt_loop.fallback_used
        route = _route_metadata_from_selection(
            selection,
            locality,
            is_fallback,
            model_name=effective_model,
            capability=CAPABILITY_CHAT,
            attempt_loop=attempt_loop,
        )
        usage = _extract_usage(response)

        # Post-execution benchmark upload for local execution
        if locality == LOCALITY_ON_DEVICE:
            _upload_benchmark_async(
                model=effective_model,
                capability=CAPABILITY_CHAT,
                engine=route.execution.engine if route.execution else None,
                policy_preset=policy_preset,
                tokens_per_second=usage.get("output_tokens", 0) / max(latency_ms / 1000, 0.001),
                latency_ms=latency_ms,
            )

        return ExecutionResult(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            model=effective_model,
            capability=CAPABILITY_CHAT,
            locality=locality,
            fallback_used=is_fallback,
            output_text=response.text,
            usage=usage,
            route=route,
        )

    async def stream_response(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response token by token."""
        defaults = self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_CHAT)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_CHAT)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_CHAT, policy_preset)

        # PR B: app-ref refusal — see _enforce_app_ref_routing_policy.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )
        routing_policy = _resolve_routing_policy(defaults)
        if _is_local_only_policy(policy):
            routing_policy = routing_policy.with_cloud_disabled()
        candidates = _selection_candidate_dicts(selection, routing_policy)
        fallback_allowed = _candidate_fallback_allowed(selection, routing_policy)
        runner_to_original = _runner_dict_to_planner_candidate(selection, candidates)
        candidates = _filter_unpreparable_local_candidates(
            self, selection, candidates, fallback_allowed=fallback_allowed
        )
        if _is_local_only_policy(policy):
            # Drop any cloud candidate the planner emitted. The
            # explicit local_only / private policy forbids cloud
            # locality regardless of what the plan recommends.
            candidates = [c for c in candidates if c.get("locality") != "cloud"]
            if not candidates:
                raise OctomilError(
                    code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                    message=(
                        f"local_runtime_unavailable: caller passed policy={policy!r} but "
                        f"no local candidate is available for {effective_model!r}."
                    ),
                )

        gen_config = GenerationConfig(
            max_tokens=max_output_tokens or 2048,
            temperature=temperature if temperature is not None else 0.7,
        )
        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.text_part(prompt)],
                ),
            ],
            generation_config=gen_config,
        )

        from octomil.runtime.routing import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=fallback_allowed, streaming=True)

        prepared_dirs: dict[tuple, Optional[str]] = {}

        t0 = time.monotonic()
        collected_text = ""
        selected_locality = LOCALITY_ON_DEVICE
        is_fallback = False
        last_error: Exception | None = None

        for idx, candidate in enumerate(candidates):
            first_token_emitted = False
            try:
                candidate_selection = _candidate_to_selection(selection, candidate)
                local_dir: Optional[str] = None
                if candidate.get("locality") == "local":
                    planner_candidate = runner_to_original.get(id(candidate))
                    local_dir = self._prepare_local_chat_artifact_cached(planner_candidate, prepared_dirs)
                router = await self._build_router(
                    effective_model,
                    CAPABILITY_CHAT,
                    defaults,
                    planner_selection=candidate_selection,
                    prepared_model_dir=local_dir,
                    cloud_execution_model=_execution_model_for_cloud_dispatch(
                        requested_model=model,
                        effective_model=effective_model,
                        planner_model=planner_model,
                        app=app,
                    ),
                )
                async for chunk in router.stream(request, policy=_routing_policy_for_candidate(candidate)):
                    text = chunk.text or ""
                    if text:
                        first_token_emitted = True
                    collected_text += text
                    yield StreamChunk(delta=text)
                selected_locality = LOCALITY_CLOUD if candidate.get("locality") == "cloud" else LOCALITY_ON_DEVICE
                is_fallback = idx > 0
                break
            except Exception as exc:
                last_error = exc
                if (
                    runner.should_fallback_after_inference_error(first_token_emitted=first_token_emitted)
                    and idx < len(candidates) - 1
                ):
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("No runtime available")

        latency_ms = (time.monotonic() - t0) * 1000

        route = _route_metadata_from_selection(
            selection,
            selected_locality,
            is_fallback,
            model_name=effective_model,
            capability=CAPABILITY_CHAT,
        )

        if selected_locality == LOCALITY_ON_DEVICE:
            _upload_benchmark_async(
                model=effective_model,
                capability=CAPABILITY_CHAT,
                engine=route.execution.engine if route.execution else None,
                policy_preset=policy_preset,
                latency_ms=latency_ms,
            )

        yield StreamChunk(
            delta="",
            done=True,
            result=ExecutionResult(
                id=f"resp_{uuid.uuid4().hex[:12]}",
                model=effective_model,
                capability=CAPABILITY_CHAT,
                locality=selected_locality,
                fallback_used=is_fallback,
                output_text=collected_text,
                route=route,
            ),
        )

    async def stream_chat_messages(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a multi-turn chat request through the shared router."""
        defaults = self._resolve(CAPABILITY_CHAT, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_CHAT)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_CHAT)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_CHAT, policy_preset)

        # PR B: app-ref refusal — see _enforce_app_ref_routing_policy.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )

        routing_policy = _resolve_routing_policy(defaults)
        if _is_local_only_policy(policy):
            routing_policy = routing_policy.with_cloud_disabled()
        candidates = _selection_candidate_dicts(selection, routing_policy)
        fallback_allowed = _candidate_fallback_allowed(selection, routing_policy)
        runner_to_original = _runner_dict_to_planner_candidate(selection, candidates)
        candidates = _filter_unpreparable_local_candidates(
            self, selection, candidates, fallback_allowed=fallback_allowed
        )
        if _is_local_only_policy(policy):
            # Drop any cloud candidate the planner emitted. The
            # explicit local_only / private policy forbids cloud
            # locality regardless of what the plan recommends.
            candidates = [c for c in candidates if c.get("locality") != "cloud"]
            if not candidates:
                raise OctomilError(
                    code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                    message=(
                        f"local_runtime_unavailable: caller passed policy={policy!r} but "
                        f"no local candidate is available for {effective_model!r}."
                    ),
                )

        gen_config = GenerationConfig(
            max_tokens=max_output_tokens or 2048,
            temperature=temperature if temperature is not None else 0.7,
        )
        request = RuntimeRequest(
            messages=_chat_messages_to_runtime_messages(messages),
            generation_config=gen_config,
        )

        from octomil.runtime.routing import CandidateAttemptRunner

        runner = CandidateAttemptRunner(fallback_allowed=fallback_allowed, streaming=True)

        prepared_dirs: dict[tuple, Optional[str]] = {}

        t0 = time.monotonic()
        collected_text = ""
        selected_locality = LOCALITY_ON_DEVICE
        is_fallback = False
        last_error: Exception | None = None

        for idx, candidate in enumerate(candidates):
            first_token_emitted = False
            try:
                candidate_selection = _candidate_to_selection(selection, candidate)
                local_dir: Optional[str] = None
                if candidate.get("locality") == "local":
                    planner_candidate = runner_to_original.get(id(candidate))
                    local_dir = self._prepare_local_chat_artifact_cached(planner_candidate, prepared_dirs)
                router = await self._build_router(
                    effective_model,
                    CAPABILITY_CHAT,
                    defaults,
                    planner_selection=candidate_selection,
                    prepared_model_dir=local_dir,
                    cloud_execution_model=_execution_model_for_cloud_dispatch(
                        requested_model=model,
                        effective_model=effective_model,
                        planner_model=planner_model,
                        app=app,
                    ),
                )
                async for chunk in router.stream(request, policy=_routing_policy_for_candidate(candidate)):
                    text = chunk.text or ""
                    if text:
                        first_token_emitted = True
                    collected_text += text
                    yield StreamChunk(delta=text)
                selected_locality = LOCALITY_CLOUD if candidate.get("locality") == "cloud" else LOCALITY_ON_DEVICE
                is_fallback = idx > 0
                break
            except Exception as exc:
                last_error = exc
                if (
                    runner.should_fallback_after_inference_error(first_token_emitted=first_token_emitted)
                    and idx < len(candidates) - 1
                ):
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("No runtime available")

        latency_ms = (time.monotonic() - t0) * 1000

        route = _route_metadata_from_selection(
            selection,
            selected_locality,
            is_fallback,
            model_name=effective_model,
            capability=CAPABILITY_CHAT,
        )

        if selected_locality == LOCALITY_ON_DEVICE:
            _upload_benchmark_async(
                model=effective_model,
                capability=CAPABILITY_CHAT,
                engine=route.execution.engine if route.execution else None,
                policy_preset=policy_preset,
                latency_ms=latency_ms,
            )

        yield StreamChunk(
            delta="",
            done=True,
            result=ExecutionResult(
                id=f"resp_{uuid.uuid4().hex[:12]}",
                model=effective_model,
                capability=CAPABILITY_CHAT,
                locality=selected_locality,
                fallback_used=is_fallback,
                output_text=collected_text,
                route=route,
            ),
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def create_embeddings(
        self,
        inputs: list[str],
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> ExecutionResult:
        """Generate embeddings for one or more text inputs."""
        defaults = self._resolve(CAPABILITY_EMBEDDING, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_EMBEDDING)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(
            effective_model=effective_model, app=app, capability=CAPABILITY_EMBEDDING
        )
        selection = _resolve_planner_selection(planner_model, CAPABILITY_EMBEDDING, policy_preset)

        # PR B follow-up: apply the same app-ref refusal gate as
        # TTS / transcription / chat. Embeddings was the lone surface
        # where ``model='nomic-...', app='private-app', policy=None``
        # plus a planner outage could still silently route to cloud.
        # The refusal is the same shape: with no explicit policy AND
        # an app-scoped request AND no planner selection, raise so
        # the caller picks ``local_only`` / ``cloud_first`` or fixes
        # planner connectivity.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )

        routing_policy = _resolve_routing_policy(defaults)

        local_available = self._can_local(effective_model, CAPABILITY_EMBEDDING)
        cloud_available = _cloud_available(defaults)
        # Same explicit local-only / private cloud-disable as the
        # other dispatchers: even if cloud creds are present in env,
        # ``policy='local_only'`` blocks promotion to cloud.
        if _is_local_only_policy(policy):
            cloud_available = False
        locality, is_fallback = _select_locality_for_capability(
            routing_policy,
            local_available=local_available,
            cloud_available=cloud_available,
            capability=CAPABILITY_EMBEDDING,
        )

        route = _route_metadata_from_selection(
            selection, locality, is_fallback, model_name=effective_model, capability=CAPABILITY_EMBEDDING
        )

        if locality == LOCALITY_CLOUD:
            assert defaults.cloud_profile is not None
            # PR B follow-up: when ``app=`` was explicit (or ``model``
            # was already an app ref), cloud dispatch must run under
            # the app identity, not the resolved underlying model.
            # Otherwise the planner sees ``@app/foo/embeddings`` but
            # the hosted call goes out under ``nomic-...`` and the
            # server can't apply the app's quota / policy / billing.
            cloud_model = _execution_model_for_cloud_dispatch(
                requested_model=model,
                effective_model=effective_model,
                planner_model=planner_model,
                app=app,
            )
            result = await self._cloud_embed(inputs, cloud_model, defaults.cloud_profile, is_fallback)
            result.route = route
            return result

        t0 = time.monotonic()
        result = await self._local_embed(inputs, effective_model, is_fallback)
        latency_ms = (time.monotonic() - t0) * 1000
        result.route = route

        _upload_benchmark_async(
            model=effective_model,
            capability=CAPABILITY_EMBEDDING,
            engine=route.execution.engine if route.execution else None,
            policy_preset=policy_preset,
            latency_ms=latency_ms,
        )

        return result

    async def _cloud_embed(
        self,
        inputs: list[str],
        model: str,
        profile: CloudProfile,
        fallback_used: bool = False,
    ) -> ExecutionResult:
        """Dispatch embeddings to cloud."""
        from octomil.embeddings import embed

        api_key = os.environ.get(profile.api_key_env, "")
        base_url = _platform_api_base_url(profile)
        if not api_key:
            raise RuntimeError(
                f"Cloud embedding requires {profile.api_key_env} to be set.\n\n"
                f"Export {profile.api_key_env} or configure a cloud profile."
            )

        result = embed(
            server_url=base_url,
            api_key=api_key,
            model_id=model,
            input=inputs if len(inputs) > 1 else inputs[0],
        )

        dims = len(result.embeddings[0]) if result.embeddings else 0
        return ExecutionResult(
            id=f"emb_{uuid.uuid4().hex[:12]}",
            model=model,
            capability=CAPABILITY_EMBEDDING,
            locality=LOCALITY_CLOUD,
            fallback_used=fallback_used,
            embeddings=result.embeddings,
            dimensions=dims,
            usage={
                "input_tokens": result.usage.prompt_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        )

    async def _local_embed(self, inputs: list[str], model: str, fallback_used: bool = False) -> ExecutionResult:
        """Dispatch embeddings to a local runtime."""
        from octomil.runtime.core.registry import ModelRuntimeRegistry

        registry = ModelRuntimeRegistry.shared()
        runtime = registry.resolve(model)
        if runtime is None:
            raise RuntimeError(f"No local runtime found for embedding model '{model}'.")

        embed_fn = getattr(runtime, "embed", None) or getattr(runtime, "create_embeddings", None)
        if embed_fn is None:
            raise RuntimeError(f"Local runtime for embedding model '{model}' does not expose an embedding interface.")

        maybe_result = embed_fn(inputs)
        if hasattr(maybe_result, "__await__"):
            maybe_result = await maybe_result

        if hasattr(maybe_result, "embeddings"):
            all_vectors = maybe_result.embeddings
            usage_obj = getattr(maybe_result, "usage", None)
            total_tokens = getattr(usage_obj, "total_tokens", 0) if usage_obj is not None else 0
        else:
            all_vectors = maybe_result
            total_tokens = sum(len(text.split()) for text in inputs)

        if not isinstance(all_vectors, list):
            raise RuntimeError(f"Local embedding runtime for '{model}' returned an invalid embedding result.")

        dims = len(all_vectors[0]) if all_vectors and all_vectors[0] else 0
        return ExecutionResult(
            id=f"emb_{uuid.uuid4().hex[:12]}",
            model=model,
            capability=CAPABILITY_EMBEDDING,
            locality=LOCALITY_ON_DEVICE,
            fallback_used=fallback_used,
            embeddings=all_vectors,
            dimensions=dims,
            usage={"input_tokens": total_tokens, "total_tokens": total_tokens},
        )

    # ------------------------------------------------------------------
    # Audio Transcription
    # ------------------------------------------------------------------

    async def transcribe_audio(
        self,
        audio_data: bytes,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
        language: Optional[str] = None,
    ) -> ExecutionResult:
        """Transcribe audio to text."""
        defaults = self._resolve(CAPABILITY_TRANSCRIPTION, model=model, policy=policy, app=app)
        effective_model = defaults.model
        if not effective_model:
            raise _no_model_error(CAPABILITY_TRANSCRIPTION)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(
            effective_model=effective_model, app=app, capability=CAPABILITY_TRANSCRIPTION
        )
        selection = _resolve_planner_selection(planner_model, CAPABILITY_TRANSCRIPTION, policy_preset)

        # PR B: same app-ref refusal + local-only forcing as TTS.
        _enforce_app_ref_routing_policy(
            requested_model=model or effective_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )

        routing_policy = _resolve_routing_policy(defaults)
        # Routing is satisfied if a backend is already staged OR if the
        # planner emitted a preparable sdk_runtime candidate the kernel
        # can materialize. BUT if the planner emits an *unpreparable*
        # prepare_required=True candidate (synthetic plan: no digest/url,
        # traversal, etc.), the SDK cannot honor the plan even if a
        # backend happens to be importable on disk — that staged-backend
        # case used to slip past via the `or` short-circuit and crash in
        # prepare(). The unpreparable-candidate veto fires first.
        if self._local_candidate_is_unpreparable(selection):
            local_available = False
        else:
            local_available = self._has_local_transcription_backend(effective_model) or (
                self._can_prepare_local_transcription(effective_model, selection)
            )
        cloud_available = _cloud_available(defaults)
        if _is_local_only_policy(policy):
            cloud_available = False
        locality, is_fallback = _select_locality_for_capability(
            routing_policy,
            local_available=local_available,
            cloud_available=cloud_available,
            capability=CAPABILITY_TRANSCRIPTION,
        )

        route = _route_metadata_from_selection(
            selection, locality, is_fallback, model_name=effective_model, capability=CAPABILITY_TRANSCRIPTION
        )

        if locality == LOCALITY_CLOUD:
            assert defaults.cloud_profile is not None
            # PR B follow-up: send the app ref to cloud when ``app=``
            # was explicit so the server keeps app-scoping. Local
            # ``effective_model`` keeps driving local backend
            # resolution below.
            cloud_model = _execution_model_for_cloud_dispatch(
                requested_model=model,
                effective_model=effective_model,
                planner_model=planner_model,
                app=app,
            )
            result = await self._cloud_transcribe(
                audio_data, cloud_model, defaults.cloud_profile, language, is_fallback
            )
            result.route = route
            return result

        # PR 10: prepare transcription artifact (whisper.cpp model file)
        # before backend load, mirroring the TTS path. The downstream
        # _resolve_local_transcription_backend threads model_dir into
        # engine.create_backend so whisper.cpp loads the prepared file
        # instead of triggering pywhispercpp's own download path.
        prepared_dir = self._prepare_local_transcription_artifact(selection)

        t0 = time.monotonic()
        result = await self._local_transcribe(
            audio_data, effective_model, language, is_fallback, prepared_model_dir=prepared_dir
        )
        latency_ms = (time.monotonic() - t0) * 1000
        result.route = route

        _upload_benchmark_async(
            model=effective_model,
            capability=CAPABILITY_TRANSCRIPTION,
            engine=route.execution.engine if route.execution else None,
            policy_preset=policy_preset,
            latency_ms=latency_ms,
        )

        return result

    async def _cloud_transcribe(
        self,
        audio_data: bytes,
        model: str,
        profile: CloudProfile,
        language: Optional[str],
        fallback_used: bool = False,
    ) -> ExecutionResult:
        """Dispatch audio transcription to a hosted OpenAI-compatible endpoint."""
        import httpx

        api_key = os.environ.get(profile.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Cloud transcription requires {profile.api_key_env} to be set.\n\n"
                f"Export {profile.api_key_env} or configure a cloud profile."
            )

        data = {"model": model}
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(base_url=_openai_base_url(profile), headers=headers, timeout=120.0) as client:
            response = await client.post(
                "/audio/transcriptions",
                data=data,
                files={"file": ("audio.wav", audio_data, "audio/wav")},
            )
            response.raise_for_status()
            payload = response.json()

        return ExecutionResult(
            id=f"txn_{uuid.uuid4().hex[:12]}",
            model=model,
            capability=CAPABILITY_TRANSCRIPTION,
            locality=LOCALITY_CLOUD,
            fallback_used=fallback_used,
            output_text=payload.get("text", ""),
            segments=payload.get("segments"),
            raw=payload,
        )

    async def _local_transcribe(
        self,
        audio_data: bytes,
        model: str,
        language: Optional[str],
        fallback_used: bool = False,
        *,
        prepared_model_dir: Optional[str] = None,
    ) -> ExecutionResult:
        """Dispatch audio transcription to a local Whisper-compatible backend.

        ``prepared_model_dir`` is the directory PrepareManager materialized
        the artifact under. When set, the resolver passes it as
        ``model_dir`` to the engine so whisper.cpp loads the prepared file
        instead of going through pywhispercpp's own download path.
        """
        backend = self._resolve_local_transcription_backend(model, prepared_model_dir=prepared_model_dir)
        if backend is None:
            raise RuntimeError(f"No local transcription runtime found for model '{model}'.")

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            os.write(fd, audio_data)
            os.close(fd)
            result = await asyncio.to_thread(backend.transcribe, tmp_path)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        segments = result.get("segments") if isinstance(result, dict) else None
        return ExecutionResult(
            id=f"txn_{uuid.uuid4().hex[:12]}",
            model=model,
            capability=CAPABILITY_TRANSCRIPTION,
            locality=LOCALITY_ON_DEVICE,
            fallback_used=fallback_used,
            output_text=text,
            segments=segments,
            raw=result if isinstance(result, dict) else None,
        )

    # ------------------------------------------------------------------
    # Audio Speech (TTS)
    # ------------------------------------------------------------------

    async def synthesize_speech(
        self,
        *,
        model: str,
        input: str,
        voice: Optional[str] = None,
        response_format: str = "wav",
        speed: float = 1.0,
        app: Optional[str] = None,
        policy: Optional[str] = None,
    ) -> Any:
        """Routed TTS synthesis. Returns a ``SpeechResponse``.

        Resolves the model ref through the same planner pipeline as the
        other capabilities. Local execution dispatches to ``SherpaTtsEngine``;
        cloud execution wraps the existing
        ``octomil.hosted.HostedSpeech.create`` so the kernel does not
        duplicate hosted request shaping or error handling.
        """
        from octomil.audio.speech import SpeechResponse, SpeechRoute

        if not input or not input.strip():
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message="`input` must be a non-empty string.",
            )

        requested_model = model
        defaults = self._resolve(CAPABILITY_TTS, model=requested_model, policy=policy, app=app)
        effective_model = defaults.model or requested_model
        if not effective_model:
            raise _no_model_error(CAPABILITY_TTS)

        policy_preset = defaults.policy_preset or "local_first"
        planner_model = _planner_model_for_request(effective_model=effective_model, app=app, capability=CAPABILITY_TTS)
        selection = _resolve_planner_selection(planner_model, CAPABILITY_TTS, policy_preset)
        runtime_model = _runtime_model_for_selection(selection, effective_model)

        planner_policy = _tts_policy_from_selection(selection)
        if planner_policy:
            policy_preset = planner_policy
            defaults.policy_preset = planner_policy
            defaults.inline_policy = None

        # PR B: refuse to silently route an @app/... ref to cloud when
        # planner resolution failed AND the caller didn't pass an
        # explicit policy. Without this, an outage of the planner +
        # an app whose intended policy is "private" lets the SDK
        # cold-route to cloud, surfacing a confusing 403 telling the
        # caller to use a local SDK that they're already using.
        # When the caller explicitly passed ``policy='local_only'``
        # or ``policy='private'``, they've expressed their intent
        # directly — honour it by forcing cloud_available=False
        # below so a planner outage cannot leak a local-only request
        # to the network.
        _enforce_app_ref_routing_policy(
            requested_model=requested_model,
            selection=selection,
            explicit_policy=policy,
            explicit_app=app,
        )

        routing_policy = _resolve_routing_policy(defaults)
        # Local routing is satisfied if the artifact is already staged OR
        # the planner emitted a preparable sdk_runtime candidate the
        # kernel can materialize via PrepareManager. The unpreparable-
        # candidate veto fires first: when the planner says "prepare this
        # artifact" but the metadata is structurally rejected by
        # ``can_prepare`` (synthetic plan), local must be unavailable
        # regardless of whether a backend happens to be staged on disk.
        # Otherwise local_first would still pick local and crash in
        # prepare() instead of falling back to cloud.
        if self._local_candidate_is_unpreparable(selection):
            local_available = False
        else:
            local_available = self._has_local_tts_backend(runtime_model) or self._can_prepare_local_tts(
                runtime_model, selection
            )
        cloud_available = _cloud_available(defaults)
        # PR B: explicit private / local_only policies force
        # cloud_available=False so a planner outage cannot leak the
        # request to a hosted backend even if cloud creds happen to
        # be present in the environment.
        if _is_local_only_policy(policy):
            cloud_available = False

        # Policy gating mirrors transcription. local_only never falls back.
        try:
            locality, is_fallback = _select_locality_for_capability(
                routing_policy,
                local_available=local_available,
                cloud_available=cloud_available,
                capability=CAPABILITY_TTS,
            )
        except RuntimeError as exc:
            if "local" in str(exc).lower():
                raise _local_tts_runtime_unavailable(runtime_model) from exc
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=str(exc),
            ) from exc

        if locality == LOCALITY_ON_DEVICE and not local_available:
            # Don't try to load — fail closed with the canonical error.
            raise _local_tts_runtime_unavailable(runtime_model)

        speech_route = SpeechRoute(
            locality="on_device" if locality == LOCALITY_ON_DEVICE else "cloud",
            engine="sherpa-onnx" if locality == LOCALITY_ON_DEVICE else None,
            policy=policy_preset,
            fallback_used=is_fallback,
        )

        if locality == LOCALITY_CLOUD:
            assert defaults.cloud_profile is not None, "cloud locality requires cloud profile"
            # PR B follow-up: when ``app=`` is explicit, send the
            # synthesized ``@app/<app>/tts`` ref to cloud so the
            # server applies the app's quota / policy / billing.
            # The pre-existing ``@app/...`` pass-through behaviour
            # for callers who put the ref in ``model=`` keeps working.
            cloud_model = _execution_model_for_cloud_dispatch(
                requested_model=requested_model,
                effective_model=runtime_model,
                planner_model=planner_model,
                app=app,
            )
            t0 = time.monotonic()
            cloud_result = await self._cloud_synthesize_speech(
                cloud_model,
                input,
                voice,
                response_format,
                speed,
                defaults.cloud_profile,
            )
            return SpeechResponse(
                audio_bytes=cloud_result["audio_bytes"],
                content_type=cloud_result["content_type"],
                format=cloud_result.get("format") or response_format,
                model=cloud_result.get("model") or runtime_model,
                provider=cloud_result.get("provider"),
                voice=voice,
                sample_rate=cloud_result.get("sample_rate"),
                duration_ms=cloud_result.get("duration_ms"),
                latency_ms=(time.monotonic() - t0) * 1000.0,
                route=speech_route,
                billed_units=cloud_result.get("billed_units"),
                unit_kind=cloud_result.get("unit_kind"),
            )

        # Local branch — sherpa-onnx.
        if response_format.lower() != "wav":
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"format_not_supported_for_local_tts: local sherpa-onnx "
                    f"returns WAV. Got response_format='{response_format}'. "
                    "Cloud-routed apps can request other formats; local apps "
                    "should request 'wav' until local transcoding ships."
                ),
            )

        self._validate_local_voice(runtime_model, voice)
        prepared_model_dir = self._prepare_local_tts_artifact(selection)
        backend = self._resolve_local_tts_backend(
            runtime_model,
            prepared_model_dir=prepared_model_dir,
        )
        if backend is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(f"local_tts_runtime_unavailable: could not load sherpa backend for model '{runtime_model}'."),
            )

        t0 = time.monotonic()
        local_result = await asyncio.to_thread(backend.synthesize, input, voice, speed)
        latency_ms = (time.monotonic() - t0) * 1000.0

        # Local execution: never write cloud_usage_logs / increment cloud quotas.
        # Route telemetry only.
        return SpeechResponse(
            audio_bytes=local_result["audio_bytes"],
            content_type=local_result.get("content_type", "audio/wav"),
            format=local_result.get("format", "wav"),
            model=runtime_model,
            provider=None,
            voice=voice or local_result.get("voice"),
            sample_rate=local_result.get("sample_rate"),
            duration_ms=local_result.get("duration_ms"),
            latency_ms=latency_ms,
            route=speech_route,
            billed_units=None,
            unit_kind=None,
        )

    async def _cloud_synthesize_speech(
        self,
        model: str,
        text: str,
        voice: Optional[str],
        response_format: str,
        speed: float,
        profile: CloudProfile,
    ) -> dict[str, Any]:
        """Wrap ``octomil.hosted.HostedSpeech.create`` for the cloud branch.

        Internal-only use of ``HostedClient``; never expose this in
        quickstarts. The kernel reuses the hosted speech transport so we
        don't duplicate request shaping or error handling.
        """
        from octomil.hosted.client import HostedClient

        api_key = os.environ.get(profile.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Cloud TTS requires {profile.api_key_env} to be set.")
        base_url = _openai_base_url(profile)
        client = HostedClient(api_key=api_key, base_url=base_url)
        # HostedSpeech.create is sync today; thread it.
        resp = await asyncio.to_thread(
            client.audio.speech.create,
            model=model,
            input=text,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )
        return {
            "audio_bytes": resp.audio_bytes,
            "content_type": resp.content_type,
            "format": response_format,
            "model": getattr(resp, "model", None),
            "provider": resp.provider,
            "billed_units": resp.billed_units,
            "unit_kind": resp.unit_kind,
        }

    def _has_local_tts_backend(self, model: str) -> bool:
        """Return True iff the local TTS engine + artifact are both ready now.

        Used for the *no-prepare* branch (engine manages its own bytes,
        planner emitted no candidate, or ``prepare_required=False``). The
        clean-device first-run case is admitted separately by
        :meth:`_can_prepare_local_tts` so that path does not require
        pre-staged files.
        """
        try:
            from octomil.runtime.engines.sherpa import is_sherpa_tts_model_staged

            return is_sherpa_tts_model_staged(model)
        except Exception:
            return False

    def _local_candidate_is_unpreparable(self, selection: Optional[Any]) -> bool:
        """Return True iff the planner's local sdk_runtime candidate is
        ``prepare_required=True`` AND ``PrepareManager.can_prepare`` rejects
        it (synthetic plan: missing digest/url, traversal, dot path, etc.).

        This is the *hard veto* the reviewer flagged. ``_has_local_*_backend``
        only checks whether a runtime is importable / files are staged;
        when the planner says "prepare this artifact" but the metadata is
        unpreparable, the SDK cannot honor that plan. Letting an
        already-staged or runtime-importable model win local routing in
        that case ignores the planner's intent and crashes at first
        prepare(). When this veto fires, the kernel must mark local
        unavailable regardless of staging so local_first falls back to
        cloud.

        ``prepare_required=False`` candidates are never blocking — those
        engines manage their own bytes (e.g. ollama). Same when the
        planner emits no local candidate at all.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None:
            return False
        if not getattr(candidate, "prepare_required", False):
            return False

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        return not manager.can_prepare(candidate)

    def _can_prepare_local_transcription(self, model: str, selection: Optional[Any]) -> bool:
        """Mirror of ``_can_prepare_local_tts`` for the transcription path.

        Returns True only when a transcription backend is importable for
        ``model`` AND ``PrepareManager.can_prepare`` says the planner
        candidate has enough metadata to actually succeed. Without this
        dry-run, ``local_first`` could commit to local on a synthetic
        candidate (prepare_required=True with no digest/url) and fail in
        prepare instead of falling back to cloud.

        We don't have a single runtime-availability helper for
        transcription engines (whisper.cpp, sherpa-ASR, etc. each detect
        themselves through the engine registry), so the import check is
        the registry walk inside ``_resolve_local_transcription_backend``
        — we simply run that resolver against an empty prepared dir and
        check whether *any* engine claims support. The resolver only
        returns when the backend exposes ``transcribe``; that's the same
        runtime check the no-prepare path uses.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None or not getattr(candidate, "prepare_required", False):
            return False

        # Runtime availability: at least one engine in the registry must
        # produce a transcription backend for ``model`` (any model_dir).
        try:
            backend = self._resolve_local_transcription_backend(model)
            if backend is None:
                return False
        except Exception:
            return False

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        return manager.can_prepare(candidate)

    def _can_prepare_local_tts(self, model: str, selection: Optional[Any]) -> bool:
        """Return True iff sherpa-onnx is importable, the model id is known,
        and ``PrepareManager.can_prepare`` confirms the planner candidate
        has enough metadata to succeed (digest + download_urls + a
        non-disabled policy + at most one required file).

        ``prepare_required`` alone is not enough: the server can emit
        synthetic prepare metadata with no urls/digest, in which case
        committing to local routing would fail at first prepare instead of
        falling back to cloud. This dry-run inspection is pure — it never
        touches disk or network.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None or not getattr(candidate, "prepare_required", False):
            return False
        try:
            from octomil.runtime.engines.sherpa import is_sherpa_tts_runtime_available

            if not is_sherpa_tts_runtime_available(model):
                return False
        except Exception:
            return False

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        return manager.can_prepare(candidate)

    def prepare(
        self,
        *,
        model: str,
        capability: str = "tts",
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> Any:
        """Resolve a planner candidate for ``model`` and pre-warm its artifact.

        Public, caller-driven equivalent of the implicit prepare path. Calls
        :meth:`PrepareManager.prepare` with ``mode=PrepareMode.EXPLICIT`` so
        candidates whose ``prepare_policy='explicit_only'`` succeed when
        invoked through this method.

        Supported capabilities: ``"tts"`` and ``"transcription"`` today.
        Both have dispatch paths that thread the prepared ``artifact_dir``
        into their backend:

        - ``tts`` -> ``SherpaTtsEngine.create_backend(model_dir=...)``.
        - ``transcription`` -> ``_WhisperBackend`` honors injected
          ``model_dir`` and prefers PrepareManager's ``<dir>/artifact``
          sentinel before falling back to ``.bin`` / ``.gguf`` / ``.ggml``.

        Embedding and chat (and responses, which routes through chat)
        will be added one at a time as their adapters learn to accept a
        ``model_dir`` kwarg — exposing ``prepare`` for them now would be
        a false success: the bytes would land on disk and the next
        inference call would still cold-start through the engine's own
        lookup. PR 10c added the kernel-side threading for chat into
        ``MLXBackend`` and ``LlamaCppBackend``, but the capability
        remains gated until (a) the public ``client.responses.create``
        facade goes through the kernel (or threads ``model_dir``
        itself), and (b) PrepareManager grows snapshot/manifest support
        for multi-file MLX artifacts. The current set is the source of
        truth; the lifecycle support fixture in octomil-contracts cites
        the e2e test that proves each cell's claim.

        Returns a :class:`PrepareOutcome`. Raises :class:`OctomilError` if
        the capability is not yet wired, the planner emits no preparable
        local candidate, or :class:`PrepareManager.prepare` rejects the
        metadata.
        """
        if capability not in _PREPAREABLE_CAPABILITIES:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"client.prepare() does not yet support capability {capability!r}. "
                    f"Supported today: {sorted(_PREPAREABLE_CAPABILITIES)}. "
                    f"Other capabilities will be added once their backends thread the "
                    f"prepared model_dir into dispatch."
                ),
            )

        defaults = self._resolve(capability, model=model, policy=policy, app=app)
        effective_model = defaults.model or model
        if not effective_model:
            raise _no_model_error(capability)

        policy_preset = defaults.policy_preset or "local_first"
        selection = _resolve_planner_selection(effective_model, capability, policy_preset)
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None:
            # PR C: fall back to a static offline recipe for canonical
            # local models (Kokoro etc.). The recipe produces the same
            # ``RuntimeCandidatePlan`` shape the planner would emit,
            # so the rest of the prepare pipeline runs unchanged. This
            # makes ``octomil prepare kokoro-82m --capability tts``
            # work without planner / network — the happy path
            # one-liner the embedded TTS bootstrap needs.
            #
            # The recipe table is deliberately narrow (canonical
            # public bundles only); models that need auth or private
            # CDNs hit the original "planner unavailable" error so we
            # never silently substitute a public mirror for a private
            # artifact.
            from octomil.runtime.lifecycle.static_recipes import static_recipe_candidate

            candidate = static_recipe_candidate(effective_model, capability)
            if candidate is None:
                raise OctomilError(
                    code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                    message=(
                        f"prepare: planner returned no local sdk_runtime candidate for "
                        f"model={effective_model!r} capability={capability!r} and the SDK has "
                        f"no static offline recipe for it. Either the model is cloud-only, "
                        f"the planner is offline, or this is a private artifact that requires "
                        f"OCTOMIL_SERVER_KEY auth. Set OCTOMIL_SERVER_KEY or use a model with "
                        f"a static recipe (e.g. 'kokoro-82m')."
                    ),
                )

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager, PrepareMode

        manager = self._prepare_manager or PrepareManager()
        return manager.prepare(candidate, mode=PrepareMode.EXPLICIT)

    def _prepare_local_transcription_artifact(self, selection: Optional[Any]) -> Optional[str]:
        """Run PrepareManager for the local transcription candidate, if any.

        Symmetric with ``_prepare_local_tts_artifact``: returns the
        prepared ``artifact_dir`` as a string when the planner emits a
        local sdk_runtime candidate with prepare_required=True; returns
        None otherwise so the existing whisper.cpp / sherpa-ASR fallback
        path runs unchanged.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None:
            return None
        if not getattr(candidate, "prepare_required", False):
            return None

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        outcome = manager.prepare(candidate)
        return str(outcome.artifact_dir)

    def _prepare_local_chat_artifact_cached(
        self,
        candidate: Optional[Any],
        cache: dict[tuple, Optional[str]],
    ) -> Optional[str]:
        """Lazy, per-call prepare for *this specific* local candidate.

        Called from inside the local branch of the candidate runner
        with the planner candidate that's about to be attempted, so
        cloud-first plans (and plans whose primary cloud candidate
        succeeds) never trigger prepare. The cache de-dupes redundant
        prepares for the same artifact shape within a single
        ``create_response`` call.

        Cache key
        ~~~~~~~~~
        ``(artifact_id, digest, format, quantization)``. Earlier shapes
        keyed only on ``artifact_id`` / ``model_id``, which let two
        local candidates that shared the logical id but carried
        different digests or formats reuse the first candidate's
        prepared directory — for example, an MLX snapshot candidate
        and a GGUF candidate that both quote ``model_id="gemma3-1b"``
        because ``artifact_id`` was omitted. The second candidate would
        receive a ``model_dir`` containing the wrong bytes. Including
        digest / format / quantization in the key ensures the cache
        only short-circuits when the planner is truly asking for the
        same verified artifact.

        Returns ``None`` (cold-load through the engine's own resolution
        path) when:
          - the candidate is missing or not a local sdk_runtime shape,
          - ``prepare_required`` is False (engine-managed, e.g. ollama),
          - the candidate has no artifact id (synthetic policy
            candidate built without planner metadata).
        ``manager.prepare`` exceptions surface unchanged so the runner
        records the failure and falls back to the next candidate — same
        contract as the TTS / transcription paths.
        """
        if candidate is None:
            return None
        if getattr(candidate, "locality", None) != "local":
            return None
        delivery_mode = getattr(candidate, "delivery_mode", None) or "sdk_runtime"
        if delivery_mode != "sdk_runtime":
            return None
        if not getattr(candidate, "prepare_required", False):
            return None
        artifact = getattr(candidate, "artifact", None)
        artifact_id = getattr(artifact, "artifact_id", None) or getattr(artifact, "model_id", None)
        if not artifact_id:
            return None
        cache_key = (
            artifact_id,
            getattr(artifact, "digest", None),
            getattr(artifact, "format", None),
            getattr(artifact, "quantization", None),
        )
        if cache_key in cache:
            return cache[cache_key]

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        outcome = manager.prepare(candidate)
        prepared = str(outcome.artifact_dir)
        cache[cache_key] = prepared
        return prepared

    def _prepare_local_tts_artifact(self, selection: Optional[Any]) -> Optional[str]:
        """Run :class:`PrepareManager` for the local TTS candidate, if any.

        Returns the prepared ``artifact_dir`` as a string when the planner
        marks the local candidate as ``delivery_mode='sdk_runtime'`` with
        ``prepare_required=True``. Returns ``None`` when no preparation is
        required (engine manages its own artifacts, ``prepare_required=False``,
        or no planner candidate was emitted) — callers fall back to the
        existing ``OCTOMIL_SHERPA_MODELS_DIR`` / ``~/.octomil/models``
        resolution path.

        Errors from PrepareManager (policy violations, contract violations,
        download failures) are surfaced unchanged so users get the actionable
        messages the manager constructs.
        """
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None:
            return None
        if not getattr(candidate, "prepare_required", False):
            return None

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager

        manager = self._prepare_manager or PrepareManager()
        outcome = manager.prepare(candidate)
        return str(outcome.artifact_dir)

    def _resolve_local_tts_backend(
        self,
        model: str,
        *,
        prepared_model_dir: Optional[str] = None,
    ) -> Optional[Any]:
        try:
            from octomil.runtime.engines.sherpa import SherpaTtsEngine

            engine = SherpaTtsEngine()
            backend_kwargs: dict[str, Any] = {}
            if prepared_model_dir:
                # PrepareManager materialized the artifact at this path; load
                # the sherpa-onnx model from there instead of the env/home
                # fallback, so the freshly-prepared bytes are what we run.
                backend_kwargs["model_dir"] = prepared_model_dir
            backend = engine.create_backend(model, **backend_kwargs)
            backend.load_model(model)
            return backend
        except Exception:
            return None

    def _validate_local_voice(self, model: str, voice: Optional[str]) -> None:
        """Pre-flight voice check for local TTS.

        Raises ``OctomilError`` with ``voice_not_supported_for_locality`` when
        the caller passed a voice that isn't in the local model's catalog.
        For cloud, voice mismatches surface post-dispatch via provider 4xx.
        """
        if not voice:
            return
        from octomil.runtime.engines.sherpa import _KOKORO_VOICES, is_sherpa_tts_model

        if not is_sherpa_tts_model(model):
            return
        # Kokoro carries a known catalog. For Piper/VITS the catalog is per-bundle;
        # defer detection to backend until we ship a per-model voices.txt scan.
        if model.lower().startswith("kokoro-") and voice.lower() not in _KOKORO_VOICES:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"voice_not_supported_for_locality: '{voice}' is not in "
                    f"the Kokoro voice catalog. Cloud voices like 'alloy' "
                    f"or 'onyx' are not valid for local Kokoro execution."
                ),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(
        self,
        capability: str,
        *,
        model: Optional[str] = None,
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> ResolvedExecutionDefaults:
        overrides = RequestOverrides(model=model, policy=policy, app_slug=app)
        return resolve_capability_defaults(capability, overrides, self._config_set)

    def _can_local(self, model: str, capability: str) -> bool:
        """Check if a local runtime is available for the given model."""
        if capability == CAPABILITY_TRANSCRIPTION:
            return self._has_local_transcription_backend(model)

        try:
            from octomil.runtime.core.registry import ModelRuntimeRegistry

            registry = ModelRuntimeRegistry.shared()
            runtime = registry.resolve(model)
            if runtime is None:
                return False
            if capability == CAPABILITY_EMBEDDING:
                return hasattr(runtime, "embed") or hasattr(runtime, "create_embeddings")
            return True
        except Exception:
            return False

    def _has_local_transcription_backend(self, model: str) -> bool:
        return self._resolve_local_transcription_backend(model) is not None

    def _resolve_local_transcription_backend(
        self,
        model: str,
        *,
        prepared_model_dir: Optional[str] = None,
    ) -> Optional[Any]:
        """Return a transcription backend for ``model``.

        ``prepared_model_dir`` is threaded into ``engine.create_backend``
        as ``model_dir=...`` so the backend loads from the directory
        PrepareManager just materialized. The whisper.cpp backend honors
        this and skips its own pywhispercpp download path; engines that
        don't recognize the kwarg ignore it (the registry's create_backend
        accepts ``**kwargs``).
        """
        try:
            from octomil.runtime.engines import get_registry

            registry = get_registry()
            backend_kwargs: dict[str, Any] = {}
            if prepared_model_dir:
                backend_kwargs["model_dir"] = prepared_model_dir
            for detection in registry.detect_all(model):
                if not detection.available:
                    continue
                backend = detection.engine.create_backend(model, **backend_kwargs)
                if hasattr(backend, "transcribe"):
                    return backend
        except Exception:
            return None
        return None

    async def _build_router(
        self,
        model: str,
        capability: str,
        defaults: ResolvedExecutionDefaults,
        *,
        planner_selection: Optional[Any] = None,
        prepared_model_dir: Optional[str] = None,
        cloud_execution_model: Optional[str] = None,
    ) -> RouterModelRuntime:
        """Build a RouterModelRuntime for the given model and capability.

        When a planner_selection is provided and recommends a specific engine,
        the local factory tries that engine first before falling back to the
        default registry resolution.

        ``prepared_model_dir`` is threaded into ``engine.create_backend``
        as ``model_dir=...`` for engines that consume prepared bytes
        (mlx-lm, llama.cpp, sherpa-onnx, whisper). Engines that do not
        recognize the kwarg ignore it (each ``create_backend`` accepts
        ``**kwargs``). When ``None`` the engine's own model resolution
        path runs unchanged.

        ``cloud_execution_model`` is the identity to send to hosted
        inference. PR B follow-up: when the caller passed ``app=`` (or
        the request was an ``@app/...`` ref), cloud dispatch must run
        under the app identity even though local backend resolution
        uses the resolved runtime model. ``None`` (default) keeps the
        legacy behaviour of running cloud under ``model``.
        """
        from octomil.runtime.core.registry import ModelRuntimeRegistry

        registry = ModelRuntimeRegistry.shared()

        # If planner says cloud, force cloud by returning None from local factory
        planner_forces_cloud = planner_selection is not None and planner_selection.locality == "cloud"
        planner_engine = (
            planner_selection.engine
            if planner_selection is not None and planner_selection.locality == "local"
            else None
        )
        backend_kwargs: dict[str, Any] = {}
        if prepared_model_dir:
            backend_kwargs["model_dir"] = prepared_model_dir

        def local_factory(hint: str):
            if planner_forces_cloud:
                return None

            # If planner recommends a specific local engine, try it first
            if planner_engine:
                try:
                    from octomil.runtime.engines import get_registry as get_engine_registry

                    engine_registry = get_engine_registry()
                    engine = engine_registry.get_engine(planner_engine)
                    if engine is not None and engine.detect() and engine.name != "echo":
                        from octomil.runtime.core.adapter import InferenceBackendAdapter
                        from octomil.runtime.core.engine_bridge import _infer_tool_call_tier
                        from octomil.runtime.core.types import RuntimeCapabilities

                        backend = engine.create_backend(model, **backend_kwargs)
                        return InferenceBackendAdapter(
                            backend=backend,
                            model_name=model,
                            capabilities=RuntimeCapabilities(
                                tool_call_tier=_infer_tool_call_tier(model),
                                supports_streaming=True,
                            ),
                        )
                except Exception:
                    logger.debug(
                        "Planner-recommended engine %s failed, falling back to registry",
                        planner_engine,
                        exc_info=True,
                    )

            resolved = registry.resolve(model)
            # Never let echo leak into user-facing execution paths
            if resolved is not None:
                backend = getattr(resolved, "_backend", None)
                backend_cls = type(backend).__name__ if backend else ""
                if backend_cls == "EchoBackend":
                    logger.debug("Rejecting echo backend from user-facing execution path")
                    return None
            return resolved

        def cloud_factory(hint: str):
            if defaults.cloud_profile is None:
                return None
            try:
                from octomil.runtime.core.cloud_runtime import CloudModelRuntime

                api_key = os.environ.get(defaults.cloud_profile.api_key_env, "")
                if not api_key:
                    return None
                # When the caller explicitly app-scoped the request,
                # send the synthesized ``@app/<app>/<cap>`` identity
                # to the hosted endpoint so server-side routing /
                # quota / billing match the planner's app
                # resolution. Falls back to the resolved ``model``
                # for non-app callers (legacy behaviour).
                cloud_model = cloud_execution_model or model
                return CloudModelRuntime(
                    base_url=_openai_base_url(defaults.cloud_profile),
                    api_key=api_key,
                    model=cloud_model,
                )
            except Exception:
                return None

        return RouterModelRuntime(
            local_factory=local_factory,
            cloud_factory=cloud_factory,
        )


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _runner_dict_to_planner_candidate(
    selection: Optional[Any],
    candidates: list[dict[str, Any]],
) -> dict[int, Any]:
    """Map each runner-dict in ``candidates`` to its planner candidate.

    ``_selection_candidate_dicts`` produces its dicts in 1:1 order with
    ``selection.candidates`` whenever the planner emitted a non-empty
    list, so we pair by position. The dict identity (``id(c)``) is the
    map key — that survives the post-filter list as long as the dicts
    aren't copied (which they aren't: filter just removes entries).

    Dicts that have no original (the policy-synthesized fallback path
    when the planner emitted nothing) are not mapped; ``cache.get``
    returns ``None`` for those, which the prepare helper treats as
    no-op.
    """
    originals = list(getattr(selection, "candidates", None) or [])
    mapping: dict[int, Any] = {}
    for idx, dict_candidate in enumerate(candidates):
        if idx < len(originals):
            mapping[id(dict_candidate)] = originals[idx]
    return mapping


def _filter_unpreparable_local_candidates(
    kernel: "ExecutionKernel",
    selection: Optional[Any],
    candidates: list[dict[str, Any]],
    *,
    fallback_allowed: bool,
) -> list[dict[str, Any]]:
    """Drop *only* the unpreparable local candidates from the runner list.

    The earlier shape of this helper looked at ``_local_candidate_is_unpreparable``
    (which inspects the *first* local sdk_runtime candidate of the
    selection) and then removed *every* local candidate. That was wrong
    in two ways:

      1. Multi-local plans like ``[bad local mlx, good local llama, cloud]``
         lost the valid llama fallback. The bad mlx candidate poisoned
         the entire local lane.
      2. Plans with ``fallback_allowed=False`` like
         ``[bad local, cloud]`` could be silently promoted to cloud:
         the local primary disappeared and cloud — which the planner
         only emitted as a non-fallback option — became the first
         remaining candidate.

    The fix evaluates each local sdk_runtime candidate independently
    against ``PrepareManager.can_prepare`` and drops only the
    individually unpreparable ones. When a *primary* candidate (index
    0) is unpreparable AND ``fallback_allowed`` is False, the helper
    returns an empty list so the runner surfaces the planner's "no
    runnable candidate" error instead of laundering the request to
    cloud.

    Cloud candidates and engine-managed (``prepare_required=False``)
    local candidates pass through untouched.
    """
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    manager = kernel._prepare_manager or PrepareManager()
    # ``_selection_candidate_dicts`` produces its list 1:1 in order with
    # ``selection.candidates`` whenever the planner emitted a non-empty
    # list, so we can pair them by position. When the dicts came from
    # the policy synthesis path (no planner candidates) the originals
    # list is shorter / empty and the corresponding entries pair to
    # ``None`` — those candidates have no prepare metadata to validate
    # so they pass through unchanged.
    originals = list(getattr(selection, "candidates", None) or [])
    paired: list[tuple[dict[str, Any], Optional[Any]]] = []
    for idx, dict_candidate in enumerate(candidates):
        original = originals[idx] if idx < len(originals) else None
        paired.append((dict_candidate, original))

    def _is_unpreparable_local(candidate_dict: dict[str, Any], original: Optional[Any]) -> bool:
        if candidate_dict.get("locality") != "local":
            return False
        delivery_mode = candidate_dict.get("delivery_mode") or "sdk_runtime"
        if delivery_mode != "sdk_runtime":
            return False
        if original is None:
            # No matching planner plan object — synthetic policy
            # candidate, no prepare metadata to validate.
            return False
        if not getattr(original, "prepare_required", False):
            return False
        try:
            return not manager.can_prepare(original)
        except Exception:
            return True

    primary_unpreparable = bool(paired) and _is_unpreparable_local(*paired[0])
    filtered = [c for c, original in paired if not _is_unpreparable_local(c, original)]

    if primary_unpreparable and not fallback_allowed:
        # Planner forbids fallback and the primary is unrunnable.
        # Raising an actionable OctomilError here is better than
        # returning ``[]`` and letting the runner emit a generic
        # "No runtime available": the latter loses the rejected
        # primary's identity (engine, artifact_id, reason), which
        # is exactly the diagnostic the caller needs to fix their
        # plan or relax fallback_allowed.
        primary_dict, primary_original = paired[0]
        engine = primary_dict.get("engine") or getattr(primary_original, "engine", None) or "?"
        artifact = getattr(primary_original, "artifact", None)
        artifact_id = getattr(artifact, "artifact_id", None) or getattr(artifact, "model_id", None) or "<unknown>"
        reason = primary_dict.get("reason") or getattr(primary_original, "reason", "") or ""
        raise OctomilError(
            code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
            message=(
                f"local_runtime_unavailable: planner emitted a local sdk_runtime candidate "
                f"(engine={engine!r}, artifact={artifact_id!r}) but PrepareManager.can_prepare "
                f"rejected the metadata (synthetic plan: missing digest/url, traversal, disabled "
                f"policy, etc.) and fallback_allowed=False blocks promotion to cloud. "
                f"{('reason=' + reason + '. ') if reason else ''}"
                f"Fix the planner candidate or relax fallback_allowed to admit cloud."
            ),
        )
    return filtered


def _local_sdk_runtime_candidate(selection: Optional[Any]) -> Optional[Any]:
    """Return the first ``locality='local', delivery_mode='sdk_runtime'``
    candidate in ``selection.candidates``, or ``None``.

    Cloud and external-endpoint candidates are skipped; PrepareManager only
    handles SDK-runtime artifacts. The caller still has to check
    ``prepare_required`` before deciding whether to invoke prepare at all.
    """
    if selection is None:
        return None
    candidates = getattr(selection, "candidates", None) or []
    for candidate in candidates:
        if getattr(candidate, "locality", None) != "local":
            continue
        delivery_mode = getattr(candidate, "delivery_mode", None) or "sdk_runtime"
        if delivery_mode != "sdk_runtime":
            continue
        return candidate
    return None


def _tts_policy_from_selection(selection: Any) -> Optional[str]:
    if selection is None:
        return None

    for attr in ("app_resolution", "resolution"):
        resolved = getattr(selection, attr, None)
        raw = getattr(resolved, "routing_policy", None)
        if raw:
            policy = getattr(raw, "value", raw)
            policy = str(policy).strip().lower()
            if policy in {
                "private",
                "local_only",
                "local_first",
                "cloud_first",
                "cloud_only",
                "performance_first",
            }:
                return policy
            if policy == "auto":
                return "auto"
    return None


_LOCAL_ONLY_POLICY_NAMES = frozenset({"private", "local_only"})


def _is_local_only_policy(policy: Optional[str]) -> bool:
    """Return True iff ``policy`` is an explicit local-only preset.

    Used by the chat / TTS / transcription dispatchers to force
    ``cloud_available=False`` when the caller has expressed a hard
    requirement that the request must not leave the device. Both
    ``"private"`` and ``"local_only"`` qualify; the SDK treats
    "private" as the privacy-strict superset of "local_only" (no
    benchmark uploads either).
    """
    if not policy:
        return False
    return policy.strip().lower() in _LOCAL_ONLY_POLICY_NAMES


def _execution_model_for_cloud_dispatch(
    *,
    requested_model: Optional[str],
    effective_model: str,
    planner_model: str,
    app: Optional[str],
) -> str:
    """Return the model id cloud dispatch should send under.

    Reviewer P1 (post PR #454): the planner now sees
    ``@app/<app>/<capability>`` when ``app=`` is explicit, but each
    capability's cloud branch was passing the *resolved* runtime
    model (e.g. ``kokoro-en-v0_19``) to the hosted endpoint. That
    means the server can't apply the app's quota / routing policy /
    billing the planner just blessed — the request reaches hosted
    inference under a different identity than the one routing was
    decided against.

    Resolution rule:

      - ``requested_model`` is already an ``@app/...`` ref → send
        the request as-is so the server keeps app-scoping.
      - ``requested_model`` is concrete BUT ``app=`` was explicit
        → send the synthesized ``@app/<app>/<capability>`` ref
        (the same one the planner saw) so cloud dispatch agrees
        with the routing decision.
      - otherwise → send the concrete ``effective_model``. Same
        behaviour as before for non-app callers.
    """
    if requested_model is not None and isinstance(requested_model, str) and requested_model.startswith("@app/"):
        return requested_model
    if app and planner_model.startswith("@app/"):
        return planner_model
    return effective_model


def _planner_visible_capability(capability: str) -> str:
    """Map an internal capability constant to the planner-visible name.

    The internal ``CAPABILITY_CHAT`` constant is ``"chat"``, but the
    planner endpoint and server-side routing model both speak of
    that public surface as ``"responses"``.
    ``RuntimePlanner.resolve`` already runs incoming capability
    strings through ``_PLANNER_CAPABILITY_MAP`` for its outbound
    request — but when the SDK *synthesizes* an app ref like
    ``@app/<app>/<capability>`` locally, that synthesized string IS
    the canonical app ref the server keys quotas / billing / app
    metadata against. If we synthesized ``@app/eternum/chat`` and
    the server only knows ``@app/eternum/responses``, the app
    resolution would fall apart even though the planner endpoint
    received ``capability="responses"``.

    Aligning both halves on the public name keeps the planner's
    parsed capability and the server-visible app ref in agreement.
    Other capabilities (tts, transcription, embedding) round-trip
    unchanged because the planner map only rewrites
    ``chat → responses``.
    """

    return _PLANNER_CAPABILITY_MAP.get(capability, capability)


def _planner_model_for_request(
    *,
    effective_model: str,
    app: Optional[str],
    capability: str,
) -> str:
    """Synthesize an ``@app/<slug>/<canonical-capability>`` ref when
    ``app=`` is set.

    Reviewer P1 on PR #454: the public facade exposes ``app=`` so a
    caller can do ``client.audio.speech.create(model='kokoro-82m',
    app='eternum')``. Previously that ``app`` value reached
    ``ResolvedExecutionDefaults`` but never propagated into the
    planner — ``RuntimePlanner.resolve()`` only sets ``app_slug`` by
    parsing an ``@app/...`` model ref. The result was that the
    request bypassed the app's routing policy entirely AND fell
    through the app-ref refusal gate (``requested_model`` was a
    concrete model id, not an ``@app/`` string).

    The synthesis uses the *planner-visible* capability name (so
    chat → ``@app/<app>/responses``, not ``@app/<app>/chat``) so the
    same string the planner endpoint receives also identifies the
    app on the server side for cloud dispatch. Otherwise
    ``RuntimePlanner.resolve`` would parse our synthesized
    ``@app/eternum/chat`` and the cloud branch would dispatch under
    a capability the server doesn't recognize.
    """
    if not app:
        return effective_model
    if isinstance(effective_model, str) and effective_model.startswith("@app/"):
        return effective_model
    canonical_capability = _planner_visible_capability(capability)
    return f"@app/{app}/{canonical_capability}"


def _enforce_app_ref_routing_policy(
    *,
    requested_model: str,
    selection: Optional[Any],
    explicit_policy: Optional[str],
    explicit_app: Optional[str] = None,
) -> None:
    """Refuse to silently cloud-route an ``@app/...`` ref on planner outage.

    Reviewer P1: when the caller asks for an app ref like
    ``@app/eternum/tts`` and the planner is unreachable (auth
    misconfig, network partition, Ren'Py / sandboxed embed without
    an HTTP transport), the SDK previously fell through to the
    routing-policy default (typically ``local_first``), which can
    silently land the request on a hosted provider. That's the worst
    DX shape: a developer who deliberately chose a private app gets
    a 403 from cloud telling them to use a local SDK that they're
    already using.

    This helper preserves caller intent:

      - When ``explicit_policy`` is ``"private"`` / ``"local_only"``
        — caller has expressed a hard local requirement. We let the
        request proceed; ``cloud_available`` is forced False
        downstream by :func:`_is_local_only_policy` so the planner
        outage cannot leak it to cloud.
      - When ``explicit_policy`` is some other value (``cloud_first``,
        ``performance_first``, etc.) — caller has accepted that cloud
        is allowed. We let the request proceed unchanged.
      - When ``explicit_policy`` is ``None`` AND the request is an
        app ref AND the planner returned no selection — the SDK
        cannot know the app's intended routing policy. Raising here
        is better than guessing: caller should either fix the
        planner / auth or pass an explicit ``policy=`` so we know
        what they want.

    Non-app refs (concrete model ids like ``kokoro-en-v0_19``) skip
    this check entirely; the caller has already named the artifact
    they want and there's no app-side policy to consult.
    """
    if explicit_policy is not None:
        return
    is_app_scoped = (isinstance(requested_model, str) and requested_model.startswith("@app/")) or bool(explicit_app)
    if not is_app_scoped:
        return
    if selection is not None:
        # Planner answered. Even if it returned a synthetic cloud
        # fallback, the request has a planner-blessed path.
        return
    descriptor = requested_model
    if explicit_app and isinstance(requested_model, str) and not requested_model.startswith("@app/"):
        descriptor = f"{requested_model!r} app={explicit_app!r}"
    raise OctomilError(
        code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
        message=(
            f"Could not resolve app routing policy for {descriptor}; "
            "the runtime planner is unavailable (network, auth, or planner-import "
            "failure). The SDK refuses to silently fall back to cloud for "
            "app-scoped requests because that surfaces a confusing 403 to "
            "local-first callers. Pass policy='local_only' to force local, "
            "policy='cloud_first' to allow cloud, or fix planner connectivity / "
            "OCTOMIL_SERVER_KEY auth."
        ),
    )


def _local_tts_runtime_unavailable(model: str) -> OctomilError:
    return OctomilError(
        code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
        message=(
            "local_tts_runtime_unavailable: sherpa-onnx is not installed "
            "or the requested model is not staged. Install sherpa-onnx "
            f"and stage '{model}' under "
            "$OCTOMIL_SHERPA_MODELS_DIR or ~/.octomil/models/sherpa/."
        ),
    )


def _no_model_error(capability: str) -> RuntimeError:
    return RuntimeError(
        f"No default model configured for {capability}.\n\n"
        f"Pass --model, run `octomil models`, or add .octomil.toml:\n\n"
        f"[capabilities.{capability}]\n"
        f'model = "your-model-name"'
    )


def _extract_usage(response: RuntimeResponse) -> dict[str, int]:
    usage: dict[str, int] = {}
    if hasattr(response, "usage") and response.usage:
        u = response.usage
        if hasattr(u, "input_tokens"):
            usage["input_tokens"] = u.input_tokens
        if hasattr(u, "output_tokens"):
            usage["output_tokens"] = u.output_tokens
        if hasattr(u, "total_tokens"):
            usage["total_tokens"] = u.total_tokens
        elif "input_tokens" in usage and "output_tokens" in usage:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _chat_messages_to_runtime_messages(messages: list[dict[str, str]]) -> list[RuntimeMessage]:
    runtime_messages: list[RuntimeMessage] = []
    for message in messages:
        raw_role = message.get("role", MessageRole.USER.value)
        try:
            role = MessageRole(raw_role)
        except ValueError:
            role = MessageRole.USER
        runtime_messages.append(
            RuntimeMessage(
                role=role,
                parts=[RuntimeContentPart.text_part(message.get("content", ""))],
            )
        )
    return runtime_messages
