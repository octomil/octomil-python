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

# Capabilities whose dispatch path can construct + cache a backend
# ahead of first inference — i.e., where ``client.warmup()`` produces
# real first-call savings instead of a false-success. A capability
# enters this set only when both:
#   (a) it's already in ``_PREPAREABLE_CAPABILITIES`` (prepare bytes
#       on disk first), AND
#   (b) the kernel's local resolver consults the warmup cache before
#       calling ``engine.create_backend`` again.
# Today: tts + transcription. Chat / responses join once their
# OctomilResponses bypass and MLX snapshot materialization gates
# clear (same blockers as prepare).
_WARMUPABLE_CAPABILITIES = frozenset({CAPABILITY_TTS, CAPABILITY_TRANSCRIPTION})


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
class WarmupOutcome:
    """Result of :meth:`ExecutionKernel.warmup`.

    ``client.warmup(model, capability)`` runs prepare (so the artifact
    bytes are on disk), constructs the local backend with the prepared
    ``model_dir``, calls ``backend.load_model``, and caches the
    instance so the next inference call reuses it. The outcome reports
    everything a caller needs to confirm warmup happened *and* drove
    real work — first-call latency savings only show up when
    ``backend_loaded`` is True.

    Fields
    ~~~~~~
    - ``capability`` / ``model``: the warmed cell.
    - ``prepare_outcome``: the underlying :class:`PrepareOutcome`. The
      same fields the prepare CLI prints (artifact_id, artifact_dir,
      cached, files map).
    - ``backend_loaded``: True when ``engine.create_backend`` +
      ``backend.load_model`` both succeeded; False when warmup ran
      but a downstream backend constructor refused (engine missing on
      this host, runtime imports failed, etc.) — prepare bytes are
      still on disk, just not loaded.
    - ``latency_ms``: wall time for the full prepare + load loop.
    """

    capability: str
    model: str
    prepare_outcome: Any
    backend_loaded: bool
    latency_ms: float


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
        # PR 11: warmup cache. ``client.warmup(model, capability)`` runs
        # prepare and then constructs + load_model's the local backend
        # so the next inference dispatch can reuse it without paying
        # cold-start latency. Keyed by ``_warmup_cache_key`` —
        # ``(capability, runtime_model, digest, format, quantization)``
        # — so distinct artifact identities (different versions, MLX
        # snapshot vs GGUF, etc.) sharing the same runtime model don't
        # alias. The local resolvers
        # (``_resolve_local_tts_backend`` /
        # ``_resolve_local_transcription_backend``) call
        # ``_lookup_warmed_backend`` which composes the same key
        # against the *current* planner selection. Reset by
        # ``release_warmed_backends`` (test helper).
        self._warmed_backends: dict[tuple, Any] = {}

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
        selection = _resolve_planner_selection(effective_model, CAPABILITY_CHAT, policy_preset)

        routing_policy = _resolve_routing_policy(defaults)
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
        selection = _resolve_planner_selection(effective_model, CAPABILITY_CHAT, policy_preset)

        routing_policy = _resolve_routing_policy(defaults)
        candidates = _selection_candidate_dicts(selection, routing_policy)
        fallback_allowed = _candidate_fallback_allowed(selection, routing_policy)
        runner_to_original = _runner_dict_to_planner_candidate(selection, candidates)
        candidates = _filter_unpreparable_local_candidates(
            self, selection, candidates, fallback_allowed=fallback_allowed
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
        selection = _resolve_planner_selection(effective_model, CAPABILITY_CHAT, policy_preset)

        routing_policy = _resolve_routing_policy(defaults)
        candidates = _selection_candidate_dicts(selection, routing_policy)
        fallback_allowed = _candidate_fallback_allowed(selection, routing_policy)
        runner_to_original = _runner_dict_to_planner_candidate(selection, candidates)
        candidates = _filter_unpreparable_local_candidates(
            self, selection, candidates, fallback_allowed=fallback_allowed
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
        selection = _resolve_planner_selection(effective_model, CAPABILITY_EMBEDDING, policy_preset)

        routing_policy = _resolve_routing_policy(defaults)

        local_available = self._can_local(effective_model, CAPABILITY_EMBEDDING)
        cloud_available = _cloud_available(defaults)
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
            result = await self._cloud_embed(inputs, effective_model, defaults.cloud_profile, is_fallback)
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
        selection = _resolve_planner_selection(effective_model, CAPABILITY_TRANSCRIPTION, policy_preset)

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
            result = await self._cloud_transcribe(
                audio_data, effective_model, defaults.cloud_profile, language, is_fallback
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
        selection = _resolve_planner_selection(effective_model, CAPABILITY_TTS, policy_preset)
        runtime_model = _runtime_model_for_selection(selection, effective_model)

        planner_policy = _tts_policy_from_selection(selection)
        if planner_policy:
            policy_preset = planner_policy
            defaults.policy_preset = planner_policy
            defaults.inline_policy = None

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
            cloud_model = requested_model if requested_model.startswith("@app/") else runtime_model
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
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"prepare: planner returned no local sdk_runtime candidate for model={effective_model!r} "
                    f"capability={capability!r}. The model is either cloud-only or the planner is offline."
                ),
            )

        from octomil.runtime.lifecycle.prepare_manager import PrepareManager, PrepareMode

        manager = self._prepare_manager or PrepareManager()
        return manager.prepare(candidate, mode=PrepareMode.EXPLICIT)

    def warmup(
        self,
        *,
        model: str,
        capability: str = "tts",
        policy: Optional[str] = None,
        app: Optional[str] = None,
    ) -> WarmupOutcome:
        """Run ``prepare`` and load the local backend so first-call is hot.

        Strict superset of :meth:`prepare`: bytes on disk *plus* the
        engine constructed and ``backend.load_model`` complete, with
        the loaded instance cached on the kernel so the next dispatch
        call reuses it. Currently supports the same capabilities as
        ``prepare`` (tts + transcription) — chat / responses inherit
        the same gates.

        Failure modes
        ~~~~~~~~~~~~~
        - Capability not in :data:`_WARMUPABLE_CAPABILITIES` →
          :class:`OctomilError` ``INVALID_INPUT`` (same actionable
          message as ``prepare``).
        - PrepareManager rejects the candidate or download fails →
          original error propagates unchanged.
        - Backend construction / load fails → returns
          ``WarmupOutcome(backend_loaded=False)`` so the prepare half
          isn't lost. Caller decides whether that's fatal; subsequent
          inference will fall through the cold path naturally.

        Returns a :class:`WarmupOutcome` with the underlying
        :class:`PrepareOutcome`, ``backend_loaded`` flag, and
        ``latency_ms`` wall time.
        """
        if capability not in _WARMUPABLE_CAPABILITIES:
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    f"client.warmup() does not yet support capability {capability!r}. "
                    f"Supported today: {sorted(_WARMUPABLE_CAPABILITIES)}. "
                    f"Other capabilities will be added once their backends thread the "
                    f"prepared model_dir into dispatch and the kernel can cache the "
                    f"loaded instance."
                ),
            )

        t0 = time.monotonic()

        # We need the planner selection up front for two reasons:
        #   1. ``_runtime_model_for_selection`` resolves ``@app/...``
        #      refs to the concrete model id (e.g. ``kokoro-en-v0_19``);
        #      that's the key the inference dispatch path uses, so the
        #      cache MUST be keyed under it (not the request alias).
        #   2. The same selection feeds the cache's artifact-identity
        #      tuple (digest/format/quantization) so a second
        #      ``warmup`` for the same runtime model but a different
        #      artifact version doesn't reuse the stale cached backend.
        defaults = self._resolve(capability, model=model, policy=policy, app=app)
        effective_model = defaults.model or model
        if not effective_model:
            raise _no_model_error(capability)
        policy_preset = defaults.policy_preset or "local_first"
        selection = _resolve_planner_selection(effective_model, capability, policy_preset)
        runtime_model = _runtime_model_for_selection(selection, effective_model)
        candidate = _local_sdk_runtime_candidate(selection)
        if candidate is None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=(
                    f"warmup: planner returned no local sdk_runtime candidate for "
                    f"model={effective_model!r} capability={capability!r}. The model is "
                    f"either cloud-only or the planner is offline."
                ),
            )

        # Run prepare via PrepareManager directly (mode=EXPLICIT) so
        # candidates with ``prepare_policy='explicit_only'`` succeed,
        # mirroring ``self.prepare``. Doing it here (instead of
        # delegating to ``self.prepare``) lets us reuse the same
        # ``selection`` / ``candidate`` for both prepare and the cache
        # key — no second planner round-trip, no risk of resolving a
        # different artifact for the cache than the one prepare ran on.
        from octomil.runtime.lifecycle.prepare_manager import PrepareManager, PrepareMode

        manager = self._prepare_manager or PrepareManager()
        prepare_outcome = manager.prepare(candidate, mode=PrepareMode.EXPLICIT)
        artifact_dir = str(prepare_outcome.artifact_dir)
        cache_key = self._warmup_cache_key(capability, runtime_model, candidate)

        backend_loaded = False
        if capability == CAPABILITY_TTS:
            backend = self._resolve_local_tts_backend(runtime_model, prepared_model_dir=artifact_dir)
            if backend is not None:
                self._warmed_backends[cache_key] = backend
                backend_loaded = True
        elif capability == CAPABILITY_TRANSCRIPTION:
            backend = self._resolve_local_transcription_backend(runtime_model, prepared_model_dir=artifact_dir)
            if backend is not None:
                # ``_resolve_local_transcription_backend`` only calls
                # ``engine.create_backend``; ``_WhisperBackend`` (and
                # most ASR backends) keep ``load_model`` lazy so the
                # actual model parse + memory allocation happens on
                # the first ``transcribe`` call. Force load_model now
                # so the warmup contract is real: by the time the
                # outcome is returned, the next ``transcribe_audio``
                # dispatch reuses an *already-loaded* instance.
                load_fn = getattr(backend, "load_model", None)
                if callable(load_fn):
                    try:
                        load_fn(runtime_model)
                    except Exception:
                        # Backend constructor returned but load_model
                        # raised (corrupt artifact, missing native
                        # lib, etc.) — keep prepare bytes on disk and
                        # report load_skipped, same shape as a failed
                        # ``create_backend``.
                        backend = None
                if backend is not None:
                    self._warmed_backends[cache_key] = backend
                    backend_loaded = True

        latency_ms = (time.monotonic() - t0) * 1000.0
        return WarmupOutcome(
            capability=capability,
            model=runtime_model,
            prepare_outcome=prepare_outcome,
            backend_loaded=backend_loaded,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _warmup_cache_key(
        capability: str,
        runtime_model: str,
        candidate: Optional[Any],
    ) -> tuple[str, str, Optional[str], Optional[str], Optional[str]]:
        """Compose a cache key that distinguishes artifact identity.

        Two private apps or two artifact versions can resolve to the
        same ``runtime_model`` but represent different artifacts
        (different digests, formats, quantizations). Keying solely by
        ``(capability, runtime_model)`` would let warmup cache the
        first artifact and then serve a *stale* backend for a later
        warmup of the same model with different bytes.

        ``digest`` is the strongest distinguishing field; ``format``
        and ``quantization`` cover cases where the planner omits the
        digest but emits separate engine artifacts (MLX safetensors
        vs GGUF, q4 vs q4_k_m, etc.). All three may be None when the
        candidate has no artifact metadata; in that case the key
        degenerates to ``(capability, runtime_model, None, None,
        None)`` and behaves like the original (capability, model)
        cache.
        """
        artifact = getattr(candidate, "artifact", None) if candidate is not None else None
        return (
            capability,
            runtime_model,
            getattr(artifact, "digest", None) if artifact is not None else None,
            getattr(artifact, "format", None) if artifact is not None else None,
            getattr(artifact, "quantization", None) if artifact is not None else None,
        )

    def _lookup_warmed_backend(
        self,
        capability: str,
        runtime_model: str,
    ) -> Optional[Any]:
        """Look up a cached warmed backend for the *current* request.

        The local resolvers call this at the top of their resolution
        path. The lookup composes the same cache key the warmup
        writer used: it consults the planner for the current request
        and pulls digest/format/quantization off the local candidate.
        That way two warmups for the same runtime model but different
        artifact identities don't collide and inference always
        retrieves a backend that was loaded against the *current*
        artifact shape.

        Falls back to a runtime-model-only lookup when no planner
        candidate is available (offline planner, synthetic tests). A
        single warmup against a runtime model still benefits the
        common case where there's only one artifact identity in play.
        """
        # Fast path: precise key, requires a planner round-trip.
        try:
            policy_preset = "local_first"
            selection = _resolve_planner_selection(runtime_model, capability, policy_preset)
            candidate = _local_sdk_runtime_candidate(selection)
            if candidate is not None:
                key = self._warmup_cache_key(capability, runtime_model, candidate)
                if key in self._warmed_backends:
                    return self._warmed_backends[key]
        except Exception:
            pass
        # Fallback: any cache entry for this (capability, runtime_model)
        # regardless of artifact-identity suffix. Covers offline
        # planner + the historical dict-shape from older callers.
        for key, backend in self._warmed_backends.items():
            if len(key) >= 2 and key[0] == capability and key[1] == runtime_model:
                return backend
        return None

    def release_warmed_backends(self) -> None:
        """Drop every cached warmed backend.

        Used by tests and long-running embedded callers that want to
        free GPU memory between phases. The next inference dispatch
        rebuilds through the normal cold path. Idempotent.
        """
        self._warmed_backends.clear()

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
        # PR 11: prefer the warmup cache. When the caller invoked
        # ``client.warmup(model, capability='tts')`` earlier in this
        # session, the loaded SherpaTts backend is already in
        # ``_warmed_backends`` and the next ``synthesize_speech`` call
        # reuses it without paying ``engine.create_backend`` +
        # ``backend.load_model`` again. The lookup composes the
        # artifact-identity cache key (digest/format/quantization)
        # against the *current* planner selection, so a stale backend
        # from an earlier artifact version never shadows a later
        # warmup against new bytes. Cold-call path (no warmup)
        # remains unchanged.
        cached = self._lookup_warmed_backend(CAPABILITY_TTS, model)
        if cached is not None:
            return cached
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
        # PR 11: warmup cache hit short-circuits the registry walk.
        # Looked up against the current planner selection so the cache
        # key encodes the requested artifact identity, not just the
        # runtime model alias.
        cached = self._lookup_warmed_backend(CAPABILITY_TRANSCRIPTION, model)
        if cached is not None:
            return cached
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
                return CloudModelRuntime(
                    base_url=_openai_base_url(defaults.cloud_profile),
                    api_key=api_key,
                    model=model,
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
