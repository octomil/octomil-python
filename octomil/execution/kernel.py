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
    CloudProfile,
    InlinePolicy,
    LoadedConfigSet,
    RequestOverrides,
    ResolvedExecutionDefaults,
    load_standalone_config,
    resolve_capability_defaults,
)

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
    ) -> None:
        self._config_set = config_set or load_standalone_config(start_dir)

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

        async def _execute_candidate(candidate: dict[str, Any]) -> RuntimeResponse:
            candidate_selection = _candidate_to_selection(selection, candidate)
            router = await self._build_router(
                effective_model,
                CAPABILITY_CHAT,
                defaults,
                planner_selection=candidate_selection,
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

        t0 = time.monotonic()
        collected_text = ""
        selected_locality = LOCALITY_ON_DEVICE
        is_fallback = False
        last_error: Exception | None = None

        for idx, candidate in enumerate(candidates):
            first_token_emitted = False
            try:
                candidate_selection = _candidate_to_selection(selection, candidate)
                router = await self._build_router(
                    effective_model,
                    CAPABILITY_CHAT,
                    defaults,
                    planner_selection=candidate_selection,
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

        t0 = time.monotonic()
        collected_text = ""
        selected_locality = LOCALITY_ON_DEVICE
        is_fallback = False
        last_error: Exception | None = None

        for idx, candidate in enumerate(candidates):
            first_token_emitted = False
            try:
                candidate_selection = _candidate_to_selection(selection, candidate)
                router = await self._build_router(
                    effective_model,
                    CAPABILITY_CHAT,
                    defaults,
                    planner_selection=candidate_selection,
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
        local_available = self._has_local_transcription_backend(effective_model)
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

        t0 = time.monotonic()
        result = await self._local_transcribe(audio_data, effective_model, language, is_fallback)
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
    ) -> ExecutionResult:
        """Dispatch audio transcription to a local Whisper-compatible backend."""
        backend = self._resolve_local_transcription_backend(model)
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

    def _resolve_local_transcription_backend(self, model: str) -> Optional[Any]:
        try:
            from octomil.runtime.engines import get_registry

            registry = get_registry()
            for detection in registry.detect_all(model):
                if not detection.available:
                    continue
                backend = detection.engine.create_backend(model)
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
    ) -> RouterModelRuntime:
        """Build a RouterModelRuntime for the given model and capability.

        When a planner_selection is provided and recommends a specific engine,
        the local factory tries that engine first before falling back to the
        default registry resolution.
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

                        backend = engine.create_backend(model)
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
