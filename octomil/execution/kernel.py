"""Shared execution kernel — the single execution path for all Octomil surfaces.

Encapsulates:
  - model resolution
  - serving-policy evaluation
  - local runtime resolution
  - cloud runtime resolution
  - fallback decisions
  - structured response generation
  - telemetry

Does NOT require an HTTP server process.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from octomil._generated.message_role import MessageRole
from octomil._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy
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
from octomil.runtime.core.policy import RoutingPolicy
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
# Result types
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


# ---------------------------------------------------------------------------
# Policy conversion
# ---------------------------------------------------------------------------


def _resolve_routing_policy(defaults: ResolvedExecutionDefaults) -> RoutingPolicy:
    """Convert resolved config defaults into a runtime RoutingPolicy."""
    preset = defaults.policy_preset
    inline = defaults.inline_policy

    if inline is not None:
        return _inline_to_routing_policy(inline)

    if preset is None or preset == "local_first":
        return RoutingPolicy.local_first()
    if preset == "private":
        return RoutingPolicy.local_only()
    if preset == "cloud_only":
        return RoutingPolicy.cloud_only()
    if preset == "cloud_first":
        return RoutingPolicy(
            mode=ContractRoutingPolicy.AUTO,
            prefer_local=False,
            fallback="local",
        )
    if preset == "performance_first":
        return RoutingPolicy.auto(prefer_local=True)

    return RoutingPolicy.local_first()


def _inline_to_routing_policy(ip: InlinePolicy) -> RoutingPolicy:
    rm = ip.routing_mode
    rp = ip.routing_preference

    if rm == "local_only":
        return RoutingPolicy.local_only()
    if rm == "cloud_only":
        return RoutingPolicy.cloud_only()
    if rp == "local":
        fb = "cloud" if ip.fallback.allow_cloud_fallback else "none"
        return RoutingPolicy.local_first(fallback=fb)
    if rp == "cloud":
        return RoutingPolicy(
            mode=ContractRoutingPolicy.AUTO,
            prefer_local=False,
            fallback="local" if ip.fallback.allow_local_fallback else "none",
        )
    if rp == "performance":
        return RoutingPolicy.auto(prefer_local=True)
    return RoutingPolicy.auto()


def _openai_base_url(profile: CloudProfile) -> str:
    """Return an OpenAI-compatible hosted base URL ending in /v1."""
    base = profile.base_url.rstrip("/")
    if base.endswith("/v1") and not base.endswith("/api/v1"):
        return base
    if base.endswith("/api/v1"):
        return base[: -len("/api/v1")] + "/v1"
    return f"{base}/v1"


def _platform_api_base_url(profile: CloudProfile) -> str:
    """Return the legacy platform API base URL ending in /api/v1."""
    base = profile.base_url.rstrip("/")
    if base.endswith("/api/v1"):
        return base
    if base.endswith("/v1"):
        return base[: -len("/v1")] + "/api/v1"
    return f"{base}/api/v1"


def _cloud_api_key(profile: Optional[CloudProfile]) -> str:
    if profile is None:
        return ""
    return os.environ.get(profile.api_key_env, "")


def _cloud_available(defaults: ResolvedExecutionDefaults) -> bool:
    return bool(defaults.cloud_profile and _cloud_api_key(defaults.cloud_profile))


def _resolve_localities(
    routing_policy: RoutingPolicy,
    *,
    local_available: bool,
    cloud_available: bool,
) -> tuple[str, Optional[str]]:
    """Return (primary_locality, fallback_locality | None).

    This mirrors ``_select_locality_for_capability`` but returns the
    configured primary and fallback localities for callers that own backend
    lifecycle.  Disabled fallbacks are exact: if the preferred locality is
    unavailable and fallback is ``"none"``, this raises instead of silently
    switching execution locations.
    """
    if routing_policy.mode == ContractRoutingPolicy.LOCAL_ONLY:
        if not local_available:
            raise RuntimeError("Private/local-only policy requires a local backend, but none is available.")
        return LOCALITY_ON_DEVICE, None

    if routing_policy.mode == ContractRoutingPolicy.CLOUD_ONLY:
        if not cloud_available:
            raise RuntimeError("Cloud-only policy requires cloud credentials, but none are configured.")
        return LOCALITY_CLOUD, None

    # AUTO or LOCAL_FIRST — determine prefer_local from policy
    prefer_local = routing_policy.prefer_local
    if routing_policy.mode == ContractRoutingPolicy.LOCAL_FIRST:
        prefer_local = True

    if prefer_local:
        if local_available:
            fallback = LOCALITY_CLOUD if routing_policy.fallback == "cloud" and cloud_available else None
            return LOCALITY_ON_DEVICE, fallback
        if routing_policy.fallback == "cloud" and cloud_available:
            return LOCALITY_CLOUD, None
        if cloud_available:
            raise RuntimeError("Local chat execution is required by policy, but cloud fallback is disabled.")
        raise RuntimeError("No local or cloud backend available for chat.")

    # cloud-first
    if cloud_available:
        fallback = LOCALITY_ON_DEVICE if routing_policy.fallback == "local" and local_available else None
        return LOCALITY_CLOUD, fallback
    if routing_policy.fallback == "local" and local_available:
        return LOCALITY_ON_DEVICE, None
    if local_available:
        raise RuntimeError("Cloud chat execution is required by policy, but local fallback is disabled.")
    raise RuntimeError("No local or cloud backend available for chat.")


def _select_locality_for_capability(
    routing_policy: RoutingPolicy,
    *,
    local_available: bool,
    cloud_available: bool,
    capability: str,
) -> tuple[str, bool]:
    """Select local/cloud for non-ModelRuntime capabilities using exact policy semantics."""
    if routing_policy.mode == ContractRoutingPolicy.LOCAL_ONLY:
        if local_available:
            return LOCALITY_ON_DEVICE, False
        raise RuntimeError(f"Local {capability} execution is required by policy, but no local runtime is available.")

    if routing_policy.mode == ContractRoutingPolicy.CLOUD_ONLY:
        if cloud_available:
            return LOCALITY_CLOUD, False
        raise RuntimeError(f"Cloud {capability} execution is required by policy, but cloud is not configured.")

    if routing_policy.mode == ContractRoutingPolicy.LOCAL_FIRST or routing_policy.prefer_local:
        if local_available:
            return LOCALITY_ON_DEVICE, False
        if routing_policy.fallback == "cloud" and cloud_available:
            return LOCALITY_CLOUD, True
        raise RuntimeError(f"No local {capability} runtime available and cloud fallback is not configured.")

    if cloud_available:
        return LOCALITY_CLOUD, False
    if routing_policy.fallback == "local" and local_available:
        return LOCALITY_ON_DEVICE, True
    raise RuntimeError(f"No cloud {capability} runtime available and local fallback is not configured.")


# ---------------------------------------------------------------------------
# Execution Kernel
# ---------------------------------------------------------------------------


class ExecutionKernel:
    """Shared execution kernel for all Octomil surfaces.

    Usage::

        kernel = ExecutionKernel()
        result = await kernel.create_response("Hello!", model="gemma-1b")
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

        routing_policy = _resolve_routing_policy(defaults)
        router = await self._build_router(effective_model, CAPABILITY_CHAT, defaults)
        locality, is_fallback = router.resolve_locality(routing_policy)

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

        response = await router.run(request, policy=routing_policy)

        return ExecutionResult(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            model=effective_model,
            capability=CAPABILITY_CHAT,
            locality=locality,
            fallback_used=is_fallback,
            output_text=response.text,
            usage=_extract_usage(response),
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

        routing_policy = _resolve_routing_policy(defaults)
        router = await self._build_router(effective_model, CAPABILITY_CHAT, defaults)
        locality, is_fallback = router.resolve_locality(routing_policy)

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

        collected_text = ""
        async for chunk in router.stream(request, policy=routing_policy):
            text = chunk.text or ""
            collected_text += text
            yield StreamChunk(delta=text)

        yield StreamChunk(
            delta="",
            done=True,
            result=ExecutionResult(
                id=f"resp_{uuid.uuid4().hex[:12]}",
                model=effective_model,
                capability=CAPABILITY_CHAT,
                locality=locality,
                fallback_used=is_fallback,
                output_text=collected_text,
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

        routing_policy = _resolve_routing_policy(defaults)

        local_available = self._can_local(effective_model, CAPABILITY_EMBEDDING)
        cloud_available = _cloud_available(defaults)
        locality, is_fallback = _select_locality_for_capability(
            routing_policy,
            local_available=local_available,
            cloud_available=cloud_available,
            capability=CAPABILITY_EMBEDDING,
        )

        if locality == LOCALITY_CLOUD:
            assert defaults.cloud_profile is not None
            return await self._cloud_embed(inputs, effective_model, defaults.cloud_profile, is_fallback)

        return await self._local_embed(inputs, effective_model, is_fallback)

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

        routing_policy = _resolve_routing_policy(defaults)
        local_available = self._has_local_transcription_backend(effective_model)
        cloud_available = _cloud_available(defaults)
        locality, is_fallback = _select_locality_for_capability(
            routing_policy,
            local_available=local_available,
            cloud_available=cloud_available,
            capability=CAPABILITY_TRANSCRIPTION,
        )

        if locality == LOCALITY_CLOUD:
            assert defaults.cloud_profile is not None
            return await self._cloud_transcribe(
                audio_data, effective_model, defaults.cloud_profile, language, is_fallback
            )

        return await self._local_transcribe(audio_data, effective_model, language, is_fallback)

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
    ) -> RouterModelRuntime:
        """Build a RouterModelRuntime for the given model and capability."""
        from octomil.runtime.core.registry import ModelRuntimeRegistry

        registry = ModelRuntimeRegistry.shared()

        def local_factory(hint: str):
            return registry.resolve(model)

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
