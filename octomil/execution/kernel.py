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

import logging
import os
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

        # Check if local embedding is even possible
        local_available = self._can_local(effective_model, CAPABILITY_EMBEDDING)
        if not local_available and routing_policy.mode == ContractRoutingPolicy.LOCAL_ONLY:
            raise RuntimeError(
                "Local embedding execution is required by policy, but no local "
                "embedding runtime is available.\n\n"
                "Pass --model with a supported local embedding model, configure "
                "a local runtime, or explicitly enable cloud."
            )

        # For now, embeddings route through cloud when cloud is available,
        # or raise a clear error when local is required but unavailable.
        if not local_available:
            # Cloud path
            cloud_profile = defaults.cloud_profile
            if cloud_profile is None:
                raise RuntimeError(
                    "No local embedding runtime available and no cloud profile "
                    "configured.\n\n"
                    "Configure a cloud profile in .octomil.toml or use "
                    "--policy private with a local embedding model."
                )
            return await self._cloud_embed(inputs, effective_model, cloud_profile)

        # Local embedding path (if a local runtime exists)
        return await self._local_embed(inputs, effective_model, routing_policy)

    async def _cloud_embed(self, inputs: list[str], model: str, profile: CloudProfile) -> ExecutionResult:
        """Dispatch embeddings to cloud."""
        from octomil.embeddings import embed

        api_key = os.environ.get(profile.api_key_env, "")
        base_url = profile.base_url
        if not api_key:
            raise RuntimeError(
                f"Cloud embedding requires {profile.api_key_env} to be set.\n\n"
                f"Export {profile.api_key_env} or configure a cloud profile."
            )

        result = embed(
            server_url=f"{base_url.rstrip('/')}/api/v1",
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
            fallback_used=False,
            embeddings=result.embeddings,
            dimensions=dims,
            usage={
                "input_tokens": result.usage.prompt_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        )

    async def _local_embed(self, inputs: list[str], model: str, policy: RoutingPolicy) -> ExecutionResult:
        """Dispatch embeddings to a local runtime."""
        from octomil.runtime.core.registry import ModelRuntimeRegistry

        registry = ModelRuntimeRegistry.shared()
        runtime = registry.resolve(model)
        if runtime is None:
            raise RuntimeError(f"No local runtime found for embedding model '{model}'.")

        all_vectors: list[list[float]] = []
        total_tokens = 0

        for text in inputs:
            request = RuntimeRequest(
                messages=[RuntimeMessage(role=MessageRole.USER, parts=[RuntimeContentPart.text_part(text)])],
                generation_config=GenerationConfig(max_tokens=0, temperature=0.0),
            )
            response = await runtime.run(request)
            # Attempt to extract embedding vector from response
            if hasattr(response, "embedding") and response.embedding:
                all_vectors.append(response.embedding)
            else:
                all_vectors.append([])
            total_tokens += getattr(response, "input_tokens", len(text.split()))

        dims = len(all_vectors[0]) if all_vectors and all_vectors[0] else 0
        return ExecutionResult(
            id=f"emb_{uuid.uuid4().hex[:12]}",
            model=model,
            capability=CAPABILITY_EMBEDDING,
            locality=LOCALITY_ON_DEVICE,
            fallback_used=False,
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

        local_available = self._can_local(effective_model, CAPABILITY_TRANSCRIPTION)
        if not local_available and routing_policy.mode == ContractRoutingPolicy.LOCAL_ONLY:
            raise RuntimeError(
                "Local transcription is required by policy, but no local "
                "transcription runtime is available.\n\n"
                "Pass --model with a supported local transcription model or "
                "change the serving policy."
            )

        if not local_available:
            raise RuntimeError(
                "No local transcription runtime available. Audio transcription "
                "currently requires a local Whisper-compatible runtime.\n\n"
                "Install whisper support or pass --model with an available model."
            )

        # Local transcription
        from octomil.runtime.core.registry import ModelRuntimeRegistry

        registry = ModelRuntimeRegistry.shared()
        runtime = registry.resolve(effective_model)
        if runtime is None:
            raise RuntimeError(f"No runtime found for transcription model '{effective_model}'.")

        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.text_part(language or "")],
                ),
            ],
            generation_config=GenerationConfig(max_tokens=0, temperature=0.0),
        )
        response = await runtime.run(request)

        return ExecutionResult(
            id=f"txn_{uuid.uuid4().hex[:12]}",
            model=effective_model,
            capability=CAPABILITY_TRANSCRIPTION,
            locality=LOCALITY_ON_DEVICE,
            fallback_used=False,
            output_text=response.text,
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
        try:
            from octomil.runtime.core.registry import ModelRuntimeRegistry

            registry = ModelRuntimeRegistry.shared()
            runtime = registry.resolve(model)
            return runtime is not None
        except Exception:
            return False

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
                    base_url=defaults.cloud_profile.base_url,
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
