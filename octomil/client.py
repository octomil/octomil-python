"""
Convenience client for the Octomil platform.

Wraps the existing SDK classes (ModelRegistry, RolloutsAPI) behind a
simpler interface designed for CLI and script usage::

    import octomil
    from octomil.auth import OrgApiKeyAuth

    client = octomil.OctomilClient(auth=OrgApiKeyAuth(api_key="edg_...", org_id="org_123"))
    client.push("model.pt", name="sentiment-v1", version="1.0.0")
    client.deploy("sentiment-v1", version="1.0.0", rollout=10)

    # Or from environment variables:
    client = octomil.OctomilClient.from_env()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Optional

if TYPE_CHECKING:
    from .agents.session import AgentSession
    from .audio import OctomilAudio
    from .capabilities_client import CapabilitiesClient
    from .chat_client import ChatClient
    from .control import OctomilControl
    from .embeddings import EmbeddingResult
    from .manifest import AppManifest
    from .manifest.catalog_service import ModelCatalogService
    from .model import Model, Prediction
    from .models_namespace import OctomilModels
    from .responses.responses import OctomilResponses
    from .runtime.core.model_runtime import ModelRuntime
    from .runtime.core.policy import RoutingPolicy
    from .serve import GenerationChunk
    from .streaming import StreamToken
    from .telemetry import TelemetryReporter
    from .telemetry_client import TelemetryClient
    from .text import OctomilText
    from .workflows import WorkflowRunner

logger = logging.getLogger(__name__)

from .auth import AuthConfig, DeviceTokenAuth, OrgApiKeyAuth  # noqa: E402

# PR C: ``ModelOpsMixin`` and the inner-package classes
# (``_ApiClient``, ``RolloutsAPI``, ``ModelRegistry``) are imported
# lazily inside ``OctomilClient.__init__`` rather than at module
# load. Eager imports here pull ``octomil.python.octomil``'s
# ``__init__``, which in turn imports ``federated_client`` /
# ``data_loader`` and crashes Ren'Py / sandboxed CPython /
# PyInstaller via pandas's ``sysconfig.get_config_var``. Thin TTS
# callers that never construct ``OctomilClient`` now skip the
# heavy chain entirely.
#
# ``ModelOpsMixin`` is the base class so it's a structural mixin
# (no methods called at import time); we still need it as a base.
# Resolve it via ``__class_getitem__``-like trick: defer the
# concrete subclassing into the constructor — too invasive for
# this PR. Instead, accept that ``model_ops`` ALSO defers its
# inner imports (handled separately in model_ops.py's lazy block).
from .model_ops import ModelOpsMixin  # noqa: E402

_DEFAULT_API_BASE = "https://api.octomil.com/api/v1"


class OctomilClient(ModelOpsMixin):
    """High-level OctomilClient for push/pull/deploy workflows.

    Args:
        auth: Authentication configuration. Use :class:`OrgApiKeyAuth` for
            API key authentication or :class:`DeviceTokenAuth` for device
            token authentication.
        device_id: Stable device identifier. When ``None``, one is derived
            automatically from the host hardware (see :mod:`octomil.device_info`).

    Example::

        from octomil import OctomilClient
        from octomil.auth import OrgApiKeyAuth

        client = OctomilClient(auth=OrgApiKeyAuth(api_key="edg_...", org_id="org_123"))

        # Or from environment variables:
        client = OctomilClient.from_env()
    """

    def __init__(
        self,
        auth: AuthConfig,
        device_id: str | None = None,
        planner_enabled: bool = True,
    ) -> None:
        from .auth import PublishableKeyAuth

        if isinstance(auth, OrgApiKeyAuth):
            self._api_key: str = auth.api_key
            self._org_id: str = auth.org_id
            self._api_base: str = auth.api_base
        elif isinstance(auth, DeviceTokenAuth):
            self._api_key = auth.bootstrap_token
            self._org_id = ""
            self._api_base = auth.api_base
        elif isinstance(auth, PublishableKeyAuth):
            self._api_key = auth.api_key
            self._org_id = ""
            self._api_base = auth.api_base
        else:
            raise TypeError(
                f"auth must be OrgApiKeyAuth, DeviceTokenAuth, or PublishableKeyAuth, got {type(auth).__name__}"
            )

        self._auth = auth
        self._device_id: str | None = device_id
        self._planner_enabled = planner_enabled
        self._models: dict[str, "Model"] = {}

        def _token_provider() -> str:
            return self._api_key

        # PR C: lazy-import the inner-package classes so plain
        # ``import octomil`` doesn't reach pandas / pyarrow on thin
        # clients that never construct an ``OctomilClient``.
        # Resolve through the module's own ``__getattr__`` so test
        # suites that ``patch("octomil.client.RolloutsAPI", ...)``
        # actually intercept construction. Direct ``from … import``
        # statements bypass module-attribute lookup, breaking patches.
        import octomil.client as _self  # noqa: PLC0415

        self._api = _self._ApiClient(
            auth_token_provider=_token_provider,
            api_base=self._api_base,
        )
        self._registry = _self.ModelRegistry(
            auth_token_provider=_token_provider,
            org_id=self._org_id,
            api_base=self._api_base,
        )
        self._rollouts = _self.RolloutsAPI(self._api)

        # Per-deployment routing policies from desired state (set automatically via control sync)
        self._routing_policies: dict[str, "RoutingPolicy"] = {}
        self._model_deployment_map: dict[str, str] = {}  # model_id → deployment_id
        self._default_routing_policy: "RoutingPolicy | None" = None

        # Lazy-initialised response, workflow, control, and models namespace APIs
        self._responses: OctomilResponses | None = None
        self._workflows: WorkflowRunner | None = None
        self._control: OctomilControl | None = None
        self._models_ns: OctomilModels | None = None
        self._chat_ns: ChatClient | None = None
        self._capabilities_ns: CapabilitiesClient | None = None
        self._telemetry_ns: TelemetryClient | None = None
        self._audio_ns: OctomilAudio | None = None
        self._text_ns: OctomilText | None = None
        self._catalog: ModelCatalogService | None = None

        # Telemetry — best-effort, never blocks or raises
        self._reporter: TelemetryReporter | None = None
        if self._api_key:
            try:
                from .telemetry import TelemetryReporter as _TR

                self._reporter = _TR(
                    api_key=self._api_key,
                    api_base=self._api_base,
                    org_id=self._org_id,
                    device_id=self._device_id,
                )
            except Exception:
                logger.debug("Failed to initialise telemetry reporter", exc_info=True)

    @classmethod
    def from_env(
        cls,
        *,
        device_id: str | None = None,
        planner_enabled: bool = True,
    ) -> OctomilClient:
        """Construct an OctomilClient from environment variables.

        Reads ``OCTOMIL_API_KEY``, ``OCTOMIL_ORG_ID``, and optionally
        ``OCTOMIL_API_BASE`` from the environment.

        Args:
            device_id: Stable device identifier. When ``None``, one is
                derived automatically from the host hardware.

        Raises:
            ValueError: If ``OCTOMIL_API_KEY`` is not set.
        """
        return cls(auth=OrgApiKeyAuth.from_env(), device_id=device_id, planner_enabled=planner_enabled)

    # ------------------------------------------------------------------
    # Device ID — stable identifier for this device
    # ------------------------------------------------------------------

    @property
    def device_id(self) -> str:
        """Stable device identifier.

        Returns the explicitly configured ``device_id`` if one was
        provided at construction time; otherwise derives one from the
        host hardware.
        """
        if self._device_id is None:
            from .device_info import get_stable_device_id

            self._device_id = get_stable_device_id()
        return self._device_id

    # ------------------------------------------------------------------
    # Models namespace — SDK Facade Contract lifecycle API
    # ------------------------------------------------------------------

    @property
    def models(self) -> "OctomilModels":
        """Model lifecycle operations (status, load, unload, list, clear_cache)."""
        if self._models_ns is None:
            from .models_namespace import OctomilModels

            self._models_ns = OctomilModels(self)
        return self._models_ns

    # ------------------------------------------------------------------
    # Capabilities namespace — device profiling
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> "CapabilitiesClient":
        """Device capabilities (runtimes, memory, accelerators)."""
        if self._capabilities_ns is None:
            from .capabilities_client import CapabilitiesClient

            self._capabilities_ns = CapabilitiesClient(self)
        return self._capabilities_ns

    # ------------------------------------------------------------------
    # Responses API — structured on-device inference
    # ------------------------------------------------------------------

    @property
    def responses(self) -> OctomilResponses:
        """Structured response API for on-device inference."""
        if self._responses is None:
            from .responses import OctomilResponses

            self._responses = OctomilResponses(
                catalog=self._catalog,
                telemetry_reporter=self._reporter,
                routing_policies=self._routing_policies,
                model_deployment_map=self._model_deployment_map,
                default_routing_policy=self._default_routing_policy,
                planner_enabled=self._planner_enabled,
            )
        return self._responses

    # ------------------------------------------------------------------
    # Workflows — multi-step orchestration
    # ------------------------------------------------------------------

    @property
    def workflows(self) -> WorkflowRunner:
        """Workflow orchestration for multi-step pipelines."""
        if self._workflows is None:
            from .workflows import WorkflowRunner

            self._workflows = WorkflowRunner(self.responses)
        return self._workflows

    # ------------------------------------------------------------------
    # Control — device registration and heartbeat
    # ------------------------------------------------------------------

    @property
    def control(self) -> "OctomilControl":
        """Device registration and heartbeat management."""
        if self._control is None:
            from .control import OctomilControl

            self._control = OctomilControl(
                api=self._api,
                org_id=self._org_id,
                telemetry=self._reporter,
                on_desired_state=self._apply_routing_from_desired_state,
            )
        return self._control

    def _apply_routing_from_desired_state(self, entries: list[dict]) -> None:
        """Extract per-deployment routing policies from desired state.

        Builds a ``deployment_id → RoutingPolicy`` map so multi-deployment
        clients route each request correctly.  Also builds a
        ``model_id → deployment_id`` map for automatic resolution from
        request model names.

        When the same model_id appears under multiple deployments with
        different routing policies, that model_id is excluded from
        auto-resolution (ambiguous).  Callers must pass explicit
        ``metadata={"deployment_id": "..."}`` to disambiguate.
        """
        from .runtime.core.policy import RoutingPolicy

        policies: dict[str, RoutingPolicy] = {}
        model_map: dict[str, str] = {}
        model_policies: dict[str, RoutingPolicy] = {}  # track policy per model_id
        ambiguous: set[str] = set()
        first_policy: RoutingPolicy | None = None
        for entry in entries:
            policy = RoutingPolicy.from_desired_state_entry(entry)
            if policy is None:
                continue
            dep_id = entry.get("deployment_id")
            model_id = entry.get("model_id")
            if dep_id:
                policies[dep_id] = policy
                if model_id:
                    if model_id in model_policies:
                        if model_policies[model_id] != policy:
                            ambiguous.add(model_id)
                    else:
                        model_policies[model_id] = policy
                        model_map[model_id] = dep_id
            if first_policy is None:
                first_policy = policy

        # Remove model_ids with conflicting routing — can't auto-resolve
        for mid in ambiguous:
            model_map.pop(mid, None)
            logger.debug(
                "Model %s has multiple deployments with different routing; "
                "pass deployment_id in metadata to disambiguate",
                mid,
            )

        self._routing_policies = policies
        self._model_deployment_map = model_map
        self._default_routing_policy = first_policy
        self._responses = None  # Reset so it picks up new policies

    # ------------------------------------------------------------------
    # Agent session factory
    # ------------------------------------------------------------------

    def agent_session(
        self,
        base_url: str | None = None,
        *,
        max_iterations: int = 10,
    ) -> "AgentSession":
        """Create an :class:`AgentSession` pre-wired with this client's
        responses instance (and therefore its desired-state routing)."""
        from .agents.session import AgentSession

        return AgentSession(
            base_url=base_url or self._api_base,
            auth_token=self._api_key,
            responses=self.responses,
            max_iterations=max_iterations,
        )

    # ------------------------------------------------------------------
    # Audio namespace — transcription APIs
    # ------------------------------------------------------------------

    @property
    def audio(self) -> "OctomilAudio":
        """Audio transcription APIs."""
        if self._audio_ns is None:
            from .audio import OctomilAudio

            self._audio_ns = OctomilAudio(runtime_resolver=self._resolve_model_ref)
        return self._audio_ns

    # ------------------------------------------------------------------
    # Text namespace — prediction APIs
    # ------------------------------------------------------------------

    @property
    def text(self) -> "OctomilText":
        """Text prediction and completion APIs."""
        if self._text_ns is None:
            from .text import OctomilText

            self._text_ns = OctomilText(runtime_resolver=self._resolve_model_ref)
        return self._text_ns

    # ------------------------------------------------------------------
    # Catalog — manifest-driven model catalog
    # ------------------------------------------------------------------

    @property
    def catalog(self) -> "ModelCatalogService | None":
        """Model catalog service, or None if not configured."""
        return self._catalog

    def configure(
        self,
        *,
        manifest: "AppManifest | None" = None,
    ) -> None:
        """Configure the client with a manifest for catalog-driven model resolution.

        Args:
            manifest: App manifest describing desired models.
        """
        if manifest is not None:
            from .manifest.catalog_service import ModelCatalogService

            self._catalog = ModelCatalogService(manifest=manifest)
            self._catalog.bootstrap()
            # Reset responses so it picks up the new catalog
            self._responses = None

    def _resolve_model_ref(self, ref: object) -> Optional[ModelRuntime]:
        """Resolve a ModelRef to a ModelRuntime via the catalog or registry."""
        from .model_ref import _ModelRefCapability, _ModelRefId
        from .runtime.core.registry import ModelRuntimeRegistry

        if self._catalog is not None:
            if isinstance(ref, (_ModelRefId, _ModelRefCapability)):
                runtime = self._catalog.runtime_for_ref(ref)
                if runtime is not None:
                    return runtime

        # Fallback to registry for ID-based refs
        if isinstance(ref, _ModelRefId):
            return ModelRuntimeRegistry.shared().resolve(ref.model_id)
        if isinstance(ref, _ModelRefCapability):
            return ModelRuntimeRegistry.shared().resolve(ref.capability.value)
        return None

    # ------------------------------------------------------------------
    # Predict — one-call download + load + infer
    # ------------------------------------------------------------------

    def predict(
        self,
        name: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        version: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> "Prediction":
        """Download, load, and run inference in one call.

        The model is cached after the first call — subsequent calls
        with the same *name* reuse the loaded backend.

        Args:
            name: Model name (e.g. ``"phi-4-mini"``).
            messages: Chat-style messages (role + content dicts).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            version: Specific version. Defaults to latest.
            engine: Override the engine (e.g. ``"mlx-lm"``).

        Returns:
            A ``Prediction`` — a ``str`` with a ``.metrics`` attribute.
        """
        from .serve import GenerationRequest

        model = self._get_model(name, version=version, engine=engine)
        req = GenerationRequest(
            model=name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return model.predict(req)

    async def predict_stream(
        self,
        name: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        version: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> "AsyncIterator[GenerationChunk]":
        """Download, load, and stream inference in one call.

        The model is cached after the first call.

        Args:
            name: Model name (e.g. ``"phi-4-mini"``).
            messages: Chat-style messages (role + content dicts).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            version: Specific version. Defaults to latest.
            engine: Override the engine (e.g. ``"mlx-lm"``).

        Yields:
            ``GenerationChunk`` objects with incremental text.
        """
        from .serve import GenerationRequest

        model = self._get_model(name, version=version, engine=engine)
        req = GenerationRequest(
            model=name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        async for chunk in model.predict_stream(req):
            yield chunk

    # ------------------------------------------------------------------
    # Cloud streaming inference (SSE)
    # ------------------------------------------------------------------

    def stream_predict(
        self,
        model_id: str,
        input_data: str | list[dict[str, str]],
        *,
        parameters: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> "Iterator[StreamToken]":
        """Stream tokens from the cloud inference endpoint (sync).

        Requires ``api_key`` and ``api_base`` to be configured.  This is
        a cloud-only method — it does **not** download or run models
        locally.

        Args:
            model_id: Model identifier (e.g. ``"phi-4-mini"``).
            input_data: A plain string prompt or chat-style messages.
            parameters: Generation parameters (temperature, max_tokens, etc.).
            timeout: HTTP timeout in seconds for the streaming connection.

        Yields:
            :class:`~octomil.streaming.StreamToken` for each SSE event.
        """
        from .streaming import stream_inference

        return stream_inference(
            server_url=self._api_base,
            api_key=self._api_key,
            model_id=model_id,
            input_data=input_data,
            parameters=parameters,
            timeout=timeout,
        )

    async def stream_predict_async(
        self,
        model_id: str,
        input_data: str | list[dict[str, str]],
        *,
        parameters: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> "AsyncIterator[StreamToken]":
        """Stream tokens from the cloud inference endpoint (async).

        Async variant of :meth:`stream_predict`.
        """
        from .streaming import stream_inference_async

        async for token in stream_inference_async(
            server_url=self._api_base,
            api_key=self._api_key,
            model_id=model_id,
            input_data=input_data,
            parameters=parameters,
            timeout=timeout,
        ):
            yield token

    # ------------------------------------------------------------------
    # Telemetry namespace — custom events + flush
    # ------------------------------------------------------------------

    @property
    def telemetry(self) -> "TelemetryClient":
        """Telemetry operations (track, flush)."""
        if self._telemetry_ns is None:
            from .telemetry_client import TelemetryClient

            self._telemetry_ns = TelemetryClient(self)
        return self._telemetry_ns

    # ------------------------------------------------------------------
    # Chat namespace — SDK Facade Contract chat API
    # ------------------------------------------------------------------

    @property
    def chat(self) -> "ChatClient":
        """Chat completions namespace (create, stream).

        Usage::

            result = client.chat.create(model="phi-4-mini", messages=[...])
        """
        if self._chat_ns is None:
            from .chat_client import ChatClient

            self._chat_ns = ChatClient(self)
        return self._chat_ns

    # ------------------------------------------------------------------
    # Chat — internal implementation (called by ChatClient)
    # ------------------------------------------------------------------

    def _chat_create(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: float = 120.0,
        **parameters: Any,
    ) -> dict[str, Any]:
        """OpenAI-compatible chat completion (non-streaming).

        Collects the full response from :meth:`stream_predict` and returns
        a dict with the assistant message and latency.

        Args:
            model_id: Model identifier (e.g. ``"phi-4-mini"``).
            messages: Chat messages (``[{"role": "user", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold.
            timeout: HTTP timeout in seconds.
            **parameters: Additional generation parameters.

        Returns:
            Dict with ``message`` (role + content), ``latency_ms``, and
            optional ``usage`` fields.
        """
        import time as _time

        params: dict[str, Any] = {**parameters}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p

        start = _time.monotonic()
        content = ""
        for token in self.stream_predict(
            model_id,
            messages,
            parameters=params or None,
            timeout=timeout,
        ):
            content += token.token

        latency_ms = (_time.monotonic() - start) * 1000
        return {
            "message": {"role": "assistant", "content": content},
            "latency_ms": latency_ms,
        }

    async def _chat_stream(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: float = 120.0,
        **parameters: Any,
    ) -> "AsyncIterator[dict[str, Any]]":
        """Streaming chat — yields chunks as they arrive (async generator).

        Each yielded dict contains ``index``, ``content``, ``done``, and
        ``role`` keys, matching the browser SDK's ``ChatChunk`` shape.

        Args:
            model_id: Model identifier (e.g. ``"phi-4-mini"``).
            messages: Chat messages (``[{"role": "user", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold.
            timeout: HTTP timeout in seconds.
            **parameters: Additional generation parameters.

        Yields:
            Dict with ``index``, ``content``, ``done``, and ``role``.
        """
        params: dict[str, Any] = {**parameters}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p

        idx = 0
        async for token in self.stream_predict_async(
            model_id,
            messages,
            parameters=params or None,
            timeout=timeout,
        ):
            yield {
                "index": idx,
                "content": token.token,
                "done": token.done,
                "role": "assistant",
            }
            idx += 1

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(
        self,
        model_id: str,
        input: str | list[str],
        *,
        timeout: float = 30.0,
    ) -> "EmbeddingResult":
        """Generate embeddings via the Octomil cloud endpoint.

        Requires ``api_key`` and ``api_base`` to be configured.

        Args:
            model_id: Embedding model identifier (e.g. ``"nomic-embed-text"``).
            input: A single string or list of strings to embed.
            timeout: HTTP timeout in seconds.

        Returns:
            :class:`~octomil.embeddings.EmbeddingResult` with dense vectors.
        """
        from .embeddings import embed

        return embed(
            server_url=self._api_base,
            api_key=self._api_key,
            model_id=model_id,
            input=input,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Close — clean up models and telemetry
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all cached models and shut down the telemetry reporter."""
        if self._control is not None:
            self._control.stop_heartbeat()
            self._control = None
        self._models.clear()
        if self._reporter:
            try:
                self._reporter.close()
            except Exception:
                logger.debug("Failed to close telemetry reporter", exc_info=True)
            self._reporter = None

    def dispose(self) -> None:
        """Deprecated: use :meth:`close` instead."""
        import warnings

        warnings.warn(
            "dispose() is deprecated, use close() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.close()


# Module-level lazy attribute access for the inner-package classes.
# Pairs with the lazy ``import octomil.client as _self`` inside
# ``OctomilClient.__init__`` so test suites can
# ``patch("octomil.client.RolloutsAPI")`` and have the patch
# actually intercept construction. Imports are deferred so plain
# ``import octomil`` still skips pandas / pyarrow on thin clients.
def __getattr__(name: str):
    if name == "RolloutsAPI":
        from .python.octomil.control_plane import RolloutsAPI

        return RolloutsAPI
    if name == "ModelRegistry":
        from .python.octomil.registry import ModelRegistry

        return ModelRegistry
    if name == "_ApiClient":
        from .python.octomil.api_client import _ApiClient

        return _ApiClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
