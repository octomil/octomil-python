"""Unified Octomil facade — simplified entry point for the SDK."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, AsyncIterator

from .auth_config import PublishableKeyAuth

if TYPE_CHECKING:
    from .auth import AuthConfig
    from .client import OctomilClient
    from .embeddings import EmbeddingResult
    from .local_runner.client import LocalRunnerClient
    from .local_runner.manager import LocalRunnerHandle
    from .responses.responses import OctomilResponses
    from .responses.types import Response


class OctomilNotInitializedError(Exception):
    def __init__(self) -> None:
        super().__init__("Octomil client is not initialized. Call await client.initialize() first.")


class FacadeResponses:
    """Convenience wrapper over OctomilResponses with a simpler call signature."""

    def __init__(self, responses: OctomilResponses) -> None:
        self._responses = responses

    async def create(
        self, request_or_model: Any = None, *, model: str | None = None, input: str | None = None, **kwargs: Any
    ) -> Response:
        from .responses.types import ResponseRequest

        if isinstance(request_or_model, ResponseRequest):
            return await self._responses.create(request_or_model)
        resolved_model = request_or_model if isinstance(request_or_model, str) else model
        if resolved_model is None or input is None:
            raise TypeError("create() requires either a ResponseRequest or model= and input= arguments")
        request = ResponseRequest.text(resolved_model, input, **kwargs)
        return await self._responses.create(request)

    async def stream(
        self, request_or_model: Any = None, *, model: str | None = None, input: str | None = None, **kwargs: Any
    ) -> AsyncIterator:
        from .responses.types import ResponseRequest

        if isinstance(request_or_model, ResponseRequest):
            request = request_or_model
        else:
            resolved_model = request_or_model if isinstance(request_or_model, str) else model
            if resolved_model is None or input is None:
                raise TypeError("stream() requires either a ResponseRequest or model= and input= arguments")
            request = ResponseRequest.text(resolved_model, input, **kwargs)
        async for event in self._responses.stream(request):
            yield event


class FacadeEmbeddings:
    """Embeddings namespace on the unified Octomil facade."""

    def __init__(self, client: OctomilClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: str | list[str],
        timeout: float = 30.0,
    ) -> EmbeddingResult:
        """Create embeddings using the initialized facade auth context."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._client.embed(model, input, timeout=timeout),
        )


class Octomil:
    """Unified facade for the Octomil SDK.

    Usage::

        client = Octomil.from_env()
        await client.initialize()
        response = await client.responses.create(model="phi-4-mini", input="Hello!")
        print(response.output_text)
    """

    def __init__(
        self,
        *,
        publishable_key: str | None = None,
        api_key: str | None = None,
        org_id: str | None = None,
        auth: AuthConfig | None = None,
        planner_routing: bool | None = None,
        _force_hosted: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Octomil facade.

        Parameters
        ----------
        publishable_key:
            Client-side publishable key (oct_pub_test_... or oct_pub_live_...).
        api_key:
            Server-side API key. Requires ``org_id``.
        org_id:
            Organization ID (required with ``api_key``).
        auth:
            Pre-built AuthConfig object.
        planner_routing:
            Explicit override for planner routing behavior.
            - ``None`` (default): ON when credentials exist, OFF otherwise.
              Respects ``OCTOMIL_DISABLE_PLANNER=1`` env var.
            - ``True``: force planner routing ON.
            - ``False``: force planner routing OFF (direct/legacy only).
        """
        self._initialized = False
        self._kwargs = kwargs
        self._client: Any = None
        self._responses_wrapper: FacadeResponses | None = None
        self._embeddings_wrapper: FacadeEmbeddings | None = None
        self._force_hosted = _force_hosted

        if auth is not None:
            self._auth: AuthConfig = auth
        elif publishable_key is not None:
            # Validate prefix eagerly via PublishableKeyAuth (raises on bad prefix)
            pub = PublishableKeyAuth(key=publishable_key)
            # Store as the auth.py PublishableKeyAuth for OctomilClient compatibility
            from .auth import PublishableKeyAuth as _PubKeyAuth

            self._auth = _PubKeyAuth(api_key=pub.key)
        elif api_key is not None:
            if org_id is None:
                raise ValueError("org_id is required when using api_key authentication")
            from .auth import OrgApiKeyAuth

            self._auth = OrgApiKeyAuth(api_key=api_key, org_id=org_id)
        else:
            raise ValueError("One of publishable_key=, api_key= + org_id=, or auth= must be provided.")

        if self._force_hosted:
            from .auth import OrgApiKeyAuth

            if not isinstance(self._auth, OrgApiKeyAuth):
                raise ValueError("Octomil.hosted_from_env() requires server-side API key authentication.")

        from .planner_defaults import resolve_planner_enabled

        self._planner_enabled = resolve_planner_enabled(
            explicit_override=planner_routing,
            auth=self._auth,
        )

    @classmethod
    def from_env(
        cls,
        *,
        server_key_var: str = "OCTOMIL_SERVER_KEY",
        legacy_api_key_var: str = "OCTOMIL_API_KEY",
        org_id_var: str = "OCTOMIL_ORG_ID",
        api_base_var: str = "OCTOMIL_API_BASE",
        **kwargs: Any,
    ) -> "Octomil":
        """Construct a server-side facade client from environment config.

        ``OCTOMIL_SERVER_KEY`` is the canonical server SDK credential. The
        older ``OCTOMIL_API_KEY`` name is still accepted as a compatibility
        fallback so existing deployments keep working.
        """
        return cls(
            auth=cls._org_auth_from_env(
                server_key_var=server_key_var,
                legacy_api_key_var=legacy_api_key_var,
                org_id_var=org_id_var,
                api_base_var=api_base_var,
                caller="Octomil.from_env()",
            ),
            **kwargs,
        )

    @classmethod
    def hosted_from_env(
        cls,
        *,
        server_key_var: str = "OCTOMIL_SERVER_KEY",
        legacy_api_key_var: str = "OCTOMIL_API_KEY",
        org_id_var: str = "OCTOMIL_ORG_ID",
        api_base_var: str = "OCTOMIL_API_BASE",
        **kwargs: Any,
    ) -> "Octomil":
        """Construct an explicit hosted/cloud-only client from environment config.

        This is the server-side escape hatch for hosted REST behavior. It
        bypasses local planner/runtime selection for Responses and dispatches
        through the hosted OpenAI-compatible gateway instead.
        """
        # False/None are accepted for shared kwargs call sites; True would
        # contradict this constructor's explicit cloud-only contract.
        planner_routing = kwargs.pop("planner_routing", None)
        if planner_routing is True:
            raise ValueError("Octomil.hosted_from_env() is always cloud-only; do not pass planner_routing=True.")

        return cls(
            auth=cls._org_auth_from_env(
                server_key_var=server_key_var,
                legacy_api_key_var=legacy_api_key_var,
                org_id_var=org_id_var,
                api_base_var=api_base_var,
                caller="Octomil.hosted_from_env()",
            ),
            planner_routing=False,
            _force_hosted=True,
            **kwargs,
        )

    @staticmethod
    def _org_auth_from_env(
        *,
        server_key_var: str,
        legacy_api_key_var: str,
        org_id_var: str,
        api_base_var: str,
        caller: str,
    ) -> AuthConfig:
        api_key = os.environ.get(server_key_var) or os.environ.get(legacy_api_key_var)
        if not api_key:
            raise ValueError(
                f"Set {server_key_var} before calling {caller} "
                f"(or set {legacy_api_key_var} for legacy compatibility)."
            )

        org_id = os.environ.get(org_id_var)
        if not org_id:
            raise ValueError(f"Set {org_id_var} before calling {caller}.")

        from .auth import OrgApiKeyAuth

        api_base = os.environ.get(api_base_var)
        if api_base:
            return OrgApiKeyAuth(api_key=api_key, org_id=org_id, api_base=api_base)
        return OrgApiKeyAuth(api_key=api_key, org_id=org_id)

    @property
    def planner_enabled(self) -> bool:
        """Whether planner routing is active for this client."""
        return self._planner_enabled

    async def initialize(self) -> None:
        """Validate auth and prepare the underlying client. Idempotent."""
        if self._initialized:
            return

        from .client import OctomilClient

        self._client = OctomilClient(
            auth=self._auth,
            planner_enabled=self._planner_enabled,
            **self._kwargs,
        )
        responses = self._build_hosted_responses() if self._force_hosted else self._client.responses
        self._responses_wrapper = FacadeResponses(responses)
        self._embeddings_wrapper = FacadeEmbeddings(self._client)
        self._initialized = True

    def _build_hosted_responses(self) -> OctomilResponses:
        """Build a Responses namespace that always dispatches through hosted cloud."""
        from .auth import OrgApiKeyAuth
        from .config.local import CloudProfile
        from .execution.cloud_dispatch import _openai_base_url
        from .responses import OctomilResponses
        from .runtime.core.cloud_runtime import CloudModelRuntime
        from .runtime.core.policy import RoutingPolicy

        if not isinstance(self._auth, OrgApiKeyAuth):
            raise ValueError("Octomil.hosted_from_env() requires server-side API key authentication.")

        base_url = _openai_base_url(CloudProfile(name="hosted", base_url=self._auth.api_base))

        def _resolve_cloud_runtime(model_id: str) -> CloudModelRuntime:
            return CloudModelRuntime(base_url=base_url, api_key=self._auth.api_key, model=model_id)

        return OctomilResponses(
            runtime_resolver=_resolve_cloud_runtime,
            telemetry_reporter=getattr(self._client, "_reporter", None),
            default_routing_policy=RoutingPolicy.cloud_only(),
            planner_enabled=False,
        )

    @property
    def responses(self) -> FacadeResponses:
        """Access the responses API. Requires initialize() to have been called."""
        if not self._initialized:
            raise OctomilNotInitializedError()
        assert self._responses_wrapper is not None
        return self._responses_wrapper

    @property
    def embeddings(self) -> FacadeEmbeddings:
        """Access the embeddings API. Requires initialize() to have been called."""
        if not self._initialized:
            raise OctomilNotInitializedError()
        assert self._embeddings_wrapper is not None
        return self._embeddings_wrapper

    @classmethod
    def local(cls, *, model: str = "default", engine: str | None = None) -> "LocalOctomil":
        """Create a local-only client backed by the invisible local runner.

        No server key required. The runner starts automatically on first use
        and shuts down after an idle timeout.

        Usage::

            client = Octomil.local()
            await client.initialize()
            response = await client.responses.create(model="default", input="Hello!")
        """
        return LocalOctomil(model=model, engine=engine)


# ---------------------------------------------------------------------------
# Local facade — backed by the invisible local runner
# ---------------------------------------------------------------------------


class LocalFacadeResponses:
    """Responses namespace backed by the local runner."""

    def __init__(self, runner_client: LocalRunnerClient, model: str) -> None:
        self._client = runner_client
        self._model = model

    async def create(
        self,
        request_or_model: Any = None,
        *,
        model: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        resolved_model = request_or_model if isinstance(request_or_model, str) else (model or self._model)
        if input is None:
            raise TypeError("create() requires input= argument")
        return await self._client.create_response(model=resolved_model, input=input, **kwargs)


class LocalFacadeEmbeddings:
    """Embeddings namespace backed by the local runner."""

    def __init__(self, runner_client: LocalRunnerClient, model: str) -> None:
        self._client = runner_client
        self._model = model

    async def create(
        self,
        *,
        model: str | None = None,
        input: str | list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        texts = [input] if isinstance(input, str) else input
        return await self._client.create_embedding(model=model or self._model, input=texts, **kwargs)


class LocalOctomil:
    """Local-only Octomil client backed by the invisible local runner.

    Does not require a server key. Uses the ``LocalRunnerManager`` to
    start/reuse a background inference server on ``127.0.0.1``.
    """

    def __init__(self, *, model: str = "default", engine: str | None = None) -> None:
        self._model = model
        self._engine = engine
        self._initialized = False
        self._handle: LocalRunnerHandle | None = None
        self._runner_client: LocalRunnerClient | None = None
        self._responses_wrapper: LocalFacadeResponses | None = None
        self._embeddings_wrapper: LocalFacadeEmbeddings | None = None

    async def initialize(self) -> None:
        """Ensure a local runner is running and initialize the client."""
        if self._initialized:
            return

        from .local_runner.manager import LocalRunnerManager

        mgr = LocalRunnerManager()

        # Resolve model from config if "default"
        effective_model = self._model
        if effective_model == "default":
            try:
                from .execution.kernel import ExecutionKernel

                kernel = ExecutionKernel()
                defaults = kernel.resolve_chat_defaults()
                if defaults and defaults.model:
                    effective_model = defaults.model
            except Exception:
                pass

        if effective_model == "default":
            raise ValueError(
                "No default chat model configured. Pass model= explicitly or set a default in .octomil.toml."
            )

        self._handle = mgr.ensure(model=effective_model, engine=self._engine)

        from .local_runner.client import LocalRunnerClient

        self._runner_client = LocalRunnerClient(self._handle.base_url, self._handle.token)
        self._responses_wrapper = LocalFacadeResponses(self._runner_client, effective_model)
        self._embeddings_wrapper = LocalFacadeEmbeddings(self._runner_client, effective_model)
        self._initialized = True

    @property
    def responses(self) -> LocalFacadeResponses:
        if not self._initialized:
            raise OctomilNotInitializedError()
        assert self._responses_wrapper is not None
        return self._responses_wrapper

    @property
    def embeddings(self) -> LocalFacadeEmbeddings:
        if not self._initialized:
            raise OctomilNotInitializedError()
        assert self._embeddings_wrapper is not None
        return self._embeddings_wrapper
