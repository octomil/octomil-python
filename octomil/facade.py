"""Unified Octomil facade — simplified entry point for the SDK."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, AsyncIterator

from .auth_config import PublishableKeyAuth

if TYPE_CHECKING:
    from .auth import AuthConfig
    from .client import OctomilClient
    from .embeddings import EmbeddingResult
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
        **kwargs: Any,
    ) -> None:
        self._initialized = False
        self._kwargs = kwargs
        self._client: Any = None
        self._responses_wrapper: FacadeResponses | None = None
        self._embeddings_wrapper: FacadeEmbeddings | None = None

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
        api_key = os.environ.get(server_key_var) or os.environ.get(legacy_api_key_var)
        if not api_key:
            raise ValueError(
                f"Set {server_key_var} before calling Octomil.from_env() "
                f"(or set {legacy_api_key_var} for legacy compatibility)."
            )

        org_id = os.environ.get(org_id_var)
        if not org_id:
            raise ValueError(f"Set {org_id_var} before calling Octomil.from_env().")

        from .auth import OrgApiKeyAuth

        api_base = os.environ.get(api_base_var)
        if api_base:
            return cls(auth=OrgApiKeyAuth(api_key=api_key, org_id=org_id, api_base=api_base), **kwargs)
        return cls(auth=OrgApiKeyAuth(api_key=api_key, org_id=org_id), **kwargs)

    async def initialize(self) -> None:
        """Validate auth and prepare the underlying client. Idempotent."""
        if self._initialized:
            return

        from .client import OctomilClient

        self._client = OctomilClient(auth=self._auth, **self._kwargs)
        self._responses_wrapper = FacadeResponses(self._client.responses)
        self._embeddings_wrapper = FacadeEmbeddings(self._client)
        self._initialized = True

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
