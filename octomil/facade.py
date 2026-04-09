"""Unified Octomil facade — simplified entry point for the SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from .auth_config import PublishableKeyAuth

if TYPE_CHECKING:
    from .auth import AuthConfig
    from .responses.responses import OctomilResponses
    from .responses.types import Response


class OctomilNotInitializedError(Exception):
    def __init__(self) -> None:
        super().__init__("Octomil client is not initialized. Call await client.initialize() first.")


class FacadeResponses:
    """Convenience wrapper over OctomilResponses with a simpler call signature."""

    def __init__(self, responses: OctomilResponses) -> None:
        self._responses = responses

    async def create(self, *, model: str, input: str, **kwargs: Any) -> Response:
        from .responses.types import ResponseRequest

        request = ResponseRequest.text(model, input, **kwargs)
        return await self._responses.create(request)

    async def stream(self, *, model: str, input: str, **kwargs: Any) -> AsyncIterator:
        from .responses.types import ResponseRequest

        request = ResponseRequest.text(model, input, **kwargs)
        async for event in self._responses.stream(request):
            yield event


class Octomil:
    """Unified facade for the Octomil SDK.

    Usage::

        client = Octomil(publishable_key="oct_pub_test_abc123")
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

    async def initialize(self) -> None:
        """Validate auth and prepare the underlying client. Idempotent."""
        if self._initialized:
            return

        from .client import OctomilClient

        self._client = OctomilClient(auth=self._auth, **self._kwargs)
        self._responses_wrapper = FacadeResponses(self._client.responses)
        self._initialized = True

    @property
    def responses(self) -> FacadeResponses:
        """Access the responses API. Requires initialize() to have been called."""
        if not self._initialized:
            raise OctomilNotInitializedError()
        assert self._responses_wrapper is not None
        return self._responses_wrapper
