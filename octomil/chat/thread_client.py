"""Thread client for managing chat threads via the server API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import ChatThread


class ThreadClient:
    """Manages chat threads via the Octomil server API.

    Accessed via ``client.chat.threads``.
    """

    def __init__(self, *, server_url: str, api_key: str) -> None:
        self._server_url = server_url.rstrip("/")
        self._api_key = api_key

    async def create(
        self,
        *,
        model: str,
        title: str | None = None,
        binding_key: str | None = None,
        storage_mode: str | None = None,
        retention_policy: str | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatThread:
        """Create a new chat thread."""
        import httpx

        from .types import ChatThread

        body: dict[str, Any] = {"model": model}
        if title is not None:
            body["title"] = title
        if binding_key is not None:
            body["bindingKey"] = binding_key
        if storage_mode is not None:
            body["storageMode"] = storage_mode
        if retention_policy is not None:
            body["retentionPolicy"] = retention_policy
        if ttl_seconds is not None:
            body["ttlSeconds"] = ttl_seconds
        if metadata is not None:
            body["metadata"] = metadata

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._server_url}/api/v1/chat/threads",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

        return ChatThread(
            id=data["id"],
            model=data["model"],
            created_at=data["createdAt"],
            updated_at=data["updatedAt"],
            title=data.get("title"),
            binding_key=data.get("bindingKey"),
            storage_mode=data.get("storageMode"),
            retention_policy=data.get("retentionPolicy"),
            metadata=data.get("metadata", {}),
        )

    async def get(self, thread_id: str) -> ChatThread:
        """Get a chat thread by ID."""
        import httpx

        from .types import ChatThread

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._server_url}/api/v1/chat/threads/{thread_id}",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

        return ChatThread(
            id=data["id"],
            model=data["model"],
            created_at=data["createdAt"],
            updated_at=data["updatedAt"],
            title=data.get("title"),
            binding_key=data.get("bindingKey"),
            storage_mode=data.get("storageMode"),
            retention_policy=data.get("retentionPolicy"),
            metadata=data.get("metadata", {}),
        )

    async def list(
        self,
        *,
        limit: int = 20,
        order: str = "desc",
    ) -> list[ChatThread]:
        """List chat threads."""
        import httpx

        from .types import ChatThread

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._server_url}/api/v1/chat/threads",
                params={"limit": limit, "order": order},
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            items = resp.json()

        return [
            ChatThread(
                id=d["id"],
                model=d["model"],
                created_at=d["createdAt"],
                updated_at=d["updatedAt"],
                title=d.get("title"),
                binding_key=d.get("bindingKey"),
                storage_mode=d.get("storageMode"),
                retention_policy=d.get("retentionPolicy"),
                metadata=d.get("metadata", {}),
            )
            for d in items
        ]
