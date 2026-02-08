"""
Device auth runtime client for bootstrap/refresh/revoke token flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

try:
    import keyring
except ImportError:  # pragma: no cover - optional dependency
    keyring = None


@dataclass
class DeviceTokenState:
    access_token: str
    refresh_token: str
    token_type: str
    expires_at: datetime
    org_id: str
    device_identifier: str
    scopes: list[str]

    @classmethod
    def from_response(cls, payload: dict[str, Any]) -> "DeviceTokenState":
        expires_at = payload.get("expires_at")
        if expires_at:
            expires_at_dt = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        else:
            expires_in = int(payload.get("expires_in", 0))
            expires_at_dt = datetime.now(timezone.utc) + timedelta(seconds=max(expires_in, 0))
        return cls(
            access_token=str(payload["access_token"]),
            refresh_token=str(payload["refresh_token"]),
            token_type=str(payload.get("token_type", "Bearer")),
            expires_at=expires_at_dt,
            org_id=str(payload["org_id"]),
            device_identifier=str(payload["device_identifier"]),
            scopes=[str(scope) for scope in payload.get("scopes", [])],
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "token_type": self.token_type,
                "expires_at": self.expires_at.isoformat(),
                "org_id": self.org_id,
                "device_identifier": self.device_identifier,
                "scopes": self.scopes,
            }
        )

    @classmethod
    def from_json(cls, value: str) -> "DeviceTokenState":
        payload = json.loads(value)
        payload["expires_at"] = datetime.fromisoformat(payload["expires_at"])
        return cls(**payload)


class DeviceAuthClient:
    """
    Device auth manager with secure token persistence in OS keyring.
    """

    def __init__(
        self,
        *,
        base_url: str,
        org_id: str,
        device_identifier: str,
        keyring_service: str = "edgeml",
        keyring_username: Optional[str] = None,
        timeout_seconds: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.org_id = org_id
        self.device_identifier = device_identifier
        self._storage_service = keyring_service
        self._storage_key = keyring_username or f"{org_id}:{device_identifier}"
        self._timeout = timeout_seconds

        if keyring is None:
            raise RuntimeError("Install optional dependency `keyring` to use DeviceAuthClient")

    def _store_token_state(self, state: DeviceTokenState) -> None:
        keyring.set_password(self._storage_service, self._storage_key, state.to_json())

    def _load_token_state(self) -> Optional[DeviceTokenState]:
        raw = keyring.get_password(self._storage_service, self._storage_key)
        if not raw:
            return None
        return DeviceTokenState.from_json(raw)

    def clear_token_state(self) -> None:
        try:
            keyring.delete_password(self._storage_service, self._storage_key)
        except Exception as e:
            logger.debug("Failed to delete password from keyring: %s", e)

    async def bootstrap(
        self,
        *,
        bootstrap_bearer_token: str,
        scopes: Optional[list[str]] = None,
        access_ttl_seconds: Optional[int] = None,
        device_id: Optional[str] = None,
    ) -> DeviceTokenState:
        payload: dict[str, Any] = {
            "org_id": self.org_id,
            "device_identifier": self.device_identifier,
            "scopes": scopes or ["devices:write"],
        }
        if access_ttl_seconds is not None:
            payload["access_ttl_seconds"] = access_ttl_seconds
        if device_id:
            payload["device_id"] = device_id

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/device-auth/bootstrap",
                json=payload,
                headers={"Authorization": f"Bearer {bootstrap_bearer_token}"},
            )
            response.raise_for_status()
            state = DeviceTokenState.from_response(response.json())
            self._store_token_state(state)
            return state

    async def refresh(self) -> DeviceTokenState:
        state = self._load_token_state()
        if not state:
            raise RuntimeError("No token state found in keyring")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/device-auth/refresh",
                json={"refresh_token": state.refresh_token},
            )
            response.raise_for_status()
            next_state = DeviceTokenState.from_response(response.json())
            self._store_token_state(next_state)
            return next_state

    async def revoke(self, reason: str = "sdk_revoke") -> None:
        state = self._load_token_state()
        if not state:
            return
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/device-auth/revoke",
                json={"refresh_token": state.refresh_token, "reason": reason},
            )
            if response.status_code not in (200, 204):
                response.raise_for_status()
        self.clear_token_state()

    async def get_access_token(self, refresh_if_expiring_within_seconds: int = 30) -> str:
        state = self._load_token_state()
        if not state:
            raise RuntimeError("No token state found in keyring")
        now = datetime.now(timezone.utc)
        if now + timedelta(seconds=refresh_if_expiring_within_seconds) >= state.expires_at:
            try:
                state = await self.refresh()
            except Exception:
                # Offline-safe fallback: keep using current token until hard expiry.
                if now < state.expires_at:
                    return state.access_token
                raise
        return state.access_token

    def get_access_token_sync(self, refresh_if_expiring_within_seconds: int = 30) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.get_access_token(refresh_if_expiring_within_seconds))
        raise RuntimeError(
            "get_access_token_sync cannot be called inside an active event loop; "
            "use await get_access_token(...) instead."
        )
