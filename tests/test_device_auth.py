import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from edgeml.python.edgeml.auth import DeviceAuthClient, DeviceTokenState


class _FakeKeyring:
    def __init__(self):
        self.store = {}

    def set_password(self, service, key, value):
        self.store[(service, key)] = value

    def get_password(self, service, key):
        return self.store.get((service, key))

    def delete_password(self, service, key):
        self.store.pop((service, key), None)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self._requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, headers=None):
        self._requests.append((url, json, headers))
        if url.endswith("/bootstrap"):
            payload = {
                "access_token": "acc_bootstrap",
                "refresh_token": "ref_bootstrap",
                "token_type": "Bearer",
                "expires_in": 900,
                "org_id": "org_1",
                "device_identifier": "device_1",
                "scopes": ["devices:write"],
            }
            return _FakeResponse(200, payload)
        if url.endswith("/refresh"):
            payload = {
                "access_token": "acc_refresh",
                "refresh_token": "ref_refresh",
                "token_type": "Bearer",
                "expires_in": 900,
                "org_id": "org_1",
                "device_identifier": "device_1",
                "scopes": ["devices:write"],
            }
            return _FakeResponse(200, payload)
        if url.endswith("/revoke"):
            return _FakeResponse(200, {})
        return _FakeResponse(404, {})


class _FailRefreshAsyncClient(_FakeAsyncClient):
    async def post(self, url, json=None, headers=None):
        if url.endswith("/refresh"):
            raise RuntimeError("network down")
        return await super().post(url, json=json, headers=headers)


class DeviceAuthClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_bootstrap_refresh_revoke_lifecycle(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.python.edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.python.edgeml.auth.httpx.AsyncClient", _FakeAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            state = await client.bootstrap(bootstrap_bearer_token="bootstrap_token")
            self.assertEqual(state.access_token, "acc_bootstrap")

            refreshed = await client.refresh()
            self.assertEqual(refreshed.access_token, "acc_refresh")

            await client.revoke()
            self.assertIsNone(client._load_token_state())

    async def test_get_access_token_offline_fallback_before_expiry(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.python.edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.python.edgeml.auth.httpx.AsyncClient", _FailRefreshAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            state = DeviceTokenState(
                access_token="still_valid",
                refresh_token="refresh_token",
                token_type="Bearer",
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                org_id="org_1",
                device_identifier="device_1",
                scopes=["devices:write"],
            )
            client._store_token_state(state)

            token = await client.get_access_token(refresh_if_expiring_within_seconds=600)
            self.assertEqual(token, "still_valid")

    async def test_get_access_token_raises_if_expired_and_refresh_fails(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.python.edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.python.edgeml.auth.httpx.AsyncClient", _FailRefreshAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            state = DeviceTokenState(
                access_token="expired",
                refresh_token="refresh_token",
                token_type="Bearer",
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                org_id="org_1",
                device_identifier="device_1",
                scopes=["devices:write"],
            )
            client._store_token_state(state)

            with self.assertRaises(RuntimeError):
                await client.get_access_token(refresh_if_expiring_within_seconds=30)


if __name__ == "__main__":
    unittest.main()
