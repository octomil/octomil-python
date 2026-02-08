import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from edgeml.auth import DeviceAuthClient, DeviceTokenState


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
    requests_log = []

    def __init__(self, *args, **kwargs):
        self._requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, headers=None):
        self._requests.append((url, json, headers))
        self.__class__.requests_log.append((url, json, headers))
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
    def setUp(self):
        _FakeAsyncClient.requests_log = []

    async def test_bootstrap_refresh_revoke_lifecycle(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.auth.httpx.AsyncClient", _FakeAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            state = await client.bootstrap(bootstrap_bearer_token="bootstrap_token")
            self.assertEqual(state.access_token, "acc_bootstrap")

            refreshed = await client.refresh()
            self.assertEqual(refreshed.access_token, "acc_refresh")

            await client.revoke()
            self.assertIsNone(client._load_token_state())

    async def test_bootstrap_sends_expected_payload_and_bearer_token(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.auth.httpx.AsyncClient", _FakeAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            await client.bootstrap(
                bootstrap_bearer_token="bootstrap_token",
                scopes=["devices:write", "heartbeat:write"],
                access_ttl_seconds=600,
                device_id="device_db_id",
            )

            bootstrap_url, bootstrap_payload, bootstrap_headers = _FakeAsyncClient.requests_log[0]
            self.assertTrue(bootstrap_url.endswith("/api/v1/device-auth/bootstrap"))
            self.assertEqual(bootstrap_headers["Authorization"], "Bearer bootstrap_token")
            self.assertEqual(bootstrap_payload["org_id"], "org_1")
            self.assertEqual(bootstrap_payload["device_identifier"], "device_1")
            self.assertEqual(bootstrap_payload["access_ttl_seconds"], 600)
            self.assertEqual(bootstrap_payload["device_id"], "device_db_id")
            self.assertEqual(bootstrap_payload["scopes"], ["devices:write", "heartbeat:write"])

    async def test_refresh_uses_rotated_refresh_token(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.auth.httpx.AsyncClient", _FakeAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            await client.bootstrap(bootstrap_bearer_token="bootstrap_token")
            await client.refresh()
            await client.refresh()

            refresh_requests = [
                payload for url, payload, _headers in _FakeAsyncClient.requests_log if url.endswith("/api/v1/device-auth/refresh")
            ]
            self.assertEqual(refresh_requests[0]["refresh_token"], "ref_bootstrap")
            self.assertEqual(refresh_requests[1]["refresh_token"], "ref_refresh")

    async def test_get_access_token_offline_fallback_before_expiry(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.auth.httpx.AsyncClient", _FailRefreshAsyncClient
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
        with patch("edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.auth.httpx.AsyncClient", _FailRefreshAsyncClient
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

    async def test_revoke_failure_preserves_stored_state(self):
        fake_keyring = _FakeKeyring()

        class _FailRevokeAsyncClient(_FakeAsyncClient):
            async def post(self, url, json=None, headers=None):
                if url.endswith("/revoke"):
                    raise RuntimeError("network down")
                return await super().post(url, json=json, headers=headers)

        with patch("edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.auth.httpx.AsyncClient", _FailRevokeAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            await client.bootstrap(bootstrap_bearer_token="bootstrap_token")

            with self.assertRaises(RuntimeError):
                await client.revoke()
            self.assertIsNotNone(client._load_token_state())


    async def test_device_token_state_from_response_with_expires_at(self):
        payload = {
            "access_token": "acc_token",
            "refresh_token": "ref_token",
            "token_type": "Bearer",
            "expires_at": "2024-12-31T23:59:59Z",
            "org_id": "org_1",
            "device_identifier": "device_1",
            "scopes": ["devices:write"],
        }
        state = DeviceTokenState.from_response(payload)
        self.assertEqual(state.access_token, "acc_token")
        self.assertEqual(state.expires_at.year, 2024)

    async def test_device_token_state_from_response_with_expires_in(self):
        payload = {
            "access_token": "acc_token",
            "refresh_token": "ref_token",
            "expires_in": 900,
            "org_id": "org_1",
            "device_identifier": "device_1",
        }
        state = DeviceTokenState.from_response(payload)
        self.assertEqual(state.access_token, "acc_token")
        self.assertEqual(state.token_type, "Bearer")
        self.assertEqual(state.scopes, [])

    async def test_device_token_state_serialization(self):
        state = DeviceTokenState(
            access_token="acc",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            org_id="org_1",
            device_identifier="device_1",
            scopes=["devices:write"],
        )
        json_str = state.to_json()
        restored = DeviceTokenState.from_json(json_str)
        self.assertEqual(restored.access_token, state.access_token)
        self.assertEqual(restored.refresh_token, state.refresh_token)
        self.assertEqual(restored.org_id, state.org_id)

    async def test_clear_token_state(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.auth.httpx.AsyncClient", _FakeAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            await client.bootstrap(bootstrap_bearer_token="bootstrap_token")
            self.assertIsNotNone(client._load_token_state())
            client.clear_token_state()
            self.assertIsNone(client._load_token_state())

    def test_get_access_token_sync_outside_loop(self):
        """Test get_access_token_sync works when called outside an event loop"""
        import asyncio
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.auth.httpx.AsyncClient", _FakeAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            state = DeviceTokenState(
                access_token="valid_token",
                refresh_token="refresh_token",
                token_type="Bearer",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                org_id="org_1",
                device_identifier="device_1",
                scopes=["devices:write"],
            )
            client._store_token_state(state)
            # This is a non-async test method, so no event loop is running
            token = client.get_access_token_sync()
            self.assertEqual(token, "valid_token")

    async def test_get_access_token_sync_raises_inside_loop(self):
        """Test get_access_token_sync raises when called inside an event loop"""
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            # This async test method runs inside an event loop, so sync call should raise
            with self.assertRaises(RuntimeError) as ctx:
                client.get_access_token_sync()
            self.assertIn("cannot be called inside an active event loop", str(ctx.exception))

    async def test_get_access_token_no_token_state(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            with self.assertRaises(RuntimeError) as ctx:
                await client.get_access_token()
            self.assertIn("No token state found", str(ctx.exception))

    async def test_refresh_no_token_state(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            with self.assertRaises(RuntimeError) as ctx:
                await client.refresh()
            self.assertIn("No token state found", str(ctx.exception))

    async def test_revoke_with_no_state_is_noop(self):
        fake_keyring = _FakeKeyring()
        with patch("edgeml.auth.keyring", fake_keyring), patch(
            "edgeml.auth.httpx.AsyncClient", _FakeAsyncClient
        ):
            client = DeviceAuthClient(base_url="https://api.example.com", org_id="org_1", device_identifier="device_1")
            await client.revoke()


if __name__ == "__main__":
    unittest.main()
