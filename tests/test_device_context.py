"""Tests for octomil.device_context — device identity and registration state."""

import time
import unittest
from pathlib import Path
from unittest.mock import patch

from octomil.device_context import (
    DeviceContext,
    RegistrationState,
    TokenState,
    get_or_create_installation_id,
)


class TestTokenState(unittest.TestCase):
    def test_none_state(self):
        ts = TokenState()
        self.assertTrue(ts.is_none)
        self.assertFalse(ts.is_valid)
        self.assertFalse(ts.is_expired)

    def test_valid_token_no_expiry(self):
        ts = TokenState(access_token="tok_abc")
        self.assertFalse(ts.is_none)
        self.assertTrue(ts.is_valid)
        self.assertFalse(ts.is_expired)

    def test_valid_token_future_expiry(self):
        ts = TokenState(access_token="tok_abc", expires_at=time.time() + 3600)
        self.assertTrue(ts.is_valid)
        self.assertFalse(ts.is_expired)

    def test_expired_token(self):
        ts = TokenState(access_token="tok_abc", expires_at=time.time() - 10)
        self.assertFalse(ts.is_valid)
        self.assertTrue(ts.is_expired)


class TestRegistrationState(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(RegistrationState.PENDING.value, "pending")
        self.assertEqual(RegistrationState.REGISTERED.value, "registered")
        self.assertEqual(RegistrationState.FAILED.value, "failed")


class TestDeviceContext(unittest.TestCase):
    def test_default_state(self):
        ctx = DeviceContext()
        self.assertIsNone(ctx.org_id)
        self.assertIsNone(ctx.app_id)
        self.assertIsNone(ctx.server_device_id)
        self.assertEqual(ctx.registration_state, RegistrationState.PENDING)
        self.assertTrue(ctx.token_state.is_none)

    def test_auth_headers_none_when_no_token(self):
        ctx = DeviceContext()
        self.assertIsNone(ctx.auth_headers())

    def test_auth_headers_with_valid_token(self):
        ctx = DeviceContext()
        ctx.token_state = TokenState(access_token="tok_xyz", expires_at=time.time() + 3600)
        headers = ctx.auth_headers()
        self.assertIsNotNone(headers)
        self.assertEqual(headers, {"Authorization": "Bearer tok_xyz"})

    def test_auth_headers_none_with_expired_token(self):
        ctx = DeviceContext()
        ctx.token_state = TokenState(access_token="tok_old", expires_at=time.time() - 10)
        self.assertIsNone(ctx.auth_headers())

    def test_telemetry_resource_keys(self):
        ctx = DeviceContext(org_id="org_test", server_device_id="dev_123")
        resource = ctx.telemetry_resource()
        self.assertEqual(resource["service.name"], "octomil-sdk")
        self.assertEqual(resource["telemetry.sdk.language"], "python")
        self.assertEqual(resource["octomil.org_id"], "org_test")
        self.assertEqual(resource["octomil.device_id"], "dev_123")
        self.assertIn("octomil.installation_id", resource)
        self.assertIn("os.type", resource)

    def test_telemetry_resource_omits_null_org(self):
        ctx = DeviceContext()
        resource = ctx.telemetry_resource()
        self.assertNotIn("octomil.org_id", resource)
        self.assertNotIn("octomil.device_id", resource)

    def test_state_transitions(self):
        ctx = DeviceContext()
        self.assertEqual(ctx.registration_state, RegistrationState.PENDING)

        ctx.registration_state = RegistrationState.REGISTERED
        self.assertEqual(ctx.registration_state, RegistrationState.REGISTERED)

        ctx.registration_state = RegistrationState.FAILED
        self.assertEqual(ctx.registration_state, RegistrationState.FAILED)


class TestInstallationId(unittest.TestCase):
    def test_get_or_create_returns_uuid_format(self):
        """installation_id should look like a UUID."""
        import uuid

        with (
            patch.object(Path, "exists", return_value=False),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_text"),
        ):
            iid = get_or_create_installation_id()
            # Should be a valid UUID
            parsed = uuid.UUID(iid)
            self.assertEqual(str(parsed), iid)

    def test_reads_existing_id(self):
        existing_uuid = "12345678-1234-5678-1234-567812345678"
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=existing_uuid + "\n"),
        ):
            iid = get_or_create_installation_id()
            self.assertEqual(iid, existing_uuid)

    def test_creates_new_id_on_invalid_file(self):
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value="not-a-uuid\n"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_text"),
        ):
            iid = get_or_create_installation_id()
            import uuid

            uuid.UUID(iid)  # should not raise

    def test_installation_id_persisted_on_context(self):
        """Once loaded, installation_id should be cached on the context."""
        with patch(
            "octomil.device_context.get_or_create_installation_id",
            return_value="cached-uuid-1234-5678-1234-567812345678",
        ):
            ctx = DeviceContext()
            iid1 = ctx.installation_id
            iid2 = ctx.installation_id
            self.assertEqual(iid1, iid2)


if __name__ == "__main__":
    unittest.main()
