"""Tests for octomil.configure — silent device registration."""

import time
import unittest
from unittest.mock import patch

from octomil.auth_config import AnonymousAuth, BootstrapTokenAuth, PublishableKeyAuth
from octomil.configure import (
    _background_register,
    _should_auto_register,
    configure,
    get_device_context,
)
from octomil.device_context import DeviceContext, RegistrationState
from octomil.monitoring_config import MonitoringConfig


class TestShouldAutoRegister(unittest.TestCase):
    def test_publishable_key(self):
        auth = PublishableKeyAuth(key="oct_pub_live_abc")
        self.assertTrue(_should_auto_register(auth))

    def test_bootstrap_token(self):
        auth = BootstrapTokenAuth(token="jwt")
        self.assertTrue(_should_auto_register(auth))

    def test_anonymous(self):
        auth = AnonymousAuth(app_id="app")
        self.assertTrue(_should_auto_register(auth))


class TestConfigure(unittest.TestCase):
    def test_returns_device_context(self):
        ctx = configure()
        self.assertIsInstance(ctx, DeviceContext)
        self.assertEqual(ctx.registration_state, RegistrationState.PENDING)

    def test_sets_module_singleton(self):
        ctx = configure()
        self.assertIs(get_device_context(), ctx)

    def test_sets_app_id_from_anonymous_auth(self):
        ctx = configure(auth=AnonymousAuth(app_id="my-app"))
        self.assertEqual(ctx.app_id, "my-app")

    @patch("octomil.configure._do_register")
    def test_background_registration_starts(self, mock_register):
        auth = PublishableKeyAuth(key="oct_pub_test_abc")
        ctx = configure(auth=auth)
        # Give the background thread a moment to run
        time.sleep(0.5)
        mock_register.assert_called_once()
        # First arg is the context
        self.assertIs(mock_register.call_args[0][0], ctx)

    @patch("octomil.configure._do_register")
    def test_no_registration_without_auth(self, mock_register):
        configure(auth=None)
        time.sleep(0.2)
        mock_register.assert_not_called()


class TestBackgroundRegister(unittest.TestCase):
    @patch("octomil.configure.time.sleep")
    @patch("octomil.configure._do_register")
    def test_success_sets_registered(self, mock_register, mock_sleep):
        ctx = DeviceContext()
        auth = PublishableKeyAuth(key="oct_pub_live_abc")
        _background_register(ctx, auth, "https://api.test.com", max_retries=1)
        self.assertEqual(ctx.registration_state, RegistrationState.REGISTERED)
        mock_register.assert_called_once()

    @patch("octomil.configure.time.sleep")
    @patch("octomil.configure._do_register", side_effect=RuntimeError("network"))
    def test_failure_sets_failed_after_retries(self, mock_register, mock_sleep):
        ctx = DeviceContext()
        auth = PublishableKeyAuth(key="oct_pub_live_abc")
        _background_register(ctx, auth, "https://api.test.com", max_retries=2)
        self.assertEqual(ctx.registration_state, RegistrationState.FAILED)
        self.assertEqual(mock_register.call_count, 2)

    @patch("octomil.configure.time.sleep")
    @patch("octomil.configure._do_register")
    def test_success_after_retry(self, mock_register, mock_sleep):
        mock_register.side_effect = [RuntimeError("fail"), None]
        ctx = DeviceContext()
        auth = PublishableKeyAuth(key="oct_pub_live_abc")
        _background_register(ctx, auth, "https://api.test.com", max_retries=3)
        self.assertEqual(ctx.registration_state, RegistrationState.REGISTERED)
        self.assertEqual(mock_register.call_count, 2)


class TestMonitoringConfig(unittest.TestCase):
    def test_defaults(self):
        mc = MonitoringConfig()
        self.assertFalse(mc.enabled)
        self.assertEqual(mc.heartbeat_interval_seconds, 300)

    def test_custom_values(self):
        mc = MonitoringConfig(enabled=True, heartbeat_interval_seconds=60)
        self.assertTrue(mc.enabled)
        self.assertEqual(mc.heartbeat_interval_seconds, 60)


if __name__ == "__main__":
    unittest.main()
