"""Tests for octomil.control — device registration and heartbeat namespace."""

import time
import unittest
from unittest.mock import patch

from octomil.control import (
    ControlSyncResult,
    DeviceRegistration,
    HeartbeatResponse,
    OctomilControl,
    _get_sdk_version,
)


class _StubApi:
    """Stub _ApiClient that records calls and returns canned responses."""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, str, dict | None]] = []
        self._responses = responses or {}

    def get(self, path, params=None):
        self.calls.append(("get", path, params))
        return self._responses.get(("get", path), {"path": path, "params": params})

    def post(self, path, payload=None):
        self.calls.append(("post", path, payload))
        return self._responses.get(("post", path), {"path": path, "payload": payload})


class TestOctomilControlRegister(unittest.TestCase):
    def test_register_sends_correct_endpoint_and_payload(self):
        register_response = {
            "id": "dev_001",
            "status": "active",
            "metadata": {"region": "us-east"},
        }
        api = _StubApi(responses={("post", "/devices/register"): register_response})
        ctrl = OctomilControl(api=api, org_id="org_test")

        result = ctrl.register(device_id="my-device-123")

        self.assertIsInstance(result, DeviceRegistration)
        self.assertEqual(result.id, "dev_001")
        self.assertEqual(result.device_identifier, "my-device-123")
        self.assertEqual(result.org_id, "org_test")
        self.assertEqual(result.status, "active")
        self.assertEqual(result.metadata, {"region": "us-east"})

        # Verify API call
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/devices/register")
        self.assertEqual(payload["device_identifier"], "my-device-123")
        self.assertEqual(payload["org_id"], "org_test")
        self.assertIn("sdk_version", payload)

    def test_register_uses_device_info_when_no_id_provided(self):
        register_response = {"id": "dev_auto", "status": "active", "metadata": {}}
        api = _StubApi(responses={("post", "/devices/register"): register_response})
        ctrl = OctomilControl(api=api, org_id="org_test")

        with patch("octomil.device_info.get_stable_device_id", return_value="auto-id-abc"):
            result = ctrl.register()

        self.assertEqual(result.device_identifier, "auto-id-abc")
        _, _, payload = api.calls[-1]
        self.assertEqual(payload["device_identifier"], "auto-id-abc")

    def test_register_sets_server_device_id(self):
        api = _StubApi(responses={("post", "/devices/register"): {"id": "srv_42"}})
        ctrl = OctomilControl(api=api, org_id="org_test")

        ctrl.register(device_id="dev_x")

        self.assertEqual(ctrl._server_device_id, "srv_42")


class TestOctomilControlHeartbeat(unittest.TestCase):
    def test_heartbeat_raises_when_not_registered(self):
        api = _StubApi()
        ctrl = OctomilControl(api=api, org_id="org_test")

        with self.assertRaises(RuntimeError) as ctx:
            ctrl.heartbeat()
        self.assertIn("not registered", str(ctx.exception).lower())

    def test_heartbeat_sends_to_correct_endpoint(self):
        heartbeat_response = {
            "status": "ok",
            "server_time": "2026-03-12T20:00:00Z",
            "metadata": {"poll_interval": 300},
        }
        api = _StubApi(
            responses={
                ("post", "/devices/register"): {"id": "dev_hb"},
                ("post", "/devices/dev_hb/heartbeat"): heartbeat_response,
            }
        )
        ctrl = OctomilControl(api=api, org_id="org_test")
        ctrl.register(device_id="my-device")

        result = ctrl.heartbeat()

        self.assertIsInstance(result, HeartbeatResponse)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.server_time, "2026-03-12T20:00:00Z")
        self.assertEqual(result.metadata, {"poll_interval": 300})

        # Check correct call
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/devices/dev_hb/heartbeat")
        self.assertEqual(payload["platform"], "python")
        self.assertIn("sdk_version", payload)
        self.assertIn("os_version", payload)

    def test_heartbeat_includes_metadata(self):
        api = _StubApi(
            responses={
                ("post", "/devices/register"): {"id": "dev_meta"},
                ("post", "/devices/dev_meta/heartbeat"): {"status": "ok"},
            }
        )
        ctrl = OctomilControl(api=api, org_id="org_test")
        ctrl.register(device_id="dev")

        ctrl.heartbeat()

        _, _, payload = api.calls[-1]
        # metadata key should be present (from DeviceInfo.update_metadata)
        self.assertIn("metadata", payload)


class TestOctomilControlRefresh(unittest.TestCase):
    def test_refresh_returns_sync_result_when_not_registered(self):
        api = _StubApi()
        ctrl = OctomilControl(api=api, org_id="org_test")

        result = ctrl.refresh()

        self.assertIsInstance(result, ControlSyncResult)
        self.assertFalse(result.updated)
        self.assertEqual(result.config_version, "")
        self.assertFalse(result.assignments_changed)
        self.assertFalse(result.rollouts_changed)
        self.assertTrue(result.fetched_at.endswith("Z"))
        self.assertEqual(len(api.calls), 0)

    def test_refresh_calls_assignments_endpoint(self):
        api = _StubApi(
            responses={
                ("post", "/devices/register"): {"id": "dev_ref"},
                ("get", "/devices/dev_ref/assignments"): {
                    "updated": True,
                    "config_version": "v3",
                    "assignments_changed": True,
                    "rollouts_changed": False,
                },
            }
        )
        ctrl = OctomilControl(api=api, org_id="org_test")
        ctrl.register(device_id="dev")

        result = ctrl.refresh()

        method, path, _ = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/devices/dev_ref/assignments")

        self.assertIsInstance(result, ControlSyncResult)
        self.assertTrue(result.updated)
        self.assertEqual(result.config_version, "v3")
        self.assertTrue(result.assignments_changed)
        self.assertFalse(result.rollouts_changed)
        self.assertTrue(result.fetched_at.endswith("Z"))

    def test_refresh_infers_assignments_changed_from_assignments_list(self):
        api = _StubApi(
            responses={
                ("post", "/devices/register"): {"id": "dev_inf"},
                ("get", "/devices/dev_inf/assignments"): {
                    "assignments": [{"id": "a1"}],
                },
            }
        )
        ctrl = OctomilControl(api=api, org_id="org_test")
        ctrl.register(device_id="dev")

        result = ctrl.refresh()

        self.assertTrue(result.assignments_changed)


class TestOctomilControlHeartbeatLoop(unittest.TestCase):
    def test_start_and_stop_heartbeat(self):
        heartbeat_count = 0

        class _CountingApi(_StubApi):
            def post(self_, path, payload=None):
                nonlocal heartbeat_count
                result = super().post(path, payload)
                if "heartbeat" in path:
                    heartbeat_count += 1
                return result

        api = _CountingApi(
            responses={
                ("post", "/devices/register"): {"id": "dev_loop"},
                ("post", "/devices/dev_loop/heartbeat"): {"status": "ok"},
            }
        )
        ctrl = OctomilControl(api=api, org_id="org_test")
        ctrl.register(device_id="dev")

        # Start with a very short interval
        ctrl.start_heartbeat(interval_seconds=0.05)
        time.sleep(0.2)
        ctrl.stop_heartbeat()

        # At least one heartbeat should have fired
        self.assertGreaterEqual(heartbeat_count, 1)
        self.assertIsNone(ctrl._heartbeat_thread)

    def test_stop_heartbeat_is_idempotent(self):
        api = _StubApi()
        ctrl = OctomilControl(api=api, org_id="org_test")

        # Calling stop without start should not raise
        ctrl.stop_heartbeat()
        ctrl.stop_heartbeat()

    def test_start_heartbeat_stops_previous(self):
        api = _StubApi(
            responses={
                ("post", "/devices/register"): {"id": "dev_restart"},
                ("post", "/devices/dev_restart/heartbeat"): {"status": "ok"},
            }
        )
        ctrl = OctomilControl(api=api, org_id="org_test")
        ctrl.register(device_id="dev")

        ctrl.start_heartbeat(interval_seconds=0.05)
        first_thread = ctrl._heartbeat_thread
        self.assertIsNotNone(first_thread)

        # Starting again should stop the previous thread
        ctrl.start_heartbeat(interval_seconds=0.05)
        second_thread = ctrl._heartbeat_thread
        self.assertIsNotNone(second_thread)
        self.assertIsNot(first_thread, second_thread)

        ctrl.stop_heartbeat()

    def test_heartbeat_loop_continues_on_error(self):
        call_count = 0

        class _FailingApi(_StubApi):
            def post(self_, path, payload=None):
                nonlocal call_count
                if "heartbeat" in path:
                    call_count += 1
                    if call_count == 1:
                        raise RuntimeError("transient error")
                return super().post(path, payload)

        api = _FailingApi(
            responses={
                ("post", "/devices/register"): {"id": "dev_err"},
                ("post", "/devices/dev_err/heartbeat"): {"status": "ok"},
            }
        )
        ctrl = OctomilControl(api=api, org_id="org_test")
        ctrl.register(device_id="dev")

        ctrl.start_heartbeat(interval_seconds=0.05)
        time.sleep(0.25)
        ctrl.stop_heartbeat()

        # Despite the first heartbeat failing, loop should have continued
        self.assertGreaterEqual(call_count, 2)


class TestGetSdkVersion(unittest.TestCase):
    def test_returns_version_string(self):
        version = _get_sdk_version()
        # Should return the actual version from octomil.__version__
        self.assertIsInstance(version, str)
        self.assertNotEqual(version, "0.0.0")


class TestDataclasses(unittest.TestCase):
    def test_device_registration_defaults(self):
        reg = DeviceRegistration(id="x", device_identifier="dev", org_id="org")
        self.assertEqual(reg.status, "active")
        self.assertEqual(reg.metadata, {})

    def test_heartbeat_response_defaults(self):
        hb = HeartbeatResponse()
        self.assertEqual(hb.status, "ok")
        self.assertIsNone(hb.server_time)
        self.assertEqual(hb.metadata, {})

    def test_control_sync_result_fields(self):
        sr = ControlSyncResult(
            updated=True,
            config_version="v2",
            assignments_changed=True,
            rollouts_changed=False,
            fetched_at="2026-03-12T12:00:00.000Z",
        )
        self.assertTrue(sr.updated)
        self.assertEqual(sr.config_version, "v2")
        self.assertTrue(sr.assignments_changed)
        self.assertFalse(sr.rollouts_changed)
        self.assertEqual(sr.fetched_at, "2026-03-12T12:00:00.000Z")


if __name__ == "__main__":
    unittest.main()
