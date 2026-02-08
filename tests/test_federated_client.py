import unittest
from unittest.mock import Mock

from octomil.api_client import OctomilClientError
from octomil.federated_client import FederatedClient


class _StubApi:
    def __init__(self):
        self.calls = []
        self._responses = {}

    def set_response(self, key, response):
        self._responses[key] = response

    def get(self, path, params=None):
        self.calls.append(("get", path, params))
        if path in self._responses:
            return self._responses[path]
        if "/models/" in path:
            return {"id": "model_1", "name": "test_model", "architecture": {"type": "neural_net"}}
        return {}

    def post(self, path, payload=None):
        self.calls.append(("post", path, payload))
        if path == "/devices/register":
            return {"id": "device_123"}
        if path == "/devices/heartbeat":
            return {}
        if "/updates" in path:
            return {"update_id": "update_456"}
        return {}

    def get_bytes(self, path, params=None):
        self.calls.append(("get_bytes", path, params))
        return b"model_weights_data"


class FederatedClientTests(unittest.TestCase):
    def test_init_with_device_identifier(self):
        client = FederatedClient(
            auth_token_provider=lambda: "token123",
            org_id="org_1",
            device_identifier="device_abc",
            platform="ios",
        )
        self.assertEqual(client.device_identifier, "device_abc")
        self.assertEqual(client.org_id, "org_1")
        self.assertEqual(client.platform, "ios")
        self.assertIsNone(client.device_id)

    def test_init_generates_device_identifier(self):
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        self.assertIsNotNone(client.device_identifier)
        self.assertTrue(client.device_identifier.startswith("client-"))

    def test_register_device(self):
        stub = _StubApi()
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        device_id = client.register(feature_schema=["feature1", "feature2"])
        self.assertEqual(device_id, "device_123")
        self.assertEqual(client.device_id, "device_123")

        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/devices/register")
        self.assertEqual(payload["org_id"], "org_1")
        self.assertEqual(payload["platform"], "python")
        self.assertEqual(payload["feature_schema"], ["feature1", "feature2"])

    def test_register_returns_existing_device_id(self):
        stub = _StubApi()
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub
        client.device_id = "existing_device"

        device_id = client.register()
        self.assertEqual(device_id, "existing_device")
        self.assertEqual(len(stub.calls), 0)  # No API call made

    def test_register_raises_if_no_device_id_in_response(self):
        stub = _StubApi()
        stub.set_response("/devices/register", {})
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        class _BrokenApi(_StubApi):
            def post(self, path, payload=None):
                self.calls.append(("post", path, payload))
                return {}

        client.api = _BrokenApi()
        with self.assertRaises(OctomilClientError) as ctx:
            client.register()
        self.assertIn("Device registration failed", str(ctx.exception))

    def test_get_model_info_caches_result(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456", {"id": "model_456", "name": "my_model", "framework": "pytorch"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        # First call should hit API
        info1 = client._get_model_info("my_model")
        initial_call_count = len([c for c in stub.calls if c[1].startswith("/models")])

        # Second call should use cache
        info2 = client._get_model_info("my_model")
        final_call_count = len([c for c in stub.calls if c[1].startswith("/models")])

        self.assertEqual(info1, info2)
        self.assertEqual(initial_call_count, final_call_count)

    def test_get_model_info_handles_error(self):
        class _ErrorApi(_StubApi):
            def get(self, path, params=None):
                self.calls.append(("get", path, params))
                if "/models/" in path and path != "/models":
                    raise OctomilClientError("Model not found")
                return {"models": [{"name": "my_model", "id": "model_456"}]}

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = _ErrorApi()

        info = client._get_model_info("my_model")
        self.assertEqual(info, {})

    def test_get_model_architecture(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456", {"id": "model_456", "architecture": {"layers": 3}})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        arch = client._get_model_architecture("my_model")
        self.assertEqual(arch, {"layers": 3})

    def test_get_model_architecture_empty_if_not_found(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456", {"id": "model_456"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        arch = client._get_model_architecture("my_model")
        self.assertEqual(arch, {})

    def test_resolve_model_id_by_name(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_789"}]})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        model_id = client._resolve_model_id("my_model")
        self.assertEqual(model_id, "model_789")

    def test_resolve_model_id_returns_input_if_not_found(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": []})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        model_id = client._resolve_model_id("unknown_model")
        self.assertEqual(model_id, "unknown_model")

    def test_heartbeat(self):
        stub = _StubApi()
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub
        client.device_id = "device_123"

        client.heartbeat()

        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/devices/heartbeat")
        self.assertEqual(payload["device_id"], "device_123")


if __name__ == "__main__":
    unittest.main()
