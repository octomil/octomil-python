import unittest

from edgeml.api_client import EdgeMLClientError
from edgeml.federation import Federation


class _StubApi:
    def __init__(self):
        self.calls = []
        self._responses = {}

    def set_response(self, key, response):
        self._responses[key] = response

    def get(self, path, params=None):
        self.calls.append(("get", path, params))
        key = (path, tuple(sorted(params.items())) if params else None)
        if key in self._responses:
            return self._responses[key]
        if path == "/federations":
            return []
        if path.endswith("/versions/latest"):
            return {"version": "1.0.0"}
        return {"models": []}

    def post(self, path, payload=None):
        self.calls.append(("post", path, payload))
        if path == "/federations":
            return {"id": "fed_123"}
        if path == "/training/aggregate":
            return {"new_version": "1.1.0", "status": "completed"}
        return {"id": "result_id"}


class FederationTests(unittest.TestCase):
    def test_init_with_existing_federation(self):
        stub = _StubApi()
        stub.set_response(
            ("/federations", (("name", "test_fed"), ("org_id", "org_1"))),
            [{"id": "fed_456", "name": "test_fed"}]
        )

        # Patch _ApiClient to return our stub, then create Federation
        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", name="test_fed", org_id="org_1")

        self.assertEqual(federation.federation_id, "fed_456")
        self.assertEqual(federation.name, "test_fed")
        self.assertEqual(federation.org_id, "org_1")

    def test_init_creates_new_federation(self):
        stub = _StubApi()
        stub.set_response(("/federations", (("name", "new_fed"), ("org_id", "org_1"))), [])

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", name="new_fed", org_id="org_1")

        self.assertEqual(federation.federation_id, "fed_123")

    def test_init_uses_default_name(self):
        stub = _StubApi()

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        self.assertEqual(federation.name, "default")

    def test_invite(self):
        stub = _StubApi()
        stub.set_response(("/federations", (("name", "default"), ("org_id", "org_1"))), [])

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        federation.invite(["org_2", "org_3"])
        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, f"/federations/{federation.federation_id}/invite")
        self.assertEqual(payload["org_ids"], ["org_2", "org_3"])

    def test_train_with_fedavg(self):
        stub = _StubApi()
        stub.set_response(("/models", (("org_id", "org_1"),)), {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response(("/federations", (("name", "default"), ("org_id", "org_1"))), [])

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        result = federation.train(
            model="my_model",
            algorithm="fedavg",
            rounds=2,
            min_updates=3,
            base_version="1.0.0",
            new_version="1.1.0",
        )

        # Verify training was called
        self.assertEqual(stub.calls[-1][0], "post")
        self.assertEqual(stub.calls[-1][1], "/training/aggregate")
        self.assertEqual(result["new_version"], "1.1.0")
        self.assertEqual(federation.last_model_id, "model_456")
        self.assertEqual(federation.last_version, "1.1.0")

    def test_train_unsupported_algorithm_raises(self):
        stub = _StubApi()
        stub.set_response(("/federations", (("name", "default"), ("org_id", "org_1"))), [])

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        with self.assertRaises(EdgeMLClientError) as ctx:
            federation.train(model="my_model", algorithm="unsupported")
        self.assertIn("Unsupported algorithm", str(ctx.exception))

    def test_train_multiple_rounds(self):
        stub = _StubApi()
        stub.set_response(("/models", (("org_id", "org_1"),)), {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response(("/federations", (("name", "default"), ("org_id", "org_1"))), [])

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        federation.train(model="my_model", rounds=3)

        aggregate_calls = [call for call in stub.calls if call[1] == "/training/aggregate"]
        self.assertEqual(len(aggregate_calls), 3)

    def test_deploy_with_explicit_params(self):
        stub = _StubApi()
        stub.set_response(("/federations", (("name", "default"), ("org_id", "org_1"))), [])

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        federation.deploy(
            model_id="model_1",
            version="2.0.0",
            rollout_percentage=20,
            target_percentage=90,
            increment_step=15,
            start_immediately=False,
        )

        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/rollouts")
        self.assertEqual(payload["version"], "2.0.0")
        self.assertEqual(payload["rollout_percentage"], 20.0)
        self.assertEqual(payload["target_percentage"], 90.0)
        self.assertEqual(payload["increment_step"], 15.0)
        self.assertEqual(payload["start_immediately"], False)

    def test_deploy_uses_last_model_and_version(self):
        stub = _StubApi()
        stub.set_response(("/federations", (("name", "default"), ("org_id", "org_1"))), [])

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        federation.last_model_id = "model_auto"
        federation.last_version = "3.0.0"
        federation.deploy()

        method, path, payload = stub.calls[-1]
        self.assertEqual(payload["version"], "3.0.0")
        self.assertIn("model_auto", path)

    def test_deploy_fetches_latest_version_if_not_set(self):
        stub = _StubApi()
        stub.set_response(("/federations", (("name", "default"), ("org_id", "org_1"))), [])
        stub.set_response(("/models/model_1/versions/latest", None), {"version": "4.0.0"})

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        federation.last_model_id = "model_1"
        federation.deploy()

        method, path, payload = stub.calls[-1]
        self.assertEqual(payload["version"], "4.0.0")

    def test_deploy_raises_if_no_model_id(self):
        stub = _StubApi()
        stub.set_response(("/federations", (("name", "default"), ("org_id", "org_1"))), [])

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        with self.assertRaises(EdgeMLClientError) as ctx:
            federation.deploy()
        self.assertIn("model_id is required", str(ctx.exception))

    def test_deploy_raises_if_no_version(self):
        stub = _StubApi()
        stub.set_response(("/federations", (("name", "default"), ("org_id", "org_1"))), [])
        stub.set_response(("/models/model_1/versions/latest", None), {})

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            federation = Federation(lambda: "token123", org_id="org_1")

        federation.last_model_id = "model_1"

        with self.assertRaises(EdgeMLClientError) as ctx:
            federation.deploy()
        self.assertIn("version is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
