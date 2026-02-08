import unittest

from edgeml.python.edgeml.control_plane import ExperimentsAPI, RolloutsAPI
from edgeml.python.edgeml.registry import ModelRegistry


class _StubApi:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append(("get", path, params))
        return {"path": path, "params": params}

    def post(self, path, payload=None):
        self.calls.append(("post", path, payload))
        return {"path": path, "payload": payload, "id": "exp_123", "version": "1.0.0"}

    def patch(self, path, payload=None):
        self.calls.append(("patch", path, payload))
        return {"path": path, "payload": payload}

    def put(self, path, payload=None):
        self.calls.append(("put", path, payload))
        return {"path": path, "payload": payload}

    def delete(self, path, params=None):
        self.calls.append(("delete", path, params))
        return {"path": path, "params": params}


class RolloutsApiTests(unittest.TestCase):
    def test_rollout_update_percentage_payload(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.update_percentage("model_1", 7, 50.0, reason="stable")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/rollouts/7/update-percentage")
        self.assertEqual(payload["percentage"], 50.0)
        self.assertEqual(payload["reason"], "stable")

    def test_rollout_delete_force_param(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.delete("model_1", 7, force=True)
        method, path, params = api.calls[-1]
        self.assertEqual(method, "delete")
        self.assertEqual(path, "/models/model_1/rollouts/7")
        self.assertEqual(params, {"force": "true"})


class ExperimentsApiTests(unittest.TestCase):
    def test_create_experiment_payload_shape(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.create(
            name="ab_test",
            model_id="model_1",
            control_version="1.0.0",
            treatment_version="1.1.0",
            control_allocation=40.0,
            treatment_allocation=60.0,
            primary_metric="accuracy",
        )
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/experiments")
        self.assertEqual(payload["name"], "ab_test")
        self.assertEqual(payload["variants"][0]["is_control"], True)
        self.assertEqual(payload["variants"][0]["traffic_allocation"], 40.0)
        self.assertEqual(payload["variants"][1]["traffic_allocation"], 60.0)

    def test_list_experiments_includes_org(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.list(model_id="model_1", status_filter="running", limit=5, offset=2)
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/experiments")
        self.assertEqual(params["org_id"], "org_1")
        self.assertEqual(params["model_id"], "model_1")
        self.assertEqual(params["status"], "running")
        self.assertEqual(params["limit"], 5)
        self.assertEqual(params["offset"], 2)


class ModelRegistryControlPlaneTests(unittest.TestCase):
    def test_deploy_version_delegates_to_rollout(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token")
        stub = _StubApi()
        registry.api = stub
        registry.rollouts = RolloutsAPI(stub)
        result = registry.deploy_version(
            model_id="model_1",
            version="1.2.3",
            rollout_percentage=15,
            target_percentage=75,
            increment_step=5,
            start_immediately=False,
        )
        self.assertEqual(result["path"], "/models/model_1/rollouts")
        self.assertEqual(result["payload"]["version"], "1.2.3")
        self.assertEqual(result["payload"]["rollout_percentage"], 15.0)
        self.assertEqual(result["payload"]["target_percentage"], 75.0)
        self.assertEqual(result["payload"]["increment_step"], 5.0)
        self.assertEqual(result["payload"]["start_immediately"], False)

    def test_update_version_metrics_endpoint(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token")
        stub = _StubApi()
        registry.api = stub
        registry.update_version_metrics("model_1", "1.0.0", {"acc": 0.9})
        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "patch")
        self.assertEqual(path, "/models/model_1/versions/1.0.0/metrics")
        self.assertEqual(payload, {"metrics": {"acc": 0.9}})


if __name__ == "__main__":
    unittest.main()

