import unittest

from edgeml.control_plane import ExperimentsAPI, RolloutsAPI
from edgeml.registry import ModelRegistry


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
    def test_rollout_create(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.create("model_1", "1.0.0", rollout_percentage=20.0, target_percentage=80.0, increment_step=15.0, start_immediately=True)
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/rollouts")
        self.assertEqual(payload["version"], "1.0.0")
        self.assertEqual(payload["rollout_percentage"], 20.0)
        self.assertEqual(payload["start_immediately"], True)

    def test_rollout_list_with_filter(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.list("model_1", status_filter="active")
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/models/model_1/rollouts")
        self.assertEqual(params["status_filter"], "active")

    def test_rollout_list_without_filter(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.list("model_1")
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertIsNone(params)

    def test_rollout_list_active(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.list_active("model_1")
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/models/model_1/rollouts/active")

    def test_rollout_get(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.get("model_1", 5)
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/models/model_1/rollouts/5")

    def test_rollout_start(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.start("model_1", 5)
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/rollouts/5/start")

    def test_rollout_pause(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.pause("model_1", 5)
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/rollouts/5/pause")

    def test_rollout_resume(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.resume("model_1", 5)
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/rollouts/5/resume")

    def test_rollout_advance_without_custom_increment(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.advance("model_1", 5)
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/rollouts/5/advance")
        self.assertEqual(payload, {})

    def test_rollout_advance_with_custom_increment(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.advance("model_1", 5, custom_increment=25.0)
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(payload["custom_increment"], 25.0)

    def test_rollout_update_percentage_payload(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.update_percentage("model_1", 7, 50.0, reason="stable")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/rollouts/7/update-percentage")
        self.assertEqual(payload["percentage"], 50.0)
        self.assertEqual(payload["reason"], "stable")

    def test_rollout_update_percentage_without_reason(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.update_percentage("model_1", 7, 50.0)
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertNotIn("reason", payload)

    def test_rollout_get_status_history(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.get_status_history("model_1", 5)
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/models/model_1/rollouts/5/status-history")

    def test_rollout_get_affected_devices(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.get_affected_devices("model_1", 5)
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/models/model_1/rollouts/5/affected-devices")

    def test_rollout_delete_force_param(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.delete("model_1", 7, force=True)
        method, path, params = api.calls[-1]
        self.assertEqual(method, "delete")
        self.assertEqual(path, "/models/model_1/rollouts/7")
        self.assertEqual(params, {"force": "true"})

    def test_rollout_delete_without_force(self):
        api = _StubApi()
        rollouts = RolloutsAPI(api)
        rollouts.delete("model_1", 7)
        method, path, params = api.calls[-1]
        self.assertEqual(method, "delete")
        self.assertEqual(params, {"force": "false"})


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

    def test_list_experiments_without_filters(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.list()
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(params["org_id"], "org_1")
        self.assertNotIn("model_id", params)
        self.assertNotIn("status", params)

    def test_get_experiment(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.get("exp_123")
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/experiments/exp_123")

    def test_start_experiment(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.start("exp_123")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/experiments/exp_123/start")

    def test_pause_experiment(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.pause("exp_123")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/experiments/exp_123/pause")

    def test_resume_experiment(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.resume("exp_123")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/experiments/exp_123/resume")

    def test_complete_experiment(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.complete("exp_123")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/experiments/exp_123/complete")

    def test_cancel_experiment(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.cancel("exp_123")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/experiments/exp_123/cancel")

    def test_update_allocations(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        variants = [{"name": "control", "traffic_allocation": 30.0}, {"name": "treatment", "traffic_allocation": 70.0}]
        experiments.update_allocations("exp_123", variants)
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "patch")
        self.assertEqual(path, "/experiments/exp_123/allocations")
        self.assertEqual(payload["variants"], variants)

    def test_get_target_groups(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.get_target_groups("exp_123")
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/experiments/exp_123/target-groups")

    def test_set_target_groups(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.set_target_groups("exp_123", ["group_1", "group_2"])
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "put")
        self.assertEqual(path, "/experiments/exp_123/target-groups")
        self.assertEqual(payload["group_ids"], ["group_1", "group_2"])

    def test_add_target_group(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.add_target_group("exp_123", "group_1")
        method, path, payload = api.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/experiments/exp_123/target-groups/group_1")

    def test_remove_target_group(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.remove_target_group("exp_123", "group_1")
        method, path, params = api.calls[-1]
        self.assertEqual(method, "delete")
        self.assertEqual(path, "/experiments/exp_123/target-groups/group_1")

    def test_get_analytics(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.get_analytics("exp_123")
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/experiments/exp_123/analytics")

    def test_get_analytics_sample_size(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.get_analytics_sample_size("exp_123")
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/experiments/exp_123/analytics/sample-size")

    def test_get_analytics_timeseries(self):
        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")
        experiments.get_analytics_timeseries("exp_123")
        method, path, params = api.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/experiments/exp_123/analytics/timeseries")


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

