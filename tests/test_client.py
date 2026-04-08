"""Tests for octomil.client — high-level OctomilClient wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from octomil.auth import OrgApiKeyAuth


class TestClientInit:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_init_with_explicit_args(self, mock_api, mock_registry, mock_rollouts):
        from octomil.client import OctomilClient

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key123", org_id="org1", api_base="https://custom.api"))
        assert c._api_key == "key123"
        assert c._org_id == "org1"
        assert c._api_base == "https://custom.api"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_init_from_env_vars(self, mock_api, mock_registry, mock_rollouts, monkeypatch):
        from octomil.client import OctomilClient

        monkeypatch.setenv("OCTOMIL_API_KEY", "env_key")
        monkeypatch.setenv("OCTOMIL_ORG_ID", "env_org")
        monkeypatch.setenv("OCTOMIL_API_BASE", "https://env.api")

        c = OctomilClient.from_env()
        assert c._api_key == "env_key"
        assert c._org_id == "env_org"
        assert c._api_base == "https://env.api"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_init_defaults(self, mock_api, mock_registry, mock_rollouts, monkeypatch):
        from octomil.client import OctomilClient

        monkeypatch.setenv("OCTOMIL_API_KEY", "placeholder")
        monkeypatch.delenv("OCTOMIL_ORG_ID", raising=False)
        monkeypatch.delenv("OCTOMIL_API_BASE", raising=False)

        c = OctomilClient.from_env()
        assert c._api_key == "placeholder"
        assert c._org_id == "default"
        assert c._api_base == "https://api.octomil.com/api/v1"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_explicit_args_override_env(self, mock_api, mock_registry, mock_rollouts, monkeypatch):
        from octomil.client import OctomilClient

        monkeypatch.setenv("OCTOMIL_API_KEY", "env_key")
        c = OctomilClient(auth=OrgApiKeyAuth(api_key="explicit_key", org_id="default"))
        assert c._api_key == "explicit_key"


class TestClientPush:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_push_calls_registry(self, mock_api, mock_registry_cls, mock_rollouts):
        """push() uploads via v2 flow: model name passed directly as model_id."""
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_registry.upload_version_from_path.return_value = {
            "model_id": "test",
            "version": "1.0.0",
            "formats": {"onnx": "ok"},
        }

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        result = c.push("model.pt", name="test", version="1.0.0")

        # v2 flow: no ensure_model call — model name goes directly to upload
        mock_registry.ensure_model.assert_not_called()
        mock_registry.upload_version_from_path.assert_called_once_with(
            model_id="test",
            file_path="model.pt",
            version="1.0.0",
            description=None,
            formats=None,
        )
        assert result["version"] == "1.0.0"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_push_with_all_options(self, mock_api, mock_registry_cls, mock_rollouts):
        """push() forwards description and formats to upload_version_from_path."""
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_registry.upload_version_from_path.return_value = {"ok": True}

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.push(
            "model.pt",
            name="test",
            version="2.0.0",
            description="A test model",
            formats="coreml,tflite",
            framework="tensorflow",
            use_case="nlp",
        )

        # v2 flow: no ensure_model call
        mock_registry.ensure_model.assert_not_called()
        mock_registry.upload_version_from_path.assert_called_once_with(
            model_id="test",
            file_path="model.pt",
            version="2.0.0",
            description="A test model",
            formats="coreml,tflite",
        )


class TestClientPull:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_pull_with_version(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.download.return_value = {"model_path": "/tmp/model.onnx"}

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        result = c.pull("my-model", version="1.0.0", format="onnx")

        mock_registry.resolve_model_id.assert_called_once_with("my-model")
        mock_registry.download.assert_called_once_with(
            model_id="model-123",
            version="1.0.0",
            destination=".",
            format="onnx",
        )
        assert result["model_path"] == "/tmp/model.onnx"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_pull_latest_version(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "3.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/model.onnx"}

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.pull("my-model")

        mock_registry.get_latest_version.assert_called_once_with("model-123")
        mock_registry.download.assert_called_once_with(
            model_id="model-123",
            version="3.0.0",
            destination=".",
            format=None,
        )


class TestClientDeploy:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_deploy_with_version(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-abc"
        mock_registry.deploy_version.return_value = {
            "id": "rollout-1",
            "status": "created",
        }

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        result = c.deploy("my-model", version="1.0.0", rollout=10, strategy="canary")

        mock_registry.deploy_version.assert_called_once_with(
            model_id="model-abc",
            version="1.0.0",
            rollout_percentage=10,
            target_percentage=100,
            increment_step=10,
            start_immediately=False,
        )
        assert result["status"] == "created"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_deploy_immediate_strategy(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-abc"
        mock_registry.get_latest_version.return_value = "2.0.0"
        mock_registry.deploy_version.return_value = {"status": "started"}

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.deploy("my-model", strategy="immediate")

        mock_registry.deploy_version.assert_called_once_with(
            model_id="model-abc",
            version="2.0.0",
            rollout_percentage=100,
            target_percentage=100,
            increment_step=10,
            start_immediately=True,
        )


class TestClientStatus:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_status(self, mock_api, mock_registry_cls, mock_rollouts_cls):
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_rollouts = mock_rollouts_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-abc"
        mock_registry.get_model.return_value = {"name": "my-model", "id": "model-abc"}
        mock_rollouts.list_active.return_value = [{"version": "1.0.0", "status": "active"}]

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        result = c.status("my-model")

        assert result["model"]["name"] == "my-model"
        assert len(result["active_rollouts"]) == 1


class TestClientTrain:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_train_basic(self, mock_api_cls, mock_registry_cls, mock_rollouts):
        from octomil.client import OctomilClient
        from octomil.models import TrainingSession

        mock_registry = mock_registry_cls.return_value
        mock_api = mock_api_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-abc"
        mock_api.post.return_value = {"session_id": "sess-001", "status": "created"}

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        result = c.train("my-model")

        mock_registry.resolve_model_id.assert_called_once_with("my-model")
        mock_api.post.assert_called_once_with(
            "/training/sessions",
            {
                "model_id": "model-abc",
                "group": "default",
                "strategy": "fedavg",
                "rounds": 10,
                "min_updates": 1,
            },
        )
        assert isinstance(result, TrainingSession)
        assert result.session_id == "sess-001"
        assert result.model_name == "my-model"
        assert result.strategy == "fedavg"
        assert result.rounds == 10
        assert result.group == "default"
        assert result.status == "created"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_train_with_all_options(self, mock_api_cls, mock_registry_cls, mock_rollouts):
        from octomil.client import OctomilClient
        from octomil.models import TrainingSession

        mock_registry = mock_registry_cls.return_value
        mock_api = mock_api_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-xyz"
        mock_api.post.return_value = {"session_id": "sess-002", "status": "started"}

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        result = c.train(
            "my-model",
            strategy="fedprox",
            rounds=50,
            group="us-west",
            privacy="dp-sgd",
            epsilon=1.0,
            min_devices=5,
        )

        mock_api.post.assert_called_once_with(
            "/training/sessions",
            {
                "model_id": "model-xyz",
                "group": "us-west",
                "strategy": "fedprox",
                "rounds": 50,
                "min_updates": 1,
                "privacy": "dp-sgd",
                "epsilon": 1.0,
                "min_devices": 5,
            },
        )
        assert isinstance(result, TrainingSession)
        assert result.session_id == "sess-002"
        assert result.group == "us-west"
        assert result.strategy == "fedprox"
        assert result.rounds == 50


class TestClientTrainStatus:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_train_status(self, mock_api_cls, mock_registry_cls, mock_rollouts):
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_api = mock_api_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-abc"
        mock_api.get.return_value = {
            "current_round": 5,
            "total_rounds": 10,
            "active_devices": 12,
            "status": "in_progress",
            "loss": 0.45,
            "accuracy": 0.87,
        }

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        result = c.train_status("my-model")

        mock_registry.resolve_model_id.assert_called_once_with("my-model")
        mock_api.get.assert_called_once_with("/training/model-abc/status")
        assert result["current_round"] == 5
        assert result["status"] == "in_progress"


class TestClientTrainStop:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_train_stop(self, mock_api_cls, mock_registry_cls, mock_rollouts):
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_api = mock_api_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-abc"
        mock_api.post.return_value = {"last_round": 5}

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        result = c.train_stop("my-model")

        mock_registry.resolve_model_id.assert_called_once_with("my-model")
        mock_api.post.assert_called_once_with("/training/model-abc/stop")
        assert result["last_round"] == 5


class TestClientListModels:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_list_models(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import OctomilClient

        mock_registry = mock_registry_cls.return_value
        mock_registry.list_models.return_value = {"models": [{"name": "m1"}]}

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        result = c.list_models()
        assert result["models"][0]["name"] == "m1"


class TestClientDesiredStateRouting:
    """OctomilClient automatically applies routing policy from desired state."""

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_desired_state_sets_routing_policy_on_responses(self, mock_api_cls, mock_registry, mock_rollouts):
        from octomil.client import OctomilClient

        mock_api = mock_api_cls.return_value

        # Simulate sync() returning desired state with routing fields
        raw_desired = {
            "models": [
                {
                    "modelId": "m1",
                    "desiredVersion": "v2",
                    "deploymentId": "dep_1",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                    "servingPolicy": {
                        "routing_mode": "auto",
                        "routing_preference": "quality",
                        "fallback": {"allow_cloud_fallback": True},
                    },
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        # Force registration so get_desired_state works
        c.control._server_device_id = "dev-test"

        # Before sync: no routing policy
        assert c._default_routing_policy is None
        assert c._routing_policies == {}

        # Fetch desired state — should trigger routing callback
        entries = c.control.get_desired_state()
        assert len(entries) == 1

        # Per-deployment policy should be set
        assert "dep_1" in c._routing_policies
        assert c._routing_policies["dep_1"].prefer_local is False  # quality preset

        # Default fallback also set (first policy found)
        assert c._default_routing_policy is not None
        assert c._default_routing_policy.prefer_local is False

        # Responses API picks up both
        assert c._responses is None  # reset by callback
        responses = c.responses
        assert responses._default_routing_policy is not None
        assert "dep_1" in responses._routing_policies

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_multi_deployment_desired_state_stores_per_deployment_policies(
        self, mock_api_cls, mock_registry, mock_rollouts
    ):
        """Multiple deployments with different routing get distinct policies."""
        from octomil.client import OctomilClient

        mock_api = mock_api_cls.return_value
        raw_desired = {
            "models": [
                {
                    "modelId": "m1",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_local",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                    "servingPolicy": {"routing_mode": "local_only"},
                },
                {
                    "modelId": "m2",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_cloud",
                    "artifactManifest": {"artifactId": "a2", "totalBytes": 200},
                    "servingPolicy": {
                        "routing_mode": "auto",
                        "routing_preference": "quality",
                        "fallback": {"allow_cloud_fallback": True},
                    },
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.control._server_device_id = "dev-test"
        c.control.get_desired_state()

        # Both deployments stored with distinct policies
        assert len(c._routing_policies) == 2
        assert c._routing_policies["dep_local"].mode == "local_only"
        assert c._routing_policies["dep_cloud"].prefer_local is False

        # Model → deployment map built
        assert c._model_deployment_map == {"m1": "dep_local", "m2": "dep_cloud"}

        # Default fallback is the first one found
        assert c._default_routing_policy is not None
        assert c._default_routing_policy.mode == "local_only"

        # Responses API has the model map for automatic resolution
        responses = c.responses
        assert responses._model_deployment_map == {"m1": "dep_local", "m2": "dep_cloud"}

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_same_model_multiple_deployments_excluded_from_model_map(self, mock_api_cls, mock_registry, mock_rollouts):
        """Same model_id under two deployments with different routing is ambiguous."""
        from octomil.client import OctomilClient

        mock_api = mock_api_cls.return_value
        raw_desired = {
            "models": [
                {
                    "modelId": "shared-model",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_a",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                    "servingPolicy": {"routing_mode": "local_only"},
                },
                {
                    "modelId": "shared-model",
                    "desiredVersion": "v2",
                    "deploymentId": "dep_b",
                    "artifactManifest": {"artifactId": "a2", "totalBytes": 200},
                    "servingPolicy": {
                        "routing_mode": "auto",
                        "routing_preference": "quality",
                        "fallback": {"allow_cloud_fallback": True},
                    },
                },
                {
                    "modelId": "unique-model",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_c",
                    "artifactManifest": {"artifactId": "a3", "totalBytes": 300},
                    "servingPolicy": {
                        "routing_mode": "auto",
                        "routing_preference": "local",
                        "fallback": {"allow_cloud_fallback": True},
                    },
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.control._server_device_id = "dev-test"
        c.control.get_desired_state()

        # All 3 deployments have policies
        assert len(c._routing_policies) == 3

        # shared-model is excluded from model map (ambiguous)
        assert "shared-model" not in c._model_deployment_map

        # unique-model is still in the model map
        assert c._model_deployment_map == {"unique-model": "dep_c"}

        # Per-deployment lookup still works via explicit deployment_id
        assert c._routing_policies["dep_a"].mode == "local_only"
        assert c._routing_policies["dep_b"].prefer_local is False

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_same_model_same_routing_stays_in_model_map(self, mock_api_cls, mock_registry, mock_rollouts):
        """Same model_id under two deployments with identical routing is NOT ambiguous."""
        from octomil.client import OctomilClient

        mock_api = mock_api_cls.return_value
        raw_desired = {
            "models": [
                {
                    "modelId": "shared-model",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_a",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                    "servingPolicy": {"routing_mode": "local_only"},
                },
                {
                    "modelId": "shared-model",
                    "desiredVersion": "v2",
                    "deploymentId": "dep_b",
                    "artifactManifest": {"artifactId": "a2", "totalBytes": 200},
                    "servingPolicy": {"routing_mode": "local_only"},
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.control._server_device_id = "dev-test"
        c.control.get_desired_state()

        # Same routing on both → model stays in map (maps to first deployment seen)
        assert "shared-model" in c._model_deployment_map
        assert c._model_deployment_map["shared-model"] == "dep_a"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_desired_state_without_routing_fields_leaves_policy_none(self, mock_api_cls, mock_registry, mock_rollouts):
        from octomil.client import OctomilClient

        mock_api = mock_api_cls.return_value
        raw_desired = {
            "models": [
                {
                    "modelId": "m1",
                    "desiredVersion": "v1",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.control._server_device_id = "dev-test"

        c.control.get_desired_state()

        # No routing fields in desired state — policy stays None
        assert c._default_routing_policy is None
        assert c._routing_policies == {}


class TestClientProductionPaths:
    """End-to-end tests proving client.chat.create / client.chat.stream
    get correct routing from desired state through to OctomilResponses."""

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_chat_create_uses_desired_state_routing(self, mock_api_cls, mock_registry, mock_rollouts):
        """client.chat.create() delegates to responses.create() with correct routing policies."""
        from octomil.client import OctomilClient
        from octomil.responses.types import Response, ResponseUsage, TextOutput

        mock_api = mock_api_cls.return_value

        # Desired state with servingPolicy quality on dep_1 for chat-model
        raw_desired = {
            "models": [
                {
                    "modelId": "chat-model",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_1",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                    "servingPolicy": {
                        "routing_mode": "auto",
                        "routing_preference": "quality",
                        "fallback": {"allow_cloud_fallback": True},
                    },
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.control._server_device_id = "dev-test"
        c.control.get_desired_state()

        # Verify routing was applied
        assert "dep_1" in c._routing_policies
        assert c._model_deployment_map == {"chat-model": "dep_1"}

        # Mock OctomilResponses.create to capture the call without actually running inference
        fake_response = Response(
            id="resp_test123",
            model="chat-model",
            output=[TextOutput(text="Hello from mock")],
            finish_reason="stop",
            usage=ResponseUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        # Bind mock directly to the responses instance
        mock_create = AsyncMock(return_value=fake_response)
        original_create = c.responses.create
        c.responses.create = mock_create

        try:
            result = c.chat.create(model="chat-model", messages=[{"role": "user", "content": "Hi"}])

            # Verify responses.create was called
            mock_create.assert_called_once()
            request = mock_create.call_args[0][0]
            assert request.model == "chat-model"

            # Verify the OctomilResponses instance has correct routing config
            responses = c.responses
            assert "dep_1" in responses._routing_policies
            assert responses._routing_policies["dep_1"].prefer_local is False  # quality preset
            assert responses._model_deployment_map == {"chat-model": "dep_1"}
            assert responses._default_routing_policy is not None

            # Verify the ChatCompletion result
            assert result.message["role"] == "assistant"
            assert result.message["content"] == "Hello from mock"
            assert result.usage["total_tokens"] == 15
        finally:
            c.responses.create = original_create

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_chat_stream_uses_desired_state_routing(self, mock_api_cls, mock_registry, mock_rollouts):
        """client.chat.stream() delegates to responses.stream() with correct routing policies."""
        from octomil.client import OctomilClient
        from octomil.responses.types import (
            DoneEvent,
            Response,
            TextDeltaEvent,
            TextOutput,
        )

        mock_api = mock_api_cls.return_value

        raw_desired = {
            "models": [
                {
                    "modelId": "stream-model",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_stream",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                    "servingPolicy": {
                        "routing_mode": "auto",
                        "routing_preference": "local",
                        "fallback": {"allow_cloud_fallback": True},
                    },
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.control._server_device_id = "dev-test"
        c.control.get_desired_state()

        # Verify routing was applied
        assert "dep_stream" in c._routing_policies
        assert c._model_deployment_map == {"stream-model": "dep_stream"}

        # Create an async generator that yields stream events
        async def fake_stream(request):
            yield TextDeltaEvent(delta="Hello ")
            yield TextDeltaEvent(delta="world")
            yield DoneEvent(
                response=Response(
                    id="resp_stream_test",
                    model="stream-model",
                    output=[TextOutput(text="Hello world")],
                    finish_reason="stop",
                )
            )

        # Replace stream on the responses instance
        responses_instance = c.responses
        original_stream = responses_instance.stream
        responses_instance.stream = fake_stream

        # Run the async stream and collect chunks
        async def collect_chunks():
            chunks = []
            async for chunk in c.chat.stream(
                model="stream-model",
                messages=[{"role": "user", "content": "Hi"}],
            ):
                chunks.append(chunk)
            return chunks

        loop = asyncio.new_event_loop()
        try:
            chunks = loop.run_until_complete(collect_chunks())
        finally:
            loop.close()

        # Verify we got the expected chunks
        assert len(chunks) == 3
        assert chunks[0].content == "Hello "
        assert chunks[0].done is False
        assert chunks[1].content == "world"
        assert chunks[1].done is False
        assert chunks[2].done is True

        # Verify the OctomilResponses instance has correct routing config
        assert "dep_stream" in responses_instance._routing_policies
        assert responses_instance._routing_policies["dep_stream"].prefer_local is True  # local preset
        assert responses_instance._model_deployment_map == {"stream-model": "dep_stream"}

        # Restore
        responses_instance.stream = original_stream

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_duplicate_model_identical_policy_auto_resolves(self, mock_api_cls, mock_registry, mock_rollouts):
        """Two deployments, same model_id, both local_only -> model stays in model_deployment_map."""
        from octomil.client import OctomilClient

        mock_api = mock_api_cls.return_value
        raw_desired = {
            "models": [
                {
                    "modelId": "shared-llm",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_x",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                    "servingPolicy": {"routing_mode": "local_only"},
                },
                {
                    "modelId": "shared-llm",
                    "desiredVersion": "v2",
                    "deploymentId": "dep_y",
                    "artifactManifest": {"artifactId": "a2", "totalBytes": 200},
                    "servingPolicy": {"routing_mode": "local_only"},
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.control._server_device_id = "dev-test"
        c.control.get_desired_state()

        # Both deployments have policies
        assert len(c._routing_policies) == 2
        assert "dep_x" in c._routing_policies
        assert "dep_y" in c._routing_policies

        # Same policy on both -> model is NOT excluded (auto-resolved to first deployment seen)
        assert "shared-llm" in c._model_deployment_map
        assert c._model_deployment_map["shared-llm"] == "dep_x"

        # The responses instance also sees this
        responses = c.responses
        assert "shared-llm" in responses._model_deployment_map
        assert responses._model_deployment_map["shared-llm"] == "dep_x"

        # Both policies are local_only
        assert c._routing_policies["dep_x"].mode.value == "local_only"
        assert c._routing_policies["dep_y"].mode.value == "local_only"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_duplicate_model_conflicting_policy_requires_explicit_deployment_id(
        self, mock_api_cls, mock_registry, mock_rollouts
    ):
        """Two deployments, same model_id, one local_only and one quality -> model excluded from
        auto-resolution but both per-deployment policies exist."""
        from octomil.client import OctomilClient

        mock_api = mock_api_cls.return_value
        raw_desired = {
            "models": [
                {
                    "modelId": "ambiguous-model",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_local",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                    "servingPolicy": {"routing_mode": "local_only"},
                },
                {
                    "modelId": "ambiguous-model",
                    "desiredVersion": "v2",
                    "deploymentId": "dep_quality",
                    "artifactManifest": {"artifactId": "a2", "totalBytes": 200},
                    "servingPolicy": {
                        "routing_mode": "auto",
                        "routing_preference": "quality",
                        "fallback": {"allow_cloud_fallback": True},
                    },
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.control._server_device_id = "dev-test"
        c.control.get_desired_state()

        # Both deployment policies exist
        assert len(c._routing_policies) == 2
        assert "dep_local" in c._routing_policies
        assert "dep_quality" in c._routing_policies

        # Model is excluded from model_deployment_map due to conflicting policies
        assert "ambiguous-model" not in c._model_deployment_map

        # Per-deployment policies have correct values
        assert c._routing_policies["dep_local"].mode.value == "local_only"
        assert c._routing_policies["dep_quality"].prefer_local is False  # quality preset

        # The responses instance also reflects this exclusion
        responses = c.responses
        assert "ambiguous-model" not in responses._model_deployment_map

        # But responses still has both deployment policies for explicit lookup
        assert "dep_local" in responses._routing_policies
        assert "dep_quality" in responses._routing_policies

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_agent_session_inherits_routing(self, mock_api_cls, mock_registry, mock_rollouts):
        """client.agent_session() returns a session with the client's routing."""
        from octomil.client import OctomilClient

        mock_api = mock_api_cls.return_value
        raw_desired = {
            "models": [
                {
                    "modelId": "agent-model",
                    "desiredVersion": "v1",
                    "deploymentId": "dep_agent",
                    "artifactManifest": {"artifactId": "a1", "totalBytes": 100},
                    "servingPolicy": {
                        "routing_mode": "auto",
                        "routing_preference": "quality",
                        "fallback": {"allow_cloud_fallback": True},
                    },
                },
            ],
        }
        mock_api.post.return_value = raw_desired

        c = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
        c.control._server_device_id = "dev-test"
        c.control.get_desired_state()

        session = c.agent_session()

        # Session's responses instance is the same as client's — shares routing
        assert session._responses is c.responses
        assert "dep_agent" in session._responses._routing_policies
        assert session._responses._model_deployment_map == {"agent-model": "dep_agent"}
