"""Tests for octomil.client — high-level Client wrapper."""

from __future__ import annotations

from unittest.mock import patch


class TestClientInit:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_init_with_explicit_args(self, mock_api, mock_registry, mock_rollouts):
        from octomil.client import Client

        c = Client(api_key="key123", org_id="org1", api_base="https://custom.api")
        assert c._api_key == "key123"
        assert c._org_id == "org1"
        assert c._api_base == "https://custom.api"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_init_from_env_vars(
        self, mock_api, mock_registry, mock_rollouts, monkeypatch
    ):
        from octomil.client import Client

        monkeypatch.setenv("OCTOMIL_API_KEY", "env_key")
        monkeypatch.setenv("OCTOMIL_ORG_ID", "env_org")
        monkeypatch.setenv("OCTOMIL_API_BASE", "https://env.api")

        c = Client()
        assert c._api_key == "env_key"
        assert c._org_id == "env_org"
        assert c._api_base == "https://env.api"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_init_defaults(self, mock_api, mock_registry, mock_rollouts, monkeypatch):
        from octomil.client import Client

        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
        monkeypatch.delenv("OCTOMIL_ORG_ID", raising=False)
        monkeypatch.delenv("OCTOMIL_API_BASE", raising=False)

        c = Client()
        assert c._api_key == ""
        assert c._org_id == "default"
        assert c._api_base == "https://api.octomil.io/api/v1"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_explicit_args_override_env(
        self, mock_api, mock_registry, mock_rollouts, monkeypatch
    ):
        from octomil.client import Client

        monkeypatch.setenv("OCTOMIL_API_KEY", "env_key")
        c = Client(api_key="explicit_key")
        assert c._api_key == "explicit_key"


class TestClientPush:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_push_calls_registry(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.ensure_model.return_value = {"id": "model-abc"}
        mock_registry.upload_version_from_path.return_value = {
            "model_id": "model-abc",
            "version": "1.0.0",
            "formats": {"onnx": "ok"},
        }

        c = Client(api_key="key")
        result = c.push("model.pt", name="test", version="1.0.0")

        mock_registry.ensure_model.assert_called_once_with(
            name="test",
            framework="pytorch",
            use_case="general",
            description=None,
        )
        mock_registry.upload_version_from_path.assert_called_once_with(
            model_id="model-abc",
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
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.ensure_model.return_value = {"id": "model-xyz"}
        mock_registry.upload_version_from_path.return_value = {"ok": True}

        c = Client(api_key="key")
        c.push(
            "model.pt",
            name="test",
            version="2.0.0",
            description="A test model",
            formats="coreml,tflite",
            framework="tensorflow",
            use_case="nlp",
        )

        mock_registry.ensure_model.assert_called_once_with(
            name="test",
            framework="tensorflow",
            use_case="nlp",
            description="A test model",
        )
        mock_registry.upload_version_from_path.assert_called_once_with(
            model_id="model-xyz",
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
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.download.return_value = {"model_path": "/tmp/model.onnx"}

        c = Client(api_key="key")
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
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "3.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/model.onnx"}

        c = Client(api_key="key")
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
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-abc"
        mock_registry.deploy_version.return_value = {
            "id": "rollout-1",
            "status": "created",
        }

        c = Client(api_key="key")
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
    def test_deploy_immediate_strategy(
        self, mock_api, mock_registry_cls, mock_rollouts
    ):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-abc"
        mock_registry.get_latest_version.return_value = "2.0.0"
        mock_registry.deploy_version.return_value = {"status": "started"}

        c = Client(api_key="key")
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
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_rollouts = mock_rollouts_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-abc"
        mock_registry.get_model.return_value = {"name": "my-model", "id": "model-abc"}
        mock_rollouts.list_active.return_value = [
            {"version": "1.0.0", "status": "active"}
        ]

        c = Client(api_key="key")
        result = c.status("my-model")

        assert result["model"]["name"] == "my-model"
        assert len(result["active_rollouts"]) == 1


class TestClientListModels:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_list_models(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.list_models.return_value = {"models": [{"name": "m1"}]}

        c = Client(api_key="key")
        result = c.list_models()
        assert result["models"][0]["name"] == "m1"
