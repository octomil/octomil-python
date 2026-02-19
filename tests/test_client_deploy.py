"""Tests for deploy orchestration, train, and rollback in edgeml.client."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import patch

import pytest

from edgeml.models import (
    DeploymentPlan,
    DeploymentResult,
    DeviceDeployment,
    DeviceDeploymentStatus,
    RollbackResult,
    TrainingSession,
)


# ===================================================================
# Data class serialization tests
# ===================================================================


class TestDataClasses:
    def test_device_deployment_defaults(self):
        dd = DeviceDeployment(
            device_id="dev-1",
            format="coreml",
            executor="coreml_npu",
            quantization="int8",
        )
        assert dd.device_id == "dev-1"
        assert dd.download_url is None
        assert dd.conversion_needed is False
        assert dd.runtime_config == {}

    def test_device_deployment_to_dict(self):
        dd = DeviceDeployment(
            device_id="dev-1",
            format="tflite",
            executor="nnapi",
            quantization="fp16",
            download_url="https://cdn.edgeml.io/model.tflite",
            conversion_needed=True,
            runtime_config={"threads": 4},
        )
        d = dd.to_dict()
        assert d["device_id"] == "dev-1"
        assert d["download_url"] == "https://cdn.edgeml.io/model.tflite"
        assert d["conversion_needed"] is True
        assert d["runtime_config"]["threads"] == 4

    def test_deployment_plan_to_dict(self):
        plan = DeploymentPlan(
            model_name="test-model",
            model_version="1.0.0",
            deployments=[
                DeviceDeployment(
                    device_id="dev-1",
                    format="coreml",
                    executor="coreml_npu",
                    quantization="int8",
                )
            ],
        )
        d = plan.to_dict()
        assert d["model_name"] == "test-model"
        assert d["model_version"] == "1.0.0"
        assert len(d["deployments"]) == 1
        assert d["deployments"][0]["device_id"] == "dev-1"

    def test_deployment_plan_empty_deployments(self):
        plan = DeploymentPlan(
            model_name="m",
            model_version="1.0",
        )
        assert plan.deployments == []
        assert plan.to_dict()["deployments"] == []

    def test_device_deployment_status_to_dict(self):
        ds = DeviceDeploymentStatus(
            device_id="dev-1",
            status="deployed",
            download_url="https://cdn.edgeml.io/model.onnx",
            error=None,
        )
        d = ds.to_dict()
        assert d["status"] == "deployed"
        assert d["error"] is None

    def test_deployment_result_to_dict(self):
        result = DeploymentResult(
            deployment_id="dep-123",
            model_name="sentiment",
            model_version="2.0.0",
            status="deploying",
            device_statuses=[
                DeviceDeploymentStatus(device_id="dev-1", status="downloading"),
                DeviceDeploymentStatus(
                    device_id="dev-2", status="failed", error="timeout"
                ),
            ],
        )
        d = result.to_dict()
        assert d["deployment_id"] == "dep-123"
        assert len(d["device_statuses"]) == 2
        assert d["device_statuses"][1]["error"] == "timeout"

    def test_training_session_to_dict(self):
        ts = TrainingSession(
            session_id="sess-1",
            model_name="classifier",
            group="default",
            strategy="fedavg",
            rounds=10,
            status="created",
        )
        d = ts.to_dict()
        assert d["session_id"] == "sess-1"
        assert d["rounds"] == 10

    def test_rollback_result_to_dict(self):
        rb = RollbackResult(
            model_name="classifier",
            from_version="2.0.0",
            to_version="1.0.0",
            rollout_id="rollout-99",
            status="rolling_back",
        )
        d = rb.to_dict()
        assert d["from_version"] == "2.0.0"
        assert d["to_version"] == "1.0.0"
        assert d["rollout_id"] == "rollout-99"

    def test_asdict_roundtrip(self):
        """Verify that dataclasses.asdict works and to_dict() matches."""
        dd = DeviceDeployment(
            device_id="x",
            format="onnx",
            executor="cpu",
            quantization="none",
        )
        assert asdict(dd) == dd.to_dict()


# ===================================================================
# Client.deploy() with device targeting
# ===================================================================


class TestDeployWithDeviceTargeting:
    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_with_devices(self, mock_api_cls, mock_reg_cls, mock_rollouts_cls):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "2.0.0"

        mock_api.post.return_value = {
            "deployment_id": "dep-001",
            "status": "deploying",
            "device_statuses": [
                {"device_id": "dev-1", "status": "downloading"},
                {"device_id": "dev-2", "status": "downloading"},
            ],
        }

        client = Client(api_key="key")
        result = client.deploy(
            "my-model",
            devices=["dev-1", "dev-2"],
            strategy="immediate",
        )

        assert isinstance(result, DeploymentResult)
        assert result.deployment_id == "dep-001"
        assert result.model_version == "2.0.0"
        assert len(result.device_statuses) == 2
        assert result.device_statuses[0].device_id == "dev-1"

        # Verify the API call
        mock_api.post.assert_called_once()
        call_args = mock_api.post.call_args
        assert call_args[0][0] == "/deploy/execute"
        payload = call_args[0][1]
        assert payload["devices"] == ["dev-1", "dev-2"]
        assert payload["strategy"] == "immediate"

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_with_group(self, mock_api_cls, mock_reg_cls, mock_rollouts_cls):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "3.0.0"

        mock_api.post.return_value = {
            "deployment_id": "dep-002",
            "status": "deploying",
            "device_statuses": [
                {"device_id": "prod-1", "status": "queued"},
            ],
        }

        client = Client(api_key="key")
        result = client.deploy("my-model", group="production")

        assert isinstance(result, DeploymentResult)
        assert result.deployment_id == "dep-002"
        assert result.model_name == "my-model"

        call_args = mock_api.post.call_args
        payload = call_args[0][1]
        assert payload["group"] == "production"
        assert "devices" not in payload

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_with_explicit_version_and_devices(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-xyz"

        mock_api.post.return_value = {
            "deployment_id": "dep-003",
            "status": "created",
            "device_statuses": [],
        }

        client = Client(api_key="key")
        result = client.deploy(
            "my-model",
            version="1.5.0",
            devices=["dev-a"],
        )

        assert isinstance(result, DeploymentResult)
        assert result.model_version == "1.5.0"
        # Should NOT have called get_latest_version since version was explicit
        mock_reg.get_latest_version.assert_not_called()

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_with_device_error_status(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "1.0.0"

        mock_api.post.return_value = {
            "deployment_id": "dep-err",
            "status": "partial_failure",
            "device_statuses": [
                {"device_id": "dev-1", "status": "deployed"},
                {
                    "device_id": "dev-2",
                    "status": "failed",
                    "error": "incompatible format",
                },
            ],
        }

        client = Client(api_key="key")
        result = client.deploy("my-model", devices=["dev-1", "dev-2"])

        assert result.status == "partial_failure"
        assert result.device_statuses[1].error == "incompatible format"


# ===================================================================
# Client.deploy() fallback (no devices/group)
# ===================================================================


class TestDeployFallbackToRollout:
    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_without_targeting_uses_rollout(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        """When no devices/group is specified, deploy falls back to rollout."""
        from edgeml.client import Client

        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "2.0.0"
        mock_reg.deploy_version.return_value = {
            "id": "rollout-1",
            "status": "created",
        }

        client = Client(api_key="key")
        result = client.deploy("my-model", rollout=10, strategy="canary")

        # Should be a plain dict, not DeploymentResult
        assert isinstance(result, dict)
        assert result["id"] == "rollout-1"

        mock_reg.deploy_version.assert_called_once_with(
            model_id="model-abc",
            version="2.0.0",
            rollout_percentage=10,
            target_percentage=100,
            increment_step=10,
            start_immediately=False,
        )

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_immediate_strategy_fallback(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "1.0.0"
        mock_reg.deploy_version.return_value = {"status": "started"}

        client = Client(api_key="key")
        client.deploy("my-model", strategy="immediate")

        mock_reg.deploy_version.assert_called_once_with(
            model_id="model-abc",
            version="1.0.0",
            rollout_percentage=100,
            target_percentage=100,
            increment_step=10,
            start_immediately=True,
        )


# ===================================================================
# Client.deploy_prepare() — dry-run
# ===================================================================


class TestDeployPrepare:
    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_prepare_with_devices(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "2.0.0"

        mock_api.post.return_value = {
            "deployments": [
                {
                    "device_id": "iphone-1",
                    "format": "coreml",
                    "executor": "coreml_npu",
                    "quantization": "int8",
                    "download_url": "https://cdn.edgeml.io/model.mlmodel",
                    "conversion_needed": False,
                    "runtime_config": {"threads": 2},
                },
                {
                    "device_id": "pixel-1",
                    "format": "tflite",
                    "executor": "nnapi",
                    "quantization": "fp16",
                    "download_url": None,
                    "conversion_needed": True,
                    "runtime_config": {},
                },
            ]
        }

        client = Client(api_key="key")
        plan = client.deploy_prepare("my-model", devices=["iphone-1", "pixel-1"])

        assert isinstance(plan, DeploymentPlan)
        assert plan.model_name == "my-model"
        assert plan.model_version == "2.0.0"
        assert len(plan.deployments) == 2
        assert plan.deployments[0].format == "coreml"
        assert plan.deployments[0].conversion_needed is False
        assert plan.deployments[1].conversion_needed is True
        assert plan.deployments[1].download_url is None

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_prepare_with_group(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "1.0.0"

        mock_api.post.return_value = {
            "deployments": [
                {
                    "device_id": "prod-001",
                    "format": "onnx",
                    "executor": "cpu",
                    "quantization": "none",
                },
            ]
        }

        client = Client(api_key="key")
        plan = client.deploy_prepare("my-model", group="production")

        assert isinstance(plan, DeploymentPlan)
        assert len(plan.deployments) == 1
        assert plan.deployments[0].device_id == "prod-001"

        call_args = mock_api.post.call_args
        assert call_args[0][0] == "/deploy/prepare"
        payload = call_args[0][1]
        assert payload["group"] == "production"

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_prepare_explicit_version(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"

        mock_api.post.return_value = {"deployments": []}

        client = Client(api_key="key")
        plan = client.deploy_prepare("my-model", version="5.0.0")

        assert plan.model_version == "5.0.0"
        mock_reg.get_latest_version.assert_not_called()

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_deploy_prepare_empty_result(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "1.0.0"

        mock_api.post.return_value = {}

        client = Client(api_key="key")
        plan = client.deploy_prepare("my-model")

        assert plan.deployments == []


# ===================================================================
# Client.train()
# ===================================================================


class TestTrain:
    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_train_creates_session(self, mock_api_cls, mock_reg_cls, mock_rollouts_cls):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"

        mock_api.post.return_value = {
            "session_id": "sess-001",
            "status": "created",
        }

        client = Client(api_key="key")
        result = client.train("my-model", rounds=5, strategy="fedavg")

        assert isinstance(result, TrainingSession)
        assert result.session_id == "sess-001"
        assert result.model_name == "my-model"
        assert result.rounds == 5
        assert result.strategy == "fedavg"
        assert result.status == "created"

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_train_with_group_and_kwargs(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-xyz"

        mock_api.post.return_value = {
            "id": "sess-002",
            "status": "started",
        }

        client = Client(api_key="key")
        result = client.train(
            "my-model",
            group="edge-devices",
            strategy="fedprox",
            rounds=20,
            min_updates=5,
            learning_rate=0.001,
        )

        assert isinstance(result, TrainingSession)
        assert result.session_id == "sess-002"
        assert result.group == "edge-devices"
        assert result.strategy == "fedprox"
        assert result.rounds == 20

        call_args = mock_api.post.call_args
        assert call_args[0][0] == "/training/sessions"
        payload = call_args[0][1]
        assert payload["model_id"] == "model-xyz"
        assert payload["group"] == "edge-devices"
        assert payload["strategy"] == "fedprox"
        assert payload["rounds"] == 20
        assert payload["min_updates"] == 5
        assert payload["learning_rate"] == 0.001

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_train_defaults(self, mock_api_cls, mock_reg_cls, mock_rollouts_cls):
        from edgeml.client import Client

        mock_api = mock_api_cls.return_value
        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"

        mock_api.post.return_value = {"session_id": "s-3", "status": "created"}

        client = Client(api_key="key")
        result = client.train("my-model")

        assert result.group == "default"
        assert result.strategy == "fedavg"
        assert result.rounds == 10

        payload = mock_api.post.call_args[0][1]
        assert payload["min_updates"] == 1


# ===================================================================
# Client.rollback()
# ===================================================================


class TestRollback:
    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_rollback_to_previous_version(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "3.0.0"
        mock_reg.list_versions.return_value = {
            "versions": [
                {"version": "3.0.0"},
                {"version": "2.0.0"},
                {"version": "1.0.0"},
            ]
        }
        mock_reg.deploy_version.return_value = {
            "id": "rollout-99",
            "status": "started",
        }

        client = Client(api_key="key")
        result = client.rollback("my-model")

        assert isinstance(result, RollbackResult)
        assert result.from_version == "3.0.0"
        assert result.to_version == "2.0.0"
        assert result.rollout_id == "rollout-99"
        assert result.status == "started"

        mock_reg.deploy_version.assert_called_once_with(
            model_id="model-abc",
            version="2.0.0",
            rollout_percentage=100,
            target_percentage=100,
            increment_step=100,
            start_immediately=True,
        )

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_rollback_to_specific_version(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client

        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "3.0.0"
        mock_reg.deploy_version.return_value = {
            "id": "rollout-50",
            "status": "started",
        }

        client = Client(api_key="key")
        result = client.rollback("my-model", to_version="1.0.0")

        assert isinstance(result, RollbackResult)
        assert result.from_version == "3.0.0"
        assert result.to_version == "1.0.0"
        assert result.rollout_id == "rollout-50"

        # Should NOT have called list_versions since to_version was explicit
        mock_reg.list_versions.assert_not_called()

    @patch("edgeml.client.RolloutsAPI")
    @patch("edgeml.client.ModelRegistry")
    @patch("edgeml.client._ApiClient")
    def test_rollback_fails_with_single_version(
        self, mock_api_cls, mock_reg_cls, mock_rollouts_cls
    ):
        from edgeml.client import Client
        from edgeml.python.edgeml.api_client import EdgeMLClientError

        mock_reg = mock_reg_cls.return_value
        mock_reg.resolve_model_id.return_value = "model-abc"
        mock_reg.get_latest_version.return_value = "1.0.0"
        mock_reg.list_versions.return_value = {"versions": [{"version": "1.0.0"}]}

        client = Client(api_key="key")
        with pytest.raises(EdgeMLClientError, match="only one version exists"):
            client.rollback("my-model")


# ===================================================================
# CLI deploy command with targeting
# ===================================================================


class TestCLIDeploy:
    @patch("edgeml.cli._get_client")
    def test_cli_deploy_with_devices(self, mock_get_client):
        from click.testing import CliRunner

        from edgeml.cli import main

        mock_client = mock_get_client.return_value
        mock_client.deploy.return_value = DeploymentResult(
            deployment_id="dep-cli-1",
            model_name="gemma-1b",
            model_version="1.0.0",
            status="deploying",
            device_statuses=[
                DeviceDeploymentStatus(device_id="dev-1", status="downloading"),
                DeviceDeploymentStatus(device_id="dev-2", status="downloading"),
            ],
        )

        runner = CliRunner()
        result = runner.invoke(main, ["deploy", "gemma-1b", "--devices", "dev-1,dev-2"])

        assert result.exit_code == 0
        assert "dep-cli-1" in result.output
        assert "deploying" in result.output
        mock_client.deploy.assert_called_once_with(
            "gemma-1b",
            version=None,
            rollout=100,
            strategy="canary",
            devices=["dev-1", "dev-2"],
            group=None,
        )

    @patch("edgeml.cli._get_client")
    def test_cli_deploy_with_group(self, mock_get_client):
        from click.testing import CliRunner

        from edgeml.cli import main

        mock_client = mock_get_client.return_value
        mock_client.deploy.return_value = DeploymentResult(
            deployment_id="dep-cli-2",
            model_name="gemma-1b",
            model_version="2.0.0",
            status="deploying",
            device_statuses=[],
        )

        runner = CliRunner()
        result = runner.invoke(main, ["deploy", "gemma-1b", "--group", "production"])

        assert result.exit_code == 0
        assert "dep-cli-2" in result.output
        mock_client.deploy.assert_called_once_with(
            "gemma-1b",
            version=None,
            rollout=100,
            strategy="canary",
            devices=None,
            group="production",
        )

    @patch("edgeml.cli._get_client")
    def test_cli_deploy_dry_run(self, mock_get_client):
        from click.testing import CliRunner

        from edgeml.cli import main

        mock_client = mock_get_client.return_value
        mock_client.deploy_prepare.return_value = DeploymentPlan(
            model_name="gemma-1b",
            model_version="1.0.0",
            deployments=[
                DeviceDeployment(
                    device_id="iphone-1",
                    format="coreml",
                    executor="coreml_npu",
                    quantization="int8",
                    conversion_needed=False,
                ),
                DeviceDeployment(
                    device_id="pixel-1",
                    format="tflite",
                    executor="nnapi",
                    quantization="fp16",
                    conversion_needed=True,
                ),
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "deploy",
                "gemma-1b",
                "--group",
                "production",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Preparing deployment plan" in result.output
        assert "Devices: 2" in result.output
        assert "iphone-1: coreml via coreml_npu" in result.output
        assert "(conversion needed)" in result.output
        mock_client.deploy.assert_not_called()

    @patch("edgeml.cli._get_client")
    def test_cli_deploy_fallback_no_targeting(self, mock_get_client):
        from click.testing import CliRunner

        from edgeml.cli import main

        mock_client = mock_get_client.return_value
        mock_client.deploy.return_value = {"id": "rollout-1", "status": "created"}

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["deploy", "sentiment-v1", "--rollout", "10", "--strategy", "canary"],
        )

        assert result.exit_code == 0
        assert "Rollout created" in result.output
        mock_client.deploy.assert_called_once_with(
            "sentiment-v1",
            version=None,
            rollout=10,
            strategy="canary",
        )

    @patch("edgeml.cli._get_client")
    def test_cli_deploy_blue_green_strategy(self, mock_get_client):
        from click.testing import CliRunner

        from edgeml.cli import main

        mock_client = mock_get_client.return_value
        mock_client.deploy.return_value = DeploymentResult(
            deployment_id="dep-bg",
            model_name="model",
            model_version="1.0.0",
            status="deploying",
            device_statuses=[],
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "deploy",
                "model",
                "--devices",
                "dev-1",
                "--strategy",
                "blue_green",
            ],
        )

        assert result.exit_code == 0


# ===================================================================
# CLI rollback command
# ===================================================================


class TestCLIRollback:
    @patch("edgeml.cli._get_client")
    def test_cli_rollback_default(self, mock_get_client):
        from click.testing import CliRunner

        from edgeml.cli import main

        mock_client = mock_get_client.return_value
        mock_client.rollback.return_value = RollbackResult(
            model_name="gemma-1b",
            from_version="2.0.0",
            to_version="1.0.0",
            rollout_id="rollout-rb",
            status="rolling_back",
        )

        runner = CliRunner()
        result = runner.invoke(main, ["rollback", "gemma-1b"])

        assert result.exit_code == 0
        assert "2.0.0 -> 1.0.0" in result.output
        assert "rollout-rb" in result.output
        mock_client.rollback.assert_called_once_with("gemma-1b", to_version=None)

    @patch("edgeml.cli._get_client")
    def test_cli_rollback_to_specific_version(self, mock_get_client):
        from click.testing import CliRunner

        from edgeml.cli import main

        mock_client = mock_get_client.return_value
        mock_client.rollback.return_value = RollbackResult(
            model_name="gemma-1b",
            from_version="3.0.0",
            to_version="1.0.0",
            rollout_id="rollout-rb2",
            status="rolling_back",
        )

        runner = CliRunner()
        result = runner.invoke(main, ["rollback", "gemma-1b", "--to-version", "1.0.0"])

        assert result.exit_code == 0
        assert "3.0.0 -> 1.0.0" in result.output
        mock_client.rollback.assert_called_once_with("gemma-1b", to_version="1.0.0")

    @patch("edgeml.cli._get_client")
    def test_cli_rollback_error(self, mock_get_client):
        from click.testing import CliRunner

        from edgeml.cli import main

        mock_client = mock_get_client.return_value
        mock_client.rollback.side_effect = RuntimeError("only one version exists")

        runner = CliRunner()
        result = runner.invoke(main, ["rollback", "gemma-1b"])

        assert result.exit_code == 1
        assert "Rollback failed" in result.output


# ===================================================================
# Parsing helpers
# ===================================================================


class TestParsingHelpers:
    def test_parse_deployment_result_with_id_key(self):
        from edgeml.client import _parse_deployment_result

        resp = {
            "id": "dep-alt",
            "status": "completed",
            "device_statuses": [
                {"device_id": "d1", "status": "ok"},
            ],
        }
        result = _parse_deployment_result(resp, "model-a", "1.0.0")
        assert result.deployment_id == "dep-alt"

    def test_parse_deployment_result_empty_statuses(self):
        from edgeml.client import _parse_deployment_result

        resp = {"deployment_id": "dep-x", "status": "empty"}
        result = _parse_deployment_result(resp, "m", "1")
        assert result.device_statuses == []

    def test_parse_deployment_plan_missing_fields(self):
        from edgeml.client import _parse_deployment_plan

        resp = {
            "deployments": [
                {"device_id": "d1"},  # minimal fields
            ]
        }
        plan = _parse_deployment_plan(resp, "m", "1")
        assert len(plan.deployments) == 1
        d = plan.deployments[0]
        assert d.format == ""
        assert d.executor == ""
        assert d.quantization == "none"
        assert d.conversion_needed is False
