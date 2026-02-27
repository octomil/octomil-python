"""Tests for telemetry wiring in octomil.client.Client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# Standard decorator stack for Client construction — suppresses real
# _ApiClient / ModelRegistry / RolloutsAPI creation.
_PATCH_ROLLOUTS = patch("octomil.client.RolloutsAPI")
_PATCH_REGISTRY = patch("octomil.client.ModelRegistry")
_PATCH_API = patch("octomil.client._ApiClient")


class TestClientTelemetryInit:
    """Verify reporter creation logic in Client.__init__."""

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_creates_reporter_when_api_key_set(
        self, mock_api, mock_registry, mock_rollouts
    ):
        with patch("octomil.telemetry.TelemetryReporter") as mock_tr_cls:
            mock_reporter = MagicMock()
            mock_tr_cls.return_value = mock_reporter

            from octomil.client import Client

            c = Client(api_key="test-key", org_id="org1", api_base="https://api.test")
            assert c._reporter is not None

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_skips_reporter_when_api_key_empty(
        self, mock_api, mock_registry, mock_rollouts, monkeypatch
    ):
        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)

        from octomil.client import Client

        c = Client(api_key="")
        assert c._reporter is None

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_skips_reporter_when_no_api_key(
        self, mock_api, mock_registry, mock_rollouts, monkeypatch
    ):
        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)

        from octomil.client import Client

        c = Client()
        assert c._reporter is None

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_reporter_creation_failure_silently_ignored(
        self, mock_api, mock_registry, mock_rollouts
    ):
        from octomil.client import Client

        with patch(
            "octomil.telemetry.TelemetryReporter",
            side_effect=RuntimeError("init boom"),
        ):
            c = Client(api_key="test-key")
            assert c._reporter is None


class TestClientPullTelemetry:
    """Verify pull() reports funnel events."""

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_pull_reports_success_event(
        self, mock_api, mock_registry_cls, mock_rollouts
    ):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.onnx"}

        mock_reporter = MagicMock()

        c = Client(api_key="key")
        c._reporter = mock_reporter

        result = c.pull("my-model")

        assert result["model_path"] == "/tmp/m.onnx"
        mock_reporter.report_funnel_event.assert_called_once()
        call_kwargs = mock_reporter.report_funnel_event.call_args[1]
        assert call_kwargs["stage"] == "model_pull"
        assert call_kwargs["success"] is True
        assert call_kwargs["model_id"] == "my-model"
        assert "duration_ms" in call_kwargs

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_pull_reports_failure_event(
        self, mock_api, mock_registry_cls, mock_rollouts
    ):
        import pytest

        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.side_effect = RuntimeError("download failed")

        mock_reporter = MagicMock()

        c = Client(api_key="key")
        c._reporter = mock_reporter

        with pytest.raises(RuntimeError, match="download failed"):
            c.pull("my-model")

        mock_reporter.report_funnel_event.assert_called_once()
        call_kwargs = mock_reporter.report_funnel_event.call_args[1]
        assert call_kwargs["stage"] == "model_pull"
        assert call_kwargs["success"] is False
        assert call_kwargs["model_id"] == "my-model"
        assert call_kwargs["failure_reason"] == "download failed"
        assert call_kwargs["failure_category"] == "download_error"
        assert "duration_ms" in call_kwargs

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_pull_works_without_reporter(
        self, mock_api, mock_registry_cls, mock_rollouts
    ):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.onnx"}

        c = Client(api_key="key")
        c._reporter = None

        result = c.pull("my-model")
        assert result["model_path"] == "/tmp/m.onnx"


class TestClientLoadModelTelemetry:
    """Verify load_model() passes reporter to Model and reports events."""

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_load_model_passes_reporter_to_model(
        self, mock_api, mock_registry_cls, mock_rollouts
    ):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.onnx"}

        mock_reporter = MagicMock()
        mock_engine = MagicMock()
        mock_engine.manages_own_download = False
        mock_engine_registry = MagicMock()
        mock_engine_registry.auto_select.return_value = (mock_engine, [])
        mock_model_instance = MagicMock()

        with patch("octomil.engines.get_registry", return_value=mock_engine_registry), \
             patch("octomil.model.Model", return_value=mock_model_instance) as mock_model_cls:

            c = Client(api_key="key")
            c._reporter = mock_reporter

            model = c.load_model("my-model")

            mock_model_cls.assert_called_once()
            call_kwargs = mock_model_cls.call_args[1]
            assert call_kwargs["_reporter"] is mock_reporter
            assert model is mock_model_instance

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_load_model_reports_funnel_event(
        self, mock_api, mock_registry_cls, mock_rollouts
    ):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.onnx"}

        mock_reporter = MagicMock()
        mock_engine = MagicMock()
        mock_engine.manages_own_download = False
        mock_engine_registry = MagicMock()
        mock_engine_registry.auto_select.return_value = (mock_engine, [])

        with patch("octomil.engines.get_registry", return_value=mock_engine_registry), \
             patch("octomil.model.Model"):

            c = Client(api_key="key")
            c._reporter = mock_reporter

            c.load_model("my-model")

            # Should have 2 calls: one from pull (model_pull), one from load_model (model_load)
            assert mock_reporter.report_funnel_event.call_count == 2
            calls = mock_reporter.report_funnel_event.call_args_list

            pull_kwargs = calls[0][1]
            assert pull_kwargs["stage"] == "model_pull"
            assert pull_kwargs["success"] is True

            load_kwargs = calls[1][1]
            assert load_kwargs["stage"] == "model_load"
            assert load_kwargs["success"] is True
            assert load_kwargs["model_id"] == "my-model"
            assert "duration_ms" in load_kwargs

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_load_model_caches_model(
        self, mock_api, mock_registry_cls, mock_rollouts
    ):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.onnx"}

        mock_engine = MagicMock()
        mock_engine.manages_own_download = False
        mock_engine_registry = MagicMock()
        mock_engine_registry.auto_select.return_value = (mock_engine, [])
        mock_model_instance = MagicMock()

        with patch("octomil.engines.get_registry", return_value=mock_engine_registry), \
             patch("octomil.model.Model", return_value=mock_model_instance):

            c = Client(api_key="key")
            c._reporter = None

            c.load_model("my-model")
            assert c._models["my-model"] is mock_model_instance


class TestClientDisposeTelemetry:
    """Verify dispose() cleans up reporter."""

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_dispose_closes_reporter(
        self, mock_api, mock_registry, mock_rollouts
    ):
        from octomil.client import Client

        mock_reporter = MagicMock()

        c = Client(api_key="key")
        c._reporter = mock_reporter

        c.dispose()

        mock_reporter.close.assert_called_once()
        assert c._reporter is None
        assert c._models == {}

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_dispose_without_reporter(
        self, mock_api, mock_registry, mock_rollouts
    ):
        from octomil.client import Client

        c = Client(api_key="key")
        c._reporter = None

        # Should not raise
        c.dispose()
        assert c._models == {}

    @_PATCH_ROLLOUTS
    @_PATCH_REGISTRY
    @_PATCH_API
    def test_dispose_reporter_close_failure_silently_ignored(
        self, mock_api, mock_registry, mock_rollouts
    ):
        from octomil.client import Client

        mock_reporter = MagicMock()
        mock_reporter.close.side_effect = RuntimeError("close boom")

        c = Client(api_key="key")
        c._reporter = mock_reporter

        # Should not raise
        c.dispose()
        assert c._reporter is None
