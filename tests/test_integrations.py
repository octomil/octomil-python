"""Tests for octomil.integrations module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from octomil.integrations import IntegrationsAPI, MetricsIntegration, LogIntegration


@pytest.fixture
def api() -> IntegrationsAPI:
    return IntegrationsAPI(api_key="test-key", org_id="org-1", api_base="http://localhost:8000/api/v1")


class TestListMetricsIntegrations:
    def test_returns_list(self, api: IntegrationsAPI) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "int-1",
                "org_id": "org-1",
                "name": "Prod Prometheus",
                "integration_type": "prometheus",
                "enabled": True,
                "config": {"prefix": "octomil"},
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ]
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(get=MagicMock(return_value=mock_resp)))
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = api.list_metrics_integrations()
        assert len(result) == 1
        assert isinstance(result[0], MetricsIntegration)
        assert result[0].name == "Prod Prometheus"

    def test_empty_list(self, api: IntegrationsAPI) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = []
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(get=MagicMock(return_value=mock_resp)))
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = api.list_metrics_integrations()
        assert result == []


class TestCreateMetricsIntegration:
    def test_creates_integration(self, api: IntegrationsAPI) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "id": "int-2",
            "org_id": "org-1",
            "name": "OTEL",
            "integration_type": "opentelemetry",
            "enabled": True,
            "config": {"endpoint": "http://collector:4318"},
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_resp)))
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = api.create_metrics_integration("OTEL", "opentelemetry", {"endpoint": "http://collector:4318"})
        assert isinstance(result, MetricsIntegration)
        assert result.id == "int-2"


class TestListLogIntegrations:
    def test_returns_list(self, api: IntegrationsAPI) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "log-1",
                "org_id": "org-1",
                "name": "Splunk",
                "integration_type": "splunk",
                "endpoint_url": "https://splunk.example.com",
                "format": "hec",
                "enabled": True,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ]
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(get=MagicMock(return_value=mock_resp)))
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = api.list_log_integrations()
        assert len(result) == 1
        assert isinstance(result[0], LogIntegration)
        assert result[0].name == "Splunk"


class TestConnectOtlpCollector:
    def test_creates_both_integrations(self, api: IntegrationsAPI) -> None:
        metrics_resp = MagicMock()
        metrics_resp.raise_for_status = MagicMock()
        metrics_resp.json.return_value = {
            "id": "m-1", "org_id": "org-1", "name": "Prod (metrics)",
            "integration_type": "opentelemetry", "enabled": True,
            "config": {"endpoint": "http://collector:4318"},
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        }
        logs_resp = MagicMock()
        logs_resp.raise_for_status = MagicMock()
        logs_resp.json.return_value = {
            "id": "l-1", "org_id": "org-1", "name": "Prod (logs)",
            "integration_type": "otlp", "endpoint_url": "http://collector:4318/v1/logs",
            "format": "otlp", "enabled": True,
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        }

        call_count = 0
        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return metrics_resp if call_count == 1 else logs_resp

        with patch("httpx.Client") as mock_client_cls:
            mock_inner = MagicMock()
            mock_inner.post = mock_post
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_inner)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = api.connect_otlp_collector("Prod", "http://collector:4318")

        assert "metrics" in result
        assert "logs" in result
        assert isinstance(result["metrics"], MetricsIntegration)
        assert isinstance(result["logs"], LogIntegration)

    def test_with_headers(self, api: IntegrationsAPI) -> None:
        metrics_resp = MagicMock()
        metrics_resp.raise_for_status = MagicMock()
        metrics_resp.json.return_value = {
            "id": "m-2", "org_id": "org-1", "name": "Grafana (metrics)",
            "integration_type": "opentelemetry", "enabled": True,
            "config": {"endpoint": "https://otlp.grafana.net", "headers": {"Authorization": "Basic abc"}},
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        }
        logs_resp = MagicMock()
        logs_resp.raise_for_status = MagicMock()
        logs_resp.json.return_value = {
            "id": "l-2", "org_id": "org-1", "name": "Grafana (logs)",
            "integration_type": "otlp", "endpoint_url": "https://otlp.grafana.net/v1/logs",
            "format": "otlp", "enabled": True,
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        }

        call_count = 0
        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return metrics_resp if call_count == 1 else logs_resp

        with patch("httpx.Client") as mock_client_cls:
            mock_inner = MagicMock()
            mock_inner.post = mock_post
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_inner)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = api.connect_otlp_collector(
                "Grafana", "https://otlp.grafana.net",
                headers={"Authorization": "Basic abc"},
            )

        assert result["metrics"].config["headers"] == {"Authorization": "Basic abc"}


class TestDataclasses:
    def test_metrics_integration_optional_field(self) -> None:
        m = MetricsIntegration(
            id="1", org_id="o", name="n", integration_type="prometheus",
            enabled=True, config={}, created_at="", updated_at="",
        )
        assert m.last_export_at is None

    def test_log_integration(self) -> None:
        lg = LogIntegration(
            id="1", org_id="o", name="n", integration_type="webhook",
            endpoint_url="http://x", format="json", enabled=True,
            created_at="", updated_at="",
        )
        assert lg.endpoint_url == "http://x"
