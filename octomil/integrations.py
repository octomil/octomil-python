"""Integration management for metrics and log export.

Provides ``IntegrationsAPI`` for managing OTLP, Prometheus, Datadog, Splunk,
and other export integrations via the Octomil API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass
class MetricsIntegration:
    """A configured metrics export integration."""
    id: str
    org_id: str
    name: str
    integration_type: str
    enabled: bool
    config: dict[str, Any]
    created_at: str
    updated_at: str
    last_export_at: Optional[str] = None


@dataclass
class LogIntegration:
    """A configured log export integration."""
    id: str
    org_id: str
    name: str
    integration_type: str
    endpoint_url: str
    format: str
    enabled: bool
    created_at: str
    updated_at: str


class IntegrationsAPI:
    """Manage metrics and log export integrations.

    Parameters
    ----------
    api_key:
        Octomil API key.
    org_id:
        Organization ID.
    api_base:
        API base URL.
    """

    def __init__(
        self,
        api_key: str,
        org_id: str,
        api_base: str = "https://api.octomil.com/api/v1",
    ) -> None:
        self._api_key = api_key
        self._org_id = org_id
        self._api_base = api_base.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _url(self, path: str) -> str:
        return f"{self._api_base}{path}"

    # ---- Metrics integrations ----

    def list_metrics_integrations(self) -> list[MetricsIntegration]:
        """List all metrics export integrations."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                self._url(f"/metrics/integrations?org_id={self._org_id}"),
                headers=self._headers(),
            )
        resp.raise_for_status()
        return [MetricsIntegration(**item) for item in resp.json()]

    def create_metrics_integration(
        self,
        name: str,
        integration_type: str,
        config: dict[str, Any],
        *,
        enabled: bool = True,
    ) -> MetricsIntegration:
        """Create a metrics export integration."""
        payload = {
            "name": name,
            "integration_type": integration_type,
            "config": config,
            "enabled": enabled,
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                self._url(f"/metrics/integrations?org_id={self._org_id}"),
                json=payload,
                headers=self._headers(),
            )
        resp.raise_for_status()
        return MetricsIntegration(**resp.json())

    def delete_metrics_integration(self, integration_id: str) -> None:
        """Delete a metrics integration."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(
                self._url(f"/metrics/integrations/{integration_id}"),
                headers=self._headers(),
            )
        resp.raise_for_status()

    def test_metrics_integration(self, integration_id: str) -> dict[str, Any]:
        """Test a metrics integration."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                self._url(f"/metrics/integrations/{integration_id}/test"),
                json={},
                headers=self._headers(),
            )
        resp.raise_for_status()
        return resp.json()

    # ---- Log integrations ----

    def list_log_integrations(self) -> list[LogIntegration]:
        """List all log export integrations."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                self._url("/log-streams/integrations"),
                headers=self._headers(),
            )
        resp.raise_for_status()
        return [LogIntegration(**item) for item in resp.json()]

    def create_log_integration(
        self,
        name: str,
        integration_type: str,
        endpoint_url: str,
        *,
        format: str = "json",
        auth_config: Optional[dict[str, Any]] = None,
    ) -> LogIntegration:
        """Create a log export integration."""
        payload: dict[str, Any] = {
            "name": name,
            "integration_type": integration_type,
            "endpoint_url": endpoint_url,
            "format": format,
        }
        if auth_config:
            payload["auth_config"] = auth_config
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                self._url("/log-streams/integrations"),
                json=payload,
                headers=self._headers(),
            )
        resp.raise_for_status()
        return LogIntegration(**resp.json())

    def delete_log_integration(self, integration_id: str) -> None:
        """Delete a log integration."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(
                self._url(f"/log-streams/integrations/{integration_id}"),
                headers=self._headers(),
            )
        resp.raise_for_status()

    def test_log_integration(self, integration_id: str) -> dict[str, Any]:
        """Test a log integration."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                self._url(f"/log-streams/integrations/{integration_id}/test"),
                json={},
                headers=self._headers(),
            )
        resp.raise_for_status()
        return resp.json()

    # ---- Unified OTLP shortcut ----

    def connect_otlp_collector(
        self,
        name: str,
        endpoint: str,
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Configure a single OTLP collector for both metrics and logs.

        Creates two integrations (metrics + logs) pointing at the same
        collector endpoint. This is the recommended way to set up OTLP.

        Parameters
        ----------
        name:
            Human-readable name (e.g. "Production Grafana").
        endpoint:
            Collector base URL (e.g. "http://otel-collector:4318").
        headers:
            Optional HTTP headers (auth tokens, API keys).

        Returns
        -------
        dict with "metrics" and "logs" integration objects.
        """
        base = endpoint.rstrip("/")

        metrics_config: dict[str, Any] = {"endpoint": base}
        if headers:
            metrics_config["headers"] = headers

        metrics = self.create_metrics_integration(
            name=f"{name} (metrics)",
            integration_type="opentelemetry",
            config=metrics_config,
        )

        auth_config: dict[str, Any] = {}
        if headers:
            auth_config = {"type": "headers", "headers": headers}

        logs = self.create_log_integration(
            name=f"{name} (logs)",
            integration_type="otlp",
            endpoint_url=f"{base}/v1/logs",
            format="otlp",
            auth_config=auth_config if auth_config else None,
        )

        return {"metrics": metrics, "logs": logs}
