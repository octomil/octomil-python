"""Tests for Client telemetry instrumentation (push, import_from_hf, rollback)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from octomil.client import Client
from octomil.telemetry import TelemetryReporter


def _make_client_with_reporter():
    """Create a Client with mocked internals and a captured reporter."""
    sent_events = []

    def mock_send(client, url, headers, payload):
        sent_events.append(payload)

    patcher = patch.object(TelemetryReporter, "_send", side_effect=mock_send)
    patcher.start()

    reporter = TelemetryReporter(
        api_key="test-key",
        api_base="https://api.test.com/api/v1",
        org_id="test-org",
        device_id="dev-1",
    )

    client = Client(api_key="test-key", org_id="test-org")

    return client, reporter, sent_events, patcher


def _extract_funnel_events(sent_envelopes):
    """Extract all funnel events from sent envelopes."""
    events = []
    for envelope in sent_envelopes:
        for event in envelope.get("events", []):
            if event["name"].startswith("funnel."):
                events.append(event)
    return events


# ---------------------------------------------------------------------------
# push() telemetry
# ---------------------------------------------------------------------------


class TestPushTelemetry:
    def test_push_reports_success(self):
        client, reporter, sent, patcher = _make_client_with_reporter()

        mock_registry = MagicMock()
        mock_registry.ensure_model.return_value = {"id": "model-123"}
        mock_registry.upload_version_from_path.return_value = {"status": "ok"}
        client._registry = mock_registry

        with patch("octomil.get_reporter", return_value=reporter):
            result = client.push(
                "/tmp/model.pt",
                name="test-model",
                version="1.0.0",
            )

        assert result == {"status": "ok"}
        time.sleep(0.15)
        reporter.close()
        patcher.stop()

        funnel_events = _extract_funnel_events(sent)
        assert len(funnel_events) == 1
        event = funnel_events[0]
        assert event["name"] == "funnel.model_push"
        attrs = event["attributes"]
        assert attrs["funnel.success"] is True
        assert attrs["model.id"] == "test-model"
        assert "funnel.duration_ms" in attrs
        assert attrs["funnel.duration_ms"] >= 0

    def test_push_reports_failure(self):
        client, reporter, sent, patcher = _make_client_with_reporter()

        mock_registry = MagicMock()
        mock_registry.ensure_model.side_effect = RuntimeError("upload failed")
        client._registry = mock_registry

        with patch("octomil.get_reporter", return_value=reporter):
            with pytest.raises(RuntimeError, match="upload failed"):
                client.push(
                    "/tmp/model.pt",
                    name="test-model",
                    version="1.0.0",
                )

        time.sleep(0.15)
        reporter.close()
        patcher.stop()

        funnel_events = _extract_funnel_events(sent)
        assert len(funnel_events) == 1
        event = funnel_events[0]
        assert event["name"] == "funnel.model_push"
        attrs = event["attributes"]
        assert attrs["funnel.success"] is False
        assert attrs["error.type"] == "upload_error"
        assert "upload failed" in attrs["error.message"]

    def test_push_works_without_reporter(self):
        """push() should work fine when no reporter is configured."""
        client = Client(api_key="test-key")

        mock_registry = MagicMock()
        mock_registry.ensure_model.return_value = {"id": "model-123"}
        mock_registry.upload_version_from_path.return_value = {"status": "ok"}
        client._registry = mock_registry

        with patch("octomil.get_reporter", return_value=None):
            result = client.push(
                "/tmp/model.pt",
                name="test-model",
                version="1.0.0",
            )

        assert result == {"status": "ok"}

    def test_push_reporter_failure_silently_swallowed(self):
        """If the reporter itself throws, push should still succeed."""
        client = Client(api_key="test-key")

        mock_registry = MagicMock()
        mock_registry.ensure_model.return_value = {"id": "model-123"}
        mock_registry.upload_version_from_path.return_value = {"status": "ok"}
        client._registry = mock_registry

        broken_reporter = MagicMock()
        broken_reporter.report_funnel_event.side_effect = RuntimeError("reporter broken")

        with patch("octomil.get_reporter", return_value=broken_reporter):
            result = client.push(
                "/tmp/model.pt",
                name="test-model",
                version="1.0.0",
            )

        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# import_from_hf() telemetry
# ---------------------------------------------------------------------------


class TestImportFromHfTelemetry:
    def test_import_reports_success(self):
        client, reporter, sent, patcher = _make_client_with_reporter()

        mock_api = MagicMock()
        mock_api.post.return_value = {"model_id": "m1", "status": "imported"}
        client._api = mock_api

        with patch("octomil.get_reporter", return_value=reporter):
            result = client.import_from_hf("microsoft/phi-4-mini")

        assert result["status"] == "imported"
        time.sleep(0.15)
        reporter.close()
        patcher.stop()

        funnel_events = _extract_funnel_events(sent)
        assert len(funnel_events) == 1
        event = funnel_events[0]
        assert event["name"] == "funnel.hf_import"
        attrs = event["attributes"]
        assert attrs["funnel.success"] is True
        assert attrs["model.id"] == "microsoft/phi-4-mini"
        assert attrs["funnel.duration_ms"] >= 0

    def test_import_uses_name_over_repo_id(self):
        client, reporter, sent, patcher = _make_client_with_reporter()

        mock_api = MagicMock()
        mock_api.post.return_value = {"status": "imported"}
        client._api = mock_api

        with patch("octomil.get_reporter", return_value=reporter):
            client.import_from_hf("microsoft/phi-4-mini", name="phi4")

        time.sleep(0.15)
        reporter.close()
        patcher.stop()

        funnel_events = _extract_funnel_events(sent)
        assert funnel_events[0]["attributes"]["model.id"] == "phi4"

    def test_import_reports_failure(self):
        client, reporter, sent, patcher = _make_client_with_reporter()

        mock_api = MagicMock()
        mock_api.post.side_effect = RuntimeError("network timeout")
        client._api = mock_api

        with patch("octomil.get_reporter", return_value=reporter):
            with pytest.raises(RuntimeError, match="network timeout"):
                client.import_from_hf("microsoft/phi-4-mini")

        time.sleep(0.15)
        reporter.close()
        patcher.stop()

        funnel_events = _extract_funnel_events(sent)
        assert len(funnel_events) == 1
        attrs = funnel_events[0]["attributes"]
        assert attrs["funnel.success"] is False
        assert attrs["error.type"] == "import_error"
        assert "network timeout" in attrs["error.message"]


# ---------------------------------------------------------------------------
# rollback() telemetry
# ---------------------------------------------------------------------------


class TestRollbackTelemetry:
    def test_rollback_reports_success(self):
        client, reporter, sent, patcher = _make_client_with_reporter()

        mock_registry = MagicMock()
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "2.0.0"
        mock_registry.list_versions.return_value = {
            "versions": [
                {"version": "2.0.0"},
                {"version": "1.0.0"},
            ]
        }
        mock_registry.deploy_version.return_value = {
            "id": "rollout-1",
            "status": "rolling_back",
        }
        client._registry = mock_registry

        with patch("octomil.get_reporter", return_value=reporter):
            result = client.rollback("test-model")

        assert result.to_version == "1.0.0"
        time.sleep(0.15)
        reporter.close()
        patcher.stop()

        funnel_events = _extract_funnel_events(sent)
        assert len(funnel_events) == 1
        event = funnel_events[0]
        assert event["name"] == "funnel.rollback"
        attrs = event["attributes"]
        assert attrs["funnel.success"] is True
        assert attrs["model.id"] == "test-model"
        assert attrs["funnel.duration_ms"] >= 0

    def test_rollback_reports_failure(self):
        client, reporter, sent, patcher = _make_client_with_reporter()

        mock_registry = MagicMock()
        mock_registry.resolve_model_id.side_effect = RuntimeError("model not found")
        client._registry = mock_registry

        with patch("octomil.get_reporter", return_value=reporter):
            with pytest.raises(RuntimeError, match="model not found"):
                client.rollback("test-model")

        time.sleep(0.15)
        reporter.close()
        patcher.stop()

        funnel_events = _extract_funnel_events(sent)
        assert len(funnel_events) == 1
        attrs = funnel_events[0]["attributes"]
        assert attrs["funnel.success"] is False
        assert attrs["error.type"] == "rollback_error"
        assert "model not found" in attrs["error.message"]

    def test_rollback_duration_positive(self):
        client, reporter, sent, patcher = _make_client_with_reporter()

        mock_registry = MagicMock()
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "2.0.0"
        mock_registry.list_versions.return_value = {
            "versions": [{"version": "2.0.0"}, {"version": "1.0.0"}]
        }
        mock_registry.deploy_version.return_value = {"id": "r1", "status": "ok"}
        client._registry = mock_registry

        with patch("octomil.get_reporter", return_value=reporter):
            client.rollback("test-model")

        time.sleep(0.15)
        reporter.close()
        patcher.stop()

        funnel_events = _extract_funnel_events(sent)
        assert funnel_events[0]["attributes"]["funnel.duration_ms"] >= 0
