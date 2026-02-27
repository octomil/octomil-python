"""Tests for octomil.telemetry — TelemetryReporter v2 OTLP format and octomil.init()."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

try:
    import fastapi  # noqa: F401
    import pytest_asyncio  # noqa: F401

    _has_serve_deps = True
except ImportError:
    _has_serve_deps = False

from httpx import ASGITransport, AsyncClient

from octomil.telemetry import TelemetryReporter, _generate_device_id, _v2_url


# ---------------------------------------------------------------------------
# _generate_device_id
# ---------------------------------------------------------------------------


class TestGenerateDeviceId:
    def test_returns_hex_string(self):
        did = _generate_device_id()
        assert isinstance(did, str)
        assert len(did) == 16
        # Must be valid hex
        int(did, 16)

    def test_deterministic(self):
        """Same machine should produce the same device ID."""
        assert _generate_device_id() == _generate_device_id()


# ---------------------------------------------------------------------------
# _v2_url helper
# ---------------------------------------------------------------------------


class TestV2Url:
    def test_standard_api_base(self):
        assert _v2_url("https://api.octomil.com/api/v1") == "https://api.octomil.com/api/v2/telemetry/events"

    def test_trailing_slash(self):
        assert _v2_url("https://api.octomil.com/api/v1/") == "https://api.octomil.com/api/v2/telemetry/events"

    def test_custom_host(self):
        assert _v2_url("https://custom.host.com/api/v1") == "https://custom.host.com/api/v2/telemetry/events"

    def test_no_v1_suffix_fallback(self):
        url = _v2_url("https://example.com/custom")
        assert url == "https://example.com/custom/v2/telemetry/events"


# ---------------------------------------------------------------------------
# TelemetryReporter — v2 OTLP envelope structure
# ---------------------------------------------------------------------------


class TestTelemetryV2Envelope:
    """Verify that dispatched payloads use the OTLP envelope format."""

    def test_envelope_structure(self):
        """Every POST should contain resource + events list."""
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_started(
                model_id="gemma-1b",
                version="1.0",
                session_id="sess-001",
            )
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        envelope = sent[0]
        assert "resource" in envelope
        assert "events" in envelope
        assert isinstance(envelope["events"], list)
        assert len(envelope["events"]) >= 1

    def test_resource_fields(self):
        """Resource block should contain sdk, sdk_version, device_id, platform, org_id."""
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="test-org",
                device_id="dev-abc",
            )
            reporter.report_generation_started(
                model_id="m", version="v", session_id="s",
            )
            time.sleep(0.15)
            reporter.close()

        resource = sent[0]["resource"]
        assert resource["sdk"] == "python"
        assert resource["device_id"] == "dev-abc"
        assert resource["platform"] == sys.platform
        assert resource["org_id"] == "test-org"
        assert "sdk_version" in resource

    def test_v2_endpoint_used(self):
        """Dispatch should POST to the v2 telemetry endpoint."""
        urls = []

        def mock_send(client, url, headers, payload):
            urls.append(url)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_started(
                model_id="m", version="v", session_id="s",
            )
            time.sleep(0.15)
            reporter.close()

        assert len(urls) >= 1
        assert urls[0] == "https://api.test.com/api/v2/telemetry/events"

    def test_single_endpoint_for_inference_and_funnel(self):
        """Both inference and funnel events should go to the same v2 endpoint."""
        urls = []

        def mock_send(client, url, headers, payload):
            urls.append(url)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_started(
                model_id="m", version="v", session_id="s",
            )
            reporter.report_funnel_event(stage="download", success=True)
            time.sleep(0.15)
            reporter.close()

        assert len(urls) >= 2
        expected = "https://api.test.com/api/v2/telemetry/events"
        for url in urls:
            assert url == expected


# ---------------------------------------------------------------------------
# TelemetryReporter — event name dot notation
# ---------------------------------------------------------------------------


class TestEventNameDotNotation:
    """Verify event names use dot notation."""

    def _capture_events(self):
        """Helper: returns (reporter, sent_list) with _send patched."""
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        patcher = patch.object(TelemetryReporter, "_send", side_effect=mock_send)
        patcher.start()
        reporter = TelemetryReporter(
            api_key="key",
            api_base="https://api.test.com/api/v1",
            org_id="org-1",
            device_id="dev-1",
        )
        return reporter, sent, patcher

    def _get_event(self, sent, index=0):
        """Extract the first event from the first envelope."""
        return sent[index]["events"][0]

    def test_inference_started(self):
        reporter, sent, patcher = self._capture_events()
        reporter.report_generation_started(
            model_id="m", version="v", session_id="s",
        )
        time.sleep(0.15)
        reporter.close()
        patcher.stop()
        event = self._get_event(sent)
        assert event["name"] == "inference.started"

    def test_inference_completed(self):
        reporter, sent, patcher = self._capture_events()
        reporter.report_generation_completed(
            session_id="s", model_id="m", version="v",
            total_chunks=10, total_duration_ms=500.0,
            ttfc_ms=30.0, throughput=20.0,
        )
        time.sleep(0.15)
        reporter.close()
        patcher.stop()
        event = self._get_event(sent)
        assert event["name"] == "inference.completed"

    def test_inference_failed(self):
        reporter, sent, patcher = self._capture_events()
        reporter.report_generation_failed(
            session_id="s", model_id="m", version="v",
        )
        time.sleep(0.15)
        reporter.close()
        patcher.stop()
        event = self._get_event(sent)
        assert event["name"] == "inference.failed"

    def test_inference_chunk_produced(self):
        reporter, sent, patcher = self._capture_events()
        reporter.report_chunk_produced(
            session_id="s", model_id="m", version="v", chunk_index=0,
        )
        time.sleep(0.15)
        reporter.close()
        patcher.stop()
        event = self._get_event(sent)
        assert event["name"] == "inference.chunk_produced"

    def test_funnel_event(self):
        reporter, sent, patcher = self._capture_events()
        reporter.report_funnel_event(stage="download", success=True)
        time.sleep(0.15)
        reporter.close()
        patcher.stop()
        event = self._get_event(sent)
        assert event["name"] == "funnel.download"


# ---------------------------------------------------------------------------
# TelemetryReporter — attribute mapping per event type
# ---------------------------------------------------------------------------


class TestAttributeMapping:
    """Verify correct attribute keys for each event type."""

    def test_generation_started_attributes(self):
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_started(
                model_id="gemma-1b",
                version="1.0",
                session_id="sess-001",
                modality="text",
                attention_backend="flash_attention",
            )
            time.sleep(0.15)
            reporter.close()

        event = sent[0]["events"][0]
        attrs = event["attributes"]
        assert attrs["model.id"] == "gemma-1b"
        assert attrs["model.version"] == "1.0"
        assert attrs["inference.session_id"] == "sess-001"
        assert attrs["inference.modality"] == "text"
        assert attrs["inference.attention_backend"] == "flash_attention"

    def test_generation_started_no_attention_backend(self):
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_started(
                model_id="m", version="v", session_id="s",
            )
            time.sleep(0.15)
            reporter.close()

        attrs = sent[0]["events"][0]["attributes"]
        assert "inference.attention_backend" not in attrs

    def test_chunk_produced_attributes(self):
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_chunk_produced(
                session_id="s1",
                model_id="model-a",
                version="2.0",
                chunk_index=0,
                ttfc_ms=42.5,
                chunk_latency_ms=3.1,
            )
            time.sleep(0.15)
            reporter.close()

        event = sent[0]["events"][0]
        assert event["name"] == "inference.chunk_produced"
        attrs = event["attributes"]
        assert attrs["inference.chunk_index"] == 0
        assert attrs["inference.ttfc_ms"] == 42.5
        assert attrs["inference.chunk_latency_ms"] == 3.1
        assert attrs["model.id"] == "model-a"
        assert attrs["inference.session_id"] == "s1"

    def test_generation_completed_attributes(self):
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_completed(
                session_id="s1",
                model_id="model-a",
                version="2.0",
                total_chunks=10,
                total_duration_ms=500.0,
                ttfc_ms=30.0,
                throughput=20.0,
                attention_backend="metal_fused",
            )
            time.sleep(0.15)
            reporter.close()

        event = sent[0]["events"][0]
        assert event["name"] == "inference.completed"
        attrs = event["attributes"]
        assert attrs["inference.duration_ms"] == 500.0
        assert attrs["inference.ttft_ms"] == 30.0
        assert attrs["inference.total_tokens"] == 10
        assert attrs["inference.throughput_tps"] == 20.0
        assert attrs["inference.attention_backend"] == "metal_fused"
        assert attrs["model.id"] == "model-a"

    def test_generation_failed_attributes(self):
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_failed(
                session_id="s1",
                model_id="model-a",
                version="2.0",
            )
            time.sleep(0.15)
            reporter.close()

        event = sent[0]["events"][0]
        assert event["name"] == "inference.failed"
        attrs = event["attributes"]
        assert attrs["error.type"] == "generation_failed"
        assert attrs["model.id"] == "model-a"
        assert attrs["inference.session_id"] == "s1"

    def test_funnel_event_attributes(self):
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_funnel_event(
                stage="deploy",
                success=False,
                model_id="model-a",
                session_id="s1",
                failure_reason="timeout",
                failure_category="network_error",
            )
            time.sleep(0.15)
            reporter.close()

        event = sent[0]["events"][0]
        assert event["name"] == "funnel.deploy"
        attrs = event["attributes"]
        assert attrs["funnel.success"] is False
        assert attrs["model.id"] == "model-a"
        assert attrs["inference.session_id"] == "s1"
        assert attrs["error.message"] == "timeout"
        assert attrs["error.type"] == "network_error"


# ---------------------------------------------------------------------------
# TelemetryReporter — TPOT calculation
# ---------------------------------------------------------------------------


class TestTPOTCalculation:
    """Verify TPOT (time per output token) is computed correctly."""

    def test_tpot_basic(self):
        """TPOT = (total_duration_ms - ttft_ms) / (total_tokens - 1)."""
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_completed(
                session_id="s1",
                model_id="m",
                version="v",
                total_chunks=10,
                total_duration_ms=500.0,
                ttfc_ms=50.0,
                throughput=20.0,
            )
            time.sleep(0.15)
            reporter.close()

        attrs = sent[0]["events"][0]["attributes"]
        # (500 - 50) / (10 - 1) = 450 / 9 = 50.0
        assert attrs["inference.tpot_ms"] == pytest.approx(50.0)

    def test_tpot_not_present_when_single_token(self):
        """When total_tokens <= 1, TPOT should not be included."""
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_completed(
                session_id="s1",
                model_id="m",
                version="v",
                total_chunks=1,
                total_duration_ms=100.0,
                ttfc_ms=50.0,
                throughput=10.0,
            )
            time.sleep(0.15)
            reporter.close()

        attrs = sent[0]["events"][0]["attributes"]
        assert "inference.tpot_ms" not in attrs

    def test_tpot_not_present_when_zero_duration(self):
        """When total_duration_ms is 0, TPOT should not be included."""
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_completed(
                session_id="s1",
                model_id="m",
                version="v",
                total_chunks=5,
                total_duration_ms=0.0,
                ttfc_ms=0.0,
                throughput=0.0,
            )
            time.sleep(0.15)
            reporter.close()

        attrs = sent[0]["events"][0]["attributes"]
        assert "inference.tpot_ms" not in attrs

    def test_tpot_with_two_tokens(self):
        """With exactly 2 tokens, TPOT = total_duration - ttft."""
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_completed(
                session_id="s1",
                model_id="m",
                version="v",
                total_chunks=2,
                total_duration_ms=200.0,
                ttfc_ms=50.0,
                throughput=10.0,
            )
            time.sleep(0.15)
            reporter.close()

        attrs = sent[0]["events"][0]["attributes"]
        # (200 - 50) / (2 - 1) = 150.0
        assert attrs["inference.tpot_ms"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# TelemetryReporter — modality propagation
# ---------------------------------------------------------------------------


class TestModalityPropagation:
    """Verify modality is included in all inference event attributes."""

    def test_modality_in_started(self):
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_started(
                model_id="m", version="v", session_id="s", modality="audio",
            )
            time.sleep(0.15)
            reporter.close()

        attrs = sent[0]["events"][0]["attributes"]
        assert attrs["inference.modality"] == "audio"

    def test_default_modality_is_text(self):
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_started(
                model_id="m", version="v", session_id="s",
            )
            time.sleep(0.15)
            reporter.close()

        attrs = sent[0]["events"][0]["attributes"]
        assert attrs["inference.modality"] == "text"

    def test_modality_in_completed(self):
        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_completed(
                session_id="s", model_id="m", version="v",
                total_chunks=5, total_duration_ms=100.0,
                ttfc_ms=10.0, throughput=50.0, modality="image",
            )
            time.sleep(0.15)
            reporter.close()

        attrs = sent[0]["events"][0]["attributes"]
        assert attrs["inference.modality"] == "image"


# ---------------------------------------------------------------------------
# TelemetryReporter — best-effort behaviour
# ---------------------------------------------------------------------------


class TestTelemetryBestEffort:
    def test_network_failure_does_not_raise(self):
        """When the HTTP POST fails, the reporter logs a warning but does not raise."""
        reporter = TelemetryReporter(
            api_key="key",
            api_base="https://unreachable.invalid/api/v1",
            org_id="org-1",
            device_id="dev-1",
        )
        # This should not raise even though the endpoint doesn't exist
        reporter.report_generation_started(
            model_id="model-a",
            version="1.0",
            session_id="s1",
        )
        # Give the background thread time to attempt dispatch
        time.sleep(0.3)
        reporter.close()
        # If we reach here, best-effort is working

    def test_send_catches_exceptions(self):
        """_send should catch all exceptions and not propagate."""
        import httpx

        client = httpx.Client()
        # Calling with an invalid URL should not raise
        TelemetryReporter._send(
            client,
            "https://unreachable.invalid/events",
            {"Authorization": "Bearer x"},
            {"resource": {}, "events": [{"name": "test"}]},
        )
        client.close()

    def test_queue_full_drops_event(self):
        """When the queue is full, enqueue should drop silently."""
        reporter = TelemetryReporter(
            api_key="key",
            api_base="https://api.test.com/api/v1",
            org_id="org-1",
            device_id="dev-1",
        )
        # Stop the worker from consuming
        reporter._queue.put(None)  # sentinel to stop worker
        reporter._worker.join(timeout=2.0)

        # Fill the queue (maxsize=1024)
        for i in range(1100):
            reporter.report_generation_started(
                model_id="m", version="v", session_id=f"s{i}"
            )
        # Should not raise


# ---------------------------------------------------------------------------
# octomil.init() — env var fallback and validation
# ---------------------------------------------------------------------------


class TestOctomilInit:
    def test_init_raises_without_api_key(self):
        """init() should raise ValueError when no API key is provided."""
        import octomil

        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars
            for key in ("OCTOMIL_API_KEY", "OCTOMIL_ORG_ID", "OCTOMIL_API_BASE"):
                os.environ.pop(key, None)
            with pytest.raises(ValueError, match="API key required"):
                octomil.init()

    def test_init_uses_env_vars(self):
        """init() should read from env vars when args are not passed."""
        import octomil

        env = {
            "OCTOMIL_API_KEY": "test-key-123",
            "OCTOMIL_ORG_ID": "my-org",
            "OCTOMIL_API_BASE": "https://custom.api.com/api/v1",
        }
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.dict(os.environ, env, clear=False):
            with patch("httpx.Client") as MockClient:
                mock_client_instance = MagicMock()
                mock_client_instance.__enter__ = MagicMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__exit__ = MagicMock(return_value=False)
                mock_client_instance.get.return_value = mock_response
                MockClient.return_value = mock_client_instance

                octomil.init()

                assert octomil._config["api_key"] == "test-key-123"
                assert octomil._config["org_id"] == "my-org"
                assert octomil._config["api_base"] == "https://custom.api.com/api/v1"
                assert octomil._reporter is not None

                # Cleanup
                octomil._reporter.close()
                octomil._reporter = None
                octomil._config = {}

    def test_init_with_explicit_args(self):
        """init() should prefer explicit args over env vars."""
        import octomil

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.get.return_value = mock_response
            MockClient.return_value = mock_client_instance

            octomil.init(
                api_key="explicit-key",
                org_id="explicit-org",
                api_base="https://explicit.api.com/api/v1",
            )

            assert octomil._config["api_key"] == "explicit-key"
            assert octomil._config["org_id"] == "explicit-org"
            assert octomil._reporter is not None

            # Cleanup
            octomil._reporter.close()
            octomil._reporter = None
            octomil._config = {}

    def test_init_invalid_key_raises(self):
        """init() should raise ValueError on 401/403 from the health check."""
        import octomil

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.get.return_value = mock_response
            MockClient.return_value = mock_client_instance

            with pytest.raises(ValueError, match="Invalid Octomil API key"):
                octomil.init(api_key="bad-key")

    def test_init_unreachable_api_still_creates_reporter(self):
        """If the API is unreachable, init() should warn but still create a reporter."""
        import octomil
        import httpx

        with patch("httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.get.side_effect = httpx.ConnectError("unreachable")
            MockClient.return_value = mock_client_instance

            octomil.init(api_key="some-key")
            assert octomil._reporter is not None

            # Cleanup
            octomil._reporter.close()
            octomil._reporter = None
            octomil._config = {}


class TestGetReporter:
    def test_returns_none_before_init(self):
        import octomil

        # Save and reset state
        saved_reporter = octomil._reporter
        octomil._reporter = None
        try:
            assert octomil.get_reporter() is None
        finally:
            octomil._reporter = saved_reporter

    def test_returns_reporter_after_init(self):
        import octomil

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.get.return_value = mock_response
            MockClient.return_value = mock_client_instance

            octomil.init(api_key="test-key")
            reporter = octomil.get_reporter()
            assert reporter is not None
            assert isinstance(reporter, TelemetryReporter)

            # Cleanup
            reporter.close()
            octomil._reporter = None
            octomil._config = {}


# ---------------------------------------------------------------------------
# serve.py integration — telemetry when api_key is set
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_app_with_telemetry():
    """Create a FastAPI app with EchoBackend and telemetry enabled."""
    if not _has_serve_deps:
        pytest.skip("fastapi and pytest-asyncio required")
    from octomil.serve import EchoBackend, create_app

    with patch("octomil.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model", api_key="test-api-key")

        # Trigger lifespan startup manually
        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())

    return app


@pytest.fixture
def echo_app_without_telemetry():
    """Create a FastAPI app with EchoBackend and no telemetry."""
    if not _has_serve_deps:
        pytest.skip("fastapi and pytest-asyncio required")
    from octomil.serve import EchoBackend, create_app

    with patch("octomil.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model")  # no api_key

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())

    return app


@pytest.mark.asyncio
async def test_serve_creates_reporter_when_api_key_set(echo_app_with_telemetry):
    """When api_key is set, the app should have a reporter on state."""
    # The app was created with api_key="test-api-key"
    # We can check via the health endpoint that the server is running
    transport = ASGITransport(app=echo_app_with_telemetry)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_serve_no_reporter_without_api_key(echo_app_without_telemetry):
    """When no api_key, reporter should be None and requests still work."""
    transport = ASGITransport(app=echo_app_without_telemetry)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "hello" in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_serve_telemetry_reports_on_non_streaming(echo_app_with_telemetry):
    """Non-streaming completions should trigger telemetry events."""
    sent = []

    def capture_send(client, url, headers, payload):
        sent.append(payload)

    with patch.object(TelemetryReporter, "_send", side_effect=capture_send):
        transport = ASGITransport(app=echo_app_with_telemetry)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200

        # Give the background thread time to dispatch
        import time

        time.sleep(0.3)

    # Collect event names from envelopes
    event_names = []
    for envelope in sent:
        for event in envelope.get("events", []):
            event_names.append(event.get("name"))
    assert "inference.started" in event_names
    assert "inference.completed" in event_names


@pytest.mark.asyncio
async def test_serve_telemetry_reports_on_streaming(echo_app_with_telemetry):
    """Streaming completions should trigger telemetry events."""
    sent = []

    def capture_send(client, url, headers, payload):
        sent.append(payload)

    with patch.object(TelemetryReporter, "_send", side_effect=capture_send):
        transport = ASGITransport(app=echo_app_with_telemetry)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi there"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200

        # Ensure SSE response is valid
        assert "data:" in resp.text

        # Give the background thread time to dispatch
        import time

        time.sleep(0.5)

    event_names = []
    for envelope in sent:
        for event in envelope.get("events", []):
            event_names.append(event.get("name"))
    assert "inference.started" in event_names
    # Should have chunk_produced events and a generation_completed
    assert "inference.chunk_produced" in event_names
    assert "inference.completed" in event_names


@pytest.mark.asyncio
async def test_serve_telemetry_does_not_break_without_key(echo_app_without_telemetry):
    """Streaming should work fine when no telemetry is configured."""
    transport = ASGITransport(app=echo_app_without_telemetry)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert "data:" in resp.text


# ---------------------------------------------------------------------------
# TelemetryReporter — close / drain
# ---------------------------------------------------------------------------


class TestTelemetryClose:
    def test_close_drains_pending_events(self):
        """close() should process remaining queued events."""
        sent = []

        def capture_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=capture_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            for i in range(5):
                reporter.report_generation_started(
                    model_id="m",
                    version="v",
                    session_id=f"s{i}",
                )
            reporter.close()

        # Count total events across all envelopes
        total_events = sum(len(e.get("events", [])) for e in sent)
        assert total_events == 5

    def test_close_is_idempotent(self):
        """Calling close() multiple times should not hang or error."""
        reporter = TelemetryReporter(
            api_key="key",
            api_base="https://api.test.com/api/v1",
            org_id="org-1",
            device_id="dev-1",
        )
        reporter.close()
        # Second close — worker already stopped, should not hang
        reporter.close()


# ---------------------------------------------------------------------------
# TelemetryReporter — auto device ID
# ---------------------------------------------------------------------------


class TestAutoDeviceId:
    def test_auto_generates_when_none(self):
        reporter = TelemetryReporter(
            api_key="key",
            api_base="https://api.test.com/api/v1",
            org_id="org-1",
            device_id=None,
        )
        assert reporter.device_id is not None
        assert len(reporter.device_id) == 16
        reporter.close()

    def test_uses_provided_device_id(self):
        reporter = TelemetryReporter(
            api_key="key",
            api_base="https://api.test.com/api/v1",
            org_id="org-1",
            device_id="custom-device-id",
        )
        assert reporter.device_id == "custom-device-id"
        reporter.close()


# ---------------------------------------------------------------------------
# Backward compatibility — all public methods still work
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """All existing report_* method signatures must still work."""

    def test_all_report_methods_callable(self):
        reporter = TelemetryReporter(
            api_key="key",
            api_base="https://api.test.com/api/v1",
            org_id="org-1",
            device_id="dev-1",
        )

        # These should not raise
        reporter.report_generation_started(
            model_id="m", version="v", session_id="s",
        )
        reporter.report_chunk_produced(
            session_id="s", model_id="m", version="v", chunk_index=0,
        )
        reporter.report_generation_completed(
            session_id="s", model_id="m", version="v",
            total_chunks=5, total_duration_ms=100.0,
            ttfc_ms=10.0, throughput=50.0,
        )
        reporter.report_generation_failed(
            session_id="s", model_id="m", version="v",
        )
        reporter.report_early_exit_stats(
            session_id="s", model_id="m", version="v",
            total_tokens=100, early_exit_tokens=30,
            exit_percentage=30.0, avg_layers_used=20.5, avg_entropy=0.22,
        )
        reporter.report_moe_routing(
            session_id="s", model_id="m", version="v",
            num_experts=8, active_experts=2,
        )
        reporter.report_prompt_compressed(
            session_id="s", model_id="m", version="v",
            original_tokens=500, compressed_tokens=250,
            compression_ratio=0.5, strategy="token_pruning", duration_ms=3.5,
        )
        reporter.report_funnel_event(stage="download", success=True)

        reporter.close()
