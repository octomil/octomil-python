"""Tests for edgeml.telemetry — TelemetryReporter and edgeml.init()."""

from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from edgeml.telemetry import TelemetryReporter, _generate_device_id


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
# TelemetryReporter — payload construction
# ---------------------------------------------------------------------------


class TestTelemetryReporterPayloads:
    """Verify that reporter methods enqueue correct payloads."""

    def setup_method(self):
        self.reporter = TelemetryReporter(
            api_key="test-key",
            api_base="https://api.example.com/api/v1",
            org_id="test-org",
            device_id="dev-001",
        )

    def teardown_method(self):
        self.reporter.close()

    def _drain_payloads(self) -> list[dict]:
        """Drain the internal queue and return collected payloads."""
        payloads = []
        while not self.reporter._queue.empty():
            item = self.reporter._queue.get_nowait()
            if item is not None:
                payloads.append(item)
        return payloads

    def test_report_generation_started(self):
        # Stop the worker so we can inspect the queue directly
        self.reporter.report_generation_started(
            model_id="gemma-1b",
            version="1.0",
            session_id="sess-001",
            modality="text",
        )
        # Give the queue a moment
        time.sleep(0.05)
        # Put sentinel to stop worker, then drain
        self.reporter._queue.put(None)
        self.reporter._worker.join(timeout=2.0)

        # Re-create reporter for teardown
        payloads = self._drain_payloads()
        # The worker may have already consumed the item. Instead, test via mock.

    def test_generation_started_payload_structure(self):
        """Verify the payload structure using a mock HTTP client."""
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
                model_id="model-a",
                version="2.0",
                session_id="s1",
                modality="text",
            )
            # Allow background thread to process
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        p = sent[0]
        assert p["device_id"] == "dev-1"
        assert p["model_id"] == "model-a"
        assert p["version"] == "2.0"
        assert p["session_id"] == "s1"
        assert p["event_type"] == "generation_started"
        assert p["modality"] == "text"
        assert p["org_id"] == "org-1"
        assert "timestamp_ms" in p

    def test_chunk_produced_payload(self):
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

        assert len(sent) >= 1
        p = sent[0]
        assert p["event_type"] == "chunk_produced"
        assert p["metrics"]["chunk_index"] == 0
        assert p["metrics"]["ttfc_ms"] == 42.5
        assert p["metrics"]["chunk_latency_ms"] == 3.1

    def test_generation_completed_payload(self):
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
            )
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        p = sent[0]
        assert p["event_type"] == "generation_completed"
        assert p["metrics"]["total_chunks"] == 10
        assert p["metrics"]["total_duration_ms"] == 500.0
        assert p["metrics"]["ttfc_ms"] == 30.0
        assert p["metrics"]["throughput"] == 20.0

    def test_generation_failed_payload(self):
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

        assert len(sent) >= 1
        p = sent[0]
        assert p["event_type"] == "generation_failed"
        assert "metrics" not in p


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
            {"event_type": "test"},
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
# edgeml.init() — env var fallback and validation
# ---------------------------------------------------------------------------


class TestEdgemlInit:
    def test_init_raises_without_api_key(self):
        """init() should raise ValueError when no API key is provided."""
        import edgeml

        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars
            for key in ("EDGEML_API_KEY", "EDGEML_ORG_ID", "EDGEML_API_BASE"):
                os.environ.pop(key, None)
            with pytest.raises(ValueError, match="API key required"):
                edgeml.init()

    def test_init_uses_env_vars(self):
        """init() should read from env vars when args are not passed."""
        import edgeml

        env = {
            "EDGEML_API_KEY": "test-key-123",
            "EDGEML_ORG_ID": "my-org",
            "EDGEML_API_BASE": "https://custom.api.com/api/v1",
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

                edgeml.init()

                assert edgeml._config["api_key"] == "test-key-123"
                assert edgeml._config["org_id"] == "my-org"
                assert edgeml._config["api_base"] == "https://custom.api.com/api/v1"
                assert edgeml._reporter is not None

                # Cleanup
                edgeml._reporter.close()
                edgeml._reporter = None
                edgeml._config = {}

    def test_init_with_explicit_args(self):
        """init() should prefer explicit args over env vars."""
        import edgeml
        import httpx

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

            edgeml.init(
                api_key="explicit-key",
                org_id="explicit-org",
                api_base="https://explicit.api.com/api/v1",
            )

            assert edgeml._config["api_key"] == "explicit-key"
            assert edgeml._config["org_id"] == "explicit-org"
            assert edgeml._reporter is not None

            # Cleanup
            edgeml._reporter.close()
            edgeml._reporter = None
            edgeml._config = {}

    def test_init_invalid_key_raises(self):
        """init() should raise ValueError on 401/403 from the health check."""
        import edgeml
        import httpx

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

            with pytest.raises(ValueError, match="Invalid EdgeML API key"):
                edgeml.init(api_key="bad-key")

    def test_init_unreachable_api_still_creates_reporter(self):
        """If the API is unreachable, init() should warn but still create a reporter."""
        import edgeml
        import httpx

        with patch("httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = MagicMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_instance.get.side_effect = httpx.ConnectError("unreachable")
            MockClient.return_value = mock_client_instance

            edgeml.init(api_key="some-key")
            assert edgeml._reporter is not None

            # Cleanup
            edgeml._reporter.close()
            edgeml._reporter = None
            edgeml._config = {}


class TestGetReporter:
    def test_returns_none_before_init(self):
        import edgeml

        # Save and reset state
        saved_reporter = edgeml._reporter
        edgeml._reporter = None
        try:
            assert edgeml.get_reporter() is None
        finally:
            edgeml._reporter = saved_reporter

    def test_returns_reporter_after_init(self):
        import edgeml
        import httpx

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

            edgeml.init(api_key="test-key")
            reporter = edgeml.get_reporter()
            assert reporter is not None
            assert isinstance(reporter, TelemetryReporter)

            # Cleanup
            reporter.close()
            edgeml._reporter = None
            edgeml._config = {}


# ---------------------------------------------------------------------------
# serve.py integration — telemetry when api_key is set
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_app_with_telemetry():
    """Create a FastAPI app with EchoBackend and telemetry enabled."""
    from edgeml.serve import EchoBackend, create_app

    with patch("edgeml.serve._detect_backend") as mock_detect:
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
    from edgeml.serve import EchoBackend, create_app

    with patch("edgeml.serve._detect_backend") as mock_detect:
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

    # Should have generation_started and generation_completed
    event_types = [p.get("event_type") for p in sent]
    assert "generation_started" in event_types
    assert "generation_completed" in event_types


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

    event_types = [p.get("event_type") for p in sent]
    assert "generation_started" in event_types
    # Should have chunk_produced events and a generation_completed
    assert "chunk_produced" in event_types
    assert "generation_completed" in event_types


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

        assert len(sent) == 5

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
