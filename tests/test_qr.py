"""Tests for octomil.qr — QR code terminal rendering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from octomil.cli import main
from octomil.qr import build_deep_link, print_qr_code, render_qr_terminal


# ---------------------------------------------------------------------------
# render_qr_terminal
# ---------------------------------------------------------------------------


class TestRenderQrTerminal:
    def test_renders_qr_string(self):
        result = render_qr_terminal("https://example.com")
        assert isinstance(result, str)
        assert len(result) > 0
        # QR uses block chars
        assert "\u2588" in result or "\u2580" in result or "\u2584" in result

    def test_contains_block_characters(self):
        result = render_qr_terminal("https://app.octomil.com/deploy/phone?code=ABC123")
        # Should contain at least some full block or half-block chars
        block_chars = {"\u2588", "\u2580", "\u2584"}
        found = any(c in result for c in block_chars)
        assert found, "QR output should contain unicode block characters"

    def test_multiline_output(self):
        result = render_qr_terminal("https://example.com")
        lines = result.strip().split("\n")
        assert len(lines) > 1, "QR code should span multiple lines"

    def test_border_parameter(self):
        small = render_qr_terminal("https://example.com", border=0)
        large = render_qr_terminal("https://example.com", border=4)
        # Larger border should produce more lines
        assert len(large.split("\n")) >= len(small.split("\n"))

    def test_fallback_without_qrcode_lib(self):
        with patch.dict("sys.modules", {"qrcode": None}):
            result = render_qr_terminal("https://example.com")
            assert "https://example.com" in result

    def test_fallback_message_contains_install_hint(self):
        with patch.dict("sys.modules", {"qrcode": None}):
            result = render_qr_terminal("https://example.com")
            assert "pip install qrcode" in result


# ---------------------------------------------------------------------------
# build_deep_link
# ---------------------------------------------------------------------------


class TestBuildDeepLink:
    def test_basic_deep_link(self):
        url = build_deep_link(token="ABC123", host="https://api.octomil.com/api/v1")
        assert url.startswith("octomil://pair?")
        assert "token=ABC123" in url
        assert "host=https%3A%2F%2Fapi.octomil.com%2Fapi%2Fv1" in url

    def test_deep_link_scheme(self):
        url = build_deep_link(token="T", host="https://example.com")
        assert url.startswith("octomil://pair?")

    def test_token_is_url_encoded(self):
        url = build_deep_link(token="a+b&c=d", host="https://example.com")
        # + & = should all be percent-encoded
        assert "a%2Bb%26c%3Dd" in url
        assert "token=a%2Bb%26c%3Dd" in url

    def test_host_is_url_encoded(self):
        url = build_deep_link(token="T", host="https://api.octomil.com/api/v1")
        assert "host=https%3A%2F%2Fapi.octomil.com%2Fapi%2Fv1" in url

    def test_both_params_present(self):
        url = build_deep_link(token="ep_7kx", host="https://api.octomil.com/api/v1")
        # Parse the URL to verify both params
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "octomil"
        assert parsed.netloc == "pair"
        params = urllib.parse.parse_qs(parsed.query)
        assert params["token"] == ["ep_7kx"]
        assert params["host"] == ["https://api.octomil.com/api/v1"]

    def test_custom_host(self):
        url = build_deep_link(token="T", host="http://localhost:8000/api/v1")
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        assert params["host"] == ["http://localhost:8000/api/v1"]

    def test_empty_token(self):
        url = build_deep_link(token="", host="https://api.octomil.com/api/v1")
        assert "token=&" in url or url.endswith("token=")


# ---------------------------------------------------------------------------
# print_qr_code
# ---------------------------------------------------------------------------


class TestPrintQrCode:
    def test_print_qr_code_outputs_to_stdout(self, capsys):
        print_qr_code("https://example.com", label="Scan me")
        captured = capsys.readouterr()
        assert "Scan me" in captured.out

    def test_print_qr_code_without_label(self, capsys):
        print_qr_code("https://example.com")
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        # No label text appended
        assert "Scan me" not in captured.out

    def test_print_qr_code_returns_qr_string(self):
        result = print_qr_code("https://example.com")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# deploy --phone QR integration
# ---------------------------------------------------------------------------


class TestDeployPhoneQr:
    @pytest.fixture(autouse=True)
    def _mock_discovery(self):
        """Mock discovery functions so deploy --phone tests skip mDNS scanning."""
        with (
            patch("octomil.discovery.scan_for_devices", return_value=[]),
            patch("octomil.discovery.detect_platform_on_network", return_value=None),
            patch("octomil.discovery.wait_for_device", return_value=None),
            patch("octomil.ollama.get_ollama_model", return_value=None),
        ):
            yield

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_shows_qr_box(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "XYZ789",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {
            "status": "done",
            "device_name": "iPhone 15",
        }

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "gemma-1b", "--phone"])

        assert result.exit_code == 0
        # Box borders should be present
        assert "\u256d" in result.output  # top-left corner
        assert "\u256f" in result.output  # bottom-right corner
        # Deep link URL should be shown with the token
        assert "XYZ789" in result.output
        assert "octomil://pair?" in result.output
        assert "Scan this QR code with your phone camera:" in result.output
        assert "Or open manually:" in result.output
        assert "Expires in 5 minutes" in result.output

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_shows_completion(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "ABC123",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        # Mock responses for httpx.get calls:
        # 1. Model registry check (GET /models/{name})
        # 2. Poll: connected
        # 3. Poll: done
        mock_model_check = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"id": "gemma-1b", "name": "gemma-1b"}),
        )
        poll_responses = [
            mock_model_check,
            MagicMock(
                status_code=200,
                json=MagicMock(
                    return_value={
                        "status": "connected",
                        "device_name": "iPhone 15 Pro",
                        "device_platform": "iOS 18.2",
                    }
                ),
            ),
            MagicMock(
                status_code=200,
                json=MagicMock(
                    return_value={
                        "status": "done",
                        "device_name": "iPhone 15 Pro",
                    }
                ),
            ),
        ]

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", side_effect=poll_responses),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "gemma-1b", "--phone"])

        assert result.exit_code == 0
        assert "iPhone 15 Pro" in result.output
        assert "Deployment complete" in result.output

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_opens_deep_link_url(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "QR1234",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        mock_open.assert_called_once()
        url = mock_open.call_args[0][0]
        assert url.startswith("octomil://pair?")
        assert "token=QR1234" in url
        assert "host=" in url

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_session_expired(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "EXP001",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "expired"}

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code != 0

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_qr_payload_is_deep_link(self, mock_open, monkeypatch):
        """The QR code payload and webbrowser.open URL must be a valid octomil:// deep link."""
        import urllib.parse

        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "DL_TEST",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        url = mock_open.call_args[0][0]
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "octomil"
        assert parsed.netloc == "pair"
        params = urllib.parse.parse_qs(parsed.query)
        assert "token" in params
        assert params["token"] == ["DL_TEST"]
        assert "host" in params
        # host should be the default API base URL
        assert "api.octomil.com" in params["host"][0]

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_deep_link_displayed_in_output(self, mock_open, monkeypatch):
        """The deep link URL should be displayed as a fallback in the terminal output."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "DISP01",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        assert "Or open manually: octomil://pair?token=DISP01&host=" in result.output

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_custom_api_base_in_deep_link(self, mock_open, monkeypatch):
        """A custom API base URL should be encoded in the deep link host param."""
        import urllib.parse

        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        monkeypatch.setenv("OCTOMIL_API_BASE", "http://localhost:8000/api/v1")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "CUST01",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        url = mock_open.call_args[0][0]
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        assert params["host"] == ["http://localhost:8000/api/v1"]
        assert params["token"] == ["CUST01"]

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_qr_fallback(self, mock_open, monkeypatch):
        """When qrcode lib is missing, should still show the box with deep link URL."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "FB0001",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch.dict("sys.modules", {"qrcode": None}),
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        # Should still show the deep link URL even without QR
        assert "FB0001" in result.output
        assert "octomil://pair?" in result.output
        assert "Scan this QR code with your phone camera:" in result.output


# ---------------------------------------------------------------------------
# deploy --phone fast-path (direct push) and first-time path
# ---------------------------------------------------------------------------


class TestDeployPhoneFastPath:
    """Tests for the direct-push flow when a device is found via mDNS."""

    @patch("octomil.commands.deploy.webbrowser.open")
    @patch("octomil.ollama.get_ollama_model", return_value=None)
    def test_fast_path_pushes_code_to_device(self, mock_ollama, mock_open, monkeypatch):
        """When mDNS finds a device, CLI should POST the pairing code directly."""
        from octomil.discovery import DiscoveredDevice

        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        device = DiscoveredDevice(
            name="Sean's iPhone",
            platform="ios",
            ip="192.168.1.42",
            port=8080,
            device_id="abc-123",
        )

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "FAST01",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done", "device_name": "Sean's iPhone"}

        with (
            patch("octomil.discovery.scan_for_devices", return_value=[device]),
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main, ["deploy", "phi-4-mini", "--phone"], input="y\n"
            )

        assert result.exit_code == 0
        assert "Sean's iPhone" in result.output
        assert "Pairing code sent" in result.output
        # Should NOT show QR box in fast path
        mock_open.assert_not_called()

    @patch("octomil.commands.deploy.webbrowser.open")
    @patch("octomil.ollama.get_ollama_model", return_value=None)
    def test_fast_path_falls_back_on_connection_error(
        self, mock_ollama, mock_open, monkeypatch
    ):
        """If direct push fails, should fall back to QR code."""
        import httpx

        from octomil.discovery import DiscoveredDevice

        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        device = DiscoveredDevice(
            name="Sean's iPhone",
            platform="ios",
            ip="192.168.1.42",
            port=8080,
            device_id="abc-123",
        )

        mock_session_resp = MagicMock()
        mock_session_resp.status_code = 200
        mock_session_resp.json.return_value = {
            "code": "FALL01",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        # httpx.post should return session response first, then raise on device push
        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_session_resp  # pairing session create
            raise httpx.ConnectError("Connection refused")  # device push fails

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("octomil.discovery.scan_for_devices", return_value=[device]),
            patch("octomil.discovery.detect_platform_on_network", return_value=None),
            patch("octomil.discovery.wait_for_device", return_value=None),
            patch("httpx.post", side_effect=mock_post_side_effect),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main, ["deploy", "phi-4-mini", "--phone"], input="y\n"
            )

        assert result.exit_code == 0
        assert "Could not reach device" in result.output
        # Should fall back to QR
        assert "Scan this QR code" in result.output


class TestDeployPhonePlatformDetection:
    """Tests for platform detection when showing app store links."""

    @pytest.fixture(autouse=True)
    def _mock_base(self):
        with (
            patch("octomil.discovery.scan_for_devices", return_value=[]),
            patch("octomil.discovery.wait_for_device", return_value=None),
            patch("octomil.ollama.get_ollama_model", return_value=None),
        ):
            yield

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_shows_ios_store_link_when_apple_detected(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "IOS001",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("octomil.discovery.detect_platform_on_network", return_value="ios"),
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        assert "App Store" in result.output
        # Should NOT show Android link when iOS detected
        assert "play.google.com" not in result.output

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_shows_both_store_links_when_no_platform_detected(
        self, mock_open, monkeypatch
    ):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "BOTH01",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("octomil.discovery.detect_platform_on_network", return_value=None),
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        assert "apps.apple.com" in result.output
        assert "play.google.com" in result.output
