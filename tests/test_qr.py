"""Tests for octomil.qr — QR code terminal rendering."""

from __future__ import annotations


from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from octomil.cli import main
from octomil.qr import build_deep_link, print_qr_code, render_qr_terminal


def _mock_deploy_http(*responses):
    """Create a mock httpx.Client whose request() returns responses in order.

    The deploy --phone command calls http_request() which uses
    httpx.Client().request(), not httpx.post/get directly.  Also mocks
    scan_for_devices and time.sleep to avoid real I/O.
    """
    mock_client = MagicMock()
    mock_client.request.side_effect = list(responses)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return (
        patch("httpx.Client", return_value=mock_client),
        patch("octomil.discovery.scan_for_devices", return_value=[]),
        patch("time.sleep"),
    )


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
    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_shows_qr_box(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check_resp = MagicMock(status_code=200, json=MagicMock(return_value={"name": "gemma-1b"}))
        mock_post_resp = MagicMock(status_code=200, json=MagicMock(return_value={
            "code": "XYZ789",
            "expires_at": "2026-02-18T12:00:00Z",
        }))
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={
            "status": "done",
            "device_name": "iPhone 15",
        }))

        p1, p2, p3 = _mock_deploy_http(mock_check_resp, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
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

        mock_check_resp = MagicMock(status_code=200, json=MagicMock(return_value={"name": "gemma-1b"}))
        mock_post_resp = MagicMock(status_code=200, json=MagicMock(return_value={
            "code": "ABC123",
            "expires_at": "2026-02-18T12:00:00Z",
        }))
        mock_connected = MagicMock(status_code=200, json=MagicMock(return_value={
            "status": "connected",
            "device_name": "iPhone 15 Pro",
            "device_platform": "iOS 18.2",
        }))
        mock_done = MagicMock(status_code=200, json=MagicMock(return_value={
            "status": "done",
            "device_name": "iPhone 15 Pro",
        }))

        p1, p2, p3 = _mock_deploy_http(mock_check_resp, mock_post_resp, mock_connected, mock_done)
        with p1, p2, p3:
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "gemma-1b", "--phone"])

        assert result.exit_code == 0
        assert "iPhone 15 Pro" in result.output
        assert "Deployment complete" in result.output

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_opens_deep_link_url(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"name": "test-model"}))
        mock_post_resp = MagicMock(status_code=200, json=MagicMock(return_value={
            "code": "QR1234",
            "expires_at": "2026-02-18T12:00:00Z",
        }))
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
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

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"name": "test-model"}))
        mock_post_resp = MagicMock(status_code=200, json=MagicMock(return_value={
            "code": "EXP001",
            "expires_at": "2026-02-18T12:00:00Z",
        }))
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "expired"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code != 0

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_qr_payload_is_deep_link(self, mock_open, monkeypatch):
        """The QR code payload and webbrowser.open URL must be a valid octomil:// deep link."""
        import urllib.parse

        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"name": "test-model"}))
        mock_post_resp = MagicMock(status_code=200, json=MagicMock(return_value={
            "code": "DL_TEST",
            "expires_at": "2026-02-18T12:00:00Z",
        }))
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
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

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"name": "test-model"}))
        mock_post_resp = MagicMock(status_code=200, json=MagicMock(return_value={
            "code": "DISP01",
            "expires_at": "2026-02-18T12:00:00Z",
        }))
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
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

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"name": "test-model"}))
        mock_post_resp = MagicMock(status_code=200, json=MagicMock(return_value={
            "code": "CUST01",
            "expires_at": "2026-02-18T12:00:00Z",
        }))
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
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

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"name": "test-model"}))
        mock_post_resp = MagicMock(status_code=200, json=MagicMock(return_value={
            "code": "FB0001",
            "expires_at": "2026-02-18T12:00:00Z",
        }))
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with (
            patch.dict("sys.modules", {"qrcode": None}),
            p1, p2, p3,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        # Should still show the deep link URL even without QR
        assert "FB0001" in result.output
        assert "octomil://pair?" in result.output
        assert "Scan this QR code with your phone camera:" in result.output
