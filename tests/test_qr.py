"""Tests for octomil.qr — QR code rendering (segno) and deep link generation."""

from __future__ import annotations

import os
import tempfile
import urllib.parse
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from octomil.cli import main
from octomil.qr import (
    build_custom_scheme_link,
    build_deep_link,
    print_qr_code,
    render_qr_terminal,
    save_qr_svg,
)


def _mock_deploy_http(*responses):
    """Create a mock httpx.Client whose request() returns responses in order."""
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
    def test_renders_nonempty_string(self):
        result = render_qr_terminal("https://example.com")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_block_characters(self):
        result = render_qr_terminal("https://example.com")
        # compact=True uses Unicode block characters
        block_chars = {"\u2580", "\u2584", "\u2588", " "}
        found = any(c in result for c in block_chars)
        assert found, "QR output should contain unicode block characters"

    def test_multiline_output(self):
        result = render_qr_terminal("https://example.com")
        lines = result.strip().split("\n")
        assert len(lines) > 1, "QR code should span multiple lines"

    def test_border_parameter(self):
        small = render_qr_terminal("https://example.com", border=0)
        large = render_qr_terminal("https://example.com", border=4)
        assert len(large.split("\n")) >= len(small.split("\n"))

    def test_fallback_without_segno(self):
        with patch.dict("sys.modules", {"segno": None}):
            result = render_qr_terminal("https://example.com")
            assert "https://example.com" in result

    def test_fallback_message_contains_install_hint(self):
        with patch.dict("sys.modules", {"segno": None}):
            result = render_qr_terminal("https://example.com")
            assert "pip install segno" in result


# ---------------------------------------------------------------------------
# save_qr_svg
# ---------------------------------------------------------------------------


class TestSaveQrSvg:
    def test_creates_svg_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.svg")
            save_qr_svg("https://example.com", path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "<svg" in content

    def test_svg_is_valid_xml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.svg")
            save_qr_svg("https://octomil.com/pair/ABC123", path)
            with open(path) as f:
                content = f.read()
            assert content.startswith("<?xml")


# ---------------------------------------------------------------------------
# build_deep_link
# ---------------------------------------------------------------------------


class TestBuildDeepLink:
    def test_default_host_uses_path_only(self):
        url = build_deep_link(token="ABC123", host="https://api.octomil.com/api/v1")
        assert url == "https://octomil.com/pair/ABC123"

    def test_custom_host_appended_as_query(self):
        url = build_deep_link(token="T", host="http://localhost:8000/api/v1")
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "octomil.com"
        assert parsed.path == "/pair/T"
        params = urllib.parse.parse_qs(parsed.query)
        assert params["host"] == ["http://localhost:8000/api/v1"]

    def test_host_is_url_encoded(self):
        url = build_deep_link(token="T", host="https://custom.example.com/api/v1")
        assert "host=https%3A%2F%2Fcustom.example.com%2Fapi%2Fv1" in url

    def test_empty_token(self):
        url = build_deep_link(token="", host="https://api.octomil.com/api/v1")
        assert url == "https://octomil.com/pair/"

    def test_path_based_format(self):
        url = build_deep_link(token="ep_7kx", host="https://api.octomil.com/api/v1")
        parsed = urllib.parse.urlparse(url)
        assert parsed.path == "/pair/ep_7kx"
        assert parsed.query == ""  # no query for default host


# ---------------------------------------------------------------------------
# build_custom_scheme_link
# ---------------------------------------------------------------------------


class TestBuildCustomSchemeLink:
    def test_path_based_format(self):
        url = build_custom_scheme_link(token="ABC123", host="https://api.octomil.com/api/v1")
        assert url == "octomil://pair/ABC123"

    def test_custom_scheme(self):
        url = build_custom_scheme_link(token="T", host="https://example.com")
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "octomil"

    def test_token_in_path(self):
        url = build_custom_scheme_link(token="XYZ", host="https://example.com")
        assert "/pair/XYZ" in url


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
    def test_deploy_phone_shows_qr_and_manual_code(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check_resp = MagicMock(status_code=200, json=MagicMock(return_value={"models": [{"name": "gemma-1b"}]}))
        mock_post_resp = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "code": "XYZ789",
                    "expires_at": "2026-02-18T12:00:00Z",
                }
            ),
        )
        mock_poll_resp = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "status": "done",
                    "device_name": "iPhone 15",
                }
            ),
        )

        p1, p2, p3 = _mock_deploy_http(mock_check_resp, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "gemma-1b", "--phone"])

        assert result.exit_code == 0
        assert "XYZ789" in result.output
        assert "Scan with your phone camera" in result.output
        assert "enter code manually" in result.output
        assert "Expires in 5 minutes" in result.output

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_shows_completion(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check_resp = MagicMock(status_code=200, json=MagicMock(return_value={"models": [{"name": "gemma-1b"}]}))
        mock_post_resp = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "code": "ABC123",
                    "expires_at": "2026-02-18T12:00:00Z",
                }
            ),
        )
        mock_connected = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "status": "connected",
                    "device_name": "iPhone 15 Pro",
                    "device_platform": "iOS 18.2",
                }
            ),
        )
        mock_done = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "status": "done",
                    "device_name": "iPhone 15 Pro",
                }
            ),
        )

        p1, p2, p3 = _mock_deploy_http(mock_check_resp, mock_post_resp, mock_connected, mock_done)
        with p1, p2, p3:
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "gemma-1b", "--phone"])

        assert result.exit_code == 0
        assert "iPhone 15 Pro" in result.output
        assert "Deployment complete" in result.output

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_opens_universal_link(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"models": [{"name": "test-model"}]}))
        mock_post_resp = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "code": "QR1234",
                    "expires_at": "2026-02-18T12:00:00Z",
                }
            ),
        )
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        mock_open.assert_called_once()
        url = mock_open.call_args[0][0]
        # Default host → path-only format
        assert url == "https://octomil.com/pair/QR1234"

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_session_expired(self, mock_open, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"models": [{"name": "test-model"}]}))
        mock_post_resp = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "code": "EXP001",
                    "expires_at": "2026-02-18T12:00:00Z",
                }
            ),
        )
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "expired"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code != 0

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_path_based_universal_link(self, mock_open, monkeypatch):
        """The QR code payload uses path-based format: https://octomil.com/pair/CODE."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"models": [{"name": "test-model"}]}))
        mock_post_resp = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "code": "DL_TEST",
                    "expires_at": "2026-02-18T12:00:00Z",
                }
            ),
        )
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        url = mock_open.call_args[0][0]
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "octomil.com"
        assert parsed.path == "/pair/DL_TEST"
        # Default host → no query params
        assert parsed.query == ""

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_custom_api_base(self, mock_open, monkeypatch):
        """A custom API base URL is appended as ?host= query param."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        monkeypatch.setenv("OCTOMIL_API_BASE", "http://localhost:8000/api/v1")

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"models": [{"name": "test-model"}]}))
        mock_post_resp = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "code": "CUST01",
                    "expires_at": "2026-02-18T12:00:00Z",
                }
            ),
        )
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        url = mock_open.call_args[0][0]
        parsed = urllib.parse.urlparse(url)
        assert parsed.path == "/pair/CUST01"
        params = urllib.parse.parse_qs(parsed.query)
        assert params["host"] == ["http://localhost:8000/api/v1"]

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_svg_saved(self, mock_open, monkeypatch):
        """An SVG QR image should be saved to a temp file."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"models": [{"name": "test-model"}]}))
        mock_post_resp = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "code": "SVG01",
                    "expires_at": "2026-02-18T12:00:00Z",
                }
            ),
        )
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with p1, p2, p3:
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        assert "QR image saved" in result.output
        # SVG file should exist
        import tempfile as tf

        svg_path = os.path.join(tf.gettempdir(), "octomil-pair-SVG01.svg")
        assert os.path.exists(svg_path)

    @patch("octomil.commands.deploy.webbrowser.open")
    def test_deploy_phone_qr_fallback(self, mock_open, monkeypatch):
        """When segno is missing, should still show the manual code and URL."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_check = MagicMock(status_code=200, json=MagicMock(return_value={"models": [{"name": "test-model"}]}))
        mock_post_resp = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "code": "FB0001",
                    "expires_at": "2026-02-18T12:00:00Z",
                }
            ),
        )
        mock_poll_resp = MagicMock(status_code=200, json=MagicMock(return_value={"status": "done"}))

        p1, p2, p3 = _mock_deploy_http(mock_check, mock_post_resp, mock_poll_resp)
        with (
            patch.dict("sys.modules", {"segno": None}),
            p1,
            p2,
            p3,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        assert "FB0001" in result.output
        assert "enter code manually" in result.output
        assert "Scan with your phone camera" in result.output
