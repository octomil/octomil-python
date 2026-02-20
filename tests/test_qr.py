"""Tests for edgeml.qr — QR code terminal rendering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from edgeml.cli import main
from edgeml.qr import print_qr_code, render_qr_terminal


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
        result = render_qr_terminal("https://app.edgeml.io/deploy/phone?code=ABC123")
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
    @patch("edgeml.cli.webbrowser.open")
    def test_deploy_phone_shows_qr_box(self, mock_open, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")

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
        # Manual code should be shown
        assert "XYZ789" in result.output
        assert "Scan this QR code" in result.output
        assert "Expires in 5 minutes" in result.output

    @patch("edgeml.cli.webbrowser.open")
    def test_deploy_phone_shows_completion(self, mock_open, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "ABC123",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        # Simulate connected -> done sequence
        poll_responses = [
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
            runner = CliRunner(mix_stderr=False)
            result = runner.invoke(main, ["deploy", "gemma-1b", "--phone"])

        assert result.exit_code == 0
        assert "iPhone 15 Pro" in result.output
        assert "Deployment complete" in result.output

    @patch("edgeml.cli.webbrowser.open")
    def test_deploy_phone_opens_pair_url(self, mock_open, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")

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
        assert "edgeml.io/pair" in url
        assert "code=QR1234" in url

    @patch("edgeml.cli.webbrowser.open")
    def test_deploy_phone_session_expired(self, mock_open, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")

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

    @patch("edgeml.cli.webbrowser.open")
    def test_deploy_phone_qr_fallback(self, mock_open, monkeypatch):
        """When qrcode lib is missing, should still show the box with URL text."""
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")

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
        # Should still show the pairing code even without QR
        assert "FB0001" in result.output
        assert "Scan this QR code" in result.output
