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


def _model_check_resp(name="test-model"):
    """Model list response with id (so version check succeeds)."""
    return MagicMock(
        status_code=200,
        json=MagicMock(return_value={"models": [{"name": name, "id": "test-model-id"}]}),
    )


def _versions_resp():
    """Versions list response (model has at least one version)."""
    return MagicMock(
        status_code=200,
        json=MagicMock(return_value={"versions": [{"version": "1.0.0"}]}),
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
    def test_default_host_uses_custom_scheme(self):
        url = build_deep_link(token="ABC123", host="https://api.octomil.com/api/v1")
        assert url == "octomil://pair?token=ABC123"

    def test_custom_host_appended_as_query(self):
        url = build_deep_link(token="T", host="http://localhost:8000/api/v1")
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "octomil"
        assert parsed.netloc == "pair"
        params = urllib.parse.parse_qs(parsed.query)
        assert params["token"] == ["T"]
        assert params["host"] == ["http://localhost:8000"]

    def test_host_is_url_encoded(self):
        url = build_deep_link(token="T", host="https://custom.example.com/api/v1")
        assert "host=https%3A%2F%2Fcustom.example.com" in url

    def test_empty_token(self):
        url = build_deep_link(token="", host="https://api.octomil.com/api/v1")
        assert url == "octomil://pair?token="

    def test_query_param_format(self):
        url = build_deep_link(token="ep_7kx", host="https://api.octomil.com/api/v1")
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "octomil"
        assert parsed.netloc == "pair"
        params = urllib.parse.parse_qs(parsed.query)
        assert params["token"] == ["ep_7kx"]
        assert "host" not in params  # no host for default


# ---------------------------------------------------------------------------
# build_custom_scheme_link
# ---------------------------------------------------------------------------


class TestBuildCustomSchemeLink:
    def test_query_param_format(self):
        url = build_custom_scheme_link(token="ABC123", host="https://api.octomil.com/api/v1")
        assert url == "octomil://pair?token=ABC123"

    def test_custom_scheme(self):
        url = build_custom_scheme_link(token="T", host="https://example.com")
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "octomil"

    def test_token_in_query(self):
        url = build_custom_scheme_link(token="XYZ", host="https://example.com")
        assert "token=XYZ" in url


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
    def test_deploy_phone_shows_qr_and_manual_code(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

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

        p1, p2, p3 = _mock_deploy_http(mock_post_resp, mock_poll_resp)
        with p1, p2, p3, patch("octomil.models.catalog.CATALOG", {"gemma-1b": True}):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "gemma-1b", "--phone"])

        assert result.exit_code == 0
        assert "XYZ789" in result.output
        assert "Scan with your phone camera" in result.output
        assert "enter code manually" in result.output
        assert "Expires in 5 minutes" in result.output

    def test_deploy_phone_shows_completion(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

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
        # Response for POST /deploy/pair/{code}/deploy trigger
        mock_deploy_trigger = MagicMock(status_code=200, json=MagicMock(return_value={"status": "deploying"}))
        mock_done = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "status": "done",
                    "device_name": "iPhone 15 Pro",
                }
            ),
        )

        p1, p2, p3 = _mock_deploy_http(
            mock_post_resp,
            mock_connected,
            mock_deploy_trigger,
            mock_done,
        )
        with p1, p2, p3, patch("octomil.models.catalog.CATALOG", {"gemma-1b": True}):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "gemma-1b", "--phone"])

        assert result.exit_code == 0
        assert "iPhone 15 Pro" in result.output
        assert "Deployment complete" in result.output

    def test_deploy_phone_session_expired(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

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

        p1, p2, p3 = _mock_deploy_http(mock_post_resp, mock_poll_resp)
        with p1, p2, p3, patch("octomil.models.catalog.CATALOG", {"test-model": True}):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code != 0

    @patch("octomil.qr.build_deep_link", wraps=build_deep_link)
    def test_deploy_phone_deep_link_format(self, mock_build, monkeypatch):
        """The deep link uses octomil:// custom scheme with token query param."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

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

        p1, p2, p3 = _mock_deploy_http(mock_post_resp, mock_poll_resp)
        with p1, p2, p3, patch("octomil.models.catalog.CATALOG", {"test-model": True}):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        # Verify deep link URL format (unit tests cover build_deep_link in detail)
        url = build_deep_link(token="DL_TEST", host="https://api.octomil.com/api/v1")
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "octomil"
        assert parsed.netloc == "pair"
        params = urllib.parse.parse_qs(parsed.query)
        assert params["token"] == ["DL_TEST"]
        assert "host" not in params

    def test_deploy_phone_custom_api_base(self, monkeypatch):
        """A custom API base URL is appended as ?host= query param."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        monkeypatch.setenv("OCTOMIL_API_BASE", "http://localhost:8000/api/v1")

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

        p1, p2, p3 = _mock_deploy_http(mock_post_resp, mock_poll_resp)
        with p1, p2, p3, patch("octomil.models.catalog.CATALOG", {"test-model": True}):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        # Verify deep link URL includes custom host (without /api/v1, since SDKs append it)
        url = build_deep_link(token="CUST01", host="http://localhost:8000/api/v1")
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "octomil"
        assert parsed.netloc == "pair"
        params = urllib.parse.parse_qs(parsed.query)
        assert params["host"] == ["http://localhost:8000"]

    def test_deploy_phone_svg_saved(self, monkeypatch):
        """An SVG QR image should be saved to a temp file."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

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

        p1, p2, p3 = _mock_deploy_http(mock_post_resp, mock_poll_resp)
        with p1, p2, p3, patch("octomil.models.catalog.CATALOG", {"test-model": True}):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "test-model", "--phone"])

        assert result.exit_code == 0
        assert "QR image saved" in result.output
        # SVG file should exist
        import tempfile as tf

        svg_path = os.path.join(tf.gettempdir(), "octomil-pair-SVG01.svg")
        assert os.path.exists(svg_path)

    def test_deploy_phone_qr_fallback(self, monkeypatch):
        """When segno is missing, should still show the manual code and URL."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

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

        p1, p2, p3 = _mock_deploy_http(mock_post_resp, mock_poll_resp)
        with (
            patch.dict("sys.modules", {"segno": None}),
            patch("octomil.models.catalog.CATALOG", {"test-model": True}),
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
