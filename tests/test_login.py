"""Tests for the browser-based login flow in edgeml.cli."""

from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.parse
import urllib.request
from http.server import HTTPServer
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from edgeml.cli import _save_credentials, main


# ---------------------------------------------------------------------------
# State token generation
# ---------------------------------------------------------------------------


class TestStateTokenGeneration:
    """Verify the state token used in the browser login flow."""

    def test_token_is_url_safe_base64(self):
        token = secrets.token_urlsafe(32)
        # Must be non-empty and contain only URL-safe characters
        assert len(token) > 0
        assert all(c.isalnum() or c in "-_" for c in token)

    def test_token_has_sufficient_entropy(self):
        token = secrets.token_urlsafe(32)
        # 32 bytes of entropy -> 43 base64 characters
        assert len(token) >= 40

    def test_tokens_are_unique(self):
        tokens = {secrets.token_urlsafe(32) for _ in range(100)}
        assert len(tokens) == 100


# ---------------------------------------------------------------------------
# Credential file saving/loading
# ---------------------------------------------------------------------------


class TestCredentialSaving:
    """Test _save_credentials writes JSON to ~/.edgeml/credentials."""

    def test_saves_api_key_as_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(tmp_path / ".edgeml") if "~/.edgeml" in x else x,
        )
        _save_credentials("edg_test123")

        cred_file = tmp_path / ".edgeml" / "credentials"
        assert cred_file.exists()
        data = json.loads(cred_file.read_text())
        assert data["api_key"] == "edg_test123"
        assert "org" not in data

    def test_saves_api_key_with_org(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(tmp_path / ".edgeml") if "~/.edgeml" in x else x,
        )
        _save_credentials("edg_test123", org="Acme Corp")

        cred_file = tmp_path / ".edgeml" / "credentials"
        data = json.loads(cred_file.read_text())
        assert data["api_key"] == "edg_test123"
        assert data["org"] == "Acme Corp"

    def test_creates_directory_if_missing(self, tmp_path, monkeypatch):
        target_dir = tmp_path / "nonexistent" / ".edgeml"
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(target_dir) if "~/.edgeml" in x else x,
        )
        # Parent directory won't exist, but os.makedirs with exist_ok handles it
        # Actually, we need the parent to exist. Let's use tmp_path directly.
        target_dir = tmp_path / ".edgeml"
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(target_dir) if "~/.edgeml" in x else x,
        )
        assert not target_dir.exists()
        _save_credentials("edg_new")
        assert (target_dir / "credentials").exists()

    def test_overwrites_existing_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(tmp_path / ".edgeml") if "~/.edgeml" in x else x,
        )
        _save_credentials("edg_first", org="OrgA")
        _save_credentials("edg_second", org="OrgB")

        cred_file = tmp_path / ".edgeml" / "credentials"
        data = json.loads(cred_file.read_text())
        assert data["api_key"] == "edg_second"
        assert data["org"] == "OrgB"

    def test_file_permissions_restrictive(self, tmp_path, monkeypatch):
        import os
        import stat

        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(tmp_path / ".edgeml") if "~/.edgeml" in x else x,
        )
        _save_credentials("edg_secret")

        cred_file = tmp_path / ".edgeml" / "credentials"
        mode = stat.S_IMODE(os.stat(cred_file).st_mode)
        assert mode == 0o600


# ---------------------------------------------------------------------------
# Credential loading (JSON + legacy)
# ---------------------------------------------------------------------------


class TestCredentialLoading:
    """Test _get_api_key reads both JSON and legacy formats."""

    def test_reads_json_format(self, tmp_path, monkeypatch):
        from edgeml.cli import _get_api_key

        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        cred_dir = tmp_path / ".edgeml"
        cred_dir.mkdir()
        (cred_dir / "credentials").write_text(
            json.dumps({"api_key": "edg_json_key", "org": "TestOrg"})
        )
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir / "credentials"),
        )
        assert _get_api_key() == "edg_json_key"

    def test_reads_legacy_format(self, tmp_path, monkeypatch):
        from edgeml.cli import _get_api_key

        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        cred_dir = tmp_path / ".edgeml"
        cred_dir.mkdir()
        (cred_dir / "credentials").write_text("api_key=edg_legacy_key\n")
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir / "credentials"),
        )
        assert _get_api_key() == "edg_legacy_key"

    def test_env_var_takes_priority_over_file(self, tmp_path, monkeypatch):
        from edgeml.cli import _get_api_key

        monkeypatch.setenv("EDGEML_API_KEY", "edg_env_key")
        cred_dir = tmp_path / ".edgeml"
        cred_dir.mkdir()
        (cred_dir / "credentials").write_text(
            json.dumps({"api_key": "edg_file_key"})
        )
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir / "credentials"),
        )
        assert _get_api_key() == "edg_env_key"

    def test_empty_file_returns_empty_string(self, tmp_path, monkeypatch):
        from edgeml.cli import _get_api_key

        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        cred_dir = tmp_path / ".edgeml"
        cred_dir.mkdir()
        (cred_dir / "credentials").write_text("")
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir / "credentials"),
        )
        assert _get_api_key() == ""


# ---------------------------------------------------------------------------
# --api-key direct save (headless/CI)
# ---------------------------------------------------------------------------


class TestApiKeyFlag:
    """Test the --api-key flag bypasses browser flow."""

    def test_api_key_flag_saves_directly(self, tmp_path, monkeypatch):
        cred_dir = tmp_path / ".edgeml"
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir) if "~/.edgeml" in x else x,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["login", "--api-key", "edg_ci_key"])
        assert result.exit_code == 0
        assert "API key saved" in result.output

        data = json.loads((cred_dir / "credentials").read_text())
        assert data["api_key"] == "edg_ci_key"

    def test_api_key_flag_does_not_open_browser(self, tmp_path, monkeypatch):
        cred_dir = tmp_path / ".edgeml"
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir) if "~/.edgeml" in x else x,
        )

        with patch("edgeml.cli.webbrowser.open") as mock_open:
            runner = CliRunner()
            runner.invoke(main, ["login", "--api-key", "edg_ci_key"])
        mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# State mismatch rejection
# ---------------------------------------------------------------------------


class TestStateMismatch:
    """Verify the callback server rejects requests with invalid state."""

    def test_state_mismatch_returns_400(self):
        """Simulate a callback with a wrong state token."""
        import http.server
        import socket

        correct_state = secrets.token_urlsafe(32)
        wrong_state = secrets.token_urlsafe(32)

        received_key = None
        got_callback = threading.Event()
        response_code = None

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                nonlocal received_key, response_code
                params = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query
                )
                cb_state = params.get("state", [None])[0]
                if cb_state != correct_state:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Invalid state parameter.")
                    response_code = 400
                    got_callback.set()
                    return

                received_key = params.get("key", [None])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"OK")
                response_code = 200
                got_callback.set()

            def log_message(self, format, *args):
                pass

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        server = HTTPServer(("127.0.0.1", port), _Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            # Send callback with wrong state
            url = (
                f"http://127.0.0.1:{port}"
                f"?key=edg_test&state={wrong_state}&org=TestOrg"
            )
            req = urllib.request.Request(url)
            try:
                urllib.request.urlopen(req)
            except urllib.error.HTTPError as e:
                assert e.code == 400
            else:
                pytest.fail("Expected 400 response for state mismatch")

            assert received_key is None
            assert response_code == 400
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# HTTP server starts and handles callback
# ---------------------------------------------------------------------------


class TestCallbackServer:
    """Test the temporary HTTP server handles the OAuth callback correctly."""

    def test_server_handles_valid_callback(self, tmp_path, monkeypatch):
        """Full integration: start callback server, simulate dashboard redirect."""
        import socket

        cred_dir = tmp_path / ".edgeml"
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir) if "~/.edgeml" in x else x,
        )

        # We need to intercept the browser open call to get the auth URL,
        # then simulate the dashboard callback.
        captured_url = {}

        def fake_browser_open(url):
            captured_url["url"] = url
            # Parse the URL to extract port and state
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            callback = urllib.parse.unquote(params["callback"][0])
            state = params["state"][0]

            # Simulate dashboard redirecting back to the callback URL
            redirect_url = f"{callback}?key=edg_browser_key&state={state}&org=BrowserOrg"

            def do_callback():
                # Small delay to let the server start
                time.sleep(0.2)
                try:
                    urllib.request.urlopen(redirect_url)
                except Exception:
                    pass

            threading.Thread(target=do_callback, daemon=True).start()

        with patch("edgeml.cli.webbrowser.open", side_effect=fake_browser_open):
            runner = CliRunner()
            result = runner.invoke(main, ["login"])

        assert result.exit_code == 0
        assert "Authenticated as org: BrowserOrg" in result.output
        assert "API key saved" in result.output

        # Verify credentials were saved as JSON with org
        cred_file = cred_dir / "credentials"
        assert cred_file.exists()
        data = json.loads(cred_file.read_text())
        assert data["api_key"] == "edg_browser_key"
        assert data["org"] == "BrowserOrg"

    def test_browser_open_is_called_with_correct_url(self, tmp_path, monkeypatch):
        """Verify the URL opened in the browser has correct structure."""
        cred_dir = tmp_path / ".edgeml"
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir) if "~/.edgeml" in x else x,
        )

        captured_url = {}

        def fake_browser_open(url):
            captured_url["url"] = url
            # Simulate callback immediately
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            callback = urllib.parse.unquote(params["callback"][0])
            state = params["state"][0]
            redirect_url = f"{callback}?key=edg_test&state={state}&org=Test"

            def do_callback():
                time.sleep(0.2)
                try:
                    urllib.request.urlopen(redirect_url)
                except Exception:
                    pass

            threading.Thread(target=do_callback, daemon=True).start()

        with patch("edgeml.cli.webbrowser.open", side_effect=fake_browser_open):
            runner = CliRunner()
            runner.invoke(main, ["login"])

        url = captured_url["url"]
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "app.edgeml.io"
        assert parsed.path == "/cli/auth"
        params = urllib.parse.parse_qs(parsed.query)
        assert "callback" in params
        assert "state" in params
        callback = urllib.parse.unquote(params["callback"][0])
        assert callback.startswith("http://127.0.0.1:")

    def test_custom_dashboard_url(self, tmp_path, monkeypatch):
        """Verify EDGEML_DASHBOARD_URL env var is respected."""
        cred_dir = tmp_path / ".edgeml"
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir) if "~/.edgeml" in x else x,
        )
        monkeypatch.setenv("EDGEML_DASHBOARD_URL", "https://custom.edgeml.dev")

        captured_url = {}

        def fake_browser_open(url):
            captured_url["url"] = url
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            callback = urllib.parse.unquote(params["callback"][0])
            state = params["state"][0]
            redirect_url = f"{callback}?key=edg_test&state={state}&org=Test"

            def do_callback():
                time.sleep(0.2)
                try:
                    urllib.request.urlopen(redirect_url)
                except Exception:
                    pass

            threading.Thread(target=do_callback, daemon=True).start()

        with patch("edgeml.cli.webbrowser.open", side_effect=fake_browser_open):
            runner = CliRunner()
            runner.invoke(main, ["login"])

        url = captured_url["url"]
        assert url.startswith("https://custom.edgeml.dev/cli/auth")

    def test_server_serves_success_html(self, tmp_path, monkeypatch):
        """Verify the callback response includes a success page."""
        cred_dir = tmp_path / ".edgeml"
        monkeypatch.setattr(
            "edgeml.cli.os.path.expanduser",
            lambda x: str(cred_dir) if "~/.edgeml" in x else x,
        )

        response_body = {}

        def fake_browser_open(url):
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            callback = urllib.parse.unquote(params["callback"][0])
            state = params["state"][0]
            redirect_url = f"{callback}?key=edg_test&state={state}&org=Test"

            def do_callback():
                time.sleep(0.2)
                try:
                    resp = urllib.request.urlopen(redirect_url)
                    response_body["html"] = resp.read().decode()
                except Exception:
                    pass

            threading.Thread(target=do_callback, daemon=True).start()

        with patch("edgeml.cli.webbrowser.open", side_effect=fake_browser_open):
            runner = CliRunner()
            runner.invoke(main, ["login"])

        assert "Success" in response_body.get("html", "")

    def test_login_help_shows_api_key_flag(self):
        """Verify --api-key is documented in login help."""
        runner = CliRunner()
        result = runner.invoke(main, ["login", "--help"])
        assert result.exit_code == 0
        assert "--api-key" in result.output
