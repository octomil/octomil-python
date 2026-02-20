"""Tests for edgeml.enterprise — Enterprise API client and CLI commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from edgeml.cli import main
from edgeml.enterprise import (
    COMPLIANCE_PRESETS,
    EnterpriseClient,
    EnterpriseClientError,
    load_config,
    save_config,
)


# ---------------------------------------------------------------------------
# Compliance presets
# ---------------------------------------------------------------------------


class TestCompliancePresets:
    def test_hipaa_preset_has_required_fields(self):
        hipaa = COMPLIANCE_PRESETS["hipaa"]
        assert hipaa["hipaa_mode"] is True
        assert hipaa["audit_retention_days"] == 2190
        assert hipaa["require_mfa_for_admin"] is True
        assert hipaa["policy_profile"] == "regulated"
        assert hipaa["session_duration_hours"] == 8
        assert hipaa["reauth_interval_minutes"] == 30

    def test_gdpr_preset_has_required_fields(self):
        gdpr = COMPLIANCE_PRESETS["gdpr"]
        assert gdpr["audit_retention_days"] == 1825
        assert gdpr["require_mfa_for_admin"] is True
        assert gdpr["policy_profile"] == "balanced"

    def test_pci_preset_has_required_fields(self):
        pci = COMPLIANCE_PRESETS["pci"]
        assert pci["require_mfa_for_admin"] is True
        assert pci["mfa_required"] is True
        assert pci["session_duration_hours"] == 8
        assert pci["require_admin_approval"] is True
        assert pci["policy_profile"] == "regulated"

    def test_soc2_preset_has_required_fields(self):
        soc2 = COMPLIANCE_PRESETS["soc2"]
        assert soc2["audit_retention_days"] == 365
        assert soc2["require_mfa_for_admin"] is True
        assert soc2["require_admin_approval"] is True
        assert soc2["require_model_approval"] is True
        assert soc2["auto_rollback_enabled"] is True
        assert soc2["policy_profile"] == "balanced"

    def test_all_presets_have_policy_profile(self):
        for name, preset in COMPLIANCE_PRESETS.items():
            assert "policy_profile" in preset, f"{name} missing policy_profile"
            assert preset["policy_profile"] in ("startup", "balanced", "regulated")


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------


class TestConfigHelpers:
    def test_save_and_load_config(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".edgeml" / "config.json"
        monkeypatch.setattr(
            "edgeml.enterprise._config_path", lambda: config_file
        )

        data = {"org_id": "test-org-123", "org_name": "Test Corp", "region": "us"}
        save_config(data)

        assert config_file.exists()
        loaded = load_config()
        assert loaded["org_id"] == "test-org-123"
        assert loaded["org_name"] == "Test Corp"
        assert loaded["region"] == "us"

    def test_load_config_returns_empty_when_missing(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".edgeml" / "config.json"
        monkeypatch.setattr(
            "edgeml.enterprise._config_path", lambda: config_file
        )
        assert load_config() == {}

    def test_load_config_returns_empty_on_bad_json(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".edgeml" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("not json at all{{{")
        monkeypatch.setattr(
            "edgeml.enterprise._config_path", lambda: config_file
        )
        assert load_config() == {}

    def test_save_config_creates_parent_dirs(self, tmp_path, monkeypatch):
        config_file = tmp_path / "deep" / "nested" / ".edgeml" / "config.json"
        monkeypatch.setattr(
            "edgeml.enterprise._config_path", lambda: config_file
        )
        save_config({"org_id": "x"})
        assert config_file.exists()

    def test_config_file_has_restricted_permissions(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".edgeml" / "config.json"
        monkeypatch.setattr(
            "edgeml.enterprise._config_path", lambda: config_file
        )
        save_config({"org_id": "x"})
        import stat
        mode = config_file.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# EnterpriseClient
# ---------------------------------------------------------------------------


class TestEnterpriseClient:
    def _mock_client(self, monkeypatch):
        """Return an EnterpriseClient with a mocked httpx.Client."""
        mock_http = MagicMock()
        client = EnterpriseClient.__new__(EnterpriseClient)
        client._http = mock_http
        return client, mock_http

    def test_create_org(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"org_id": "acme-corp", "workspace_type": "enterprise"}
        mock_http.put.return_value = resp

        result = client.create_org("Acme Corp", region="us")
        assert result["org_id"] == "acme-corp"
        mock_http.put.assert_called_once()
        call_args = mock_http.put.call_args
        assert "/onboarding/" in call_args[0][0]
        assert call_args[1]["json"]["workspace_name"] == "Acme Corp"

    def test_set_policy_profile(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"policy_profile": "regulated"}
        mock_http.put.return_value = resp

        result = client.set_policy_profile("org-1", "regulated")
        assert result["policy_profile"] == "regulated"
        mock_http.put.assert_called_once_with(
            "/onboarding/org-1/policy",
            json={"policy_profile": "regulated"},
        )

    def test_set_compliance_hipaa(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"hipaa_mode": True}
        mock_http.put.return_value = resp

        result = client.set_compliance("org-1", "hipaa")
        # Should have called put twice: once for policy profile, once for settings
        assert mock_http.put.call_count == 2

    def test_set_compliance_unknown_raises(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        with pytest.raises(EnterpriseClientError, match="Unknown compliance preset"):
            client.set_compliance("org-1", "unknown_preset")

    def test_invite_member(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 201
        resp.json.return_value = {"email": "alice@acme.com", "role": "admin"}
        mock_http.post.return_value = resp

        result = client.invite_member("org-1", "alice@acme.com", role="admin")
        assert result["email"] == "alice@acme.com"
        assert result["role"] == "admin"
        mock_http.post.assert_called_once()

    def test_list_members(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [
            {"email": "alice@acme.com", "role": "admin"},
            {"email": "bob@acme.com", "role": "member"},
        ]
        mock_http.get.return_value = resp

        members = client.list_members("org-1")
        assert len(members) == 2
        mock_http.get.assert_called_once_with("/orgs/org-1/members")

    def test_create_api_key(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 201
        resp.json.return_value = {
            "api_key": "edg_test1234567890",
            "prefix": "edg_test1234",
            "name": "deploy-key",
        }
        mock_http.post.return_value = resp

        result = client.create_api_key("org-1", "deploy-key", scopes={"devices": "write"})
        assert "api_key" in result
        assert result["name"] == "deploy-key"

    def test_list_api_keys(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [
            {"name": "key1", "prefix": "edg_abc"},
            {"name": "key2", "prefix": "edg_def"},
        ]
        mock_http.get.return_value = resp

        keys = client.list_api_keys("org-1")
        assert len(keys) == 2

    def test_revoke_api_key(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"name": "old-key", "revoked_at": "2026-01-01T00:00:00Z"}
        mock_http.post.return_value = resp

        result = client.revoke_api_key("key-123")
        assert result["revoked_at"] is not None
        mock_http.post.assert_called_once_with("/api-keys/key-123/revoke")

    def test_get_settings(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "org_id": "org-1",
            "audit_retention_days": 90,
            "require_mfa_for_admin": False,
        }
        mock_http.get.return_value = resp

        result = client.get_settings("org-1")
        assert result["audit_retention_days"] == 90

    def test_update_settings(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"require_mfa_for_admin": True}
        mock_http.put.return_value = resp

        result = client.update_settings("org-1", require_mfa_for_admin=True)
        assert result["require_mfa_for_admin"] is True
        mock_http.put.assert_called_once_with(
            "/settings/org-1",
            json={"require_mfa_for_admin": True},
        )

    def test_get_onboarding_status(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"org_id": "org-1", "current_step": "sdk_setup"}
        mock_http.get.return_value = resp

        result = client.get_onboarding_status("org-1")
        assert result["current_step"] == "sdk_setup"

    def test_complete_onboarding(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"current_step": "complete", "onboarding_complete": True}
        mock_http.post.return_value = resp

        result = client.complete_onboarding("org-1")
        assert result["onboarding_complete"] is True

    def test_raise_for_status_on_error(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 403
        resp.json.return_value = {"detail": "Forbidden"}
        resp.text = '{"detail": "Forbidden"}'
        mock_http.get.return_value = resp

        with pytest.raises(EnterpriseClientError, match="403"):
            client.get_settings("org-1")

    def test_raise_for_status_non_json_error(self, monkeypatch):
        client, mock_http = self._mock_client(monkeypatch)
        resp = MagicMock()
        resp.status_code = 500
        resp.json.side_effect = ValueError("not json")
        resp.text = "Internal Server Error"
        mock_http.get.return_value = resp

        with pytest.raises(EnterpriseClientError, match="Internal Server Error"):
            client.get_settings("org-1")


# ---------------------------------------------------------------------------
# CLI: edgeml init
# ---------------------------------------------------------------------------


class TestInitCommand:
    @patch("edgeml.cli._get_api_key", return_value="test-key")
    @patch("edgeml.enterprise.EnterpriseClient")
    @patch("edgeml.enterprise.save_config")
    @patch("edgeml.enterprise.load_config", return_value={})
    def test_init_creates_org(self, mock_load, mock_save, MockClient, mock_key):
        mock_instance = MockClient.return_value
        mock_instance.create_org.return_value = {"org_id": "acme-corp"}

        runner = CliRunner()
        result = runner.invoke(main, ["init", "Acme Corp", "--region", "us"])

        assert result.exit_code == 0
        assert "acme-corp" in result.output
        mock_instance.create_org.assert_called_once_with(
            "Acme Corp", region="us", workspace_type="enterprise"
        )
        mock_save.assert_called_once()
        saved_config = mock_save.call_args[0][0]
        assert saved_config["org_id"] == "acme-corp"
        assert saved_config["org_name"] == "Acme Corp"

    @patch("edgeml.cli._get_api_key", return_value="test-key")
    @patch("edgeml.enterprise.EnterpriseClient")
    @patch("edgeml.enterprise.save_config")
    @patch("edgeml.enterprise.load_config", return_value={})
    def test_init_with_compliance(self, mock_load, mock_save, MockClient, mock_key):
        mock_instance = MockClient.return_value
        mock_instance.create_org.return_value = {"org_id": "acme-corp"}
        mock_instance.set_compliance.return_value = {}

        runner = CliRunner()
        result = runner.invoke(
            main, ["init", "Acme Corp", "--compliance", "hipaa", "--region", "eu"]
        )

        assert result.exit_code == 0
        assert "HIPAA" in result.output
        mock_instance.set_compliance.assert_called_once_with("acme-corp", "hipaa")
        saved_config = mock_save.call_args[0][0]
        assert saved_config["compliance"] == "hipaa"
        assert saved_config["region"] == "eu"

    @patch("edgeml.cli._get_api_key", return_value="")
    def test_init_requires_api_key(self, mock_key):
        runner = CliRunner()
        result = runner.invoke(main, ["init", "Acme Corp"])
        assert result.exit_code != 0
        assert "No API key" in result.output

    def test_init_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "compliance" in result.output
        assert "region" in result.output


# ---------------------------------------------------------------------------
# CLI: edgeml team
# ---------------------------------------------------------------------------


class TestTeamCommands:
    def test_team_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["team", "--help"])
        assert result.exit_code == 0
        assert "add" in result.output
        assert "list" in result.output
        assert "set-policy" in result.output

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_team_add(self, mock_get_client, mock_org_id):
        mock_client = MagicMock()
        mock_client.invite_member.return_value = {
            "email": "alice@acme.com",
            "role": "admin",
        }
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main, ["team", "add", "alice@acme.com", "--role", "admin"]
        )
        assert result.exit_code == 0
        assert "alice@acme.com" in result.output
        assert "admin" in result.output
        mock_client.invite_member.assert_called_once_with(
            "org-123", "alice@acme.com", role="admin", name=None
        )

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_team_list(self, mock_get_client, mock_org_id):
        mock_client = MagicMock()
        mock_client.list_members.return_value = [
            {"email": "alice@acme.com", "role": "admin", "name": "Alice"},
            {"email": "bob@acme.com", "role": "member", "name": "Bob"},
        ]
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["team", "list"])
        assert result.exit_code == 0
        assert "alice@acme.com" in result.output
        assert "bob@acme.com" in result.output
        assert "admin" in result.output
        assert "member" in result.output
        assert "2 member(s)" in result.output

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_team_list_empty(self, mock_get_client, mock_org_id):
        mock_client = MagicMock()
        mock_client.list_members.return_value = []
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["team", "list"])
        assert result.exit_code == 0
        assert "No team members found" in result.output

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_team_set_policy_mfa(self, mock_get_client, mock_org_id):
        mock_client = MagicMock()
        mock_client.update_settings.return_value = {}
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main, ["team", "set-policy", "--require-mfa", "--session-hours", "8"]
        )
        assert result.exit_code == 0
        mock_client.update_settings.assert_called_once_with(
            "org-123",
            require_mfa_for_admin=True,
            session_duration_hours=8,
        )

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_team_set_policy_auto_rollback(self, mock_get_client, mock_org_id):
        mock_client = MagicMock()
        mock_client.update_settings.return_value = {}
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main, ["team", "set-policy", "--auto-rollback", "--audit-retention-days", "365"]
        )
        assert result.exit_code == 0
        mock_client.update_settings.assert_called_once_with(
            "org-123",
            auto_rollback_enabled=True,
            audit_retention_days=365,
        )

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_team_set_policy_no_changes(self, mock_get_client, mock_org_id):
        runner = CliRunner()
        result = runner.invoke(main, ["team", "set-policy"])
        assert result.exit_code == 0
        assert "No policy changes" in result.output


# ---------------------------------------------------------------------------
# CLI: edgeml keys
# ---------------------------------------------------------------------------


class TestKeysCommands:
    def test_keys_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["keys", "--help"])
        assert result.exit_code == 0
        assert "create" in result.output
        assert "list" in result.output
        assert "revoke" in result.output

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_keys_create(self, mock_get_client, mock_org_id):
        mock_client = MagicMock()
        mock_client.create_api_key.return_value = {
            "api_key": "edg_test1234567890abcdef",
            "prefix": "edg_test1234",
            "name": "deploy-key",
        }
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["keys", "create", "deploy-key", "--scope", "devices:write", "--scope", "models:read"],
        )
        assert result.exit_code == 0
        assert "deploy-key" in result.output
        assert "edg_test1234567890abcdef" in result.output
        assert "Save this key" in result.output

        # Verify scopes were passed correctly
        call_kwargs = mock_client.create_api_key.call_args
        assert call_kwargs[0] == ("org-123", "deploy-key")
        scopes = call_kwargs[1]["scopes"]
        assert scopes["devices"] == "write"
        assert scopes["models"] == "read"

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_keys_create_no_scopes(self, mock_get_client, mock_org_id):
        mock_client = MagicMock()
        mock_client.create_api_key.return_value = {
            "api_key": "edg_abc123",
            "prefix": "edg_abc1",
            "name": "admin-key",
        }
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["keys", "create", "admin-key"])
        assert result.exit_code == 0
        call_kwargs = mock_client.create_api_key.call_args
        assert call_kwargs[1]["scopes"] is None

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_keys_list(self, mock_get_client, mock_org_id):
        mock_client = MagicMock()
        mock_client.list_api_keys.return_value = [
            {
                "name": "deploy-key",
                "prefix": "edg_abc12345",
                "created_at": "2026-01-15T10:00:00Z",
                "revoked_at": None,
            },
            {
                "name": "old-key",
                "prefix": "edg_def67890",
                "created_at": "2025-06-01T10:00:00Z",
                "revoked_at": "2025-12-01T10:00:00Z",
            },
        ]
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["keys", "list"])
        assert result.exit_code == 0
        assert "deploy-key" in result.output
        assert "old-key" in result.output
        assert "2 key(s)" in result.output

    @patch("edgeml.cli._require_org_id", return_value="org-123")
    @patch("edgeml.cli._get_enterprise_client")
    def test_keys_list_empty(self, mock_get_client, mock_org_id):
        mock_client = MagicMock()
        mock_client.list_api_keys.return_value = []
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["keys", "list"])
        assert result.exit_code == 0
        assert "No API keys found" in result.output

    @patch("edgeml.cli._get_enterprise_client")
    def test_keys_revoke(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.revoke_api_key.return_value = {
            "name": "old-key",
            "prefix": "edg_abc",
            "revoked_at": "2026-02-19T00:00:00Z",
        }
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["keys", "revoke", "key-abc-123", "--yes"])
        assert result.exit_code == 0
        assert "Revoked" in result.output
        assert "old-key" in result.output
        mock_client.revoke_api_key.assert_called_once_with("key-abc-123")


# ---------------------------------------------------------------------------
# CLI: edgeml org
# ---------------------------------------------------------------------------


class TestOrgCommand:
    @patch("edgeml.enterprise.get_org_id", return_value="")
    def test_org_no_config(self, mock_org):
        runner = CliRunner()
        result = runner.invoke(main, ["org"])
        assert result.exit_code == 0
        assert "No organization configured" in result.output

    @patch("edgeml.cli._get_api_key", return_value="test-key")
    @patch("edgeml.cli._get_enterprise_client")
    @patch("edgeml.enterprise.get_org_id", return_value="org-123")
    @patch("edgeml.enterprise.load_config", return_value={
        "org_name": "Test Corp",
        "region": "us",
        "compliance": "soc2",
    })
    def test_org_shows_info(self, mock_load, mock_org_id, mock_get_client, mock_key):
        mock_client = MagicMock()
        mock_client.get_settings.return_value = {
            "audit_retention_days": 365,
            "require_mfa_for_admin": True,
            "require_admin_approval": True,
            "require_model_approval": True,
            "auto_rollback_enabled": True,
            "session_duration_hours": 12,
            "reauth_interval_minutes": 60,
        }
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, ["org"])
        assert result.exit_code == 0
        assert "Test Corp" in result.output
        assert "org-123" in result.output
        assert "SOC2" in result.output
        assert "365" in result.output


# ---------------------------------------------------------------------------
# get_org_id
# ---------------------------------------------------------------------------


class TestGetOrgId:
    def test_from_env(self, monkeypatch):
        from edgeml.enterprise import get_org_id

        monkeypatch.setenv("EDGEML_ORG_ID", "env-org-42")
        assert get_org_id() == "env-org-42"

    def test_from_config(self, monkeypatch, tmp_path):
        from edgeml.enterprise import get_org_id

        monkeypatch.delenv("EDGEML_ORG_ID", raising=False)
        config_file = tmp_path / ".edgeml" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps({"org_id": "config-org-99"}))
        monkeypatch.setattr(
            "edgeml.enterprise._config_path", lambda: config_file
        )

        assert get_org_id() == "config-org-99"

    def test_empty_when_no_env_no_config(self, monkeypatch, tmp_path):
        from edgeml.enterprise import get_org_id

        monkeypatch.delenv("EDGEML_ORG_ID", raising=False)
        config_file = tmp_path / ".edgeml" / "config.json"
        monkeypatch.setattr(
            "edgeml.enterprise._config_path", lambda: config_file
        )

        assert get_org_id() == ""
