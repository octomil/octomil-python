"""Tests for the ``edgeml federation`` CLI command group."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from edgeml.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(api_responses: dict | None = None):
    """Build a mock Client with a mock _api that returns preconfigured data.

    ``api_responses`` maps ``(method, path_prefix)`` tuples to return values.
    For simple cases you can also just set return values on the mock directly.
    """
    mock = MagicMock()
    mock._org_id = "org_test_123"

    api = MagicMock()
    if api_responses:

        def _get(path, params=None):
            for (m, prefix), val in api_responses.items():
                if m == "get" and path.startswith(prefix):
                    return val
            return []

        def _post(path, payload=None):
            for (m, prefix), val in api_responses.items():
                if m == "post" and path.startswith(prefix):
                    return val
            return {}

        api.get = MagicMock(side_effect=_get)
        api.post = MagicMock(side_effect=_post)

    mock._api = api
    return mock


# ---------------------------------------------------------------------------
# federation create
# ---------------------------------------------------------------------------


class TestFederationCreate:
    @patch("edgeml.cli._get_client")
    def test_create_success(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        client._api.post.return_value = {"id": "fed_abc123", "name": "my-fed"}
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["federation", "create", "my-fed", "--description", "Test federation"],
        )

        assert result.exit_code == 0
        assert "Federation created: my-fed" in result.output
        assert "fed_abc123" in result.output

        # Verify the API was called with the right payload
        client._api.post.assert_called_once_with(
            "/federations",
            {
                "name": "my-fed",
                "org_id": "org_test_123",
                "description": "Test federation",
            },
        )

    @patch("edgeml.cli._get_client")
    def test_create_without_description(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        client._api.post.return_value = {"id": "fed_xyz"}
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "create", "bare-fed"])

        assert result.exit_code == 0
        assert "bare-fed" in result.output
        client._api.post.assert_called_once_with(
            "/federations",
            {"name": "bare-fed", "org_id": "org_test_123"},
        )

    def test_create_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        runner = CliRunner()
        result = runner.invoke(main, ["federation", "create", "test"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# federation invite
# ---------------------------------------------------------------------------


class TestFederationInvite:
    @patch("edgeml.cli._get_client")
    def test_invite_multiple_orgs(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        # First call: resolve federation name -> ID
        client._api.get.return_value = [{"id": "fed_001", "name": "my-fed"}]
        # Second call: invite response
        client._api.post.return_value = [
            {"org_id": "org_a", "status": "invited"},
            {"org_id": "org_b", "status": "invited"},
        ]
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["federation", "invite", "my-fed", "--org", "org_a", "--org", "org_b"],
        )

        assert result.exit_code == 0
        assert "Invited 2 org(s)" in result.output
        client._api.post.assert_called_once_with(
            "/federations/fed_001/invite",
            {"org_ids": ["org_a", "org_b"]},
        )

    @patch("edgeml.cli._get_client")
    def test_invite_single_org(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        client._api.get.return_value = [{"id": "fed_002", "name": "test-fed"}]
        client._api.post.return_value = [{"org_id": "org_x", "status": "invited"}]
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["federation", "invite", "test-fed", "--org", "org_x"],
        )

        assert result.exit_code == 0
        assert "Invited 1 org(s)" in result.output

    def test_invite_requires_org_flag(self, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        runner = CliRunner()
        result = runner.invoke(main, ["federation", "invite", "my-fed"])
        assert result.exit_code != 0

    @patch("edgeml.cli._get_client")
    def test_invite_resolves_federation_by_id_fallback(
        self, mock_get_client, monkeypatch
    ):
        """When name lookup returns empty, the raw value is used as ID."""
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        # Name lookup returns nothing
        client._api.get.return_value = []
        client._api.post.return_value = {"invited": [{"org_id": "org_z"}]}
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["federation", "invite", "fed_raw_id_999", "--org", "org_z"],
        )

        assert result.exit_code == 0
        # Should have called invite with the raw ID
        client._api.post.assert_called_once_with(
            "/federations/fed_raw_id_999/invite",
            {"org_ids": ["org_z"]},
        )


# ---------------------------------------------------------------------------
# federation join
# ---------------------------------------------------------------------------


class TestFederationJoin:
    @patch("edgeml.cli._get_client")
    def test_join_success(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        client._api.get.return_value = [{"id": "fed_join_01", "name": "target-fed"}]
        client._api.post.return_value = {"status": "joined"}
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "join", "target-fed"])

        assert result.exit_code == 0
        assert "Joined federation: target-fed" in result.output
        client._api.post.assert_called_once_with(
            "/federations/fed_join_01/join",
            {"org_id": "org_test_123"},
        )

    @patch("edgeml.cli._get_client")
    def test_join_with_raw_id(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        client._api.get.return_value = []  # name not found, fallback to raw ID
        client._api.post.return_value = {"status": "joined"}
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "join", "fed_direct_id"])

        assert result.exit_code == 0
        assert "Joined federation: fed_direct_id" in result.output


# ---------------------------------------------------------------------------
# federation list
# ---------------------------------------------------------------------------


class TestFederationList:
    @patch("edgeml.cli._get_client")
    def test_list_populated(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        client._api.get.return_value = [
            {"id": "fed_1", "name": "alpha-fed", "description": "First"},
            {"id": "fed_2", "name": "beta-fed", "description": "Second"},
        ]
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "list"])

        assert result.exit_code == 0
        assert "alpha-fed" in result.output
        assert "beta-fed" in result.output
        assert "fed_1" in result.output
        assert "fed_2" in result.output
        assert "First" in result.output
        assert "Second" in result.output

    @patch("edgeml.cli._get_client")
    def test_list_empty(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        client._api.get.return_value = []
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "list"])

        assert result.exit_code == 0
        assert "No federations found" in result.output

    @patch("edgeml.cli._get_client")
    def test_list_passes_org_id(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()
        client._api.get.return_value = []
        mock_get_client.return_value = client

        runner = CliRunner()
        runner.invoke(main, ["federation", "list"])

        client._api.get.assert_called_once_with(
            "/federations",
            params={"org_id": "org_test_123"},
        )


# ---------------------------------------------------------------------------
# federation show
# ---------------------------------------------------------------------------


class TestFederationShow:
    @patch("edgeml.cli._get_client")
    def test_show_success(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()

        # First get: name resolution
        # Second get: federation details
        call_count = [0]

        def _side_effect(path, params=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # name resolution
                return [{"id": "fed_show_01", "name": "show-fed"}]
            # detail fetch
            return {
                "id": "fed_show_01",
                "name": "show-fed",
                "description": "Detailed desc",
                "created_at": "2026-01-15T10:00:00Z",
                "org_id": "org_test_123",
            }

        client._api.get.side_effect = _side_effect
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "show", "show-fed"])

        assert result.exit_code == 0
        assert "show-fed" in result.output
        assert "fed_show_01" in result.output
        assert "Detailed desc" in result.output
        assert "2026-01-15" in result.output

    @patch("edgeml.cli._get_client")
    def test_show_with_no_description(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()

        call_count = [0]

        def _side_effect(path, params=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return [{"id": "fed_nd", "name": "no-desc"}]
            return {
                "id": "fed_nd",
                "name": "no-desc",
                "description": None,
                "created_at": "2026-02-01T00:00:00Z",
                "org_id": "org_test_123",
            }

        client._api.get.side_effect = _side_effect
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "show", "no-desc"])

        assert result.exit_code == 0
        # Description should show '-' for None
        assert "Description: -" in result.output


# ---------------------------------------------------------------------------
# federation members
# ---------------------------------------------------------------------------


class TestFederationMembers:
    @patch("edgeml.cli._get_client")
    def test_members_populated(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()

        call_count = [0]

        def _side_effect(path, params=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return [{"id": "fed_mem_01", "name": "mem-fed"}]
            return [
                {
                    "org_id": "org_aaa",
                    "status": "active",
                    "joined_at": "2026-01-10T00:00:00Z",
                },
                {"org_id": "org_bbb", "status": "invited", "joined_at": None},
            ]

        client._api.get.side_effect = _side_effect
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "members", "mem-fed"])

        assert result.exit_code == 0
        assert "org_aaa" in result.output
        assert "active" in result.output
        assert "org_bbb" in result.output
        assert "invited" in result.output

    @patch("edgeml.cli._get_client")
    def test_members_empty(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()

        call_count = [0]

        def _side_effect(path, params=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return [{"id": "fed_empty", "name": "empty-fed"}]
            return []

        client._api.get.side_effect = _side_effect
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "members", "empty-fed"])

        assert result.exit_code == 0
        assert "No members found" in result.output

    @patch("edgeml.cli._get_client")
    def test_members_resolves_federation_name(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()

        call_count = [0]

        def _side_effect(path, params=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return [{"id": "fed_resolved", "name": "named-fed"}]
            return [{"org_id": "org_x", "status": "active", "joined_at": "2026-01-01"}]

        client._api.get.side_effect = _side_effect
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(main, ["federation", "members", "named-fed"])

        assert result.exit_code == 0
        # Verify name resolution call
        first_call = client._api.get.call_args_list[0]
        assert first_call[0][0] == "/federations"
        assert first_call[1]["params"]["name"] == "named-fed"
        # Verify members call with resolved ID
        second_call = client._api.get.call_args_list[1]
        assert second_call[0][0] == "/federations/fed_resolved/members"


# ---------------------------------------------------------------------------
# federation share
# ---------------------------------------------------------------------------


class TestFederationShare:
    @patch("edgeml.cli._get_client")
    def test_share_model_success(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()

        call_count = [0]

        def _get_side_effect(path, params=None):
            call_count[0] += 1
            if "/federations" in path and params and "name" in params:
                return [{"id": "fed_abc", "name": "my-fed"}]
            if "/models" in path:
                return [{"id": "model_123", "name": "radiology-v1"}]
            return []

        client._api.get.side_effect = _get_side_effect
        client._api.post.return_value = {"id": "model_123", "federation_id": "fed_abc"}
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["federation", "share", "radiology-v1", "--federation", "my-fed"],
        )

        assert result.exit_code == 0
        assert "shared with federation" in result.output

    @patch("edgeml.cli._get_client")
    def test_share_model_not_found(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()

        def _get_side_effect(path, params=None):
            if "/federations" in path and params and "name" in params:
                return [{"id": "fed_abc", "name": "my-fed"}]
            if "/models" in path:
                return []  # no models
            return []

        client._api.get.side_effect = _get_side_effect
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["federation", "share", "nonexistent", "--federation", "my-fed"],
        )

        assert result.exit_code != 0
        assert "not found" in result.output

    @patch("edgeml.cli._get_client")
    def test_share_calls_correct_api(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")
        client = _mock_client()

        def _get_side_effect(path, params=None):
            if "/federations" in path and params and "name" in params:
                return [{"id": "fed_xyz", "name": "test-fed"}]
            if "/models" in path:
                return [{"id": "model_456", "name": "my-model"}]
            return []

        client._api.get.side_effect = _get_side_effect
        client._api.post.return_value = {}
        mock_get_client.return_value = client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["federation", "share", "my-model", "--federation", "test-fed"],
        )

        assert result.exit_code == 0
        # Verify the POST call to share the model
        post_calls = client._api.post.call_args_list
        assert len(post_calls) == 1
        assert post_calls[0][0][0] == "/federations/fed_xyz/models"
        assert post_calls[0][0][1] == {"model_id": "model_456"}


# ---------------------------------------------------------------------------
# federation help
# ---------------------------------------------------------------------------


class TestFederationHelp:
    def test_federation_group_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["federation", "--help"])
        assert result.exit_code == 0
        assert "create" in result.output
        assert "invite" in result.output
        assert "join" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "members" in result.output
        assert "share" in result.output

    def test_federation_create_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["federation", "create", "--help"])
        assert result.exit_code == 0
        assert "--description" in result.output

    def test_federation_share_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["federation", "share", "--help"])
        assert result.exit_code == 0
        assert "--federation" in result.output
