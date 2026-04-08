"""Tests for octomil.commands.agent — CLI agent command routing integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


class TestAgentCommandRun:
    """Verify that the agent CLI command constructs OctomilClient,
    best-effort fetches desired state, and uses client.agent_session()."""

    @patch("octomil.client.OctomilClient")
    @patch("octomil.auth.OrgApiKeyAuth")
    def test_constructs_octomil_client(self, mock_auth_cls, mock_client_cls):
        """_run() creates OctomilClient with OrgApiKeyAuth."""
        from octomil.commands.agent import _run

        mock_auth = MagicMock()
        mock_auth_cls.return_value = mock_auth

        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.summary = "Test result"
        mock_result.confidence = None
        mock_result.evidence = []
        mock_result.next_steps = []
        mock_result.session_id = "sess_123"
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_client.agent_session.return_value = mock_session
        mock_client_cls.return_value = mock_client

        asyncio.run(_run("https://api.octomil.com", "test-key", "advisor", "test query", "qwen2.5:7b"))

        mock_auth_cls.assert_called_once_with(
            api_key="test-key",
            org_id="default",
            api_base="https://api.octomil.com",
        )
        mock_client_cls.assert_called_once_with(auth=mock_auth)

    @patch("octomil.client.OctomilClient")
    @patch("octomil.auth.OrgApiKeyAuth")
    def test_best_effort_desired_state(self, mock_auth_cls, mock_client_cls):
        """_run() calls control.register() and control.get_desired_state() best-effort."""
        from octomil.commands.agent import _run

        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.summary = "OK"
        mock_result.confidence = None
        mock_result.evidence = []
        mock_result.next_steps = []
        mock_result.session_id = "sess_456"
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_client.agent_session.return_value = mock_session
        mock_client_cls.return_value = mock_client

        asyncio.run(_run("https://api.octomil.com", "test-key", "advisor", "q", "qwen2.5:7b"))

        mock_client.control.register.assert_called_once()
        mock_client.control.get_desired_state.assert_called_once()

    @patch("octomil.client.OctomilClient")
    @patch("octomil.auth.OrgApiKeyAuth")
    def test_uses_agent_session_factory(self, mock_auth_cls, mock_client_cls):
        """_run() uses client.agent_session() — not bare AgentSession."""
        from octomil.commands.agent import _run

        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.summary = "Done"
        mock_result.confidence = 0.95
        mock_result.evidence = ["ev1"]
        mock_result.next_steps = ["step1"]
        mock_result.session_id = "sess_789"
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_client.agent_session.return_value = mock_session
        mock_client_cls.return_value = mock_client

        result = asyncio.run(_run("https://api.octomil.com", "test-key", "advisor", "deploy?", "phi-4"))

        mock_client.agent_session.assert_called_once_with(base_url="https://api.octomil.com")
        mock_session.run.assert_called_once_with("advisor", "deploy?", model="phi-4")
        assert result.session_id == "sess_789"

    @patch("octomil.client.OctomilClient")
    @patch("octomil.auth.OrgApiKeyAuth")
    def test_works_when_desired_state_fails(self, mock_auth_cls, mock_client_cls):
        """_run() continues even when control.register() or get_desired_state() raises."""
        from octomil.commands.agent import _run

        mock_client = MagicMock()
        mock_client.control.register.side_effect = ConnectionError("offline")
        mock_client.control.get_desired_state.side_effect = ConnectionError("offline")

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.summary = "Ran without routing"
        mock_result.confidence = None
        mock_result.evidence = []
        mock_result.next_steps = []
        mock_result.session_id = "sess_fallback"
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_client.agent_session.return_value = mock_session
        mock_client_cls.return_value = mock_client

        result = asyncio.run(_run("https://api.octomil.com", "test-key", "advisor", "hello", "qwen2.5:7b"))

        # Should still succeed
        assert result.session_id == "sess_fallback"
        mock_client.agent_session.assert_called_once()
        mock_session.run.assert_called_once()
