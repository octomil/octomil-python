"""Tests for Octomil.local() facade backed by the invisible local runner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from octomil.facade import LocalOctomil, Octomil, OctomilNotInitializedError


class TestOctomilLocalFactory:
    def test_local_returns_local_octomil(self) -> None:
        client = Octomil.local(model="gemma-1b")
        assert isinstance(client, LocalOctomil)

    def test_local_default_model(self) -> None:
        client = Octomil.local()
        assert isinstance(client, LocalOctomil)
        assert client._model == "default"


class TestLocalOctomilNotInitialized:
    def test_responses_raises_before_initialize(self) -> None:
        client = LocalOctomil(model="gemma-1b")
        with pytest.raises(OctomilNotInitializedError):
            _ = client.responses

    def test_embeddings_raises_before_initialize(self) -> None:
        client = LocalOctomil(model="gemma-1b")
        with pytest.raises(OctomilNotInitializedError):
            _ = client.embeddings


class TestLocalOctomilInitialize:
    @pytest.mark.asyncio
    async def test_initialize_starts_runner(self) -> None:
        mock_handle = MagicMock(
            base_url="http://127.0.0.1:51200",
            token="test-token",
            model="gemma-1b",
            engine="auto",
        )
        with (
            patch("octomil.local_runner.manager.LocalRunnerManager") as MockMgr,
            patch("octomil.local_runner.client.LocalRunnerClient"),
        ):
            MockMgr.return_value.ensure.return_value = mock_handle
            client = LocalOctomil(model="gemma-1b")
            await client.initialize()

            MockMgr.return_value.ensure.assert_called_once_with(model="gemma-1b", engine=None)
            assert client._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        mock_handle = MagicMock(
            base_url="http://127.0.0.1:51200",
            token="test-token",
            model="gemma-1b",
            engine="auto",
        )
        with (
            patch("octomil.local_runner.manager.LocalRunnerManager") as MockMgr,
            patch("octomil.local_runner.client.LocalRunnerClient"),
        ):
            MockMgr.return_value.ensure.return_value = mock_handle
            client = LocalOctomil(model="gemma-1b")
            await client.initialize()
            await client.initialize()  # second call is no-op
            assert MockMgr.return_value.ensure.call_count == 1


class TestLocalResponses:
    @pytest.mark.asyncio
    async def test_create_response(self) -> None:
        mock_handle = MagicMock(
            base_url="http://127.0.0.1:51200",
            token="test-token",
        )
        mock_runner_client = MagicMock()
        mock_runner_client.create_response = AsyncMock(return_value={"choices": [{"message": {"content": "Hello!"}}]})
        with (
            patch("octomil.local_runner.manager.LocalRunnerManager") as MockMgr,
            patch("octomil.local_runner.client.LocalRunnerClient", return_value=mock_runner_client),
        ):
            MockMgr.return_value.ensure.return_value = mock_handle
            client = LocalOctomil(model="gemma-1b")
            await client.initialize()

            result = await client.responses.create(input="Hi there")
            assert result["choices"][0]["message"]["content"] == "Hello!"
            mock_runner_client.create_response.assert_called_once_with(model="gemma-1b", input="Hi there")


class TestLocalEmbeddings:
    @pytest.mark.asyncio
    async def test_create_embedding(self) -> None:
        mock_handle = MagicMock(
            base_url="http://127.0.0.1:51200",
            token="test-token",
        )
        mock_runner_client = MagicMock()
        mock_runner_client.create_embedding = AsyncMock(return_value={"data": [{"embedding": [0.1, 0.2]}]})
        with (
            patch("octomil.local_runner.manager.LocalRunnerManager") as MockMgr,
            patch("octomil.local_runner.client.LocalRunnerClient", return_value=mock_runner_client),
        ):
            MockMgr.return_value.ensure.return_value = mock_handle
            client = LocalOctomil(model="gemma-1b")
            await client.initialize()

            result = await client.embeddings.create(input="test text")
            assert "data" in result
            mock_runner_client.create_embedding.assert_called_once_with(model="gemma-1b", input=["test text"])


class TestLocalPolicyBehavior:
    def test_no_server_key_required(self) -> None:
        """Octomil.local() should not require OCTOMIL_SERVER_KEY."""
        with patch.dict("os.environ", {}, clear=True):
            client = Octomil.local(model="gemma-1b")
            assert isinstance(client, LocalOctomil)

    def test_local_never_creates_cloud_client(self) -> None:
        """LocalOctomil should not import or use OctomilClient (hosted)."""
        client = LocalOctomil(model="gemma-1b")
        assert client._client is None if hasattr(client, "_client") else True
