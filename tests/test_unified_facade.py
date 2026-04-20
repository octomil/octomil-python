"""Tests for the unified Octomil facade."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import octomil
from octomil.embeddings import EmbeddingResult, EmbeddingUsage
from octomil.errors import OctomilError
from octomil.facade import FacadeEmbeddings, FacadeResponses, Octomil, OctomilNotInitializedError
from octomil.responses.types import Response, ResponseToolCall, TextOutput, ToolCallOutput


class TestTopLevelFacadeExport:
    def test_top_level_octomil_exports_unified_facade(self):
        from octomil import Octomil as TopLevelOctomil

        assert TopLevelOctomil is Octomil
        assert hasattr(TopLevelOctomil, "from_env")
        assert inspect.getsourcefile(TopLevelOctomil) == inspect.getsourcefile(Octomil)
        assert octomil.__version__ == "4.6.1"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructorPublishableKey:
    def test_valid_test_key(self):
        client = Octomil(publishable_key="oct_pub_test_abc123")
        assert client._initialized is False

    def test_valid_live_key(self):
        client = Octomil(publishable_key="oct_pub_live_xyz789")
        assert client._initialized is False

    def test_invalid_prefix_raises(self):
        with pytest.raises(OctomilError):
            Octomil(publishable_key="bad_key_prefix")

    def test_empty_key_raises(self):
        with pytest.raises(OctomilError):
            Octomil(publishable_key="")


class TestConstructorApiKey:
    def test_api_key_with_org_id(self):
        client = Octomil(api_key="edg_test_123", org_id="org_abc")
        assert client._initialized is False

    def test_api_key_without_org_id_raises(self):
        with pytest.raises(ValueError, match="org_id is required"):
            Octomil(api_key="edg_test_123")


class TestConstructorAuth:
    def test_auth_passthrough(self):
        from octomil.auth import OrgApiKeyAuth

        auth = OrgApiKeyAuth(api_key="edg_test", org_id="org_1")
        client = Octomil(auth=auth)
        assert client._auth is auth


class TestConstructorNoArgs:
    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="One of"):
            Octomil()


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_sets_initialized(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        assert client._initialized is False
        asyncio.run(client.initialize())
        assert client._initialized is True

    def test_idempotent(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        asyncio.run(client.initialize())
        # Second call should not raise
        asyncio.run(client.initialize())
        assert client._initialized is True


# ---------------------------------------------------------------------------
# responses property — guards
# ---------------------------------------------------------------------------


class TestResponsesGuard:
    def test_create_before_init_raises(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        with pytest.raises(OctomilNotInitializedError):
            _ = client.responses

    def test_stream_before_init_raises(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        with pytest.raises(OctomilNotInitializedError):
            _ = client.responses


# ---------------------------------------------------------------------------
# FacadeResponses.create / stream
# ---------------------------------------------------------------------------


class TestFacadeResponsesCreate:
    def test_delegates_to_underlying(self):
        fake_response = Response(
            id="resp_1",
            model="phi-4-mini",
            output=[TextOutput(text="Hello world")],
            finish_reason="stop",
        )
        mock_responses = MagicMock()
        mock_responses.create = AsyncMock(return_value=fake_response)

        facade = FacadeResponses(mock_responses)
        result = asyncio.run(facade.create(model="phi-4-mini", input="hi"))

        assert result is fake_response
        mock_responses.create.assert_called_once()
        request = mock_responses.create.call_args[0][0]
        assert request.model == "phi-4-mini"


class TestFacadeResponsesStream:
    def test_delegates_to_underlying(self):
        from octomil.responses.types import DoneEvent, TextDeltaEvent

        fake_done = DoneEvent(
            response=Response(
                id="resp_s1",
                model="phi-4-mini",
                output=[TextOutput(text="AB")],
                finish_reason="stop",
            )
        )

        async def _fake_stream(request):
            yield TextDeltaEvent(delta="A")
            yield TextDeltaEvent(delta="B")
            yield fake_done

        mock_responses = MagicMock()
        mock_responses.stream = _fake_stream

        facade = FacadeResponses(mock_responses)

        async def _run():
            events = []
            async for event in facade.stream(model="phi-4-mini", input="hi"):
                events.append(event)
            return events

        events = asyncio.run(_run())
        assert len(events) == 3
        assert isinstance(events[0], TextDeltaEvent)
        assert isinstance(events[2], DoneEvent)


# ---------------------------------------------------------------------------
# Response.output_text
# ---------------------------------------------------------------------------


class TestOutputText:
    def test_concatenates_text_items(self):
        resp = Response(
            id="r1",
            model="m",
            output=[TextOutput(text="Hello "), TextOutput(text="world")],
            finish_reason="stop",
        )
        assert resp.output_text == "Hello world"

    def test_empty_when_no_text_items(self):
        resp = Response(
            id="r2",
            model="m",
            output=[ToolCallOutput(tool_call=ResponseToolCall(id="tc1", name="fn", arguments="{}"))],
            finish_reason="tool_calls",
        )
        assert resp.output_text == ""

    def test_empty_output_list(self):
        resp = Response(
            id="r3",
            model="m",
            output=[],
            finish_reason="stop",
        )
        assert resp.output_text == ""

    def test_mixed_output_types(self):
        resp = Response(
            id="r4",
            model="m",
            output=[
                TextOutput(text="part1"),
                ToolCallOutput(tool_call=ResponseToolCall(id="tc1", name="fn", arguments="{}")),
                TextOutput(text="part2"),
            ],
            finish_reason="stop",
        )
        assert resp.output_text == "part1part2"


# ---------------------------------------------------------------------------
# Embeddings namespace
# ---------------------------------------------------------------------------


class TestEmbeddingsNamespace:
    def test_embeddings_namespace_exists(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        asyncio.run(client.initialize())
        assert isinstance(client.embeddings, FacadeEmbeddings)

    def test_embeddings_before_init_raises(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        with pytest.raises(OctomilNotInitializedError):
            _ = client.embeddings


class TestEmbeddingsCreate:
    def test_embeddings_create_delegates(self):
        fake_result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="nomic-embed-text-v1.5",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )
        client = Octomil(publishable_key="oct_pub_test_abc")
        asyncio.run(client.initialize())

        with patch.object(client._client, "embed", return_value=fake_result) as mock_embed:
            result = asyncio.run(
                client.embeddings.create(
                    model="nomic-embed-text-v1.5",
                    input="On-device AI inference at scale",
                )
            )

        assert result is fake_result
        mock_embed.assert_called_once_with(
            "nomic-embed-text-v1.5",
            "On-device AI inference at scale",
            timeout=30.0,
        )

    def test_embeddings_create_batch_input(self):
        fake_result = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="nomic-embed-text-v1.5",
            usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
        )
        client = Octomil(publishable_key="oct_pub_test_abc")
        asyncio.run(client.initialize())

        with patch.object(client._client, "embed", return_value=fake_result) as mock_embed:
            result = asyncio.run(
                client.embeddings.create(
                    model="nomic-embed-text-v1.5",
                    input=["hello", "world"],
                )
            )

        assert result is fake_result
        assert len(result.embeddings) == 2
        mock_embed.assert_called_once_with(
            "nomic-embed-text-v1.5",
            ["hello", "world"],
            timeout=30.0,
        )
