"""Tests for cloud streaming inference (SSE parsing + OctomilClient integration)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from octomil.streaming import (
    StreamToken,
    _build_payload,
    _parse_sse_line,
    _parse_sse_lines,
    stream_inference,
)


# ------------------------------------------------------------------
# StreamToken dataclass
# ------------------------------------------------------------------


class StreamTokenTests(unittest.TestCase):
    def test_defaults(self):
        tok = StreamToken(token="hello", done=False)
        self.assertEqual(tok.token, "hello")
        self.assertFalse(tok.done)
        self.assertIsNone(tok.provider)
        self.assertIsNone(tok.latency_ms)
        self.assertIsNone(tok.session_id)

    def test_all_fields(self):
        tok = StreamToken(
            token="world",
            done=True,
            provider="ollama",
            latency_ms=42.5,
            session_id="abc-123",
        )
        self.assertEqual(tok.token, "world")
        self.assertTrue(tok.done)
        self.assertEqual(tok.provider, "ollama")
        self.assertAlmostEqual(tok.latency_ms, 42.5)
        self.assertEqual(tok.session_id, "abc-123")


# ------------------------------------------------------------------
# _build_payload
# ------------------------------------------------------------------


class BuildPayloadTests(unittest.TestCase):
    def test_string_input(self):
        p = _build_payload("phi-4-mini", "Hello", None)
        self.assertEqual(p["model_id"], "phi-4-mini")
        self.assertEqual(p["input_data"], "Hello")
        self.assertNotIn("messages", p)
        self.assertNotIn("parameters", p)

    def test_messages_input(self):
        msgs = [{"role": "user", "content": "Hi"}]
        p = _build_payload("phi-4-mini", msgs, None)
        self.assertEqual(p["messages"], msgs)
        self.assertNotIn("input_data", p)

    def test_with_parameters(self):
        params = {"temperature": 0.7, "max_tokens": 512}
        p = _build_payload("phi-4-mini", "test", params)
        self.assertEqual(p["parameters"], params)

    def test_empty_parameters_omitted(self):
        p = _build_payload("phi-4-mini", "test", {})
        self.assertNotIn("parameters", p)


# ------------------------------------------------------------------
# _parse_sse_line
# ------------------------------------------------------------------


class ParseSSELineTests(unittest.TestCase):
    def test_normal_token(self):
        line = 'data: {"token": "The", "done": false, "provider": "ollama"}'
        tok = _parse_sse_line(line)
        self.assertIsNotNone(tok)
        self.assertEqual(tok.token, "The")
        self.assertFalse(tok.done)
        self.assertEqual(tok.provider, "ollama")

    def test_done_token(self):
        line = 'data: {"done": true, "latency_ms": 1234.5, "session_id": "abc-123"}'
        tok = _parse_sse_line(line)
        self.assertIsNotNone(tok)
        self.assertTrue(tok.done)
        self.assertEqual(tok.token, "")
        self.assertAlmostEqual(tok.latency_ms, 1234.5)
        self.assertEqual(tok.session_id, "abc-123")

    def test_empty_line_returns_none(self):
        self.assertIsNone(_parse_sse_line(""))
        self.assertIsNone(_parse_sse_line("   "))

    def test_non_data_line_returns_none(self):
        self.assertIsNone(_parse_sse_line("event: message"))
        self.assertIsNone(_parse_sse_line("id: 1"))
        self.assertIsNone(_parse_sse_line(": comment"))

    def test_empty_data_returns_none(self):
        self.assertIsNone(_parse_sse_line("data:"))
        self.assertIsNone(_parse_sse_line("data:   "))

    def test_invalid_json_returns_none(self):
        self.assertIsNone(_parse_sse_line("data: not-json"))

    def test_leading_trailing_whitespace(self):
        line = '  data: {"token": "x", "done": false}  '
        tok = _parse_sse_line(line)
        self.assertIsNotNone(tok)
        self.assertEqual(tok.token, "x")


# ------------------------------------------------------------------
# _parse_sse_lines
# ------------------------------------------------------------------


class ParseSSELinesTests(unittest.TestCase):
    def test_multiple_lines(self):
        lines = [
            'data: {"token": "The", "done": false}',
            'data: {"token": " answer", "done": false}',
            "",
            'data: {"done": true, "latency_ms": 100.0, "session_id": "s1"}',
        ]
        tokens = list(_parse_sse_lines(iter(lines)))
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0].token, "The")
        self.assertEqual(tokens[1].token, " answer")
        self.assertTrue(tokens[2].done)

    def test_empty_stream(self):
        tokens = list(_parse_sse_lines(iter([])))
        self.assertEqual(tokens, [])


# ------------------------------------------------------------------
# stream_inference (mocked HTTP)
# ------------------------------------------------------------------


class StreamInferenceTests(unittest.TestCase):
    def test_stream_inference_yields_tokens(self):
        sse_lines = [
            'data: {"token": "Hello", "done": false, "provider": "ollama"}',
            'data: {"token": " world", "done": false, "provider": "ollama"}',
            'data: {"done": true, "latency_ms": 50.0, "session_id": "sess-1"}',
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = iter(sse_lines)
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("octomil.streaming.httpx.Client", return_value=mock_client):
            tokens = list(
                stream_inference(
                    server_url="https://api.octomil.com/api/v1",
                    api_key="test-key",
                    model_id="phi-4-mini",
                    input_data="Hello",
                )
            )

        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0].token, "Hello")
        self.assertFalse(tokens[0].done)
        self.assertEqual(tokens[1].token, " world")
        self.assertTrue(tokens[2].done)
        self.assertAlmostEqual(tokens[2].latency_ms, 50.0)

    def test_stream_inference_sends_correct_request(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = iter([])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("octomil.streaming.httpx.Client", return_value=mock_client):
            list(
                stream_inference(
                    server_url="https://api.octomil.com/api/v1/",
                    api_key="my-key",
                    model_id="phi-4-mini",
                    input_data="test prompt",
                    parameters={"temperature": 0.5},
                )
            )

        mock_client.stream.assert_called_once()
        call_args = mock_client.stream.call_args
        self.assertEqual(call_args[0][0], "POST")
        self.assertEqual(
            call_args[0][1], "https://api.octomil.com/api/v1/inference/stream"
        )
        self.assertEqual(call_args[1]["json"]["model_id"], "phi-4-mini")
        self.assertEqual(call_args[1]["json"]["input_data"], "test prompt")
        self.assertEqual(call_args[1]["json"]["parameters"]["temperature"], 0.5)
        self.assertEqual(call_args[1]["headers"]["Authorization"], "Bearer my-key")

    def test_stream_inference_with_messages(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = iter([])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        msgs = [{"role": "user", "content": "Hi"}]
        with patch("octomil.streaming.httpx.Client", return_value=mock_client):
            list(
                stream_inference(
                    server_url="https://api.octomil.com/api/v1",
                    api_key="key",
                    model_id="phi-4-mini",
                    input_data=msgs,
                )
            )

        call_json = mock_client.stream.call_args[1]["json"]
        self.assertEqual(call_json["messages"], msgs)
        self.assertNotIn("input_data", call_json)


# ------------------------------------------------------------------
# stream_inference_async (mocked HTTP)
# ------------------------------------------------------------------


class StreamInferenceAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_inference_async_yields_tokens(self):
        from unittest.mock import AsyncMock

        from octomil.streaming import stream_inference_async

        sse_lines = [
            'data: {"token": "Async", "done": false}',
            'data: {"token": " test", "done": false}',
            'data: {"done": true, "session_id": "s2"}',
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream_ctx

        mock_client_ctx = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("octomil.streaming.httpx.AsyncClient", return_value=mock_client_ctx):
            tokens = []
            async for tok in stream_inference_async(
                server_url="https://api.octomil.com/api/v1",
                api_key="key",
                model_id="phi-4-mini",
                input_data="Hello",
            ):
                tokens.append(tok)

        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0].token, "Async")
        self.assertTrue(tokens[2].done)


# ------------------------------------------------------------------
# OctomilClient.stream_predict integration (mocked)
# ------------------------------------------------------------------


class ClientStreamPredictTests(unittest.TestCase):
    def test_stream_predict_delegates_to_stream_inference(self):
        from octomil.client import OctomilClient

        expected_tokens = [
            StreamToken(token="Hi", done=False),
            StreamToken(token="", done=True, session_id="s1"),
        ]

        with patch(
            "octomil.streaming.stream_inference", return_value=iter(expected_tokens)
        ) as mock_fn:
            client = OctomilClient(
                api_key="test-key", api_base="https://api.test.com/api/v1"
            )
            tokens = list(
                client.stream_predict(
                    "phi-4-mini",
                    "Hello",
                    parameters={"temperature": 0.7},
                )
            )

        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].token, "Hi")
        mock_fn.assert_called_once_with(
            server_url="https://api.test.com/api/v1",
            api_key="test-key",
            model_id="phi-4-mini",
            input_data="Hello",
            parameters={"temperature": 0.7},
            timeout=120.0,
        )


class ClientStreamPredictAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_predict_async_delegates(self):
        from octomil.client import OctomilClient

        expected_tokens = [
            StreamToken(token="Async", done=False),
            StreamToken(token="", done=True),
        ]

        async def fake_stream_async(**kwargs):
            for tok in expected_tokens:
                yield tok

        with patch(
            "octomil.streaming.stream_inference_async",
            side_effect=lambda **kwargs: fake_stream_async(**kwargs),
        ):
            client = OctomilClient(
                api_key="test-key", api_base="https://api.test.com/api/v1"
            )
            tokens = []
            async for tok in client.stream_predict_async("phi-4-mini", "Hello"):
                tokens.append(tok)

        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].token, "Async")


if __name__ == "__main__":
    unittest.main()
