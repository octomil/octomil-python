"""Tests for cloud embeddings (embed function + Client integration)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from octomil.embeddings import (
    EmbeddingResult,
    EmbeddingUsage,
    embed,
)


# ------------------------------------------------------------------
# EmbeddingResult / EmbeddingUsage dataclasses
# ------------------------------------------------------------------


class EmbeddingUsageTests(unittest.TestCase):
    def test_fields(self):
        u = EmbeddingUsage(prompt_tokens=5, total_tokens=5)
        self.assertEqual(u.prompt_tokens, 5)
        self.assertEqual(u.total_tokens, 5)


class EmbeddingResultTests(unittest.TestCase):
    def test_fields(self):
        r = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="nomic-embed-text",
            usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
        )
        self.assertEqual(len(r.embeddings), 2)
        self.assertEqual(r.model, "nomic-embed-text")
        self.assertEqual(r.usage.prompt_tokens, 10)


# ------------------------------------------------------------------
# embed() — validation
# ------------------------------------------------------------------


class EmbedValidationTests(unittest.TestCase):
    def test_empty_server_url_raises(self):
        with self.assertRaises(ValueError, msg="server_url"):
            embed(server_url="", api_key="key", model_id="m", input="hi")

    def test_empty_api_key_raises(self):
        with self.assertRaises(ValueError, msg="api_key"):
            embed(
                server_url="https://api.test.com/api/v1",
                api_key="",
                model_id="m",
                input="hi",
            )


# ------------------------------------------------------------------
# embed() — mocked HTTP
# ------------------------------------------------------------------


class EmbedTests(unittest.TestCase):
    def _mock_response(self, json_data: dict, status_code: int = 200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.raise_for_status = MagicMock()
        resp.json.return_value = json_data
        return resp

    def test_embed_single_string(self):
        api_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "nomic-embed-text",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        mock_client = MagicMock()
        mock_client.post.return_value = self._mock_response(api_response)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("octomil.embeddings.httpx.Client", return_value=mock_client):
            result = embed(
                server_url="https://api.octomil.com/api/v1",
                api_key="test-key",
                model_id="nomic-embed-text",
                input="hello world",
            )

        self.assertEqual(len(result.embeddings), 1)
        self.assertEqual(result.embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(result.model, "nomic-embed-text")
        self.assertEqual(result.usage.prompt_tokens, 5)
        self.assertEqual(result.usage.total_tokens, 5)

    def test_embed_multiple_strings(self):
        api_response = {
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ],
            "model": "nomic-embed-text",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        mock_client = MagicMock()
        mock_client.post.return_value = self._mock_response(api_response)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("octomil.embeddings.httpx.Client", return_value=mock_client):
            result = embed(
                server_url="https://api.octomil.com/api/v1",
                api_key="test-key",
                model_id="nomic-embed-text",
                input=["hello", "world"],
            )

        self.assertEqual(len(result.embeddings), 2)
        self.assertEqual(result.embeddings[0], [0.1, 0.2])
        self.assertEqual(result.embeddings[1], [0.3, 0.4])

    def test_embed_sends_correct_request(self):
        api_response = {
            "data": [{"embedding": [0.1], "index": 0}],
            "model": "nomic-embed-text",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

        mock_client = MagicMock()
        mock_client.post.return_value = self._mock_response(api_response)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("octomil.embeddings.httpx.Client", return_value=mock_client):
            embed(
                server_url="https://api.octomil.com/api/v1/",
                api_key="my-key",
                model_id="nomic-embed-text",
                input="test",
            )

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        self.assertEqual(call_args[1]["json"]["model_id"], "nomic-embed-text")
        self.assertEqual(call_args[1]["json"]["input"], "test")
        self.assertIn("Bearer my-key", call_args[1]["headers"]["Authorization"])
        # URL should strip trailing slash
        self.assertEqual(call_args[0][0], "https://api.octomil.com/api/v1/embeddings")

    def test_embed_missing_usage_defaults_to_zero(self):
        api_response = {
            "data": [{"embedding": [0.1], "index": 0}],
            "model": "nomic-embed-text",
        }

        mock_client = MagicMock()
        mock_client.post.return_value = self._mock_response(api_response)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("octomil.embeddings.httpx.Client", return_value=mock_client):
            result = embed(
                server_url="https://api.octomil.com/api/v1",
                api_key="key",
                model_id="nomic-embed-text",
                input="test",
            )

        self.assertEqual(result.usage.prompt_tokens, 0)
        self.assertEqual(result.usage.total_tokens, 0)


# ------------------------------------------------------------------
# Client.embed integration (mocked)
# ------------------------------------------------------------------


class ClientEmbedTests(unittest.TestCase):
    def test_client_embed_delegates_to_embed_function(self):
        from octomil.client import Client

        expected = EmbeddingResult(
            embeddings=[[0.1, 0.2]],
            model="nomic-embed-text",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )

        with patch("octomil.embeddings.embed", return_value=expected) as mock_fn:
            client = Client(api_key="test-key", api_base="https://api.test.com/api/v1")
            result = client.embed("nomic-embed-text", "hello world")

        self.assertEqual(result.embeddings, [[0.1, 0.2]])
        mock_fn.assert_called_once_with(
            server_url="https://api.test.com/api/v1",
            api_key="test-key",
            model_id="nomic-embed-text",
            input="hello world",
            timeout=30.0,
        )

    def test_client_embed_with_list_input(self):
        from octomil.client import Client

        expected = EmbeddingResult(
            embeddings=[[0.1], [0.2]],
            model="nomic-embed-text",
            usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
        )

        with patch("octomil.embeddings.embed", return_value=expected) as mock_fn:
            client = Client(api_key="test-key", api_base="https://api.test.com/api/v1")
            result = client.embed("nomic-embed-text", ["hello", "world"], timeout=60.0)

        self.assertEqual(len(result.embeddings), 2)
        mock_fn.assert_called_once_with(
            server_url="https://api.test.com/api/v1",
            api_key="test-key",
            model_id="nomic-embed-text",
            input=["hello", "world"],
            timeout=60.0,
        )


if __name__ == "__main__":
    unittest.main()
