import unittest
from unittest.mock import patch

from edgeml.api_client import EdgeMLClientError, _ApiClient


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict = None, text_data: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        if text_data:
            self.text = text_data
        elif json_data:
            self.text = str(json_data)
        else:
            self.text = ""
        self.content = b""

    def json(self):
        return self._json_data


class _FakeHttpxClient:
    def __init__(self, response: _FakeResponse, *args, **kwargs):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def get(self, url, params=None, headers=None):
        return self._response

    def post(self, url, json=None, headers=None):
        return self._response

    def put(self, url, json=None, headers=None):
        return self._response

    def patch(self, url, json=None, headers=None):
        return self._response

    def delete(self, url, params=None, headers=None):
        return self._response


class ApiClientTests(unittest.TestCase):
    def test_init(self):
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com/",
            timeout=30.0,
        )
        self.assertEqual(client.api_base, "https://api.example.com")
        self.assertEqual(client.timeout, 30.0)

    def test_headers_with_valid_token(self):
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        headers = client._headers()
        self.assertEqual(headers["Authorization"], "Bearer token123")

    def test_headers_with_empty_token_raises(self):
        client = _ApiClient(
            auth_token_provider=lambda: "",
            api_base="https://api.example.com",
        )
        with self.assertRaises(EdgeMLClientError) as ctx:
            client._headers()
        self.assertIn("empty token", str(ctx.exception))

    def test_get_success(self):
        response = _FakeResponse(200, {"result": "success"})
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            result = client.get("/path", params={"key": "value"})
        self.assertEqual(result["result"], "success")

    def test_get_error(self):
        response = _FakeResponse(404, text_data="Not found")
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            with self.assertRaises(EdgeMLClientError) as ctx:
                client.get("/path")
            self.assertIn("Not found", str(ctx.exception))

    def test_post_success(self):
        response = _FakeResponse(200, {"id": "123"})
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            result = client.post("/path", payload={"data": "value"})
        self.assertEqual(result["id"], "123")

    def test_post_error(self):
        response = _FakeResponse(400, text_data="Bad request")
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            with self.assertRaises(EdgeMLClientError) as ctx:
                client.post("/path", payload={})
            self.assertIn("Bad request", str(ctx.exception))

    def test_put_success_with_response(self):
        response = _FakeResponse(200, {"updated": True})
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            result = client.put("/path", payload={"field": "value"})
        self.assertTrue(result["updated"])

    def test_put_success_empty_response(self):
        response = _FakeResponse(200, text_data="")
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            result = client.put("/path")
        self.assertEqual(result, {})

    def test_put_error(self):
        response = _FakeResponse(403, text_data="Forbidden")
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            with self.assertRaises(EdgeMLClientError) as ctx:
                client.put("/path")
            self.assertIn("Forbidden", str(ctx.exception))

    def test_patch_success_with_response(self):
        response = _FakeResponse(200, {"patched": True})
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            result = client.patch("/path", payload={"field": "value"})
        self.assertTrue(result["patched"])

    def test_patch_success_empty_response(self):
        response = _FakeResponse(204, text_data="")
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            result = client.patch("/path")
        self.assertEqual(result, {})

    def test_patch_error(self):
        response = _FakeResponse(500, text_data="Server error")
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            with self.assertRaises(EdgeMLClientError) as ctx:
                client.patch("/path")
            self.assertIn("Server error", str(ctx.exception))

    def test_delete_success_with_response(self):
        response = _FakeResponse(200, {"deleted": True})
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            result = client.delete("/path", params={"id": "123"})
        self.assertTrue(result["deleted"])

    def test_delete_success_empty_response(self):
        response = _FakeResponse(204, text_data="")
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            result = client.delete("/path")
        self.assertEqual(result, {})

    def test_delete_error(self):
        response = _FakeResponse(404, text_data="Not found")
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            with self.assertRaises(EdgeMLClientError) as ctx:
                client.delete("/path")
            self.assertIn("Not found", str(ctx.exception))

    def test_get_bytes_success(self):
        response = _FakeResponse(200)
        response.content = b"binary data"
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            result = client.get_bytes("/path", params={"format": "bin"})
        self.assertEqual(result, b"binary data")

    def test_get_bytes_error(self):
        response = _FakeResponse(401, text_data="Unauthorized")
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        with patch("edgeml.api_client.httpx.Client", lambda timeout: _FakeHttpxClient(response, timeout=timeout)):
            with self.assertRaises(EdgeMLClientError) as ctx:
                client.get_bytes("/path")
            self.assertIn("Unauthorized", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
