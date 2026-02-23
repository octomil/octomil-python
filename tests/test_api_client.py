import unittest
from unittest.mock import patch

from octomil.api_client import OctomilClientError, _ApiClient


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
    """Fake httpx.Client that supports both legacy per-method calls and .request()."""

    def __init__(self, response: _FakeResponse, *args, **kwargs):
        self._response = response
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @property
    def is_closed(self):
        return self._closed

    def close(self):
        self._closed = True

    def request(self, method, url, **kwargs):
        return self._response

    def get(self, url, params=None, headers=None):
        return self._response

    def post(self, url, json=None, headers=None, content=None):
        return self._response

    def put(self, url, json=None, headers=None):
        return self._response

    def patch(self, url, json=None, headers=None):
        return self._response

    def delete(self, url, params=None, headers=None):
        return self._response


def _patch_httpx(response):
    """Patch httpx.Client to return a _FakeHttpxClient with the given response."""
    return patch(
        "octomil.api_client.httpx.Client",
        lambda **kwargs: _FakeHttpxClient(response, **kwargs),
    )


class ApiClientTests(unittest.TestCase):
    def setUp(self):
        """Reset client state between tests."""
        pass

    def _make_client(self, **kwargs):
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
            **kwargs,
        )
        # Reset any pooled connection so each test gets a fresh client
        client.close()
        return client

    def test_init(self):
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com/",
            timeout=30.0,
        )
        self.assertEqual(client.api_base, "https://api.example.com")
        self.assertEqual(client.timeout, 30.0)

    def test_init_defaults(self):
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        self.assertEqual(client.timeout, 20.0)
        self.assertEqual(client.download_timeout, 120.0)
        self.assertEqual(client.max_retries, 3)
        self.assertEqual(client.backoff_base, 0.5)

    def test_headers_with_valid_token(self):
        client = self._make_client()
        headers = client._headers()
        self.assertEqual(headers["Authorization"], "Bearer token123")

    def test_headers_with_empty_token_raises(self):
        client = _ApiClient(
            auth_token_provider=lambda: "",
            api_base="https://api.example.com",
        )
        with self.assertRaises(OctomilClientError) as ctx:
            client._headers()
        self.assertIn("empty token", str(ctx.exception))

    def test_get_success(self):
        response = _FakeResponse(200, {"result": "success"})
        client = self._make_client()
        with _patch_httpx(response):
            result = client.get("/path", params={"key": "value"})
        self.assertEqual(result["result"], "success")

    def test_get_error(self):
        response = _FakeResponse(404, text_data="Not found")
        client = self._make_client()
        with _patch_httpx(response):
            with self.assertRaises(OctomilClientError) as ctx:
                client.get("/path")
            self.assertIn("Not found", str(ctx.exception))

    def test_post_success(self):
        response = _FakeResponse(200, {"id": "123"})
        client = self._make_client()
        with _patch_httpx(response):
            result = client.post("/path", payload={"data": "value"})
        self.assertEqual(result["id"], "123")

    def test_post_error(self):
        response = _FakeResponse(400, text_data="Bad request")
        client = self._make_client()
        with _patch_httpx(response):
            with self.assertRaises(OctomilClientError) as ctx:
                client.post("/path", payload={})
            self.assertIn("Bad request", str(ctx.exception))

    def test_put_success_with_response(self):
        response = _FakeResponse(200, {"updated": True})
        client = self._make_client()
        with _patch_httpx(response):
            result = client.put("/path", payload={"field": "value"})
        self.assertTrue(result["updated"])

    def test_put_success_empty_response(self):
        response = _FakeResponse(200, text_data="")
        client = self._make_client()
        with _patch_httpx(response):
            result = client.put("/path")
        self.assertEqual(result, {})

    def test_put_error(self):
        response = _FakeResponse(403, text_data="Forbidden")
        client = self._make_client()
        with _patch_httpx(response):
            with self.assertRaises(OctomilClientError) as ctx:
                client.put("/path")
            self.assertIn("Forbidden", str(ctx.exception))

    def test_patch_success_with_response(self):
        response = _FakeResponse(200, {"patched": True})
        client = self._make_client()
        with _patch_httpx(response):
            result = client.patch("/path", payload={"field": "value"})
        self.assertTrue(result["patched"])

    def test_patch_success_empty_response(self):
        response = _FakeResponse(204, text_data="")
        client = self._make_client()
        with _patch_httpx(response):
            result = client.patch("/path")
        self.assertEqual(result, {})

    def test_patch_error(self):
        response = _FakeResponse(500, text_data="Server error")
        client = self._make_client()
        with _patch_httpx(response):
            with self.assertRaises(OctomilClientError) as ctx:
                client.patch("/path")
            self.assertIn("Server error", str(ctx.exception))

    def test_delete_success_with_response(self):
        response = _FakeResponse(200, {"deleted": True})
        client = self._make_client()
        with _patch_httpx(response):
            result = client.delete("/path", params={"id": "123"})
        self.assertTrue(result["deleted"])

    def test_delete_success_empty_response(self):
        response = _FakeResponse(204, text_data="")
        client = self._make_client()
        with _patch_httpx(response):
            result = client.delete("/path")
        self.assertEqual(result, {})

    def test_delete_error(self):
        response = _FakeResponse(404, text_data="Not found")
        client = self._make_client()
        with _patch_httpx(response):
            with self.assertRaises(OctomilClientError) as ctx:
                client.delete("/path")
            self.assertIn("Not found", str(ctx.exception))

    def test_get_bytes_success(self):
        response = _FakeResponse(200)
        response.content = b"binary data"
        client = self._make_client()
        with _patch_httpx(response):
            result = client.get_bytes("/path", params={"format": "bin"})
        self.assertEqual(result, b"binary data")

    def test_get_bytes_error(self):
        response = _FakeResponse(401, text_data="Unauthorized")
        client = self._make_client()
        with _patch_httpx(response):
            with self.assertRaises(OctomilClientError) as ctx:
                client.get_bytes("/path")
            self.assertIn("Unauthorized", str(ctx.exception))

    def test_close(self):
        client = self._make_client()
        response = _FakeResponse(200, {"ok": True})
        with _patch_httpx(response):
            client.get("/test")
            self.assertIsNotNone(client._client)
            client.close()
            self.assertIsNone(client._client)


class RetryTests(unittest.TestCase):
    """Tests for exponential backoff retry logic."""

    def _make_client(self, **kwargs):
        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
            backoff_base=0.0,  # No actual sleep in tests
            **kwargs,
        )
        client.close()
        return client

    def test_retry_on_502(self):
        """502 should be retried, succeeding on second attempt."""
        call_count = 0

        class _RetryClient(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _FakeResponse(502, text_data="Bad Gateway")
                return _FakeResponse(200, {"ok": True})

        client = self._make_client()
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _RetryClient(_FakeResponse(200), **kwargs),
        ):
            result = client.get("/test")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count, 2)

    def test_retry_on_503(self):
        """503 should be retried."""
        call_count = 0

        class _RetryClient(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return _FakeResponse(503, text_data="Service Unavailable")
                return _FakeResponse(200, {"ok": True})

        client = self._make_client()
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _RetryClient(_FakeResponse(200), **kwargs),
        ):
            result = client.get("/test")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count, 3)

    def test_retry_exhausted_raises(self):
        """After max_retries, should raise OctomilClientError."""

        class _AlwaysFail(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                return _FakeResponse(503, text_data="Always failing")

        client = self._make_client(max_retries=3)
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _AlwaysFail(_FakeResponse(503), **kwargs),
        ):
            with self.assertRaises(OctomilClientError) as ctx:
                client.get("/test")
            self.assertIn("Always failing", str(ctx.exception))

    def test_no_retry_on_400(self):
        """400 errors should not be retried."""
        call_count = 0

        class _ClientError(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                return _FakeResponse(400, text_data="Bad request")

        client = self._make_client()
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _ClientError(_FakeResponse(400), **kwargs),
        ):
            with self.assertRaises(OctomilClientError):
                client.get("/test")
        self.assertEqual(call_count, 1)

    def test_no_retry_on_404(self):
        """404 errors should not be retried."""
        call_count = 0

        class _NotFound(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                return _FakeResponse(404, text_data="Not found")

        client = self._make_client()
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _NotFound(_FakeResponse(404), **kwargs),
        ):
            with self.assertRaises(OctomilClientError):
                client.get("/test")
        self.assertEqual(call_count, 1)

    def test_retry_on_429(self):
        """429 (rate limited) should be retried."""
        call_count = 0

        class _RateLimited(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _FakeResponse(429, text_data="Too Many Requests")
                return _FakeResponse(200, {"ok": True})

        client = self._make_client()
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _RateLimited(_FakeResponse(200), **kwargs),
        ):
            result = client.get("/test")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count, 2)

    def test_retry_on_connection_error(self):
        """Connection errors should be retried."""
        import httpx as httpx_mod

        call_count = 0

        class _ConnError(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise httpx_mod.ConnectError("Connection refused")
                return _FakeResponse(200, {"ok": True})

        client = self._make_client()
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _ConnError(_FakeResponse(200), **kwargs),
        ):
            result = client.get("/test")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count, 2)

    def test_retry_on_timeout(self):
        """Timeout errors should be retried."""
        import httpx as httpx_mod

        call_count = 0

        class _Timeout(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise httpx_mod.ReadTimeout("Read timed out")
                return _FakeResponse(200, {"ok": True})

        client = self._make_client()
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _Timeout(_FakeResponse(200), **kwargs),
        ):
            result = client.get("/test")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count, 2)

    def test_connection_error_exhausted_raises(self):
        """Connection errors exhausting retries should raise OctomilClientError."""
        import httpx as httpx_mod

        class _AlwaysTimeout(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                raise httpx_mod.ConnectError("Connection refused")

        client = self._make_client(max_retries=2)
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _AlwaysTimeout(_FakeResponse(200), **kwargs),
        ):
            with self.assertRaises(OctomilClientError) as ctx:
                client.get("/test")
            self.assertIn("2 attempts", str(ctx.exception))

    def test_max_retries_configurable(self):
        """max_retries=1 should not retry at all."""
        call_count = 0

        class _Fail(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                return _FakeResponse(503, text_data="Fail")

        client = self._make_client(max_retries=1)
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _Fail(_FakeResponse(503), **kwargs),
        ):
            with self.assertRaises(OctomilClientError):
                client.get("/test")
        self.assertEqual(call_count, 1)

    def test_non_retryable_500_fails_immediately_on_last_attempt(self):
        """A plain 500 error should fail after exhausting retries (it's not in _RETRYABLE_STATUS_CODES)."""
        call_count = 0

        class _ServerError(_FakeHttpxClient):
            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                return _FakeResponse(500, text_data="Internal Server Error")

        client = self._make_client(max_retries=1)
        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _ServerError(_FakeResponse(500), **kwargs),
        ):
            with self.assertRaises(OctomilClientError) as ctx:
                client.get("/test")
            self.assertIn("Internal Server Error", str(ctx.exception))
        self.assertEqual(call_count, 1)


class ConnectionPoolingTests(unittest.TestCase):
    """Tests for connection pooling behavior."""

    def test_client_reused_across_requests(self):
        """Multiple requests should reuse the same httpx.Client."""
        _response = _FakeResponse(200, {"ok": True})
        create_count = 0
        original_init = _FakeHttpxClient.__init__

        class _TrackingClient(_FakeHttpxClient):
            def __init__(self, *args, **kwargs):
                nonlocal create_count
                create_count += 1
                original_init(self, _FakeResponse(200), **kwargs)

        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        client.close()

        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _TrackingClient(**kwargs),
        ):
            client.get("/path1")
            client.get("/path2")
            client.get("/path3")

        # Client should be created only once (pooled)
        self.assertEqual(create_count, 1)

    def test_client_recreated_after_close(self):
        """After close(), the next request should create a new client."""
        create_count = 0

        class _TrackingClient(_FakeHttpxClient):
            def __init__(self, *args, **kwargs):
                nonlocal create_count
                create_count += 1
                super().__init__(_FakeResponse(200), **kwargs)

        client = _ApiClient(
            auth_token_provider=lambda: "token123",
            api_base="https://api.example.com",
        )
        client.close()

        with patch(
            "octomil.api_client.httpx.Client",
            lambda **kwargs: _TrackingClient(**kwargs),
        ):
            client.get("/path1")
            client.close()
            client.get("/path2")

        self.assertEqual(create_count, 2)


if __name__ == "__main__":
    unittest.main()
