import tempfile
import unittest
from unittest.mock import Mock, patch

from edgeml.api_client import EdgeMLClientError
from edgeml.registry import ModelRegistry


class _StubApi:
    def __init__(self):
        self.calls = []
        self._responses = {}
        self.timeout = 60.0  # Add timeout attribute
        self.api_base = "https://api.edgeml.io/api/v1"

    def set_response(self, key, response):
        self._responses[key] = response

    def get(self, path, params=None):
        self.calls.append(("get", path, params))
        key = (path, tuple(sorted(params.items())) if params else None)
        if key in self._responses:
            return self._responses[key]
        return {"models": [], "versions": []}

    def post(self, path, payload=None):
        self.calls.append(("post", path, payload))
        if path.startswith("/models") and "versions" in path and not path.endswith("/versions"):
            return {"version": "1.0.0"}
        return {"id": "model_123", "version": "1.0.0"}

    def patch(self, path, payload=None):
        self.calls.append(("patch", path, payload))
        return {"updated": True}

    def delete(self, path, params=None):
        self.calls.append(("delete", path, params))
        return {"deleted": True}

    def _headers(self):
        return {"Authorization": "Bearer token123"}


class ModelRegistryTests(unittest.TestCase):
    def test_init(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123", org_id="org_1")
        self.assertEqual(registry.org_id, "org_1")
        self.assertIsNotNone(registry.api)
        self.assertIsNotNone(registry.rollouts)
        self.assertIsNotNone(registry.experiments)

    def test_resolve_model_id_by_name(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123", org_id="org_1")
        stub = _StubApi()
        stub.set_response(("/models", (("org_id", "org_1"),)), {"models": [{"name": "my_model", "id": "model_456"}]})
        registry.api = stub
        result = registry.resolve_model_id("my_model")
        self.assertEqual(result, "model_456")

    def test_resolve_model_id_returns_input_if_not_found(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123", org_id="org_1")
        stub = _StubApi()
        stub.set_response(("/models", (("org_id", "org_1"),)), {"models": []})
        registry.api = stub
        result = registry.resolve_model_id("unknown_model")
        self.assertEqual(result, "unknown_model")

    def test_get_latest_version_info(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        stub.set_response(("/models/model_1/versions/latest", None), {"version": "2.0.0", "status": "published"})
        registry.api = stub
        result = registry.get_latest_version_info("model_1")
        self.assertEqual(result["version"], "2.0.0")

    def test_list_models_with_filters(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123", org_id="org_1")
        stub = _StubApi()
        registry.api = stub
        registry.list_models(framework="pytorch", use_case="classification", limit=50, offset=10)
        method, path, params = stub.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/models")
        self.assertEqual(params["framework"], "pytorch")
        self.assertEqual(params["use_case"], "classification")
        self.assertEqual(params["limit"], 50)
        self.assertEqual(params["offset"], 10)

    def test_list_models_without_filters(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123", org_id="org_1")
        stub = _StubApi()
        registry.api = stub
        registry.list_models()
        method, path, params = stub.calls[-1]
        self.assertNotIn("framework", params)
        self.assertNotIn("use_case", params)

    def test_get_model(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.get_model("model_1")
        method, path, params = stub.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/models/model_1")

    def test_update_model(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.update_model("model_1", name="new_name", description="new_desc")
        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "patch")
        self.assertEqual(path, "/models/model_1")
        self.assertEqual(payload["name"], "new_name")
        self.assertEqual(payload["description"], "new_desc")

    def test_delete_model(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.delete_model("model_1")
        method, path, params = stub.calls[-1]
        self.assertEqual(method, "delete")
        self.assertEqual(path, "/models/model_1")

    def test_get_latest_version(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        stub.set_response(("/models/model_1/versions/latest", None), {"version": "3.1.0"})
        registry.api = stub
        result = registry.get_latest_version("model_1")
        self.assertEqual(result, "3.1.0")

    def test_get_latest_version_raises_if_not_found(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        stub.set_response(("/models/model_1/versions/latest", None), {})
        registry.api = stub
        with self.assertRaises(EdgeMLClientError) as ctx:
            registry.get_latest_version("model_1")
        self.assertIn("Latest version not found", str(ctx.exception))

    def test_list_versions_with_filter(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.list_versions("model_1", status_filter="published", limit=20, offset=5)
        method, path, params = stub.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/models/model_1/versions")
        self.assertEqual(params["status"], "published")
        self.assertEqual(params["limit"], 20)
        self.assertEqual(params["offset"], 5)

    def test_list_versions_without_filter(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.list_versions("model_1")
        method, path, params = stub.calls[-1]
        self.assertNotIn("status", params)

    def test_get_version(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.get_version("model_1", "1.0.0")
        method, path, params = stub.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(path, "/models/model_1/versions/1.0.0")

    def test_create_version(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        result = registry.create_version(
            model_id="model_1",
            version="2.0.0",
            storage_path="s3://bucket/model.onnx",
            fmt="onnx",
            checksum="abc123",
            size_bytes=1024,
            description="test version",
            metrics={"accuracy": 0.95},
        )
        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/versions")
        self.assertEqual(payload["version"], "2.0.0")
        self.assertEqual(payload["storage_path"], "s3://bucket/model.onnx")
        self.assertEqual(payload["format"], "onnx")
        self.assertEqual(payload["checksum"], "abc123")
        self.assertEqual(payload["size_bytes"], 1024)
        self.assertEqual(payload["description"], "test version")
        self.assertEqual(payload["metrics"], {"accuracy": 0.95})

    def test_create_version_without_optional_fields(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.create_version(
            model_id="model_1",
            version="2.0.0",
            storage_path="s3://bucket/model.onnx",
            fmt="onnx",
            checksum="abc123",
            size_bytes=1024,
        )
        method, path, payload = stub.calls[-1]
        self.assertNotIn("description", payload)
        self.assertNotIn("metrics", payload)

    def test_get_download_url(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        stub.set_response(("/models/model_1/versions/1.0.0/download-url", (("format", "onnx"),)), {"url": "https://download.com/model.onnx"})
        registry.api = stub
        result = registry.get_download_url("model_1", "1.0.0", fmt="onnx")
        method, path, params = stub.calls[-1]
        self.assertEqual(method, "get")
        self.assertEqual(params["format"], "onnx")

    def test_download_version(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        stub.set_response(("/models/model_1/versions/1.0.0/download-url", (("format", "onnx"),)), {"url": "https://download.com/model.onnx"})
        registry.api = stub

        class _FakeResponse:
            status_code = 200
            content = b"model data"

        class _FakeHttpxClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def get(self, url):
                return _FakeResponse()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        with patch("edgeml.registry.httpx.Client", _FakeHttpxClient):
            result = registry.download_version("model_1", "1.0.0", "onnx", tmp_path)

        self.assertEqual(result, tmp_path)
        with open(tmp_path, "rb") as f:
            self.assertEqual(f.read(), b"model data")

        import os
        os.unlink(tmp_path)

    def test_download_version_missing_url_raises(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        stub.set_response(("/models/model_1/versions/1.0.0/download-url", (("format", "onnx"),)), {})
        registry.api = stub
        with self.assertRaises(EdgeMLClientError) as ctx:
            registry.download_version("model_1", "1.0.0", "onnx", "/tmp/model.onnx")
        self.assertIn("Download URL missing", str(ctx.exception))

    def test_download_version_http_error_raises(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        stub.set_response(("/models/model_1/versions/1.0.0/download-url", (("format", "onnx"),)), {"url": "https://download.com/model.onnx"})
        registry.api = stub

        class _FakeResponse:
            status_code = 404
            text = "Not found"

        class _FakeHttpxClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def get(self, url):
                return _FakeResponse()

        with patch("edgeml.registry.httpx.Client", _FakeHttpxClient):
            with self.assertRaises(EdgeMLClientError) as ctx:
                registry.download_version("model_1", "1.0.0", "onnx", "/tmp/model.onnx")
        self.assertIn("Not found", str(ctx.exception))

    def test_ensure_model_creates_if_not_exists(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123", org_id="org_1")
        stub = _StubApi()
        stub.set_response(("/models", (("org_id", "org_1"),)), {"models": []})
        registry.api = stub
        result = registry.ensure_model(
            name="new_model",
            framework="pytorch",
            use_case="detection",
            description="test model",
            model_contract={"input": "tensor"},
            data_contract={"format": "json"},
        )
        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(payload["name"], "new_model")
        self.assertEqual(payload["framework"], "pytorch")
        self.assertEqual(payload["use_case"], "detection")
        self.assertEqual(payload["description"], "test model")
        self.assertEqual(payload["model_contract"], {"input": "tensor"})
        self.assertEqual(payload["data_contract"], {"format": "json"})

    def test_ensure_model_returns_existing(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123", org_id="org_1")
        stub = _StubApi()
        existing_model = {"name": "existing_model", "id": "model_789"}
        stub.set_response(("/models", (("org_id", "org_1"),)), {"models": [existing_model]})
        registry.api = stub
        result = registry.ensure_model(name="existing_model", framework="pytorch", use_case="detection")
        self.assertEqual(result, existing_model)
        self.assertEqual(len(stub.calls), 1)  # Only GET, no POST

    def test_ensure_model_without_contracts(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123", org_id="org_1")
        stub = _StubApi()
        stub.set_response(("/models", (("org_id", "org_1"),)), {"models": []})
        registry.api = stub
        registry.ensure_model(name="new_model", framework="pytorch", use_case="detection")
        method, path, payload = stub.calls[-1]
        self.assertNotIn("model_contract", payload)
        self.assertNotIn("data_contract", payload)

    def test_publish_version(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.publish_version("model_1", "1.0.0")
        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/versions/1.0.0/publish")

    def test_deprecate_version(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.deprecate_version("model_1", "1.0.0")
        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/versions/1.0.0/deprecate")

    def test_update_version_metrics(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.update_version_metrics("model_1", "1.0.0", {"accuracy": 0.99})
        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "patch")
        self.assertEqual(path, "/models/model_1/versions/1.0.0/metrics")
        self.assertEqual(payload["metrics"], {"accuracy": 0.99})

    def test_delete_version(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        registry.delete_version("model_1", "1.0.0")
        method, path, params = stub.calls[-1]
        self.assertEqual(method, "delete")
        self.assertEqual(path, "/models/model_1/versions/1.0.0")

    def test_create_rollout(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub
        from edgeml.control_plane import RolloutsAPI
        registry.rollouts = RolloutsAPI(stub)
        registry.create_rollout(
            model_id="model_1",
            version="1.0.0",
            rollout_percentage=20,
            target_percentage=80,
            increment_step=15,
            start_immediately=False,
        )
        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/models/model_1/rollouts")
        self.assertEqual(payload["rollout_percentage"], 20.0)

    def test_upload_version_from_path(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub

        class _FakeResponse:
            status_code = 200
            text = ""
            def json(self):
                return {"version": "1.0.0", "id": "version_123"}

        class _FakeHttpxClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def post(self, url, data=None, files=None, headers=None):
                return _FakeResponse()

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp:
            tmp.write(b"model data")
            tmp_path = tmp.name

        try:
            with patch("edgeml.registry.httpx.Client", _FakeHttpxClient):
                result = registry.upload_version_from_path(
                    model_id="model_1",
                    file_path=tmp_path,
                    version="1.0.0",
                    description="test upload",
                    formats="onnx",
                    architecture="resnet",
                    input_dim=224,
                    hidden_dim=512,
                    output_dim=10,
                )
            self.assertEqual(result["version"], "1.0.0")
        finally:
            import os
            os.unlink(tmp_path)

    def test_upload_version_from_path_with_onnx_data(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub

        class _FakeResponse:
            status_code = 200
            text = ""
            def json(self):
                return {"version": "1.0.0"}

        class _FakeHttpxClient:
            def __init__(self, *args, **kwargs):
                self.post_called_with_files = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def post(self, url, data=None, files=None, headers=None):
                self.post_called_with_files = files
                return _FakeResponse()

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp:
            tmp.write(b"model data")
            tmp_path = tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx.data") as tmp2:
            tmp2.write(b"onnx data")
            tmp2_path = tmp2.name

        try:
            with patch("edgeml.registry.httpx.Client", _FakeHttpxClient):
                result = registry.upload_version_from_path(
                    model_id="model_1",
                    file_path=tmp_path,
                    version="1.0.0",
                    onnx_data_path=tmp2_path,
                )
            self.assertEqual(result["version"], "1.0.0")
        finally:
            import os
            os.unlink(tmp_path)
            os.unlink(tmp2_path)

    def test_upload_version_from_path_http_error(self):
        registry = ModelRegistry(auth_token_provider=lambda: "token123")
        stub = _StubApi()
        registry.api = stub

        class _FakeResponse:
            status_code = 400
            text = "Bad request"

        class _FakeHttpxClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def post(self, url, data=None, files=None, headers=None):
                return _FakeResponse()

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp:
            tmp.write(b"model data")
            tmp_path = tmp.name

        try:
            with patch("edgeml.registry.httpx.Client", _FakeHttpxClient):
                with self.assertRaises(EdgeMLClientError) as ctx:
                    registry.upload_version_from_path(
                        model_id="model_1",
                        file_path=tmp_path,
                        version="1.0.0",
                    )
                self.assertIn("Bad request", str(ctx.exception))
        finally:
            import os
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
