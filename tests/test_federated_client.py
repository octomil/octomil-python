import unittest
import tempfile
import io
from unittest.mock import Mock, patch
import numpy as np

from edgeml.api_client import EdgeMLClientError
from edgeml.federated_client import FederatedClient, compute_state_dict_delta


class _StubApi:
    def __init__(self):
        self.calls = []
        self._responses = {}

    def set_response(self, key, response):
        self._responses[key] = response

    def get(self, path, params=None):
        self.calls.append(("get", path, params))
        if path in self._responses:
            return self._responses[path]
        if "/models/" in path:
            return {"id": "model_1", "name": "test_model", "architecture": {"type": "neural_net"}}
        return {}

    def post(self, path, payload=None):
        self.calls.append(("post", path, payload))
        if path == "/devices/register":
            return {"id": "device_123"}
        if path == "/devices/heartbeat":
            return {}
        if "/updates" in path or "weights" in path:
            return {"update_id": "update_456", "status": "accepted"}
        return {}

    def get_bytes(self, path, params=None):
        self.calls.append(("get_bytes", path, params))
        # Return serialized torch state dict
        try:
            import torch
            state_dict = {"weight": torch.randn(3, 3), "bias": torch.randn(3)}
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            return buffer.getvalue()
        except ImportError:
            return b"model_weights_data"


class FederatedClientTests(unittest.TestCase):
    def test_init_with_device_identifier(self):
        client = FederatedClient(
            auth_token_provider=lambda: "token123",
            org_id="org_1",
            device_identifier="device_abc",
            platform="ios",
        )
        self.assertEqual(client.device_identifier, "device_abc")
        self.assertEqual(client.org_id, "org_1")
        self.assertEqual(client.platform, "ios")
        self.assertIsNone(client.device_id)

    def test_init_generates_device_identifier(self):
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        self.assertIsNotNone(client.device_identifier)
        self.assertTrue(client.device_identifier.startswith("client-"))

    def test_register_device(self):
        stub = _StubApi()
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        device_id = client.register(feature_schema=["feature1", "feature2"])
        self.assertEqual(device_id, "device_123")
        self.assertEqual(client.device_id, "device_123")

        method, path, payload = stub.calls[-1]
        self.assertEqual(method, "post")
        self.assertEqual(path, "/devices/register")
        self.assertEqual(payload["org_id"], "org_1")
        self.assertEqual(payload["platform"], "python")
        self.assertEqual(payload["feature_schema"], ["feature1", "feature2"])

    def test_register_returns_existing_device_id(self):
        stub = _StubApi()
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub
        client.device_id = "existing_device"

        device_id = client.register()
        self.assertEqual(device_id, "existing_device")
        self.assertEqual(len(stub.calls), 0)  # No API call made

    def test_register_raises_if_no_device_id_in_response(self):
        stub = _StubApi()
        stub.set_response("/devices/register", {})
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        class _BrokenApi(_StubApi):
            def post(self, path, payload=None):
                self.calls.append(("post", path, payload))
                return {}

        client.api = _BrokenApi()
        with self.assertRaises(EdgeMLClientError) as ctx:
            client.register()
        self.assertIn("Device registration failed", str(ctx.exception))

    def test_get_model_info_caches_result(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456", {"id": "model_456", "name": "my_model", "framework": "pytorch"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        # First call should hit API
        info1 = client._get_model_info("my_model")
        initial_call_count = len([c for c in stub.calls if c[1].startswith("/models")])

        # Second call should use cache
        info2 = client._get_model_info("my_model")
        final_call_count = len([c for c in stub.calls if c[1].startswith("/models")])

        self.assertEqual(info1, info2)
        self.assertEqual(initial_call_count, final_call_count)

    def test_get_model_info_handles_error(self):
        class _ErrorApi(_StubApi):
            def get(self, path, params=None):
                self.calls.append(("get", path, params))
                if "/models/" in path and path != "/models":
                    raise EdgeMLClientError("Model not found")
                return {"models": [{"name": "my_model", "id": "model_456"}]}

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = _ErrorApi()

        info = client._get_model_info("my_model")
        self.assertEqual(info, {})

    def test_get_model_architecture(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456", {"id": "model_456", "architecture": {"layers": 3}})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        arch = client._get_model_architecture("my_model")
        self.assertEqual(arch, {"layers": 3})

    def test_get_model_architecture_empty_if_not_found(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456", {"id": "model_456"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        arch = client._get_model_architecture("my_model")
        self.assertEqual(arch, {})

    def test_resolve_model_id_by_name(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_789"}]})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        model_id = client._resolve_model_id("my_model")
        self.assertEqual(model_id, "model_789")

    def test_resolve_model_id_returns_input_if_not_found(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": []})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        model_id = client._resolve_model_id("unknown_model")
        self.assertEqual(model_id, "unknown_model")

    def test_serialize_weights_bytes(self):
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        data = b"raw_bytes"
        result = client._serialize_weights(data)
        self.assertEqual(result, data)

    def test_serialize_weights_bytearray(self):
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        data = bytearray(b"raw_bytes")
        result = client._serialize_weights(data)
        self.assertEqual(result, bytes(data))

    def test_serialize_weights_numpy_array(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        arr = np.array([[1, 2], [3, 4]])
        result = client._serialize_weights(arr)
        self.assertIsInstance(result, bytes)

        # Verify round-trip
        buffer = io.BytesIO(result)
        restored = np.load(buffer)
        np.testing.assert_array_equal(restored, arr)

    def test_serialize_weights_torch_module(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch not installed")

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

        model = SimpleModel()
        result = client._serialize_weights(model)
        self.assertIsInstance(result, bytes)

    def test_serialize_weights_state_dict(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        state_dict = {"weight": torch.randn(3, 3), "bias": torch.randn(3)}
        result = client._serialize_weights(state_dict)
        self.assertIsInstance(result, bytes)

    def test_serialize_weights_invalid_type_raises(self):
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        with self.assertRaises(EdgeMLClientError) as ctx:
            client._serialize_weights("invalid_string_data")
        self.assertIn("must be bytes", str(ctx.exception))

    def test_deserialize_weights(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")

        # Create a state dict
        original_state = {"weight": torch.randn(3, 3), "bias": torch.randn(3)}

        # Serialize it
        buffer = io.BytesIO()
        torch.save(original_state, buffer)
        serialized = buffer.getvalue()

        # Deserialize it
        restored_state = client._deserialize_weights(serialized)
        self.assertIsInstance(restored_state, dict)
        self.assertIn("weight", restored_state)
        self.assertIn("bias", restored_state)

    def test_pull_model(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456/versions/latest", {"version": "1.0.0"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        result = client.pull_model("my_model", version="1.0.0")
        self.assertIsInstance(result, bytes)

        # Check that download endpoint was called
        download_calls = [c for c in stub.calls if "download" in str(c[1])]
        self.assertGreater(len(download_calls), 0)

    def test_pull_model_resolves_latest_version(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456/versions/latest", {"version": "2.5.0"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        result = client.pull_model("my_model")  # No version specified
        self.assertIsInstance(result, bytes)

    def test_compute_state_dict_delta(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        base_state = {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([1.0, 2.0]),
        }

        updated_state = {
            "weight": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            "bias": torch.tensor([2.0, 3.0]),
        }

        delta = compute_state_dict_delta(base_state, updated_state)

        self.assertIn("weight", delta)
        self.assertIn("bias", delta)

        # Verify delta is the difference
        expected_weight_delta = updated_state["weight"] - base_state["weight"]
        torch.testing.assert_close(delta["weight"], expected_weight_delta)

    def test_inference_property_returns_streaming_inference_client(self):
        from edgeml.inference import StreamingInferenceClient

        stub = _StubApi()
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        inference = client.inference
        self.assertIsInstance(inference, StreamingInferenceClient)

    def test_inference_property_is_cached(self):
        stub = _StubApi()
        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        inference1 = client.inference
        inference2 = client.inference
        self.assertIs(inference1, inference2)

    def test_inference_property_uses_device_id_when_registered(self):
        from edgeml.inference import StreamingInferenceClient

        stub = _StubApi()
        client = FederatedClient(
            auth_token_provider=lambda: "token123",
            org_id="org_1",
            device_identifier="my_identifier",
        )
        client.api = stub
        client.device_id = "registered_device_id"

        inference = client.inference
        self.assertIsInstance(inference, StreamingInferenceClient)
        # The client should use device_id (not device_identifier)
        self.assertEqual(inference.device_id, "registered_device_id")

    def test_inference_property_uses_device_identifier_when_not_registered(self):
        from edgeml.inference import StreamingInferenceClient

        stub = _StubApi()
        client = FederatedClient(
            auth_token_provider=lambda: "token123",
            org_id="org_1",
            device_identifier="my_identifier",
        )
        client.api = stub
        # device_id is None (not registered)

        inference = client.inference
        self.assertIsInstance(inference, StreamingInferenceClient)
        self.assertEqual(inference.device_id, "my_identifier")

    def test_train_no_version_found_raises(self):
        stub = _StubApi()
        stub.set_response("/models/model_456/versions/latest", {})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        with self.assertRaises(EdgeMLClientError) as ctx:
            client.train("model_456", data=b"some_weights")
        self.assertIn("Failed to resolve model version", str(ctx.exception))

    def test_train_with_callable_data(self):
        stub = _StubApi()
        stub.set_response("/models/model_456/versions/latest", {"version": "1.0.0"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        data_fn = lambda: (b"weights_from_callable", 100, {"loss": 0.5})

        results = client.train("model_456", data=data_fn, rounds=1)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

        # Verify the posted payload has the correct sample_count and metrics
        post_calls = [c for c in stub.calls if c[0] == "post" and c[1] == "/training/weights"]
        self.assertEqual(len(post_calls), 1)
        payload = post_calls[0][2]
        self.assertEqual(payload["sample_count"], 100)
        self.assertEqual(payload["metrics"], {"loss": 0.5})

    def test_train_with_raw_bytes_data(self):
        stub = _StubApi()
        stub.set_response("/models/model_456/versions/latest", {"version": "1.0.0"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        results = client.train("model_456", data=b"raw_weights", rounds=1)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

        # Verify post was made
        post_calls = [c for c in stub.calls if c[0] == "post" and c[1] == "/training/weights"]
        self.assertEqual(len(post_calls), 1)
        payload = post_calls[0][2]
        self.assertEqual(payload["model_id"], "model_456")
        self.assertEqual(payload["version"], "1.0.0")
        self.assertEqual(payload["sample_count"], 0)  # no sample_count provided

    def test_get_model_architecture_public_method(self):
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456", {"id": "model_456", "architecture": {"layers": 5, "type": "cnn"}})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        arch = client.get_model_architecture("my_model")
        self.assertEqual(arch, {"layers": 5, "type": "cnn"})

    def test_pull_model_no_version_found_raises(self):
        stub = _StubApi()
        stub.set_response("/models/model_456/versions/latest", {})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        with self.assertRaises(EdgeMLClientError) as ctx:
            client.pull_model("model_456")
        self.assertIn("Failed to resolve model version", str(ctx.exception))

    def test_train_from_remote(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/models/model_456/versions/latest", {"version": "1.0.0"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        def local_train_fn(base_state):
            # Modify the state dict
            updated_state = {k: v + 1.0 for k, v in base_state.items()}
            return updated_state, 50, {"loss": 0.3}

        results = client.train_from_remote(
            "model_456",
            local_train_fn=local_train_fn,
            rounds=1,
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

        # Verify the posted payload
        post_calls = [c for c in stub.calls if c[0] == "post" and c[1] == "/training/weights"]
        self.assertEqual(len(post_calls), 1)
        payload = post_calls[0][2]
        self.assertEqual(payload["model_id"], "model_456")
        self.assertEqual(payload["version"], "1.0.0")
        self.assertEqual(payload["sample_count"], 50)
        self.assertEqual(payload["metrics"], {"loss": 0.3})
        self.assertEqual(payload["update_format"], "weights")

    def test_train_from_remote_delta_format(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/models/model_456/versions/latest", {"version": "2.0.0"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        def local_train_fn(base_state):
            updated_state = {k: v + 0.5 for k, v in base_state.items()}
            return updated_state, 25, {"accuracy": 0.95}

        results = client.train_from_remote(
            "model_456",
            local_train_fn=local_train_fn,
            rounds=1,
            update_format="delta",
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

        # Verify delta format was used in the payload
        post_calls = [c for c in stub.calls if c[0] == "post" and c[1] == "/training/weights"]
        self.assertEqual(len(post_calls), 1)
        payload = post_calls[0][2]
        self.assertEqual(payload["update_format"], "delta")
        self.assertEqual(payload["sample_count"], 25)
        self.assertEqual(payload["metrics"], {"accuracy": 0.95})

    def test_train_with_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "my_model", "id": "model_456"}]})
        stub.set_response("/models/model_456", {
            "id": "model_456",
            "architecture": {"output_type": "binary", "output_dim": 1}
        })
        stub.set_response("/models/model_456/versions/latest", {"version": "1.0.0"})

        client = FederatedClient(auth_token_provider=lambda: "token123", org_id="org_1")
        client.api = stub

        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "target": [0, 1, 0, 1],
        })

        results = client.train("my_model", df, target_col="target", rounds=1)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)


class PrepareTrainingDataTests(unittest.TestCase):
    """Tests for FederatedClient._prepare_training_data."""

    def _make_client(self, architecture=None):
        """Create a FederatedClient with a stubbed API that returns the given architecture."""
        stub = _StubApi()
        stub.set_response("/models", {"models": [{"name": "test_model", "id": "model_1"}]})
        model_info = {"id": "model_1"}
        if architecture is not None:
            model_info["architecture"] = architecture
        stub.set_response("/models/model_1", model_info)

        client = FederatedClient(
            auth_token_provider=lambda: "token",
            org_id="default",
        )
        client.api = stub
        return client

    def test_prepare_training_data_success(self):
        """Provide a DataFrame with features + target column, verify returned tuple."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        client = self._make_client(architecture={})

        df = pd.DataFrame({
            "feat_a": [10, 20, 30],
            "feat_b": [40, 50, 60],
            "target": [0, 1, 0],
        })

        result_df, features, target_col, sample_count = client._prepare_training_data(
            model="test_model",
            data=df,
            target_col="target",
        )

        self.assertEqual(target_col, "target")
        self.assertEqual(sample_count, 3)
        self.assertEqual(sorted(features), ["feat_a", "feat_b"])
        self.assertNotIn("target", features)
        self.assertEqual(len(result_df), 3)

    def test_prepare_training_data_load_error(self):
        """Mock load_data to raise DataLoadError, verify EdgeMLClientError is raised."""
        from edgeml.data_loader import DataLoadError

        client = self._make_client(architecture={})

        with patch(
            "edgeml.federated_client.load_data",
            side_effect=DataLoadError("file not found"),
        ):
            with self.assertRaises(EdgeMLClientError) as ctx:
                client._prepare_training_data(
                    model="test_model",
                    data="nonexistent.csv",
                    target_col="target",
                )
            self.assertIn("Failed to load data", str(ctx.exception))
            self.assertIn("file not found", str(ctx.exception))

    def test_prepare_training_data_missing_target_col(self):
        """Provide DataFrame without the target column, verify error lists available columns."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        client = self._make_client(architecture={})

        df = pd.DataFrame({
            "alpha": [1, 2],
            "beta": [3, 4],
        })

        with self.assertRaises(EdgeMLClientError) as ctx:
            client._prepare_training_data(
                model="test_model",
                data=df,
                target_col="nonexistent",
            )
        msg = str(ctx.exception)
        self.assertIn("nonexistent", msg)
        self.assertIn("alpha", msg)
        self.assertIn("beta", msg)

    def test_prepare_training_data_default_target_col(self):
        """Pass target_col=None, verify it defaults to architecture's target_col or 'target'."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        # Case 1: architecture specifies a custom target_col
        client = self._make_client(architecture={"target_col": "label"})
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "label": [0, 1, 0],
        })
        _, _, resolved_col, _ = client._prepare_training_data(
            model="test_model",
            data=df,
            target_col=None,
        )
        self.assertEqual(resolved_col, "label")

        # Case 2: architecture has no target_col key -- falls back to "target"
        client2 = self._make_client(architecture={})
        df2 = pd.DataFrame({
            "x": [1, 2, 3],
            "target": [0, 1, 0],
        })
        _, _, resolved_col2, _ = client2._prepare_training_data(
            model="test_model",
            data=df2,
            target_col=None,
        )
        self.assertEqual(resolved_col2, "target")

    def test_prepare_training_data_with_architecture_validation(self):
        """Provide architecture info, verify validate_target is called with correct args."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        architecture = {
            "output_type": "multiclass",
            "output_dim": 3,
        }
        client = self._make_client(architecture=architecture)

        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "target": [0, 1, 2],
        })

        with patch(
            "edgeml.federated_client.validate_target",
            return_value=df,
        ) as mock_validate:
            client._prepare_training_data(
                model="test_model",
                data=df,
                target_col="target",
            )

            mock_validate.assert_called_once()
            call_kwargs = mock_validate.call_args
            # validate_target(df, target_col=..., output_type=..., output_dim=...)
            self.assertEqual(call_kwargs.kwargs.get("target_col") or call_kwargs[1].get("target_col"), "target")
            self.assertEqual(call_kwargs.kwargs.get("output_type") or call_kwargs[1].get("output_type"), "multiclass")
            self.assertEqual(call_kwargs.kwargs.get("output_dim") or call_kwargs[1].get("output_dim"), 3)


if __name__ == "__main__":
    unittest.main()
