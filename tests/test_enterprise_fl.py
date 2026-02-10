"""Enterprise federated learning tests.

Tests for enterprise FL features in the Python SDK:
- SecAgg integration in round participation
- FedProx + filter pipeline interactions
- Ditto / FedPer strategy integration
- Federation algorithm validation
- Privacy budget edge cases
- SecAgg session and dropout handling
"""

import io
import struct
import unittest
from unittest.mock import patch, MagicMock

from edgeml.api_client import EdgeMLClientError
from edgeml.federated_client import (
    FederatedClient,
    apply_filters,
    compute_state_dict_delta,
    _apply_fedprox_correction,
)
from edgeml.secagg import (
    SecAggClient,
    SecAggConfig,
    ShamirShare,
    generate_shares,
    reconstruct_secret,
    model_bytes_to_field_elements,
    field_elements_to_model_bytes,
    _derive_mask_elements,
    DEFAULT_FIELD_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubApi:
    """Stub API client that records calls and returns preconfigured responses."""

    def __init__(self):
        self.calls = []
        self._responses = {}
        self._bytes_responses = {}

    def set_response(self, key, response):
        self._responses[key] = response

    def set_bytes_response(self, key, data):
        self._bytes_responses[key] = data

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
        if "/updates" in path or "weights" in path:
            return {"update_id": "update_456", "status": "accepted"}
        return {}

    def get_bytes(self, path, params=None):
        self.calls.append(("get_bytes", path, params))
        if path in self._bytes_responses:
            return self._bytes_responses[path]
        try:
            import torch
            state_dict = {"weight": torch.randn(3, 3), "bias": torch.randn(3)}
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            return buffer.getvalue()
        except ImportError:
            return b"model_weights_data"

    def post_bytes(self, path, data):
        self.calls.append(("post_bytes", path, data))
        return {"status": "ok"}

    def secagg_get_session(self, round_id, device_id):
        self.calls.append(("secagg_get_session", round_id, device_id))
        return {
            "session_id": "sess_1",
            "threshold": 2,
            "total_clients": 3,
            "field_size": DEFAULT_FIELD_SIZE,
            "key_length": 256,
        }

    def secagg_submit_shares(self, round_id, device_id, shares_data):
        self.calls.append(("secagg_submit_shares", round_id, device_id, shares_data))
        return {"status": "ok"}


def _make_client(stub=None, secure_aggregation=False):
    """Create a FederatedClient with stubbed API."""
    client = FederatedClient(
        auth_token_provider=lambda: "token123",
        org_id="org_1",
        secure_aggregation=secure_aggregation,
    )
    client.api = stub or _StubApi()
    client.device_id = "device_123"
    return client


# ---------------------------------------------------------------------------
# SecAgg integration in participate_in_round
# ---------------------------------------------------------------------------


class SecAggRoundIntegrationTests(unittest.TestCase):
    """Test SecAgg activation paths in participate_in_round."""

    def test_secagg_activated_by_constructor_flag(self):
        """SecAgg enabled via constructor should mask update and submit shares."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {"model_id": "model_456", "version": "1.0.0"},
        })

        client = _make_client(stub=stub, secure_aggregation=True)

        def local_train_fn(base_state):
            return {k: v + 0.1 for k, v in base_state.items()}, 100, {"loss": 0.5}

        result = client.participate_in_round("r1", local_train_fn)
        self.assertEqual(result["status"], "accepted")

        # Verify SecAgg calls were made
        secagg_session_calls = [c for c in stub.calls if c[0] == "secagg_get_session"]
        secagg_share_calls = [c for c in stub.calls if c[0] == "secagg_submit_shares"]
        self.assertEqual(len(secagg_session_calls), 1)
        self.assertEqual(len(secagg_share_calls), 1)
        self.assertEqual(secagg_session_calls[0][1], "r1")

    def test_secagg_activated_by_round_config(self):
        """SecAgg enabled via round config should mask update even if constructor flag is off."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {
                "model_id": "model_456",
                "version": "1.0.0",
                "secure_aggregation": True,
            },
        })

        client = _make_client(stub=stub, secure_aggregation=False)

        def local_train_fn(base_state):
            return {k: v + 0.1 for k, v in base_state.items()}, 50, {}

        result = client.participate_in_round("r1", local_train_fn)
        self.assertEqual(result["status"], "accepted")

        secagg_session_calls = [c for c in stub.calls if c[0] == "secagg_get_session"]
        self.assertEqual(len(secagg_session_calls), 1)

    def test_secagg_not_activated_when_both_disabled(self):
        """No SecAgg calls when both constructor and config are off."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {"model_id": "model_456", "version": "1.0.0"},
        })

        client = _make_client(stub=stub, secure_aggregation=False)

        def local_train_fn(base_state):
            return {k: v + 0.1 for k, v in base_state.items()}, 50, {}

        client.participate_in_round("r1", local_train_fn)

        secagg_calls = [c for c in stub.calls if c[0].startswith("secagg_")]
        self.assertEqual(len(secagg_calls), 0)

    def test_secagg_mask_changes_uploaded_data(self):
        """Verify that SecAgg masking actually changes the uploaded payload."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {"model_id": "model_456", "version": "1.0.0"},
        })

        # Run without SecAgg
        client_plain = _make_client(stub=stub, secure_aggregation=False)

        def local_train_fn(base_state):
            return {k: v + 1.0 for k, v in base_state.items()}, 100, {}

        client_plain.participate_in_round("r1", local_train_fn)
        plain_post = [c for c in stub.calls if c[0] == "post" and "weights" in c[1]][-1]
        plain_weights = plain_post[2]["weights_data"]

        # Run with SecAgg
        stub2 = _StubApi()
        stub2.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {"model_id": "model_456", "version": "1.0.0"},
        })
        client_secagg = _make_client(stub=stub2, secure_aggregation=True)

        client_secagg.participate_in_round("r1", local_train_fn)
        secagg_post = [c for c in stub2.calls if c[0] == "post" and "weights" in c[1]][-1]
        secagg_weights = secagg_post[2]["weights_data"]

        # Masked data should differ from plain data
        self.assertNotEqual(plain_weights, secagg_weights)


# ---------------------------------------------------------------------------
# SecAgg session and dropout edge cases
# ---------------------------------------------------------------------------


class SecAggSessionTests(unittest.TestCase):
    """Test SecAgg session handling edge cases."""

    def test_secagg_client_with_custom_field_size(self):
        """SecAggConfig with non-default field_size should work."""
        small_prime = 2147483647  # Mersenne prime 2^31 - 1
        config = SecAggConfig(
            session_id="s1",
            round_id="r1",
            threshold=2,
            total_clients=3,
            field_size=small_prime,
        )
        client = SecAggClient(config)
        shares = client.generate_key_shares()
        self.assertEqual(len(shares), 3)

        raw = struct.pack(">IIII", 10, 20, 30, 40)
        masked = client.mask_model_update(raw)
        self.assertNotEqual(masked, raw)

    def test_secagg_client_with_noise_scale(self):
        """SecAggConfig with noise_scale should be stored correctly."""
        config = SecAggConfig(
            session_id="s1",
            round_id="r1",
            threshold=2,
            total_clients=3,
            noise_scale=0.01,
        )
        self.assertEqual(config.noise_scale, 0.01)

    def test_secagg_dropout_multiple_peers(self):
        """Simulate multiple peer dropouts -- shares for different peers are distinct."""
        config = SecAggConfig(
            session_id="s1",
            round_id="r1",
            threshold=2,
            total_clients=5,
        )
        client = SecAggClient(config)
        shares = client.generate_key_shares()

        share_1 = client.get_seed_share_for_peer(1)
        share_3 = client.get_seed_share_for_peer(3)
        share_5 = client.get_seed_share_for_peer(5)

        self.assertIsNotNone(share_1)
        self.assertIsNotNone(share_3)
        self.assertIsNotNone(share_5)
        self.assertNotEqual(share_1.value, share_3.value)
        self.assertNotEqual(share_3.value, share_5.value)

    def test_secagg_reconstruct_seed_after_dropout(self):
        """Simulate 1 of 5 clients dropping out. Remaining 4 reconstruct dropped client's seed."""
        n_clients = 5
        threshold = 3
        field_size = DEFAULT_FIELD_SIZE

        clients = []
        for _ in range(n_clients):
            cfg = SecAggConfig(
                session_id="s1", round_id="r1",
                threshold=threshold, total_clients=n_clients,
                field_size=field_size,
            )
            clients.append(SecAggClient(cfg))

        all_shares = [c.generate_key_shares() for c in clients]

        # Client 2 (index 1) drops out.
        dropped_idx = 1
        dropped_seed_int = int.from_bytes(clients[dropped_idx]._seed, "big") % field_size

        # Remaining clients contribute their share for the dropped client.
        contributing = [i for i in range(n_clients) if i != dropped_idx]
        shares_for_dropped = [all_shares[dropped_idx][i] for i in contributing[:threshold]]

        reconstructed = reconstruct_secret(shares_for_dropped)
        self.assertEqual(reconstructed, dropped_seed_int)

    def test_secagg_share_serialization_large_batch(self):
        """Verify serialization/deserialization of a large batch of shares."""
        config = SecAggConfig(
            session_id="s1", round_id="r1",
            threshold=5, total_clients=20,
        )
        client = SecAggClient(config)
        shares = client.generate_key_shares()

        serialized = SecAggClient.serialize_shares(shares)
        restored = SecAggClient.deserialize_shares(serialized)

        self.assertEqual(len(restored), 20)
        for orig, rest in zip(shares, restored):
            self.assertEqual(orig.index, rest.index)
            self.assertEqual(orig.value, rest.value)


# ---------------------------------------------------------------------------
# FedProx + filter pipeline interactions
# ---------------------------------------------------------------------------


class FedProxFilterInteractionTests(unittest.TestCase):
    """Test FedProx correction combined with filter pipeline."""

    def test_fedprox_then_gradient_clip(self):
        """FedProx correction followed by gradient clipping."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {
                "model_id": "model_456",
                "version": "1.0.0",
                "proximal_mu": 1.0,
                "filters": [{"type": "gradient_clip", "max_norm": 0.5}],
            },
        })

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            # Large update to trigger both FedProx and clipping
            return {k: v + 10.0 for k, v in base_state.items()}, 100, {}

        result = client.participate_in_round("r1", local_train_fn)
        self.assertEqual(result["status"], "accepted")

    def test_fedprox_then_sparsification(self):
        """FedProx correction followed by sparsification."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {
                "model_id": "model_456",
                "version": "1.0.0",
                "proximal_mu": 0.5,
                "filters": [{"type": "sparsification", "top_k_percent": 50.0}],
            },
        })

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            return {k: v + 1.0 for k, v in base_state.items()}, 50, {}

        result = client.participate_in_round("r1", local_train_fn)
        self.assertEqual(result["status"], "accepted")

    def test_fedprox_then_noise_then_quantization(self):
        """Three-stage pipeline: FedProx -> noise -> quantization."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {
                "model_id": "model_456",
                "version": "1.0.0",
                "proximal_mu": 0.1,
                "filters": [
                    {"type": "gaussian_noise", "stddev": 0.001},
                    {"type": "quantization", "bits": 8},
                ],
            },
        })

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            return {k: v + 0.5 for k, v in base_state.items()}, 200, {}

        result = client.participate_in_round("r1", local_train_fn)
        self.assertEqual(result["status"], "accepted")

    def test_clip_norm_injected_before_filters(self):
        """clip_norm in config should be injected before the filter list."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {
                "model_id": "model_456",
                "version": "1.0.0",
                "clip_norm": 0.1,
                "filters": [{"type": "gaussian_noise", "stddev": 0.01}],
            },
        })

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            return {k: v + 100.0 for k, v in base_state.items()}, 10, {}

        result = client.participate_in_round("r1", local_train_fn)
        self.assertEqual(result["status"], "accepted")

    def test_fedprox_with_secagg(self):
        """FedProx + SecAgg together -- correction applied before masking."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {
                "model_id": "model_456",
                "version": "1.0.0",
                "proximal_mu": 0.5,
                "secure_aggregation": True,
            },
        })

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            return {k: v + 2.0 for k, v in base_state.items()}, 100, {}

        result = client.participate_in_round("r1", local_train_fn)
        self.assertEqual(result["status"], "accepted")

        # Both FedProx (implicit) and SecAgg should have been applied
        secagg_calls = [c for c in stub.calls if c[0] == "secagg_get_session"]
        self.assertEqual(len(secagg_calls), 1)


# ---------------------------------------------------------------------------
# FedProx correction edge cases
# ---------------------------------------------------------------------------


class FedProxEdgeCaseTests(unittest.TestCase):
    """Additional edge cases for FedProx correction."""

    def test_fedprox_very_large_mu(self):
        """Very large mu should scale delta close to zero."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([100.0, 200.0])}
        result = _apply_fedprox_correction(delta, 1000.0)

        # scale = 1/1001 ~ 0.001
        self.assertAlmostEqual(result["w"][0].item(), 100.0 / 1001.0, places=3)
        self.assertAlmostEqual(result["w"][1].item(), 200.0 / 1001.0, places=3)

    def test_fedprox_with_empty_delta(self):
        """Empty delta should return empty dict."""
        result = _apply_fedprox_correction({}, 1.0)
        self.assertEqual(result, {})

    def test_fedprox_mixed_tensor_types(self):
        """Delta with int tensors and float tensors."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {
            "float_w": torch.tensor([4.0, 6.0]),
            "int_w": torch.tensor([10, 20]),
            "metadata": "some_string",
        }
        result = _apply_fedprox_correction(delta, 1.0)

        # Float tensors scaled by 0.5
        torch.testing.assert_close(result["float_w"], torch.tensor([2.0, 3.0]))
        # Int tensors also scaled (torch handles int * float)
        self.assertEqual(result["int_w"][0].item(), 5)
        # Non-tensor preserved
        self.assertEqual(result["metadata"], "some_string")


# ---------------------------------------------------------------------------
# Ditto and FedPer strategy integration
# ---------------------------------------------------------------------------


class DittoIntegrationTests(unittest.TestCase):
    """Test Ditto personalization strategy."""

    def test_ditto_large_lambda_pulls_toward_global(self):
        """Large lambda should make personal weights very close to global."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        client = _make_client()

        global_model = {"w": torch.tensor([1.0, 2.0, 3.0])}
        personal_model = {"w": torch.tensor([10.0, 20.0, 30.0])}

        def local_train_fn(base_state):
            return {k: v + 0.1 for k, v in base_state.items()}, 100, {}

        _, personal_updated, _, _ = client.train_ditto(
            global_model=global_model,
            personal_model=personal_model,
            local_train_fn=local_train_fn,
            lambda_ditto=0.99,
        )

        # personal = p - 0.99 * (p - g) = p * 0.01 + g * 0.99
        for i in range(3):
            expected = personal_model["w"][i].item() * 0.01 + global_model["w"][i].item() * 0.99
            self.assertAlmostEqual(personal_updated["w"][i].item(), expected, places=4)

    def test_ditto_mismatched_keys(self):
        """Personal model with extra keys not in global should preserve them."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        client = _make_client()

        global_model = {"shared": torch.tensor([1.0, 2.0])}
        personal_model = {
            "shared": torch.tensor([5.0, 6.0]),
            "personal_only": torch.tensor([10.0]),
        }

        def local_train_fn(base_state):
            return {k: v + 1.0 for k, v in base_state.items()}, 50, {}

        _, personal_updated, _, _ = client.train_ditto(
            global_model=global_model,
            personal_model=personal_model,
            local_train_fn=local_train_fn,
            lambda_ditto=0.5,
        )

        # "shared" should be regularized
        self.assertIn("shared", personal_updated)
        # "personal_only" has no global counterpart -- should be passed through
        self.assertIn("personal_only", personal_updated)
        torch.testing.assert_close(
            personal_updated["personal_only"],
            personal_model["personal_only"],
        )

    def test_ditto_with_non_tensor_values(self):
        """Non-tensor values in personal model should be preserved."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        client = _make_client()

        global_model = {"w": torch.tensor([1.0]), "version": "1.0"}
        personal_model = {"w": torch.tensor([5.0]), "version": "personal_v2"}

        def local_train_fn(base_state):
            return {k: v + 0.5 if torch.is_tensor(v) else v for k, v in base_state.items()}, 10, {}

        _, personal_updated, _, _ = client.train_ditto(
            global_model=global_model,
            personal_model=personal_model,
            local_train_fn=local_train_fn,
            lambda_ditto=0.5,
        )

        # Non-tensor should be preserved as-is
        self.assertEqual(personal_updated["version"], "personal_v2")


class FedPerIntegrationTests(unittest.TestCase):
    """Test FedPer personalization strategy."""

    def test_fedper_multiple_head_prefixes(self):
        """Multiple head layer prefixes should all be excluded."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        client = _make_client()

        model = {
            "encoder.layer1.weight": torch.tensor([1.0]),
            "encoder.layer2.weight": torch.tensor([2.0]),
            "decoder.weight": torch.tensor([3.0]),
            "classifier.weight": torch.tensor([4.0]),
        }

        def local_train_fn(state):
            return {k: v + 0.1 for k, v in state.items()}, 100, {}

        delta, sample_count, _ = client.train_fedper(
            model=model,
            head_layers=["decoder", "classifier"],
            local_train_fn=local_train_fn,
        )

        self.assertIn("encoder.layer1.weight", delta)
        self.assertIn("encoder.layer2.weight", delta)
        self.assertNotIn("decoder.weight", delta)
        self.assertNotIn("classifier.weight", delta)
        self.assertEqual(sample_count, 100)

    def test_fedper_all_layers_are_head(self):
        """If all layers are head layers, delta should be empty."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        client = _make_client()

        model = {
            "head.fc1.weight": torch.tensor([1.0]),
            "head.fc2.weight": torch.tensor([2.0]),
        }

        def local_train_fn(state):
            return {k: v + 1.0 for k, v in state.items()}, 50, {}

        delta, _, _ = client.train_fedper(
            model=model,
            head_layers=["head"],
            local_train_fn=local_train_fn,
        )

        self.assertEqual(len(delta), 0)

    def test_fedper_delta_values_correct(self):
        """Verify FedPer delta values are correct (updated - base)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        client = _make_client()

        model = {
            "body.weight": torch.tensor([1.0, 2.0, 3.0]),
            "head.weight": torch.tensor([4.0, 5.0]),
        }

        def local_train_fn(state):
            updated = {k: v + 0.5 for k, v in state.items()}
            return updated, 30, {"loss": 0.1}

        delta, sample_count, metrics = client.train_fedper(
            model=model,
            head_layers=["head"],
            local_train_fn=local_train_fn,
        )

        self.assertEqual(sample_count, 30)
        self.assertEqual(metrics, {"loss": 0.1})
        torch.testing.assert_close(
            delta["body.weight"],
            torch.tensor([0.5, 0.5, 0.5]),
        )


# ---------------------------------------------------------------------------
# Federation algorithm and round management
# ---------------------------------------------------------------------------


class FederationAlgorithmTests(unittest.TestCase):
    """Test Federation.train with different algorithm parameters."""

    def test_fedavg_algorithm_accepted(self):
        """fedavg algorithm should work."""
        from edgeml.federation import Federation

        stub = _StubApi()
        stub._responses["/federations"] = []

        def stub_post(path, payload=None):
            stub.calls.append(("post", path, payload))
            if path == "/federations":
                return {"id": "fed_123"}
            if path == "/training/aggregate":
                return {"new_version": "1.1.0", "status": "completed"}
            return {}

        stub.post = stub_post
        stub._responses[("/models", (("org_id", "org_1"),))] = {"models": [{"name": "m", "id": "m1"}]}

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            fed = Federation(lambda: "t", name="test", org_id="org_1")

        result = fed.train(model="m", algorithm="fedavg", rounds=1)
        self.assertEqual(result["status"], "completed")

    def test_unsupported_algorithm_raises(self):
        """Unsupported algorithms should raise EdgeMLClientError."""
        from edgeml.federation import Federation

        stub = _StubApi()
        stub._responses["/federations"] = []

        def stub_post(path, payload=None):
            stub.calls.append(("post", path, payload))
            if path == "/federations":
                return {"id": "fed_123"}
            return {}

        stub.post = stub_post

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            fed = Federation(lambda: "t", org_id="org_1")

        for algo in ["fedprox", "krum", "fedmedian", "fedtrimmedavg", "scaffold"]:
            with self.assertRaises(EdgeMLClientError) as ctx:
                fed.train(model="m", algorithm=algo)
            self.assertIn("Unsupported algorithm", str(ctx.exception))

    def test_train_version_progression(self):
        """Multi-round training should use the new version as base for next round."""
        from edgeml.federation import Federation

        stub = _StubApi()
        stub._responses["/federations"] = []
        round_counter = [0]

        def stub_post(path, payload=None):
            stub.calls.append(("post", path, payload))
            if path == "/federations":
                return {"id": "fed_123"}
            if path == "/training/aggregate":
                round_counter[0] += 1
                return {"new_version": f"1.{round_counter[0]}.0", "status": "completed"}
            return {}

        stub.post = stub_post
        stub._responses[("/models", (("org_id", "org_1"),))] = {"models": [{"name": "m", "id": "m1"}]}

        with unittest.mock.patch("edgeml.federation._ApiClient") as mock_api:
            mock_api.return_value = stub
            fed = Federation(lambda: "t", org_id="org_1")

        fed.train(model="m", rounds=3, base_version="1.0.0")

        aggregate_calls = [c for c in stub.calls if c[1] == "/training/aggregate"]
        self.assertEqual(len(aggregate_calls), 3)

        # First call uses base_version="1.0.0"
        self.assertEqual(aggregate_calls[0][2]["base_version"], "1.0.0")
        # Second call uses new_version from first round
        self.assertEqual(aggregate_calls[1][2]["base_version"], "1.1.0")
        # Third call uses new_version from second round
        self.assertEqual(aggregate_calls[2][2]["base_version"], "1.2.0")


# ---------------------------------------------------------------------------
# Privacy budget edge cases
# ---------------------------------------------------------------------------


class PrivacyBudgetEdgeCaseTests(unittest.TestCase):
    """Test privacy budget retrieval edge cases."""

    def test_privacy_budget_zero_remaining(self):
        """Zero budget remaining should still return valid response."""
        stub = _StubApi()
        stub.set_response("/federations/fed_1/privacy", {
            "epsilon": 5.0,
            "delta": 1e-5,
            "rounds_consumed": 100,
            "budget_remaining": 0.0,
        })

        client = _make_client(stub=stub)
        result = client.get_privacy_budget("fed_1")

        self.assertEqual(result["budget_remaining"], 0.0)
        self.assertEqual(result["rounds_consumed"], 100)

    def test_privacy_budget_api_error(self):
        """API error when fetching privacy budget should propagate."""
        class _ErrorApi(_StubApi):
            def get(self, path, params=None):
                self.calls.append(("get", path, params))
                if "/privacy" in path:
                    raise EdgeMLClientError("Privacy service unavailable")
                return super().get(path, params)

        client = _make_client(stub=_ErrorApi())

        with self.assertRaises(EdgeMLClientError) as ctx:
            client.get_privacy_budget("fed_1")
        self.assertIn("Privacy service unavailable", str(ctx.exception))

    def test_privacy_budget_large_epsilon(self):
        """Very large epsilon (relaxed privacy) should be returned as-is."""
        stub = _StubApi()
        stub.set_response("/federations/fed_1/privacy", {
            "epsilon": 1000.0,
            "delta": 0.1,
            "rounds_consumed": 1,
            "budget_remaining": 999.0,
        })

        client = _make_client(stub=stub)
        result = client.get_privacy_budget("fed_1")
        self.assertEqual(result["epsilon"], 1000.0)


# ---------------------------------------------------------------------------
# Filter pipeline adversarial cases
# ---------------------------------------------------------------------------


class FilterPipelineAdversarialTests(unittest.TestCase):
    """Adversarial and edge-case tests for filter pipeline."""

    def test_all_five_filters_chained(self):
        """Apply all 5 filter types in sequence."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        torch.manual_seed(42)
        delta = {"w": torch.randn(20) * 10}
        result = apply_filters(delta, [
            {"type": "gradient_clip", "max_norm": 5.0},
            {"type": "gaussian_noise", "stddev": 0.001},
            {"type": "norm_validation", "max_norm": 100.0},
            {"type": "sparsification", "top_k_percent": 50.0},
            {"type": "quantization", "bits": 4},
        ])

        self.assertIn("w", result)
        # After all filters, result should have the same shape
        self.assertEqual(result["w"].shape[0], 20)

    def test_norm_validation_drops_all_tensors(self):
        """If all tensors exceed max_norm, delta should be empty."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {
            "w1": torch.tensor([100.0, 200.0]),
            "w2": torch.tensor([300.0, 400.0]),
        }
        result = apply_filters(delta, [{"type": "norm_validation", "max_norm": 0.001}])

        # All tensors should be dropped
        self.assertNotIn("w1", result)
        self.assertNotIn("w2", result)

    def test_sparsification_100_percent_keeps_all(self):
        """top_k_percent=100 should keep all values."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}
        result = apply_filters(delta, [{"type": "sparsification", "top_k_percent": 100.0}])

        non_zero = (result["w"] != 0).sum().item()
        self.assertEqual(non_zero, 5)

    def test_quantization_1_bit(self):
        """1-bit quantization should produce binary-like output."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"w": torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])}
        result = apply_filters(delta, [{"type": "quantization", "bits": 1}])

        # 1 bit = 1 level, so values should be at min or max
        unique_vals = result["w"].unique()
        self.assertLessEqual(len(unique_vals), 2)

    def test_filter_with_scalar_tensor(self):
        """Filters should handle scalar (0-dim) tensors."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"scalar": torch.tensor(5.0)}
        result = apply_filters(delta, [{"type": "gradient_clip", "max_norm": 1.0}])

        # Scalar with norm 5.0 should be clipped to norm 1.0
        self.assertAlmostEqual(result["scalar"].item(), 1.0, places=4)

    def test_filter_with_multidimensional_tensor(self):
        """Filters should handle 3D+ tensors (e.g., conv weight shapes)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        delta = {"conv": torch.randn(3, 3, 3, 3) * 10}
        result = apply_filters(delta, [{"type": "gradient_clip", "max_norm": 1.0}])

        clipped_norm = torch.norm(result["conv"].float().flatten(), dim=0)
        self.assertAlmostEqual(clipped_norm.item(), 1.0, places=4)


# ---------------------------------------------------------------------------
# Round participation edge cases
# ---------------------------------------------------------------------------


class RoundParticipationEdgeCaseTests(unittest.TestCase):
    """Edge cases for participate_in_round."""

    def test_participate_with_empty_config(self):
        """Round with minimal config (only model_id) should work."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {"model_id": "model_456"},
        })
        stub.set_response("/models/model_456/versions/latest", {"version": "1.0.0"})

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            return {k: v + 0.1 for k, v in base_state.items()}, 10, {}

        result = client.participate_in_round("r1", local_train_fn)
        self.assertEqual(result["status"], "accepted")

    def test_participate_uploads_correct_round_id(self):
        """Verify round_id is correctly included in the upload payload."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/round_xyz/status", {
            "round_id": "round_xyz",
            "config": {"model_id": "model_456", "version": "2.0.0"},
        })

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            return {k: v + 0.1 for k, v in base_state.items()}, 5, {"acc": 0.9}

        client.participate_in_round("round_xyz", local_train_fn)

        post_calls = [c for c in stub.calls if c[0] == "post" and "weights" in c[1]]
        self.assertEqual(len(post_calls), 1)
        payload = post_calls[0][2]
        self.assertEqual(payload["round_id"], "round_xyz")
        self.assertEqual(payload["sample_count"], 5)
        self.assertEqual(payload["metrics"], {"acc": 0.9})
        self.assertEqual(payload["update_format"], "delta")

    def test_participate_version_from_round_config(self):
        """Version should be taken from round config when present."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {"model_id": "model_456", "version": "3.0.0"},
        })

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            return {k: v + 0.1 for k, v in base_state.items()}, 10, {}

        client.participate_in_round("r1", local_train_fn)

        post_calls = [c for c in stub.calls if c[0] == "post" and "weights" in c[1]]
        payload = post_calls[0][2]
        self.assertEqual(payload["version"], "3.0.0")

    def test_participate_version_falls_back_to_latest(self):
        """When round config has no version, should fall back to /versions/latest."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {"model_id": "model_456"},
        })
        stub.set_response("/models/model_456/versions/latest", {"version": "5.0.0"})

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            return {k: v + 0.1 for k, v in base_state.items()}, 10, {}

        client.participate_in_round("r1", local_train_fn)

        post_calls = [c for c in stub.calls if c[0] == "post" and "weights" in c[1]]
        payload = post_calls[0][2]
        self.assertEqual(payload["version"], "5.0.0")


# ---------------------------------------------------------------------------
# End-to-end enterprise scenario
# ---------------------------------------------------------------------------


class EndToEndEnterpriseTests(unittest.TestCase):
    """End-to-end scenario tests combining multiple enterprise features."""

    def test_full_enterprise_round_fedprox_filters_secagg(self):
        """Simulate a complete enterprise round: FedProx + filters + SecAgg."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        stub = _StubApi()
        stub.set_response("/training/rounds/r1/status", {
            "round_id": "r1",
            "config": {
                "model_id": "model_456",
                "version": "1.0.0",
                "proximal_mu": 0.5,
                "clip_norm": 2.0,
                "filters": [
                    {"type": "gaussian_noise", "stddev": 0.001},
                    {"type": "quantization", "bits": 8},
                ],
                "secure_aggregation": True,
            },
        })

        client = _make_client(stub=stub)

        def local_train_fn(base_state):
            return {k: v + 5.0 for k, v in base_state.items()}, 200, {"loss": 0.25}

        result = client.participate_in_round("r1", local_train_fn)
        self.assertEqual(result["status"], "accepted")

        # Verify the full pipeline ran
        secagg_calls = [c for c in stub.calls if c[0] == "secagg_get_session"]
        self.assertEqual(len(secagg_calls), 1)

        post_calls = [c for c in stub.calls if c[0] == "post" and "weights" in c[1]]
        self.assertEqual(len(post_calls), 1)
        payload = post_calls[0][2]
        self.assertEqual(payload["round_id"], "r1")
        self.assertEqual(payload["sample_count"], 200)
        self.assertEqual(payload["metrics"], {"loss": 0.25})

    def test_secagg_multi_client_aggregate_with_dropout(self):
        """Full SecAgg simulation: 5 clients, 1 dropout, verify aggregate recovery."""
        n_clients = 5
        threshold = 3
        field_size = DEFAULT_FIELD_SIZE

        # Each client has a 4-element update
        raw_updates = [
            struct.pack(">IIII", 10, 20, 30, 40),
            struct.pack(">IIII", 1, 2, 3, 4),
            struct.pack(">IIII", 100, 200, 300, 400),
            struct.pack(">IIII", 5, 10, 15, 20),
            struct.pack(">IIII", 50, 60, 70, 80),
        ]

        clients = []
        for _ in range(n_clients):
            cfg = SecAggConfig(
                session_id="s1", round_id="r1",
                threshold=threshold, total_clients=n_clients,
                field_size=field_size,
            )
            clients.append(SecAggClient(cfg))

        # Phase 1: generate shares
        all_shares = [c.generate_key_shares() for c in clients]

        # Phase 2: mask updates
        masked_updates = [c.mask_model_update(raw) for c, raw in zip(clients, raw_updates)]

        # Client 4 (index 3) drops out after masking
        dropped_idx = 3
        active_indices = [i for i in range(n_clients) if i != dropped_idx]

        # Server: sum all masked elements (including the dropped client's)
        n_elements = 4
        agg_masked = [0] * n_elements
        for masked in masked_updates:
            elems = model_bytes_to_field_elements(masked, field_size)
            for j in range(n_elements):
                agg_masked[j] = (agg_masked[j] + elems[j]) % field_size

        # Server: reconstruct each active client's mask and subtract
        for i in active_indices:
            seed_int = int.from_bytes(clients[i]._seed, "big") % field_size
            shares_for_i = all_shares[i][:threshold]
            reconstructed = reconstruct_secret(shares_for_i)
            self.assertEqual(reconstructed, seed_int)

            mask = _derive_mask_elements(clients[i]._seed, n_elements, field_size)
            for j in range(n_elements):
                agg_masked[j] = (agg_masked[j] - mask[j]) % field_size

        # Server: reconstruct dropped client's mask and subtract
        shares_for_dropped = [all_shares[dropped_idx][i] for i in active_indices[:threshold]]
        dropped_seed_int = reconstruct_secret(shares_for_dropped)
        dropped_seed = clients[dropped_idx]._seed
        expected_seed_int = int.from_bytes(dropped_seed, "big") % field_size
        self.assertEqual(dropped_seed_int, expected_seed_int)

        dropped_mask = _derive_mask_elements(dropped_seed, n_elements, field_size)
        for j in range(n_elements):
            agg_masked[j] = (agg_masked[j] - dropped_mask[j]) % field_size

        # Verify: aggregate should equal sum of all raw updates
        expected = [0] * n_elements
        for raw in raw_updates:
            elems = model_bytes_to_field_elements(raw, field_size)
            for j in range(n_elements):
                expected[j] = (expected[j] + elems[j]) % field_size

        self.assertEqual(agg_masked, expected)

        # Verify actual values
        agg_bytes = field_elements_to_model_bytes(agg_masked)
        vals = struct.unpack(">IIII", agg_bytes)
        self.assertEqual(vals, (166, 292, 418, 544))


if __name__ == "__main__":
    unittest.main()
