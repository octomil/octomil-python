"""Tests for client-side Secure Aggregation (SecAgg)."""

import struct
import unittest

from octomil.secagg import (
    DEFAULT_FIELD_SIZE,
    SecAggClient,
    SecAggConfig,
    ShamirShare,
    _derive_mask_elements,
    _mod_inverse,
    field_elements_to_model_bytes,
    generate_shares,
    model_bytes_to_field_elements,
    reconstruct_secret,
)


# ---------------------------------------------------------------------------
# Shamir secret sharing
# ---------------------------------------------------------------------------


class ShamirSharingTests(unittest.TestCase):
    """Core Shamir secret-sharing algebra."""

    def test_share_and_reconstruct_basic(self):
        secret = 42
        shares = generate_shares(secret, threshold=3, total_shares=5)
        self.assertEqual(len(shares), 5)
        reconstructed = reconstruct_secret(shares[:3])
        self.assertEqual(reconstructed, secret)

    def test_reconstruct_with_any_subset(self):
        secret = 12345
        shares = generate_shares(secret, threshold=3, total_shares=5)
        # Any 3-of-5 subset should work.
        for combo in [[0, 1, 2], [0, 2, 4], [1, 3, 4], [2, 3, 4]]:
            subset = [shares[i] for i in combo]
            self.assertEqual(reconstruct_secret(subset), secret)

    def test_insufficient_shares_gives_wrong_result(self):
        secret = 99999
        shares = generate_shares(secret, threshold=3, total_shares=5)
        # Only 2 shares -- reconstruction should (overwhelmingly) fail.
        result = reconstruct_secret(shares[:2])
        # Extremely unlikely to equal the secret by chance.
        self.assertNotEqual(result, secret)

    def test_threshold_one(self):
        """Threshold-1 sharing means every share equals the secret."""
        secret = 7
        shares = generate_shares(secret, threshold=1, total_shares=4)
        for s in shares:
            self.assertEqual(reconstruct_secret([s]), secret)

    def test_threshold_equals_total(self):
        secret = 2**64
        shares = generate_shares(secret, threshold=5, total_shares=5)
        self.assertEqual(reconstruct_secret(shares), secret)

    def test_zero_secret(self):
        shares = generate_shares(0, threshold=2, total_shares=3)
        self.assertEqual(reconstruct_secret(shares[:2]), 0)

    def test_large_secret(self):
        secret = DEFAULT_FIELD_SIZE - 1
        shares = generate_shares(secret, threshold=3, total_shares=5)
        self.assertEqual(reconstruct_secret(shares[:3]), secret)

    def test_invalid_threshold_zero(self):
        with self.assertRaises(ValueError):
            generate_shares(1, threshold=0, total_shares=3)

    def test_invalid_threshold_exceeds_total(self):
        with self.assertRaises(ValueError):
            generate_shares(1, threshold=4, total_shares=3)


class ModInverseTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual((_mod_inverse(3, 7) * 3) % 7, 1)

    def test_large_prime(self):
        a = 123456789
        inv = _mod_inverse(a, DEFAULT_FIELD_SIZE)
        self.assertEqual((a * inv) % DEFAULT_FIELD_SIZE, 1)


# ---------------------------------------------------------------------------
# Field-element encoding
# ---------------------------------------------------------------------------


class FieldEncodingTests(unittest.TestCase):
    def test_roundtrip(self):
        data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        elements = model_bytes_to_field_elements(data)
        result = field_elements_to_model_bytes(elements)
        self.assertEqual(result, data)

    def test_padding(self):
        """Data whose length is not a multiple of 4 is zero-padded."""
        data = b"\x01\x02\x03"
        elements = model_bytes_to_field_elements(data)
        result = field_elements_to_model_bytes(elements)
        self.assertEqual(result, b"\x01\x02\x03\x00")

    def test_empty(self):
        self.assertEqual(model_bytes_to_field_elements(b""), [])
        self.assertEqual(field_elements_to_model_bytes([]), b"")


# ---------------------------------------------------------------------------
# Mask derivation
# ---------------------------------------------------------------------------


class MaskDerivationTests(unittest.TestCase):
    def test_deterministic(self):
        seed = b"test-seed-1234567890123456789012"
        m1 = _derive_mask_elements(seed, 10)
        m2 = _derive_mask_elements(seed, 10)
        self.assertEqual(m1, m2)

    def test_different_seeds(self):
        m1 = _derive_mask_elements(b"seed_a_32_bytes_1234567890123456", 10)
        m2 = _derive_mask_elements(b"seed_b_32_bytes_1234567890123456", 10)
        self.assertNotEqual(m1, m2)

    def test_requested_count(self):
        mask = _derive_mask_elements(b"x" * 32, 100)
        self.assertEqual(len(mask), 100)


# ---------------------------------------------------------------------------
# ShamirShare serialisation
# ---------------------------------------------------------------------------


class ShareSerializationTests(unittest.TestCase):
    def test_roundtrip_single(self):
        share = ShamirShare(index=3, value=999, modulus=DEFAULT_FIELD_SIZE)
        data = share.to_bytes()
        restored, offset = ShamirShare.from_bytes(data)
        self.assertEqual(restored.index, share.index)
        self.assertEqual(restored.value, share.value)

    def test_roundtrip_batch(self):
        shares = [
            ShamirShare(index=i, value=i * 1000, modulus=DEFAULT_FIELD_SIZE)
            for i in range(1, 6)
        ]
        data = SecAggClient.serialize_shares(shares)
        restored = SecAggClient.deserialize_shares(data)
        self.assertEqual(len(restored), 5)
        for orig, rest in zip(shares, restored):
            self.assertEqual(orig.index, rest.index)
            self.assertEqual(orig.value, rest.value)


# ---------------------------------------------------------------------------
# SecAggClient high-level
# ---------------------------------------------------------------------------


class SecAggClientTests(unittest.TestCase):
    def _make_config(self, **overrides):
        defaults = dict(
            session_id="sess-1",
            round_id="round-1",
            threshold=2,
            total_clients=3,
        )
        defaults.update(overrides)
        return SecAggConfig(**defaults)

    def test_generate_key_shares_count(self):
        sac = SecAggClient(self._make_config(total_clients=5, threshold=3))
        shares = sac.generate_key_shares()
        self.assertEqual(len(shares), 5)

    def test_mask_changes_data(self):
        sac = SecAggClient(self._make_config())
        sac.generate_key_shares()
        raw = b"\x00\x00\x00\x01" * 10
        masked = sac.mask_model_update(raw)
        self.assertNotEqual(masked, raw)

    def test_mask_is_deterministic_per_instance(self):
        """Two calls to mask_model_update with the same client use the same seed."""
        sac = SecAggClient(self._make_config())
        sac.generate_key_shares()
        raw = b"\x00\x00\x00\x05" * 4
        masked1 = sac.mask_model_update(raw)
        masked2 = sac.mask_model_update(raw)
        self.assertEqual(masked1, masked2)

    def test_different_clients_produce_different_masks(self):
        cfg = self._make_config()
        sac1 = SecAggClient(cfg)
        sac2 = SecAggClient(cfg)
        sac1.generate_key_shares()
        sac2.generate_key_shares()
        raw = b"\x00\x00\x00\x0a" * 8
        self.assertNotEqual(sac1.mask_model_update(raw), sac2.mask_model_update(raw))

    def test_get_seed_share_for_peer(self):
        sac = SecAggClient(self._make_config(total_clients=5, threshold=3))
        _shares = sac.generate_key_shares()
        peer_share = sac.get_seed_share_for_peer(3)
        self.assertIsNotNone(peer_share)
        self.assertEqual(peer_share.index, 3)

    def test_get_seed_share_for_unknown_peer(self):
        sac = SecAggClient(self._make_config(total_clients=3, threshold=2))
        sac.generate_key_shares()
        self.assertIsNone(sac.get_seed_share_for_peer(99))

    def test_get_seed_share_before_keygen(self):
        sac = SecAggClient(self._make_config())
        self.assertIsNone(sac.get_seed_share_for_peer(1))


# ---------------------------------------------------------------------------
# End-to-end: mask + reconstruct
# ---------------------------------------------------------------------------


class EndToEndSecAggTests(unittest.TestCase):
    """Simulate multiple clients masking, then the server unmasking the aggregate."""

    def test_aggregate_three_clients(self):
        """Three clients mask their updates. The sum of masks can be removed to
        recover the sum of raw updates (mod 2^32 per element)."""
        n_clients = 3
        threshold = 2
        field_size = DEFAULT_FIELD_SIZE

        # Each client has a 4-element update.
        raw_updates = [
            struct.pack(">IIII", 10, 20, 30, 40),
            struct.pack(">IIII", 1, 2, 3, 4),
            struct.pack(">IIII", 100, 200, 300, 400),
        ]

        clients = []
        for _ in range(n_clients):
            cfg = SecAggConfig(
                session_id="s1",
                round_id="r1",
                threshold=threshold,
                total_clients=n_clients,
                field_size=field_size,
            )
            clients.append(SecAggClient(cfg))

        # Phase 1: each client generates shares.
        all_shares = []
        for c in clients:
            all_shares.append(c.generate_key_shares())

        # Phase 2: each client masks its update.
        masked_updates = []
        for c, raw in zip(clients, raw_updates):
            masked_updates.append(c.mask_model_update(raw))

        # Server: sum masked elements.
        n_elements = 4
        masked_elements = [
            model_bytes_to_field_elements(m, field_size) for m in masked_updates
        ]
        agg_masked = [0] * n_elements
        for elems in masked_elements:
            for j in range(n_elements):
                agg_masked[j] = (agg_masked[j] + elems[j]) % field_size

        # Server: reconstruct each client's seed from their shares and subtract masks.
        for i, c in enumerate(clients):
            # Collect threshold shares for client i's seed.
            # In the real protocol other clients provide these; here we grab them directly.
            seed_int = int.from_bytes(c._seed, "big") % field_size
            shares_for_i = all_shares[i][:threshold]
            reconstructed_seed_int = reconstruct_secret(shares_for_i)
            self.assertEqual(reconstructed_seed_int, seed_int)

            mask = _derive_mask_elements(c._seed, n_elements, field_size)
            for j in range(n_elements):
                agg_masked[j] = (agg_masked[j] - mask[j]) % field_size

        # The result should equal the element-wise sum of raw updates (mod 2^32).
        raw_elements = [
            model_bytes_to_field_elements(r, field_size) for r in raw_updates
        ]
        expected = [0] * n_elements
        for elems in raw_elements:
            for j in range(n_elements):
                expected[j] = (expected[j] + elems[j]) % field_size

        self.assertEqual(agg_masked, expected)

        # Convert back to bytes and verify.
        agg_bytes = field_elements_to_model_bytes(agg_masked)
        vals = struct.unpack(">IIII", agg_bytes)
        self.assertEqual(vals, (111, 222, 333, 444))


# ---------------------------------------------------------------------------
# FederatedClient integration (SecAgg flag)
# ---------------------------------------------------------------------------


class FederatedClientSecAggFlagTests(unittest.TestCase):
    """Verify that FederatedClient passes the secure_aggregation flag."""

    def test_default_off(self):
        from octomil.federated_client import FederatedClient

        client = FederatedClient(auth_token_provider=lambda: "t", org_id="o")
        self.assertFalse(client.secure_aggregation)

    def test_explicit_on(self):
        from octomil.federated_client import FederatedClient

        client = FederatedClient(
            auth_token_provider=lambda: "t",
            org_id="o",
            secure_aggregation=True,
        )
        self.assertTrue(client.secure_aggregation)


if __name__ == "__main__":
    unittest.main()
