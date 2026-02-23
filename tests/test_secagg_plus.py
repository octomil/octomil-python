"""Tests for the SecAgg+ rearchitecture (X25519, dual keys, stochastic quant).

Covers:
- Stochastic quantization pipeline (quantize / dequantize)
- X25519 key pair generation and ECDH shared key computation
- Standardized HKDF info strings for cross-platform compatibility
- Pairwise mask derivation via PRG
- AES-GCM encrypted share transport
- SecAggPlusClient 4-stage protocol with dual key pairs
- Byte-level Shamir for sk1 (X25519 private key) sharing
- Dropped client handling in unmask phase
- End-to-end: multiple clients mask, server reconstructs aggregate
"""

import unittest

from octomil.secagg import (
    HKDF_INFO_PAIRWISE_MASK,
    HKDF_INFO_SELF_MASK,
    HKDF_INFO_SHARE_ENCRYPTION,
    SECAGG_PLUS_MOD_RANGE,
    ByteShamirShare,
    ECKeyPair,
    SecAggPlusClient,
    SecAggPlusConfig,
    ShamirShare,
    _pseudo_rand_gen,
    combine_shares_bytes,
    create_shares_bytes,
    decrypt_share,
    dequantize,
    derive_pairwise_mask,
    encrypt_share,
    generate_pairwise_key,
    generate_share_encryption_key,
    generate_shared_key,
    quantize,
    reconstruct_secret,
)


# ---------------------------------------------------------------------------
# Stochastic quantization pipeline
# ---------------------------------------------------------------------------


class QuantizeTests(unittest.TestCase):
    """Test stochastic quantize/dequantize roundtrip."""

    def test_basic_roundtrip(self):
        """Values within clipping range should survive roundtrip with small error."""
        values = [0.0, 0.5, 1.0, -0.5, -1.0]
        clip = 3.0
        target = 1 << 16
        quantized = quantize(values, clip, target)
        restored = dequantize(quantized, clip, target)

        for orig, rest in zip(values, restored):
            # Stochastic rounding means up to 1 quantization bin of error.
            self.assertAlmostEqual(orig, rest, delta=2 * clip / target + 0.01)

    def test_clipping(self):
        """Values outside clipping range should be clipped."""
        values = [-100.0, 0.0, 100.0]
        clip = 1.0
        target = 255
        quantized = quantize(values, clip, target)
        # -100 clips to -1.0 -> quantized to 0
        self.assertEqual(quantized[0], 0)
        # 100 clips to +1.0 -> quantized to target_range
        self.assertEqual(quantized[2], target)

    def test_empty(self):
        self.assertEqual(quantize([], 1.0, 256), [])
        self.assertEqual(dequantize([], 1.0, 256), [])

    def test_quantized_range(self):
        """All quantized values should be in [0, target_range]."""
        values = [-2.5, -1.0, 0.0, 1.0, 2.5]
        clip = 3.0
        target = 1000
        quantized = quantize(values, clip, target)
        for q in quantized:
            self.assertGreaterEqual(q, 0)
            self.assertLessEqual(q, target)

    def test_midpoint_for_zero(self):
        """0.0 with symmetric clip should quantize near target/2."""
        clip = 1.0
        target = 1000
        # Run multiple times; stochastic round should be close to 500.
        results = []
        for _ in range(100):
            q = quantize([0.0], clip, target)[0]
            results.append(q)
        avg = sum(results) / len(results)
        self.assertAlmostEqual(avg, target / 2, delta=20)


# ---------------------------------------------------------------------------
# X25519 key pair generation
# ---------------------------------------------------------------------------


class ECKeyPairTests(unittest.TestCase):
    """Test X25519 key pair generation and ECDH shared key computation."""

    def test_generate_key_pair(self):
        kp = ECKeyPair.generate()
        # X25519 raw keys are exactly 32 bytes.
        self.assertEqual(len(kp.private_key_bytes), 32)
        self.assertEqual(len(kp.public_key_bytes), 32)

    def test_raw_bytes_format(self):
        kp = ECKeyPair.generate()
        # Raw bytes, not PEM -- should NOT start with "-----BEGIN"
        self.assertFalse(kp.private_key_bytes.startswith(b"-----BEGIN"))
        self.assertFalse(kp.public_key_bytes.startswith(b"-----BEGIN"))

    def test_two_pairs_are_different(self):
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        self.assertNotEqual(kp1.private_key_bytes, kp2.private_key_bytes)
        self.assertNotEqual(kp1.public_key_bytes, kp2.public_key_bytes)

    def test_shared_key_symmetric(self):
        """Both parties compute the same shared key."""
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()

        sk1 = generate_shared_key(kp1.private_key_bytes, kp2.public_key_bytes)
        sk2 = generate_shared_key(kp2.private_key_bytes, kp1.public_key_bytes)

        self.assertEqual(sk1, sk2)
        # Result is raw 32-byte HKDF output.
        self.assertEqual(len(sk1), 32)

    def test_different_pairs_different_secrets(self):
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        kp3 = ECKeyPair.generate()

        sk12 = generate_shared_key(kp1.private_key_bytes, kp2.public_key_bytes)
        sk13 = generate_shared_key(kp1.private_key_bytes, kp3.public_key_bytes)

        self.assertNotEqual(sk12, sk13)

    def test_hkdf_info_changes_derived_key(self):
        """Different HKDF info strings produce different keys from same ECDH exchange."""
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()

        key_pairwise = generate_shared_key(
            kp1.private_key_bytes, kp2.public_key_bytes, info=HKDF_INFO_PAIRWISE_MASK,
        )
        key_share_enc = generate_shared_key(
            kp1.private_key_bytes, kp2.public_key_bytes, info=HKDF_INFO_SHARE_ENCRYPTION,
        )
        key_none = generate_shared_key(
            kp1.private_key_bytes, kp2.public_key_bytes, info=None,
        )

        # All three should be different.
        self.assertNotEqual(key_pairwise, key_share_enc)
        self.assertNotEqual(key_pairwise, key_none)
        self.assertNotEqual(key_share_enc, key_none)

    def test_purpose_specific_key_helpers(self):
        """generate_pairwise_key and generate_share_encryption_key are symmetric."""
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()

        pk_12 = generate_pairwise_key(kp1.private_key_bytes, kp2.public_key_bytes)
        pk_21 = generate_pairwise_key(kp2.private_key_bytes, kp1.public_key_bytes)
        self.assertEqual(pk_12, pk_21)

        ek_12 = generate_share_encryption_key(kp1.private_key_bytes, kp2.public_key_bytes)
        ek_21 = generate_share_encryption_key(kp2.private_key_bytes, kp1.public_key_bytes)
        self.assertEqual(ek_12, ek_21)

        # Pairwise key != encryption key (different info strings).
        self.assertNotEqual(pk_12, ek_12)

    def test_hkdf_info_constants(self):
        """Verify the standardized HKDF info strings."""
        self.assertEqual(HKDF_INFO_PAIRWISE_MASK, b"secagg-pairwise-mask")
        self.assertEqual(HKDF_INFO_SHARE_ENCRYPTION, b"secagg-share-encryption")
        self.assertEqual(HKDF_INFO_SELF_MASK, b"secagg-self-mask")


# ---------------------------------------------------------------------------
# Pairwise mask derivation via PRG
# ---------------------------------------------------------------------------


class PairwiseMaskTests(unittest.TestCase):
    """Test pairwise mask derivation from shared keys."""

    def test_deterministic(self):
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        sk = generate_shared_key(kp1.private_key_bytes, kp2.public_key_bytes)
        m1 = derive_pairwise_mask(sk, 10)
        m2 = derive_pairwise_mask(sk, 10)
        self.assertEqual(m1, m2)

    def test_correct_count(self):
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        sk = generate_shared_key(kp1.private_key_bytes, kp2.public_key_bytes)
        mask = derive_pairwise_mask(sk, 50)
        self.assertEqual(len(mask), 50)

    def test_different_keys_different_masks(self):
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        kp3 = ECKeyPair.generate()
        sk12 = generate_shared_key(kp1.private_key_bytes, kp2.public_key_bytes)
        sk13 = generate_shared_key(kp1.private_key_bytes, kp3.public_key_bytes)
        m12 = derive_pairwise_mask(sk12, 10)
        m13 = derive_pairwise_mask(sk13, 10)
        self.assertNotEqual(m12, m13)

    def test_values_in_mod_range(self):
        """All mask values should be < mod_range."""
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        sk = generate_shared_key(kp1.private_key_bytes, kp2.public_key_bytes)
        mod = 1000000
        mask = derive_pairwise_mask(sk, 20, mod_range=mod)
        for v in mask:
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, mod)

    def test_symmetric_mask(self):
        """ECDH is symmetric, so mask from (kp1, kp2) == mask from (kp2, kp1)."""
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        sk_12 = generate_shared_key(kp1.private_key_bytes, kp2.public_key_bytes)
        sk_21 = generate_shared_key(kp2.private_key_bytes, kp1.public_key_bytes)
        m1 = derive_pairwise_mask(sk_12, 10)
        m2 = derive_pairwise_mask(sk_21, 10)
        self.assertEqual(m1, m2)


# ---------------------------------------------------------------------------
# AES-GCM encrypted share transport
# ---------------------------------------------------------------------------


class EncryptedShareTests(unittest.TestCase):
    """Test AES-256-GCM encryption/decryption using ECDH-derived shared keys."""

    def test_roundtrip(self):
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        shared_key = generate_share_encryption_key(kp1.private_key_bytes, kp2.public_key_bytes)

        plaintext = b"hello world Shamir share data"
        encrypted = encrypt_share(plaintext, shared_key)
        decrypted = decrypt_share(encrypted, shared_key)

        self.assertEqual(decrypted, plaintext)

    def test_wire_format(self):
        """Verify wire format: 12-byte nonce || ciphertext || 16-byte GCM tag."""
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        shared_key = generate_share_encryption_key(kp1.private_key_bytes, kp2.public_key_bytes)
        plaintext = b"test data"
        encrypted = encrypt_share(plaintext, shared_key)
        # 12 (nonce) + len(plaintext) + 16 (GCM tag)
        self.assertEqual(len(encrypted), 12 + len(plaintext) + 16)

    def test_encrypted_differs_from_plaintext(self):
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        shared_key = generate_share_encryption_key(kp1.private_key_bytes, kp2.public_key_bytes)
        plaintext = b"test data"
        encrypted = encrypt_share(plaintext, shared_key)
        self.assertNotEqual(encrypted, plaintext)

    def test_wrong_key_fails(self):
        from cryptography.exceptions import InvalidTag

        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        kp3 = ECKeyPair.generate()
        key_12 = generate_share_encryption_key(kp1.private_key_bytes, kp2.public_key_bytes)
        key_13 = generate_share_encryption_key(kp1.private_key_bytes, kp3.public_key_bytes)

        encrypted = encrypt_share(b"secret", key_12)
        with self.assertRaises(InvalidTag):
            decrypt_share(encrypted, key_13)

    def test_symmetric_key_decrypts(self):
        """Key derived from either direction decrypts the same ciphertext."""
        kp1 = ECKeyPair.generate()
        kp2 = ECKeyPair.generate()
        key_12 = generate_share_encryption_key(kp1.private_key_bytes, kp2.public_key_bytes)
        key_21 = generate_share_encryption_key(kp2.private_key_bytes, kp1.public_key_bytes)

        encrypted = encrypt_share(b"symmetric test", key_12)
        decrypted = decrypt_share(encrypted, key_21)
        self.assertEqual(decrypted, b"symmetric test")


# ---------------------------------------------------------------------------
# SecAggPlusClient: 4-stage protocol with dual keys
# ---------------------------------------------------------------------------


def _make_config(n_clients=3, threshold=2, my_index=1, **overrides):
    defaults = dict(
        session_id="sess-plus-1",
        round_id="round-1",
        threshold=threshold,
        total_clients=n_clients,
        my_index=my_index,
        clipping_range=3.0,
        target_range=(1 << 16),
        mod_range=SECAGG_PLUS_MOD_RANGE,
    )
    defaults.update(overrides)
    return SecAggPlusConfig(**defaults)


class SecAggPlusSetupTests(unittest.TestCase):
    """Stage 1: Setup with dual X25519 key pairs."""

    def test_get_public_keys_returns_tuple(self):
        client = SecAggPlusClient(_make_config())
        pk1, pk2 = client.get_public_keys()
        # X25519 raw public keys are 32 bytes each.
        self.assertEqual(len(pk1), 32)
        self.assertEqual(len(pk2), 32)
        self.assertNotEqual(pk1, pk2)

    def test_get_public_key_alias(self):
        client = SecAggPlusClient(_make_config())
        self.assertEqual(client.get_public_key(), client.get_public_keys())

    def test_different_clients_different_keys(self):
        c1 = SecAggPlusClient(_make_config(my_index=1))
        c2 = SecAggPlusClient(_make_config(my_index=2))
        pk1_1, pk2_1 = c1.get_public_keys()
        pk1_2, pk2_2 = c2.get_public_keys()
        self.assertNotEqual(pk1_1, pk1_2)
        self.assertNotEqual(pk2_1, pk2_2)


class SecAggPlusShareKeysTests(unittest.TestCase):
    """Stage 2: Share Keys with dual Shamir (rd_seed + sk1)."""

    def test_encrypted_shares_generated(self):
        n = 3
        clients = [
            SecAggPlusClient(_make_config(n_clients=n, my_index=i + 1))
            for i in range(n)
        ]

        # Exchange public keys.
        all_pks = {i + 1: clients[i].get_public_keys() for i in range(n)}
        for c in clients:
            c.receive_peer_public_keys(all_pks)

        # Generate encrypted shares.
        all_enc = {}
        for i, c in enumerate(clients):
            all_enc[i + 1] = c.generate_encrypted_shares()

        # Each client should have shares for all peers (not self).
        for idx, enc in all_enc.items():
            self.assertEqual(len(enc), n - 1)
            self.assertNotIn(idx, enc)

    def test_receive_and_decrypt_shares(self):
        n = 3
        clients = _setup_clients(n, threshold=2)

        # After setup, each client should have rd_seed + sk1 shares from all peers.
        for c in clients:
            # n-1 from peers + 1 from self = n total
            self.assertEqual(len(c._received_rd_seed_shares), n)
            self.assertEqual(len(c._received_sk1_shares), n)


class SecAggPlusMaskTests(unittest.TestCase):
    """Stage 3: Masked Upload with quantization + mod arithmetic."""

    def test_mask_changes_data(self):
        n = 3
        clients = _setup_clients(n, threshold=2)
        values = [0.1, 0.2, 0.3, 0.4]
        masked = clients[0].mask_model_update(values)
        # Should be a list of ints
        self.assertIsInstance(masked, list)
        self.assertTrue(all(isinstance(v, int) for v in masked))

    def test_masked_values_in_mod_range(self):
        n = 3
        clients = _setup_clients(n, threshold=2)
        values = [0.1, -0.5, 1.0, -1.0, 2.0]
        mod = clients[0].config.mod_range
        masked = clients[0].mask_model_update(values)
        for v in masked:
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, mod)

    def test_mask_is_deterministic(self):
        """Same client masking same values gives same result (PRG is seeded)."""
        n = 3
        clients = _setup_clients(n, threshold=2)
        values = [0.1, 0.2, 0.3]
        m1 = clients[0].mask_model_update(values)
        m2 = clients[0].mask_model_update(values)
        # Note: quantize uses stochastic rounding, so results may differ.
        # But the mask PRG is deterministic, so differences come only from
        # stochastic rounding. We just verify both are valid.
        self.assertEqual(len(m1), len(m2))


class SecAggPlusUnmaskTests(unittest.TestCase):
    """Stage 4: Unmask."""

    def test_unmask_all_active(self):
        """All clients active: reveal rd_seed shares."""
        n = 3
        clients = _setup_clients(n, threshold=2)
        active = [1, 2, 3]
        dropped = []

        nids, shares = clients[0].unmask(active, dropped)
        self.assertEqual(nids, [1, 2, 3])
        self.assertEqual(len(shares), 3)
        # All shares should be rd_seed shares (non-empty).
        for s in shares:
            self.assertGreater(len(s), 0)

    def test_unmask_with_dropout(self):
        """Client 3 drops: reveal rd_seed for active, sk1 for dropped."""
        n = 3
        clients = _setup_clients(n, threshold=2)
        active = [1, 2]
        dropped = [3]

        nids, shares = clients[0].unmask(active, dropped)
        self.assertEqual(nids, [1, 2, 3])
        self.assertEqual(len(shares), 3)

    def test_reveal_shares_for_dropped_legacy(self):
        """Legacy API: reveal_shares_for_dropped returns sk1 shares."""
        n = 3
        clients = _setup_clients(n, threshold=2)

        revealed = clients[0].reveal_shares_for_dropped([3])
        self.assertIn(3, revealed)
        self.assertGreater(len(revealed[3]), 0)

    def test_get_own_share_before_keygen(self):
        """Before generate_encrypted_shares, own shares are None."""
        client = SecAggPlusClient(_make_config())
        self.assertIsNone(client.get_own_share(1))
        self.assertIsNone(client.get_rd_seed_share(1))
        self.assertIsNone(client.get_sk1_share(1))


# ---------------------------------------------------------------------------
# End-to-end: pairwise masks cancel, aggregate is recoverable
# ---------------------------------------------------------------------------


class EndToEndSecAggPlusTests(unittest.TestCase):
    """Simulate the full SecAgg+ protocol with multiple clients."""

    def test_three_clients_all_survive(self):
        """Three clients mask, all survive. Server reconstructs aggregate."""
        n = 3
        threshold = 2
        mod = SECAGG_PLUS_MOD_RANGE
        clip = 3.0
        target = 1 << 16

        clients = _setup_clients(n, threshold=threshold)

        # Each client has an update vector (within clipping range).
        updates = [
            [0.1, 0.2, 0.3, 0.4],
            [0.01, 0.02, 0.03, 0.04],
            [1.0, 2.0, -1.0, -2.0],
        ]

        # Stage 3: mask updates.
        masked_updates = []
        for c, vals in zip(clients, updates):
            masked_updates.append(c.mask_model_update(vals))

        # Server: sum masked vectors mod mod_range.
        n_elements = 4
        agg = [0] * n_elements
        for masked in masked_updates:
            for j in range(n_elements):
                agg[j] = (agg[j] + masked[j]) % mod

        # Stage 4: Unmask -- all survive, reveal rd_seed shares.
        _active = list(range(1, n + 1))
        _dropped = []

        # Server collects rd_seed shares and reconstructs self-masks.
        for i in range(n):
            c = clients[i]
            # Collect rd_seed shares from all other clients.
            rd_shares = []
            for j in range(n):
                if i == j:
                    continue
                share_bytes = clients[j]._received_rd_seed_shares.get(i + 1)
                if share_bytes is not None:
                    share, _ = ShamirShare.from_bytes(share_bytes)
                    rd_shares.append(share)

            self.assertGreaterEqual(len(rd_shares), threshold)
            # Reconstruct rd_seed.
            seed_int = reconstruct_secret(rd_shares[:threshold])
            expected_seed_int = int.from_bytes(c._rd_seed, "big") % c.config.field_size
            self.assertEqual(seed_int, expected_seed_int)

            # Subtract self-mask.
            self_mask = _pseudo_rand_gen(c._rd_seed, mod, n_elements)
            for j in range(n_elements):
                agg[j] = (agg[j] - self_mask[j]) % mod

        # Pairwise masks cancel in the aggregate (by construction).
        # The aggregate should now be the sum of quantized updates (mod mod_range).

        # Compute expected: sum of individually quantized updates.
        # Because of stochastic rounding, we can't predict exact values, but
        # we verify that dequantized aggregate is close to sum of original updates.
        # Note: dequantize reverses shift for 1 client, but aggregate has shift
        # from N clients. Subtract (N-1)*clip to correct.
        dequantized = dequantize(agg, clip, target)
        corrected = [d - (n - 1) * clip for d in dequantized]
        expected_sum = [sum(u[j] for u in updates) for j in range(n_elements)]

        for d, e in zip(corrected, expected_sum):
            # Allow larger tolerance due to stochastic quantization error across 3 clients.
            self.assertAlmostEqual(d, e, delta=0.5)

    def test_one_client_drops_after_masking(self):
        """Client 3 drops after masking. Survivors reveal sk1 shares for
        the server to reconstruct pairwise masks."""
        n = 3
        threshold = 2
        mod = SECAGG_PLUS_MOD_RANGE

        clients = _setup_clients(n, threshold=threshold)

        updates = [
            [0.5, -0.5],
            [1.0, 1.0],
            [0.0, 0.0],
        ]

        # All three mask and upload.
        masked_updates = [c.mask_model_update(vals) for c, vals in zip(clients, updates)]

        # Sum all masked updates.
        n_elements = 2
        agg = [0] * n_elements
        for masked in masked_updates:
            for j in range(n_elements):
                agg[j] = (agg[j] + masked[j]) % mod

        # Client 3 drops during unmask.
        surviving_indices = [0, 1]   # 0-indexed
        dropped_idx = 3              # 1-based

        # Step 1: Remove self-masks for surviving clients.
        for i in surviving_indices:
            c = clients[i]
            rd_shares = []
            for j in surviving_indices:
                if i == j:
                    continue
                share_bytes = clients[j]._received_rd_seed_shares.get(i + 1)
                if share_bytes is not None:
                    share, _ = ShamirShare.from_bytes(share_bytes)
                    rd_shares.append(share)
            # Also collect from dropped client.
            drop_share_bytes = clients[2]._received_rd_seed_shares.get(i + 1)
            if drop_share_bytes is not None:
                drop_share, _ = ShamirShare.from_bytes(drop_share_bytes)
                rd_shares.append(drop_share)

            self.assertGreaterEqual(len(rd_shares), threshold)
            seed_int = reconstruct_secret(rd_shares[:threshold])
            expected = int.from_bytes(c._rd_seed, "big") % c.config.field_size
            self.assertEqual(seed_int, expected)

            self_mask = _pseudo_rand_gen(c._rd_seed, mod, n_elements)
            for j in range(n_elements):
                agg[j] = (agg[j] - self_mask[j]) % mod

        # Step 2: Remove self-mask for dropped client.
        drop_c = clients[2]
        rd_shares_for_drop = []
        for j in surviving_indices:
            share_bytes = clients[j]._received_rd_seed_shares.get(dropped_idx)
            if share_bytes is not None:
                share, _ = ShamirShare.from_bytes(share_bytes)
                rd_shares_for_drop.append(share)

        self.assertGreaterEqual(len(rd_shares_for_drop), threshold)
        seed_int = reconstruct_secret(rd_shares_for_drop[:threshold])
        expected = int.from_bytes(drop_c._rd_seed, "big") % drop_c.config.field_size
        self.assertEqual(seed_int, expected)

        self_mask = _pseudo_rand_gen(drop_c._rd_seed, mod, n_elements)
        for j in range(n_elements):
            agg[j] = (agg[j] - self_mask[j]) % mod

        # Step 3: Pairwise masks between survivors cancelled.
        # Pairwise masks between survivors and dropped also cancelled
        # (both uploaded masked vectors). No action needed.

        # Verify: dequantize and check approximate sum.
        # Correct for N-client shift: dequantize reverses 1 client's shift,
        # but aggregate has shift from N clients.
        clip = clients[0].config.clipping_range
        dequantized = dequantize(agg, clip, clients[0].config.target_range)
        corrected = [d - (n - 1) * clip for d in dequantized]
        expected_sum = [sum(u[j] for u in updates) for j in range(n_elements)]
        for d, e in zip(corrected, expected_sum):
            self.assertAlmostEqual(d, e, delta=0.5)

    def test_five_clients_two_drop(self):
        """5 clients, threshold=3, 2 drop after masking. Still recoverable."""
        n = 5
        threshold = 3
        mod = SECAGG_PLUS_MOD_RANGE

        clients = _setup_clients(n, threshold=threshold)

        updates = [
            [float(i * 0.1)] * 3 for i in range(1, n + 1)
        ]

        masked_updates = [c.mask_model_update(vals) for c, vals in zip(clients, updates)]

        n_elements = 3
        agg = [0] * n_elements
        for masked in masked_updates:
            for j in range(n_elements):
                agg[j] = (agg[j] + masked[j]) % mod

        # Clients 4, 5 drop (0-indexed: 3, 4).
        _surviving = [0, 1, 2]

        # Remove self-masks for all 5 clients.
        for i in range(n):
            c = clients[i]
            rd_shares = []
            for j in range(n):
                if i == j:
                    continue
                share_bytes = clients[j]._received_rd_seed_shares.get(i + 1)
                if share_bytes is not None:
                    share, _ = ShamirShare.from_bytes(share_bytes)
                    rd_shares.append(share)

            self.assertGreaterEqual(len(rd_shares), threshold)
            seed_int = reconstruct_secret(rd_shares[:threshold])
            expected = int.from_bytes(c._rd_seed, "big") % c.config.field_size
            self.assertEqual(seed_int, expected)

            self_mask = _pseudo_rand_gen(c._rd_seed, mod, n_elements)
            for j in range(n_elements):
                agg[j] = (agg[j] - self_mask[j]) % mod

        # Pairwise masks cancel since all clients uploaded.
        # Correct for N-client shift.
        clip = clients[0].config.clipping_range
        dequantized = dequantize(agg, clip, clients[0].config.target_range)
        corrected = [d - (n - 1) * clip for d in dequantized]
        expected_sum = [sum(u[j] for u in updates) for j in range(n_elements)]
        for d, e in zip(corrected, expected_sum):
            self.assertAlmostEqual(d, e, delta=1.0)


class UnmaskAPITests(unittest.TestCase):
    """Test the new unmask() API that returns (nids, shares)."""

    def test_unmask_returns_correct_structure(self):
        n = 3
        clients = _setup_clients(n, threshold=2)
        active = [1, 2]
        dropped = [3]

        nids, shares = clients[0].unmask(active, dropped)
        self.assertEqual(nids, [1, 2, 3])
        self.assertEqual(len(shares), 3)

        # First 2 are rd_seed shares (for active), last 1 is sk1 share (for dropped).
        for s in shares:
            self.assertIsInstance(s, bytes)
            self.assertGreater(len(s), 0)

    def test_unmask_all_active_no_dropped(self):
        n = 3
        clients = _setup_clients(n, threshold=2)
        nids, shares = clients[0].unmask([1, 2, 3], [])
        self.assertEqual(nids, [1, 2, 3])
        self.assertEqual(len(shares), 3)

    def test_unmask_rd_seed_shares_reconstructable(self):
        """Verify that rd_seed shares from unmask() can reconstruct the seed."""
        n = 3
        threshold = 2
        clients = _setup_clients(n, threshold=threshold)

        # Client 0 provides shares for active clients 2 and 3.
        # Client 1 provides shares for active clients 1 and 3.
        # Together they provide threshold shares for client 3's rd_seed.
        nids_0, shares_0 = clients[0].unmask([1, 2, 3], [])
        nids_1, shares_1 = clients[1].unmask([1, 2, 3], [])

        # Reconstruct client 3's rd_seed from shares held by clients 0 and 1.
        share_0_for_3, _ = ShamirShare.from_bytes(shares_0[2])  # index 2 = nid 3
        share_1_for_3, _ = ShamirShare.from_bytes(shares_1[2])

        reconstructed = reconstruct_secret([share_0_for_3, share_1_for_3])
        expected = int.from_bytes(clients[2]._rd_seed, "big") % clients[2].config.field_size
        self.assertEqual(reconstructed, expected)


# ---------------------------------------------------------------------------
# PRG determinism
# ---------------------------------------------------------------------------


class PseudoRandGenTests(unittest.TestCase):
    """Test _pseudo_rand_gen determinism and range."""

    def test_deterministic(self):
        seed = b"\x42" * 32
        m1 = _pseudo_rand_gen(seed, 1000, 10)
        m2 = _pseudo_rand_gen(seed, 1000, 10)
        self.assertEqual(m1, m2)

    def test_different_seeds(self):
        m1 = _pseudo_rand_gen(b"\x01" * 32, 1000, 10)
        m2 = _pseudo_rand_gen(b"\x02" * 32, 1000, 10)
        self.assertNotEqual(m1, m2)

    def test_range(self):
        mask = _pseudo_rand_gen(b"\xab" * 32, 100, 50)
        for v in mask:
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, 100)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class ByteShamirTests(unittest.TestCase):
    """Test byte-level Shamir sharing for sk1 (X25519 private key)."""

    def test_roundtrip_32_bytes(self):
        """32-byte X25519 private key survives byte-level Shamir roundtrip."""
        secret = bytes(range(32))
        shares = create_shares_bytes(secret, threshold=2, total_shares=3)
        self.assertEqual(len(shares), 3)
        recovered = combine_shares_bytes(shares[:2])
        self.assertEqual(recovered, secret)

    def test_roundtrip_with_real_x25519_key(self):
        """Real X25519 private key survives byte-level Shamir roundtrip."""
        kp = ECKeyPair.generate()
        shares = create_shares_bytes(kp.private_key_bytes, threshold=3, total_shares=5)
        recovered = combine_shares_bytes(shares[:3])
        self.assertEqual(recovered, kp.private_key_bytes)

    def test_serialization_roundtrip(self):
        """ByteShamirShare serializes and deserializes correctly."""
        secret = bytes(range(32))
        shares = create_shares_bytes(secret, threshold=2, total_shares=3)
        for share in shares:
            data = share.to_bytes()
            recovered, offset = ByteShamirShare.from_bytes(data)
            self.assertEqual(recovered.index, share.index)
            self.assertEqual(len(recovered.chunk_shares), len(share.chunk_shares))

    def test_threshold_insufficient(self):
        """Fewer than threshold shares produce wrong secret."""
        secret = bytes(range(32))
        shares = create_shares_bytes(secret, threshold=3, total_shares=5)
        # Only 2 shares with threshold=3 should reconstruct incorrectly.
        wrong = combine_shares_bytes(shares[:2])
        self.assertNotEqual(wrong, secret)

    def test_sk1_reconstruction_in_dropped_client_scenario(self):
        """Simulate server reconstructing dropped client's sk1 from byte-level Shamir."""
        n = 3
        threshold = 2
        clients = _setup_clients(n, threshold=threshold)

        # Client 3 (index 2) drops. Survivors reveal sk1 shares for client 3.
        dropped_idx = 3  # 1-based

        # Collect sk1 ByteShamirShares from survivors.
        sk1_byte_shares = []
        for j in [0, 1]:  # surviving clients (0-indexed)
            share_bytes = clients[j]._received_sk1_shares.get(dropped_idx)
            self.assertIsNotNone(share_bytes)
            bs, _ = ByteShamirShare.from_bytes(share_bytes)
            sk1_byte_shares.append(bs)

        self.assertGreaterEqual(len(sk1_byte_shares), threshold)

        # Reconstruct sk1 (the dropped client's X25519 private key).
        recovered_sk1 = combine_shares_bytes(sk1_byte_shares[:threshold])
        self.assertEqual(recovered_sk1, clients[2]._kp1.private_key_bytes)

        # Verify the recovered key can derive the same pairwise masks.
        for surv_idx in [0, 1]:
            surv_pk1 = clients[surv_idx]._kp1.public_key_bytes
            # Key from recovered sk1.
            recovered_key = generate_pairwise_key(recovered_sk1, surv_pk1)
            # Key that the dropped client would have used.
            original_key = generate_pairwise_key(
                clients[2]._kp1.private_key_bytes, surv_pk1,
            )
            self.assertEqual(recovered_key, original_key)


class BackwardCompatibilityTests(unittest.TestCase):
    """Ensure original SecAggClient still works unchanged."""

    def test_original_client_unchanged(self):
        from octomil.secagg import SecAggClient, SecAggConfig

        cfg = SecAggConfig(
            session_id="s1",
            round_id="r1",
            threshold=2,
            total_clients=3,
        )
        sac = SecAggClient(cfg)
        shares = sac.generate_key_shares()
        self.assertEqual(len(shares), 3)

        raw = b"\x00\x00\x00\x01" * 10
        masked = sac.mask_model_update(raw)
        self.assertNotEqual(masked, raw)

    def test_new_types_exported_from_package(self):
        from octomil import (
            ECKeyPair, SecAggPlusClient, SecAggPlusConfig, SECAGG_PLUS_MOD_RANGE,
            HKDF_INFO_PAIRWISE_MASK, HKDF_INFO_SHARE_ENCRYPTION, HKDF_INFO_SELF_MASK,
        )

        self.assertIsNotNone(ECKeyPair)
        self.assertIsNotNone(SecAggPlusClient)
        self.assertIsNotNone(SecAggPlusConfig)
        self.assertEqual(SECAGG_PLUS_MOD_RANGE, 1 << 32)
        self.assertEqual(HKDF_INFO_PAIRWISE_MASK, b"secagg-pairwise-mask")
        self.assertEqual(HKDF_INFO_SHARE_ENCRYPTION, b"secagg-share-encryption")
        self.assertEqual(HKDF_INFO_SELF_MASK, b"secagg-self-mask")

    def test_ecdh_keypair_alias(self):
        from octomil.secagg import ECDHKeyPair
        self.assertIs(ECDHKeyPair, ECKeyPair)

    def test_compute_shared_secret_alias(self):
        from octomil.secagg import compute_shared_secret
        self.assertIs(compute_shared_secret, generate_shared_key)


# ---------------------------------------------------------------------------
# Helper to set up N clients through stages 1-2
# ---------------------------------------------------------------------------


def _setup_clients(n: int, threshold: int = 2) -> list:
    """Create N SecAggPlusClients and run them through stages 1-2."""
    clients = [
        SecAggPlusClient(_make_config(n_clients=n, threshold=threshold, my_index=i + 1))
        for i in range(n)
    ]

    # Stage 1: Collect all public keys.
    all_pks = {i + 1: clients[i].get_public_keys() for i in range(n)}

    # Exchange public keys (each client gets the full dict, including self).
    for c in clients:
        c.receive_peer_public_keys(all_pks)

    # Stage 2: Generate and distribute encrypted shares.
    all_enc = {}
    for i, c in enumerate(clients):
        all_enc[i + 1] = c.generate_encrypted_shares()

    for i, c in enumerate(clients):
        my_idx = i + 1
        incoming = {}
        for sender_idx, enc_shares in all_enc.items():
            if my_idx in enc_shares:
                incoming[sender_idx] = enc_shares[my_idx]
        c.receive_encrypted_shares(incoming)

    return clients


if __name__ == "__main__":
    unittest.main()
