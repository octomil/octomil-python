"""SecAgg+ client — full protocol following Bonawitz et al.

Implements :class:`SecAggPlusClient` and :class:`SecAggPlusConfig` with
dual X25519 key pairs, AES-GCM encrypted share transport, stochastic
quantization, and pairwise + self masking.
"""

from __future__ import annotations

import logging
import secrets
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .crypto import (
    ECKeyPair,
    _pseudo_rand_gen,
    decrypt_share,
    derive_pairwise_mask,
    encrypt_share,
    generate_pairwise_key,
    generate_share_encryption_key,
)
from .quantization import quantize
from .shamir import (
    DEFAULT_FIELD_SIZE,
    ByteShamirShare,
    ShamirShare,
    create_shares_bytes,
    generate_shares,
)

logger = logging.getLogger(__name__)

# Default mod range matching Flower: 2^32 for integer-mod masking.
SECAGG_PLUS_MOD_RANGE = 1 << 32


@dataclass
class SecAggPlusConfig:
    """Configuration for the SecAgg+ protocol.

    Follows the Flower SecAgg+ parameter naming.
    """

    session_id: str
    round_id: str
    threshold: int  # Shamir reconstruction threshold
    total_clients: int  # Total number of participating clients
    my_index: int  # 1-based index of this client
    clipping_range: float = 3.0  # Symmetric clip range for quantization
    target_range: int = 1 << 16  # Quantization target range
    mod_range: int = SECAGG_PLUS_MOD_RANGE  # Modular arithmetic range
    # Shamir still uses a large prime for share reconstruction.
    field_size: int = DEFAULT_FIELD_SIZE


class SecAggPlusClient:
    """Client-side SecAgg+ state machine implementing the 4-stage Flower protocol.

    **Dual key pairs** (X25519, 32-byte raw keys):

    - ``(sk1, pk1)`` -- Used to compute pairwise masks via
      ``ECDH(sk1_i, pk1_j)`` + ``HKDF(info="secagg-pairwise-mask")``
    - ``(sk2, pk2)`` -- Used to encrypt Shamir shares via
      ``AES-GCM(ECDH(sk2_i, pk2_j) + HKDF(info="secagg-share-encryption"))``

    **Shamir-shared secrets**:

    - ``rd_seed`` -- random seed for the self-mask (private mask)
    - ``sk1`` -- pairwise mask private key (needed for dropped-client recovery),
      shared using byte-level Shamir (16-byte chunks) to handle the full 32 bytes

    Stage 1 -- Setup:
        Generate two X25519 key pairs, publish ``pk1`` and ``pk2``.

    Stage 2 -- Share Keys:
        After receiving all peers' ``(pk1, pk2)`` pairs:
        - Generate ``rd_seed`` and Shamir-share it (integer Shamir).
        - Byte-level Shamir-share ``sk1`` (the pairwise mask private key).
        - Encrypt each ``(rd_seed_share, sk1_share)`` pair with AES-256-GCM
          using the shared key derived from ``ECDH(sk2, peer_pk2)`` +
          ``HKDF(info="secagg-share-encryption")``.

    Stage 3 -- Collect Masked Vectors:
        - Clip + stochastic quantize model update to integers.
        - Add self-mask (PRG(``rd_seed``)) mod ``mod_range``.
        - Add/subtract pairwise masks (PRG(ECDH(``sk1``, peer ``pk1``)))
          mod ``mod_range``.
        - Upload the masked quantized vector.

    Stage 4 -- Unmask:
        - For surviving (active) peers: reveal ``rd_seed`` shares so the
          server can reconstruct and remove self-masks.
        - For dropped (dead) peers: reveal ``sk1`` byte-level Shamir shares
          so the server can reconstruct ``sk1``, recompute pairwise masks,
          and remove them.

    Example::

        client = SecAggPlusClient(config)

        # Stage 1: Setup
        pk1, pk2 = client.get_public_keys()
        # ... exchange public keys via server ...

        # Stage 2: Share Keys
        client.receive_peer_public_keys(peer_keys)  # {idx: (pk1, pk2)}
        encrypted_shares = client.generate_encrypted_shares()
        # ... send encrypted_shares[peer_idx] to each peer via server ...
        client.receive_encrypted_shares(shares_from_peers)

        # Stage 3: Collect Masked Vectors
        masked = client.mask_model_update(float_values)
        # ... upload masked ...

        # Stage 4: Unmask
        nids, shares = client.unmask(active_indices, dropped_indices)
        # ... send to server ...
    """

    def __init__(self, config: SecAggPlusConfig) -> None:
        self.config = config
        # Dual key pairs (X25519, 32-byte raw keys).
        self._kp1: ECKeyPair = ECKeyPair.generate()  # pairwise masks
        self._kp2: ECKeyPair = ECKeyPair.generate()  # share encryption

        # Self-mask seed (Shamir-shared as rd_seed).
        self._rd_seed: bytes = secrets.token_bytes(32)

        # Shamir shares of rd_seed (integer Shamir) and sk1 (byte-level Shamir).
        self._rd_seed_shares: Optional[List[ShamirShare]] = None
        self._sk1_shares: Optional[List[ByteShamirShare]] = None

        # Peer state (populated in stage 2).
        self._peer_public_keys: Dict[int, Tuple[bytes, bytes]] = {}  # idx -> (pk1, pk2)
        self._ss2_dict: Dict[int, bytes] = {}  # idx -> shared_key from (sk2, pk2)

        # Shares received from peers (populated in stage 2).
        self._received_rd_seed_shares: Dict[int, bytes] = {}  # sender_idx -> share_bytes
        self._received_sk1_shares: Dict[int, bytes] = {}  # sender_idx -> share_bytes

    # ------------------------------------------------------------------
    # Stage 1: Setup
    # ------------------------------------------------------------------

    def get_public_keys(self) -> Tuple[bytes, bytes]:
        """Return this client's two public keys ``(pk1, pk2)`` as 32-byte raw X25519 keys."""
        return self._kp1.public_key_bytes, self._kp2.public_key_bytes

    # Convenience alias for single-key API migration.
    def get_public_key(self) -> Tuple[bytes, bytes]:
        """Alias for :meth:`get_public_keys`."""
        return self.get_public_keys()

    # ------------------------------------------------------------------
    # Stage 2: Share Keys
    # ------------------------------------------------------------------

    def receive_peer_public_keys(
        self,
        peer_keys: Dict[int, Tuple[bytes, bytes]],
    ) -> None:
        """Store public keys received from all peers (including self).

        *peer_keys* maps peer index (1-based) to ``(pk1_raw, pk2_raw)``
        where each key is a 32-byte raw X25519 public key.
        """
        self._peer_public_keys = dict(peer_keys)

        # Compute ECDH shared keys with each peer using sk2/pk2 + share-encryption info.
        for idx, (_, pk2) in self._peer_public_keys.items():
            if idx == self.config.my_index:
                continue
            self._ss2_dict[idx] = generate_share_encryption_key(
                self._kp2.private_key_bytes,
                pk2,
            )

    def generate_encrypted_shares(self) -> Dict[int, bytes]:
        """Shamir-share ``rd_seed`` and ``sk1``, encrypt, and return.

        Returns a dict mapping peer index -> encrypted payload.
        Each payload contains ``(rd_seed_share, sk1_share)`` encrypted with
        AES-256-GCM using the key derived from ``ECDH(sk2, peer_pk2)`` +
        ``HKDF(info="secagg-share-encryption")``.

        ``rd_seed`` uses integer Shamir (fits in GF(2^127-1) after mod).
        ``sk1`` uses byte-level Shamir (32-byte X25519 key -> 2 chunks of 16 bytes).
        """
        # Shamir-share rd_seed (integer Shamir).
        seed_int = int.from_bytes(self._rd_seed, "big") % self.config.field_size
        self._rd_seed_shares = generate_shares(
            secret=seed_int,
            threshold=self.config.threshold,
            total_shares=self.config.total_clients,
            modulus=self.config.field_size,
        )

        # Byte-level Shamir-share sk1 (32-byte X25519 private key).
        self._sk1_shares = create_shares_bytes(
            secret=self._kp1.private_key_bytes,
            threshold=self.config.threshold,
            total_shares=self.config.total_clients,
            modulus=self.config.field_size,
        )

        encrypted_outgoing: Dict[int, bytes] = {}
        for i in range(self.config.total_clients):
            peer_idx = i + 1  # 1-based
            rd_share = self._rd_seed_shares[i]
            sk1_share = self._sk1_shares[i]

            if peer_idx == self.config.my_index:
                # Keep own shares locally.
                self._received_rd_seed_shares[self.config.my_index] = rd_share.to_bytes()
                self._received_sk1_shares[self.config.my_index] = sk1_share.to_bytes()
                continue

            shared_key = self._ss2_dict.get(peer_idx)
            if shared_key is None:
                continue

            # Concatenate rd_seed_share and sk1_share with a length prefix.
            rd_bytes = rd_share.to_bytes()
            sk1_bytes = sk1_share.to_bytes()
            plaintext = struct.pack(">I", len(rd_bytes)) + rd_bytes + sk1_bytes
            encrypted_outgoing[peer_idx] = encrypt_share(plaintext, shared_key)

        return encrypted_outgoing

    def receive_encrypted_shares(self, shares: Dict[int, bytes]) -> None:
        """Receive and decrypt share pairs from peers.

        *shares* maps sender index -> encrypted payload.
        """
        for sender_idx, encrypted in shares.items():
            shared_key = self._ss2_dict.get(sender_idx)
            if shared_key is None:
                logger.warning(
                    "No shared key for peer %d, skipping share",
                    sender_idx,
                )
                continue
            plaintext = decrypt_share(encrypted, shared_key)

            # Parse: <4 bytes rd_len><rd_share_bytes><sk1_share_bytes>
            rd_len = struct.unpack(">I", plaintext[:4])[0]
            rd_bytes = plaintext[4 : 4 + rd_len]
            sk1_bytes = plaintext[4 + rd_len :]

            self._received_rd_seed_shares[sender_idx] = rd_bytes
            self._received_sk1_shares[sender_idx] = sk1_bytes

    # ------------------------------------------------------------------
    # Stage 3: Collect Masked Vectors
    # ------------------------------------------------------------------

    def mask_model_update(self, values: List[float]) -> List[int]:
        """Clip, stochastic-quantize, and mask a model update vector.

        Returns a list of masked integers (mod ``mod_range``).

        The pairwise mask between clients ``i`` and ``j`` is derived from
        ``ECDH(sk1_i, pk1_j)``.  For ``i > j`` the mask is **added**; for
        ``i < j`` it is **subtracted** (matching Flower convention).  These
        cancel in the aggregate sum.

        The self-mask is derived from ``rd_seed`` via PRG and **added** by
        every client.  The server must reconstruct it from Shamir shares
        and subtract it.
        """
        mod = self.config.mod_range
        my_idx = self.config.my_index
        n = len(values)

        # Step 1: Clip + stochastic quantize.
        quantized = quantize(values, self.config.clipping_range, self.config.target_range)

        # Step 2: Add self-mask (private mask from rd_seed).
        self_mask = _pseudo_rand_gen(self._rd_seed, mod, n)
        masked = [(q + m) % mod for q, m in zip(quantized, self_mask)]

        # Step 3: Add/subtract pairwise masks.
        for peer_idx, (pk1, _) in self._peer_public_keys.items():
            if peer_idx == my_idx:
                continue
            pairwise_key = generate_pairwise_key(self._kp1.private_key_bytes, pk1)
            pairwise_mask = derive_pairwise_mask(pairwise_key, n, mod)
            if my_idx > peer_idx:
                # Add
                masked = [(m + p) % mod for m, p in zip(masked, pairwise_mask)]
            else:
                # Subtract
                masked = [(m - p) % mod for m, p in zip(masked, pairwise_mask)]

        return masked

    # ------------------------------------------------------------------
    # Stage 4: Unmask
    # ------------------------------------------------------------------

    def unmask(
        self,
        active_indices: List[int],
        dropped_indices: List[int],
    ) -> Tuple[List[int], List[bytes]]:
        """Reveal shares for the unmask phase.

        For **active** peers: reveal ``rd_seed`` shares (so the server can
        reconstruct each peer's self-mask seed and subtract it).

        For **dropped** peers: reveal ``sk1`` shares (so the server can
        reconstruct the dropped peer's ``sk1``, recompute their pairwise
        masks with each surviving client, and cancel them).

        Returns ``(node_ids, share_bytes_list)`` matching the Flower wire
        format.
        """
        all_nids = list(active_indices) + list(dropped_indices)
        shares: List[bytes] = []

        for nid in active_indices:
            share_bytes = self._received_rd_seed_shares.get(nid, b"")
            shares.append(share_bytes)

        for nid in dropped_indices:
            share_bytes = self._received_sk1_shares.get(nid, b"")
            shares.append(share_bytes)

        return all_nids, shares

    # Legacy helpers for backward compatibility.

    def reveal_shares_for_dropped(
        self,
        dropped_indices: List[int],
    ) -> Dict[int, bytes]:
        """Reveal ``sk1`` shares for dropped peers.

        Returns a dict mapping dropped peer index -> serialized share bytes.
        """
        revealed: Dict[int, bytes] = {}
        for idx in dropped_indices:
            share_bytes = self._received_sk1_shares.get(idx)
            if share_bytes is not None:
                revealed[idx] = share_bytes
        return revealed

    def get_rd_seed_share(self, peer_index: int) -> Optional[ShamirShare]:
        """Return the Shamir share of this client's ``rd_seed`` for *peer_index*."""
        if self._rd_seed_shares is None:
            return None
        for share in self._rd_seed_shares:
            if share.index == peer_index:
                return share
        return None

    def get_sk1_share(self, peer_index: int) -> Optional[ByteShamirShare]:
        """Return the byte-level Shamir share of this client's ``sk1`` for *peer_index*."""
        if self._sk1_shares is None:
            return None
        for share in self._sk1_shares:
            if share.index == peer_index:
                return share
        return None

    # Alias for backward compat.
    def get_own_share(self, peer_index: int) -> Optional[ShamirShare]:
        """Alias for :meth:`get_rd_seed_share`."""
        return self.get_rd_seed_share(peer_index)
