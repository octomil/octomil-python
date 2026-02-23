"""Client-side Secure Aggregation (SecAgg / SecAgg+) for federated learning.

Implements two protocol variants:

**SecAgg (basic)** -- :class:`SecAggClient`
  Single-seed masking with Shamir secret sharing.  Each client generates a
  random seed, splits it via Shamir, and adds a PRG-derived mask.  Simple but
  lacks pairwise masking and encrypted share transport.

**SecAgg+** -- :class:`SecAggPlusClient`
  Full protocol following Bonawitz et al. and the Flower SecAgg+ reference:
    1. **Setup** -- Generate two ECDH key pairs on X25519:
       ``(sk1, pk1)`` for pairwise masks, ``(sk2, pk2)`` for share encryption.
    2. **Share keys** -- Shamir-share both ``rd_seed`` (self-mask seed) and
       ``sk1`` (pairwise mask key).  Encrypt each pair of shares with AES-GCM
       using the ECDH shared secret derived from ``(sk2, peer_pk2)``.
    3. **Masked upload** -- clip + stochastic quantize → add pairwise masks
       (from ``ECDH(sk1, peer_pk1)``) → add self-mask (from ``rd_seed``) →
       take ``mod mod_range`` (default ``2**32``).
    4. **Unmask** -- surviving clients reveal ``rd_seed`` shares for active
       peers and ``sk1`` shares for dropped peers, so the server can
       reconstruct self-masks and pairwise masks respectively.

  Cross-platform: all platforms (Python, Android, iOS) use X25519 with 32-byte
  raw keys and standardized HKDF info strings for key derivation.

  Requires the ``cryptography`` package (``pip install octomil[secagg]``).

Shared utilities: Shamir secret sharing, field-element encoding, mask
derivation, and quantization.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default Mersenne prime used as finite-field modulus (2^127 - 1).
DEFAULT_FIELD_SIZE = (1 << 127) - 1

# Chunk size used when converting model bytes <-> field elements.
_CHUNK_BYTES = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SecAggConfig:
    """Client-side view of a SecAgg session configuration."""

    session_id: str
    round_id: str
    threshold: int
    total_clients: int
    field_size: int = DEFAULT_FIELD_SIZE
    key_length: int = 256
    noise_scale: Optional[float] = None


@dataclass
class ShamirShare:
    """A single Shamir secret share."""

    index: int
    value: int
    modulus: int

    def to_bytes(self) -> bytes:
        mod_bytes = self.modulus.to_bytes(16, "big")
        return (
            struct.pack(">I", self.index)
            + self.value.to_bytes(16, "big")
            + struct.pack(">I", len(mod_bytes))
            + mod_bytes
        )

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple["ShamirShare", int]:
        index = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        value = int.from_bytes(data[offset : offset + 16], "big")
        offset += 16
        mod_len = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        modulus = int.from_bytes(data[offset : offset + mod_len], "big")
        offset += mod_len
        return cls(index=index, value=value, modulus=modulus), offset


# ---------------------------------------------------------------------------
# Shamir helpers (pure functions, no I/O)
# ---------------------------------------------------------------------------


def _mod_inverse(a: int, m: int) -> int:
    """Modular multiplicative inverse via extended Euclidean algorithm."""

    def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = _extended_gcd(b % a, a)
        return gcd, y1 - (b // a) * x1, x1

    gcd, x, _ = _extended_gcd(a % m, m)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
    return (x % m + m) % m


def _evaluate_polynomial(coefficients: List[int], x: int, modulus: int) -> int:
    """Evaluate polynomial at *x* using Horner's method in GF(*modulus*)."""
    result = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        result = (result * x + coefficients[i]) % modulus
    return result


def generate_shares(
    secret: int,
    threshold: int,
    total_shares: int,
    modulus: int = DEFAULT_FIELD_SIZE,
) -> List[ShamirShare]:
    """Split *secret* into *total_shares* Shamir shares.

    Any *threshold* shares are sufficient to reconstruct the secret.
    """
    if threshold < 1:
        raise ValueError("threshold must be >= 1")
    if threshold > total_shares:
        raise ValueError("threshold must be <= total_shares")

    # Build random polynomial with secret as the constant term.
    coefficients = [secret % modulus]
    for _ in range(threshold - 1):
        coefficients.append(secrets.randbelow(modulus))

    shares: List[ShamirShare] = []
    for i in range(1, total_shares + 1):
        y = _evaluate_polynomial(coefficients, i, modulus)
        shares.append(ShamirShare(index=i, value=y, modulus=modulus))
    return shares


def reconstruct_secret(shares: List[ShamirShare]) -> int:
    """Reconstruct the secret from a list of shares via Lagrange interpolation at x=0."""
    if not shares:
        raise ValueError("Need at least one share")

    modulus = shares[0].modulus
    result = 0

    for i, share_i in enumerate(shares):
        numerator = 1
        denominator = 1
        for j, share_j in enumerate(shares):
            if i != j:
                numerator = (numerator * (0 - share_j.index)) % modulus
                denominator = (denominator * (share_i.index - share_j.index)) % modulus

        lagrange_coeff = (numerator * _mod_inverse(denominator, modulus)) % modulus
        result = (result + share_i.value * lagrange_coeff) % modulus

    return result


# ---------------------------------------------------------------------------
# Byte-level Shamir sharing (for arbitrary-length secrets like PEM keys)
# ---------------------------------------------------------------------------

# Shamir chunk size in bytes (matches Flower's 16-byte AES block approach).
_SHAMIR_CHUNK_SIZE = 15  # Must fit within GF(2^127-1): 15 bytes = 120 bits < 127 bits


def _zero_pad_to_chunk_size(data: bytes) -> bytes:
    """Zero-pad *data* to the next multiple of ``_SHAMIR_CHUNK_SIZE``."""
    remainder = len(data) % _SHAMIR_CHUNK_SIZE
    if remainder == 0 and len(data) > 0:
        return data
    return data + b"\x00" * (_SHAMIR_CHUNK_SIZE - remainder)


@dataclass
class ByteShamirShare:
    """A Shamir share of an arbitrary-length byte secret.

    Internally stores one :class:`ShamirShare` per 16-byte chunk.
    """

    index: int
    chunk_shares: List[ShamirShare]  # one per chunk
    secret_length: int = 0  # original secret length for reconstruction

    def to_bytes(self) -> bytes:
        """Serialize to a single byte string."""
        # Header: 4-byte index, 4-byte num_chunks, 4-byte secret_length
        buf = struct.pack(">III", self.index, len(self.chunk_shares), self.secret_length)
        for cs in self.chunk_shares:
            buf += cs.to_bytes()
        return buf

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple["ByteShamirShare", int]:
        index, num_chunks, secret_length = struct.unpack(">III", data[offset : offset + 12])
        offset += 12
        chunks: List[ShamirShare] = []
        for _ in range(num_chunks):
            cs, offset = ShamirShare.from_bytes(data, offset)
            chunks.append(cs)
        return cls(index=index, chunk_shares=chunks, secret_length=secret_length), offset


def create_shares_bytes(
    secret: bytes,
    threshold: int,
    total_shares: int,
    modulus: int = DEFAULT_FIELD_SIZE,
) -> List[ByteShamirShare]:
    """Shamir-share arbitrary bytes by splitting into 16-byte chunks.

    Each chunk is independently shared as an integer mod *modulus*.
    Zero-padding is applied so the secret length need not be a multiple
    of 16.  The original bytes are exactly recovered by
    :func:`combine_shares_bytes` using the stored secret length.
    """
    padded = _zero_pad_to_chunk_size(secret)
    num_chunks = len(padded) // _SHAMIR_CHUNK_SIZE

    # Pre-create result containers.
    result = [
        ByteShamirShare(index=i + 1, chunk_shares=[], secret_length=len(secret))
        for i in range(total_shares)
    ]

    for c in range(num_chunks):
        chunk = padded[c * _SHAMIR_CHUNK_SIZE : (c + 1) * _SHAMIR_CHUNK_SIZE]
        chunk_int = int.from_bytes(chunk, "big")
        shares = generate_shares(chunk_int, threshold, total_shares, modulus)
        for s in shares:
            result[s.index - 1].chunk_shares.append(s)

    return result


def combine_shares_bytes(
    shares: List[ByteShamirShare],
) -> bytes:
    """Reconstruct the original byte secret from byte-level Shamir shares."""
    if not shares:
        raise ValueError("Need at least one share")

    num_chunks = len(shares[0].chunk_shares)
    secret_length = shares[0].secret_length
    reconstructed = bytearray()

    for c in range(num_chunks):
        chunk_shares = [s.chunk_shares[c] for s in shares]
        chunk_int = reconstruct_secret(chunk_shares)
        # Clamp to chunk size bytes (may overflow with insufficient shares)
        chunk_bytes = (chunk_int % (1 << (_SHAMIR_CHUNK_SIZE * 8))).to_bytes(
            _SHAMIR_CHUNK_SIZE, "big"
        )
        reconstructed.extend(chunk_bytes)

    return bytes(reconstructed[:secret_length]) if secret_length > 0 else bytes(reconstructed)


# ---------------------------------------------------------------------------
# Field-element encoding / decoding
# ---------------------------------------------------------------------------


def model_bytes_to_field_elements(data: bytes, modulus: int = DEFAULT_FIELD_SIZE) -> List[int]:
    """Convert raw bytes into a list of finite-field elements (4-byte chunks)."""
    elements: List[int] = []
    for i in range(0, len(data), _CHUNK_BYTES):
        chunk = data[i : i + _CHUNK_BYTES]
        if len(chunk) < _CHUNK_BYTES:
            chunk = chunk + b"\x00" * (_CHUNK_BYTES - len(chunk))
        value = struct.unpack(">I", chunk)[0]
        elements.append(value % modulus)
    return elements


def field_elements_to_model_bytes(elements: List[int]) -> bytes:
    """Convert finite-field elements back to raw bytes."""
    parts: List[bytes] = []
    for elem in elements:
        parts.append(struct.pack(">I", elem % (1 << 32)))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Mask generation from a seed
# ---------------------------------------------------------------------------


def _derive_mask_elements(
    seed: bytes,
    count: int,
    modulus: int = DEFAULT_FIELD_SIZE,
) -> List[int]:
    """Derive *count* pseudorandom mask elements from *seed* using HMAC-SHA256."""
    elements: List[int] = []
    counter = 0
    while len(elements) < count:
        h = hashlib.sha256(seed + struct.pack(">I", counter)).digest()
        # Each hash gives 32 bytes -> 8 four-byte field elements.
        for off in range(0, len(h), _CHUNK_BYTES):
            if len(elements) >= count:
                break
            val = struct.unpack(">I", h[off : off + _CHUNK_BYTES])[0]
            elements.append(val % modulus)
        counter += 1
    return elements


# ---------------------------------------------------------------------------
# High-level client helper
# ---------------------------------------------------------------------------


class SecAggClient:
    """Client-side SecAgg state machine.

    Typical usage inside ``FederatedClient.participate_in_round``::

        sac = SecAggClient(config)
        shares_for_peers = sac.generate_key_shares()
        # ... send shares_for_peers to server ...
        masked_update = sac.mask_model_update(raw_update_bytes)
        # ... upload masked_update ...
    """

    def __init__(self, config: SecAggConfig) -> None:
        self.config = config
        self._seed: bytes = secrets.token_bytes(32)
        self._shares: Optional[List[ShamirShare]] = None
        self._mask_elements: Optional[List[int]] = None

    # Phase 1 -- share keys ------------------------------------------------

    def generate_key_shares(self) -> List[ShamirShare]:
        """Split this client's random seed into Shamir shares.

        Returns a list of ``total_clients`` shares. Share *i* should be
        delivered to participant *i* (1-indexed).
        """
        # Convert seed to a single big integer for sharing.
        secret_int = int.from_bytes(self._seed, "big") % self.config.field_size
        self._shares = generate_shares(
            secret=secret_int,
            threshold=self.config.threshold,
            total_shares=self.config.total_clients,
            modulus=self.config.field_size,
        )
        return list(self._shares)

    # Phase 2 -- masked upload ----------------------------------------------

    def mask_model_update(self, update_bytes: bytes) -> bytes:
        """Add a pseudorandom mask to the serialised model update.

        The mask is derived deterministically from ``self._seed`` so that the
        server can reconstruct it (via Shamir) and remove it from the aggregate.
        """
        elements = model_bytes_to_field_elements(update_bytes, self.config.field_size)
        mask = _derive_mask_elements(self._seed, len(elements), self.config.field_size)
        self._mask_elements = mask

        masked: List[int] = []
        for e, m in zip(elements, mask):
            masked.append((e + m) % self.config.field_size)

        return field_elements_to_model_bytes(masked)

    # Phase 3 -- unmask (dropout handling) ----------------------------------

    def get_seed_share_for_peer(self, peer_index: int) -> Optional[ShamirShare]:
        """Return the share destined for *peer_index* (1-based).

        Called when a peer drops out and the server requests this client's
        share of that peer's seed so it can reconstruct the dropout's mask.
        """
        if self._shares is None:
            return None
        for share in self._shares:
            if share.index == peer_index:
                return share
        return None

    # Serialisation helpers -------------------------------------------------

    @staticmethod
    def serialize_shares(shares: List[ShamirShare]) -> bytes:
        """Serialize a list of shares for network transmission."""
        buf = struct.pack(">I", len(shares))
        for s in shares:
            buf += s.to_bytes()
        return buf

    @staticmethod
    def deserialize_shares(data: bytes) -> List[ShamirShare]:
        """Deserialize shares received from the server or a peer."""
        count = struct.unpack(">I", data[:4])[0]
        offset = 4
        shares: List[ShamirShare] = []
        for _ in range(count):
            share, offset = ShamirShare.from_bytes(data, offset)
            shares.append(share)
        return shares


# ---------------------------------------------------------------------------
# Quantization pipeline (float <-> integer)
# ---------------------------------------------------------------------------


def _stochastic_round(values: List[float]) -> List[int]:
    """Stochastic rounding: ``ceil(x)`` with probability ``x - floor(x)``.

    Matches the Flower SecAgg+ stochastic rounding implementation.
    """
    import math
    import random

    result: List[int] = []
    for v in values:
        c = math.ceil(v)
        # Probability of rounding down = ceil(v) - v
        if random.random() < (c - v):
            result.append(c - 1)
        else:
            result.append(c)
    return result


def quantize(
    values: List[float],
    clipping_range: float,
    target_range: int,
) -> List[int]:
    """Stochastic quantize floats to integers in ``[0, target_range]``.

    Follows the Flower SecAgg+ quantization scheme:
      1. Clip values to ``[-clipping_range, +clipping_range]``
      2. Shift to ``[0, 2 * clipping_range]``
      3. Scale to ``[0, target_range]``
      4. Stochastic round to integers

    The inverse is :func:`dequantize`.
    """
    if not values:
        return []

    quantizer = target_range / (2.0 * clipping_range) if clipping_range != 0 else 0.0
    pre_quantized = [
        (max(-clipping_range, min(clipping_range, v)) + clipping_range) * quantizer for v in values
    ]
    return _stochastic_round(pre_quantized)


def dequantize(
    quantized: List[int],
    clipping_range: float,
    target_range: int,
) -> List[float]:
    """Reverse :func:`quantize` — map integers back to floats in
    ``[-clipping_range, +clipping_range]``.
    """
    if not quantized:
        return []

    scale = (2.0 * clipping_range) / target_range if target_range != 0 else 0.0
    shift = -clipping_range
    return [q * scale + shift for q in quantized]


# ---------------------------------------------------------------------------
# ECDH key exchange helpers (X25519)
# ---------------------------------------------------------------------------

# Standardized HKDF info strings for cross-platform compatibility.
# All platforms (Python, Android, iOS) MUST use these exact strings.
HKDF_INFO_PAIRWISE_MASK = b"secagg-pairwise-mask"
HKDF_INFO_SHARE_ENCRYPTION = b"secagg-share-encryption"
HKDF_INFO_SELF_MASK = b"secagg-self-mask"


def _require_cryptography():
    """Import and return the ``cryptography`` package or raise a clear error."""
    try:
        import cryptography  # noqa: F401

        return cryptography
    except ImportError as exc:
        raise ImportError(
            "SecAgg+ requires the 'cryptography' package. "
            "Install it with: pip install octomil[secagg]"
        ) from exc


@dataclass
class ECKeyPair:
    """An X25519 key pair for ECDH key exchange.

    Keys are stored as 32-byte raw byte strings for cross-platform
    compatibility with Android (JCA X25519) and iOS (CryptoKit Curve25519).
    """

    private_key_bytes: bytes  # 32-byte raw private key
    public_key_bytes: bytes  # 32-byte raw public key

    @classmethod
    def generate(cls) -> "ECKeyPair":
        """Generate a fresh X25519 key pair."""
        _require_cryptography()
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
            PublicFormat,
        )

        private_key = X25519PrivateKey.generate()
        priv_bytes = private_key.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        )
        pub_bytes = private_key.public_key().public_bytes(
            Encoding.Raw,
            PublicFormat.Raw,
        )
        return cls(private_key_bytes=priv_bytes, public_key_bytes=pub_bytes)


# Keep old name as alias for backward compatibility.
ECDHKeyPair = ECKeyPair


def generate_shared_key(
    my_private_raw: bytes,
    peer_public_raw: bytes,
    info: Optional[bytes] = None,
) -> bytes:
    """Compute a raw 32-byte key via X25519 ECDH + HKDF-SHA256.

    Uses X25519 key exchange -> HKDF-SHA256(32 bytes, info) -> raw bytes.

    The ``info`` parameter selects the purpose-specific HKDF context:
      - ``HKDF_INFO_PAIRWISE_MASK`` for pairwise mask derivation
      - ``HKDF_INFO_SHARE_ENCRYPTION`` for share encryption keys (AES-256-GCM)
      - ``None`` for backward compatibility (not recommended)

    Returns exactly 32 raw bytes suitable for AES-256-GCM or PRG seeding.
    """
    _require_cryptography()

    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    private_key = X25519PrivateKey.from_private_bytes(my_private_raw)
    peer_public = X25519PublicKey.from_public_bytes(peer_public_raw)
    raw_shared = private_key.exchange(peer_public)
    return HKDF(
        algorithm=SHA256(),
        length=32,
        salt=None,
        info=info,
    ).derive(raw_shared)


def generate_pairwise_key(
    my_private_raw: bytes,
    peer_public_raw: bytes,
) -> bytes:
    """Derive a 32-byte pairwise-mask key via X25519 ECDH + HKDF(info="secagg-pairwise-mask")."""
    return generate_shared_key(my_private_raw, peer_public_raw, HKDF_INFO_PAIRWISE_MASK)


def generate_share_encryption_key(
    my_private_raw: bytes,
    peer_public_raw: bytes,
) -> bytes:
    """Derive a 32-byte AES-GCM key via X25519 ECDH + HKDF(info="secagg-share-encryption")."""
    return generate_shared_key(my_private_raw, peer_public_raw, HKDF_INFO_SHARE_ENCRYPTION)


# Backward-compatible alias.
compute_shared_secret = generate_shared_key


def _pseudo_rand_gen(
    seed: bytes,
    num_range: int,
    count: int,
) -> List[int]:
    """SHA-256 counter mode PRG. Cross-platform compatible with server/Android/iOS.

    For each index i in [0, count), computes SHA-256(seed || i.to_bytes(4, 'big')),
    takes the first 4 bytes as a big-endian uint32, and mods by num_range.
    """
    import hashlib

    masks: List[int] = []
    for i in range(count):
        h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        val = int.from_bytes(h[:4], "big") % num_range
        masks.append(val)
    return masks


def derive_pairwise_mask(
    shared_key: bytes,
    count: int,
    mod_range: int = (1 << 32),
) -> List[int]:
    """Derive a pairwise mask from the ECDH-derived shared key.

    Uses the Flower-compatible PRG: fold the shared key into a 32-bit seed,
    then generate ``count`` random ints in ``[0, mod_range - 1]``.

    ``shared_key`` is a raw 32-byte key from :func:`generate_pairwise_key`.
    """
    # Pad to multiple of 4 if needed (should already be 32 bytes).
    raw = shared_key
    if len(raw) % 4 != 0:
        raw = raw + b"\x00" * (4 - len(raw) % 4)
    return _pseudo_rand_gen(raw, mod_range, count)


# ---------------------------------------------------------------------------
# AES-GCM encrypted shares
# ---------------------------------------------------------------------------

# Wire format: nonce (12 bytes) || ciphertext || GCM tag (16 bytes)
# This matches the Android (JCA) and iOS (CryptoKit) AES-GCM wire formats.
_AES_GCM_NONCE_SIZE = 12


def encrypt_share(plaintext: bytes, shared_key: bytes) -> bytes:
    """Encrypt plaintext using AES-256-GCM with the ECDH-derived shared key.

    ``shared_key`` is a raw 32-byte key from :func:`generate_share_encryption_key`.

    Wire format: ``nonce (12 bytes) || ciphertext + GCM tag``.
    Cross-platform compatible with Android JCA and iOS CryptoKit AES-GCM.
    """
    _require_cryptography()
    import os

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(_AES_GCM_NONCE_SIZE)
    aesgcm = AESGCM(shared_key)
    # AESGCM.encrypt returns ciphertext || tag (16 bytes appended).
    ct_with_tag = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ct_with_tag


def decrypt_share(ciphertext: bytes, shared_key: bytes) -> bytes:
    """Decrypt ciphertext using AES-256-GCM with the ECDH-derived shared key.

    ``shared_key`` is a raw 32-byte key from :func:`generate_share_encryption_key`.

    Expects wire format: ``nonce (12 bytes) || ciphertext + GCM tag``.
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = ciphertext[:_AES_GCM_NONCE_SIZE]
    ct_with_tag = ciphertext[_AES_GCM_NONCE_SIZE:]
    aesgcm = AESGCM(shared_key)
    return aesgcm.decrypt(nonce, ct_with_tag, None)


# ---------------------------------------------------------------------------
# SecAgg+ configuration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SecAgg+ client state machine
# ---------------------------------------------------------------------------


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
