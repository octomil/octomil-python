"""ECDH key exchange (X25519), pairwise mask derivation, and AES-GCM share encryption.

Cross-platform compatible with Android (JCA X25519) and iOS (CryptoKit Curve25519).
All platforms use X25519 with 32-byte raw keys and standardized HKDF info strings.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

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
