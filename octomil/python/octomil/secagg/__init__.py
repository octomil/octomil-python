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
    3. **Masked upload** -- clip + stochastic quantize -> add pairwise masks
       (from ``ECDH(sk1, peer_pk1)``) -> add self-mask (from ``rd_seed``) ->
       take ``mod mod_range`` (default ``2**32``).
    4. **Unmask** -- surviving clients reveal ``rd_seed`` shares for active
       peers and ``sk1`` shares for dropped peers, so the server can
       reconstruct self-masks and pairwise masks respectively.

  Cross-platform: all platforms (Python, Android, iOS) use X25519 with 32-byte
  raw keys and standardized HKDF info strings for key derivation.

  Requires the ``cryptography`` package (``pip install octomil[secagg]``).

Shared utilities: Shamir secret sharing, field-element encoding, mask
derivation, and quantization.

This package is a refactored version of the original ``secagg.py`` monolith.
Sub-modules: client, crypto, encoding, plus_client, quantization, shamir.
"""

from .client import SecAggClient, SecAggConfig  # noqa: F401
from .crypto import (  # noqa: F401
    HKDF_INFO_PAIRWISE_MASK,
    HKDF_INFO_SELF_MASK,
    HKDF_INFO_SHARE_ENCRYPTION,
    ECDHKeyPair,
    ECKeyPair,
    compute_shared_secret,
    decrypt_share,
    derive_pairwise_mask,
    encrypt_share,
    generate_pairwise_key,
    generate_share_encryption_key,
    generate_shared_key,
)
from .encoding import (  # noqa: F401
    field_elements_to_model_bytes,
    model_bytes_to_field_elements,
)
from .plus_client import SECAGG_PLUS_MOD_RANGE, SecAggPlusClient, SecAggPlusConfig  # noqa: F401
from .quantization import dequantize, quantize  # noqa: F401
from .shamir import (  # noqa: F401
    DEFAULT_FIELD_SIZE,
    ByteShamirShare,
    ShamirShare,
    combine_shares_bytes,
    create_shares_bytes,
    generate_shares,
    reconstruct_secret,
)

__all__ = [
    # Shamir
    "DEFAULT_FIELD_SIZE",
    "ShamirShare",
    "ByteShamirShare",
    "generate_shares",
    "reconstruct_secret",
    "create_shares_bytes",
    "combine_shares_bytes",
    # Encoding
    "model_bytes_to_field_elements",
    "field_elements_to_model_bytes",
    # Quantization
    "quantize",
    "dequantize",
    # Crypto
    "ECKeyPair",
    "ECDHKeyPair",
    "HKDF_INFO_PAIRWISE_MASK",
    "HKDF_INFO_SHARE_ENCRYPTION",
    "HKDF_INFO_SELF_MASK",
    "generate_shared_key",
    "compute_shared_secret",
    "generate_pairwise_key",
    "generate_share_encryption_key",
    "derive_pairwise_mask",
    "encrypt_share",
    "decrypt_share",
    # Client
    "SecAggConfig",
    "SecAggClient",
    # Plus Client
    "SECAGG_PLUS_MOD_RANGE",
    "SecAggPlusConfig",
    "SecAggPlusClient",
]
