"""Shamir secret sharing — data classes and pure functions.

Provides integer-level and byte-level Shamir splitting and reconstruction
over a finite field (default modulus: Mersenne prime 2^127 - 1).
"""

from __future__ import annotations

import secrets
import struct
from dataclasses import dataclass
from typing import List, Tuple

# Default Mersenne prime used as finite-field modulus (2^127 - 1).
DEFAULT_FIELD_SIZE = (1 << 127) - 1

# Shamir chunk size in bytes (matches Flower's 16-byte AES block approach).
_SHAMIR_CHUNK_SIZE = 15  # Must fit within GF(2^127-1): 15 bytes = 120 bits < 127 bits


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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


def _zero_pad_to_chunk_size(data: bytes) -> bytes:
    """Zero-pad *data* to the next multiple of ``_SHAMIR_CHUNK_SIZE``."""
    remainder = len(data) % _SHAMIR_CHUNK_SIZE
    if remainder == 0 and len(data) > 0:
        return data
    return data + b"\x00" * (_SHAMIR_CHUNK_SIZE - remainder)


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
