"""Field-element encoding/decoding and mask derivation.

Converts raw model bytes to/from finite-field elements (4-byte chunks)
and derives pseudorandom mask vectors from seeds.
"""

from __future__ import annotations

import hashlib
import struct
from typing import List

from .shamir import DEFAULT_FIELD_SIZE

# Chunk size used when converting model bytes <-> field elements.
_CHUNK_BYTES = 4


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
