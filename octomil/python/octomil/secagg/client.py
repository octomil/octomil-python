"""SecAgg client — basic single-seed masking with Shamir secret sharing.

Implements :class:`SecAggClient` and :class:`SecAggConfig`.
"""

from __future__ import annotations

import secrets
import struct
from dataclasses import dataclass
from typing import List, Optional

from .encoding import (
    _derive_mask_elements,
    field_elements_to_model_bytes,
    model_bytes_to_field_elements,
)
from .shamir import DEFAULT_FIELD_SIZE, ShamirShare, generate_shares


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


class SecAggClient:
    """Client-side SecAgg state machine.

    Typical usage inside ``FederatedClient.join_round``::

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
