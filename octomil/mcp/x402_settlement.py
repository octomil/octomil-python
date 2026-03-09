"""Batch settlement for x402 micro-payments.

Accumulates signed EIP-3009 TransferWithAuthorization payloads in-memory
and submits them as a single batch when the running total hits a configurable
threshold (default: $1 USDC = 1,000,000 base units).  One multicall
transaction amortises gas across ~1000 micro-payments.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PendingAuthorization:
    """A verified but unsettled payment authorization."""

    authorization: dict[str, Any]
    signature: str
    payer: str
    amount: int  # base units (e.g. 1000 = 0.001 USDC)
    request_path: str
    timestamp: float = field(default_factory=time.time)


class SettlementStore:
    """Thread-safe in-memory store for pending payment authorizations.

    Mirrors the ``NonceTracker`` pattern — all access guarded by a lock.
    """

    def __init__(self, threshold: int = 1_000_000) -> None:
        self._pending: OrderedDict[str, PendingAuthorization] = OrderedDict()
        self._lock = threading.Lock()
        self._total: int = 0
        self._threshold: int = threshold
        self._settled_count: int = 0
        self._settled_total: int = 0

    def add(self, nonce: str, auth: PendingAuthorization) -> bool:
        """Store a pending authorization.

        Returns True if the running total has reached the settlement threshold.
        """
        with self._lock:
            self._pending[nonce] = auth
            self._total += auth.amount
            return self._total >= self._threshold

    def pop_batch(self) -> list[PendingAuthorization]:
        """Atomically drain all pending authorizations.

        Resets the running total and increments lifetime counters.
        """
        with self._lock:
            batch = list(self._pending.values())
            self._settled_count += len(batch)
            self._settled_total += self._total
            self._pending.clear()
            self._total = 0
            return batch

    def requeue(self, items: list[PendingAuthorization]) -> None:
        """Re-add authorizations after a failed settlement attempt."""
        with self._lock:
            for item in items:
                nonce = item.authorization.get("nonce", "")
                if isinstance(nonce, bytes):
                    nonce = nonce.hex()
                nonce = str(nonce)
                if nonce not in self._pending:
                    self._pending[nonce] = item
                    self._total += item.amount

    def discard(self, nonce: str) -> None:
        """Remove a single authorization (e.g. on service error / refund)."""
        with self._lock:
            auth = self._pending.pop(nonce, None)
            if auth is not None:
                self._total -= auth.amount

    def stats(self) -> dict[str, Any]:
        """Return settlement statistics."""
        with self._lock:
            return {
                "pending_count": len(self._pending),
                "pending_total": self._total,
                "settled_count": self._settled_count,
                "settled_total": self._settled_total,
                "threshold": self._threshold,
            }


async def settle_batch(
    store: SettlementStore,
    facilitator_url: str,
    config: Any,
) -> None:
    """Submit a batch of authorizations to settle402 for on-chain settlement.

    If ``facilitator_url`` is set, POSTs the batch to ``{facilitator_url}/settle``.
    Otherwise logs the batch for manual settlement (the signed authorizations
    remain valid on-chain until their ``validBefore`` timestamp).

    Handles partial failures: only re-queues authorizations that failed.
    """
    batch = store.pop_batch()
    if not batch:
        return

    logger.info(
        "x402: settling batch of %d authorizations (total=%d base units)",
        len(batch),
        sum(a.amount for a in batch),
    )

    if not facilitator_url:
        logger.info(
            "x402: no facilitator_url configured — %d authorizations logged for manual settlement",
            len(batch),
        )
        return

    payload = {
        "network": getattr(config, "network", "base"),
        "chainId": getattr(config, "resolved_chain_id", lambda: 8453)(),
        "tokenContract": getattr(config, "resolved_token_contract", lambda: "")(),
        "authorizations": [
            {
                "authorization": a.authorization,
                "signature": a.signature,
                "payer": a.payer,
                "amount": a.amount,
            }
            for a in batch
        ],
    }

    headers: dict[str, str] = {}
    settler_token = getattr(config, "settler_token", "")
    if settler_token:
        headers["x-settler-token"] = settler_token

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{facilitator_url}/settle", json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        status = result.get("status", "")
        succeeded = result.get("total_succeeded", 0)
        failed = result.get("total_failed", 0)

        if status == "settled":
            logger.info("x402: batch settled — %d/%d succeeded", succeeded, len(batch))
        elif status == "partial":
            logger.warning("x402: partial settlement — %d succeeded, %d failed", succeeded, failed)
            # Re-queue only the failed authorizations
            failed_indices = {r["index"] for r in result.get("results", []) if not r.get("success")}
            failed_auths = [batch[i] for i in failed_indices if i < len(batch)]
            if failed_auths:
                store.requeue(failed_auths)
        else:
            logger.error("x402: batch settlement failed — re-queuing %d auths", len(batch))
            store.requeue(batch)

    except Exception as exc:
        logger.error("x402: settle402 request failed: %s — re-queuing %d auths", exc, len(batch))
        store.requeue(batch)
