"""x402 payment gating middleware for the Octomil HTTP agent server.

Implements the x402 payment protocol:
- Returns 402 + payment requirements header when payment is needed
- Accepts x-payment (x402 spec) and payment-signature (legacy) headers
- EIP-712 signature verification (with graceful degradation)
- Expiry enforcement, replay protection, amount validation
- Optional facilitator forwarding for on-chain settlement
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .x402_settlement import SettlementStore

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_DEFAULT_EXPIRY_SECONDS = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Chain / token mappings
# ---------------------------------------------------------------------------

CHAIN_IDS: dict[str, int] = {
    "base": 8453,
    "base-sepolia": 84532,
    "ethereum": 1,
    "sepolia": 11155111,
    "polygon": 137,
    "arbitrum": 42161,
    "optimism": 10,
}

USDC_CONTRACTS: dict[str, str] = {
    "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "base-sepolia": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "polygon": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    "arbitrum": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    "optimism": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class X402Config:
    """Configuration for x402 payment gating."""

    price_per_call: str = "1000"  # base units (e.g. 1000 = 0.001 USDC with 6 decimals)
    cloud_price_per_call: str = "1000"  # base units = $0.001 USDC (same as local)
    currency: str = "USDC"
    network: str = "base"
    payment_address: str = "0x7BeEa3e83033e5399FfAAdfb8bf731eBb36126F3"
    token_contract: str = ""  # auto-resolved from network if empty
    verify_signatures: bool = True  # False for dev/testing
    facilitator_url: str = ""  # settle402 service URL
    settler_token: str = ""  # X-Settler-Token for settle402 auth
    settlement_threshold: int = 1_000_000  # base units = $1 USDC (6 decimals)
    enable_settlement: bool = True
    protected_prefixes: list[str] = field(default_factory=lambda: ["/api/v1/"])
    exempt_paths: list[str] = field(
        default_factory=lambda: [
            # Discovery & readiness — agents need these to bootstrap
            "/.well-known/agent-card.json",
            "/health",
            "/api/v1/ready",
            "/api/v1/warmup",
            "/docs",
            "/openapi.json",
            "/redoc",
            # Platform metadata tools — no GPU, no cost
            "/api/v1/metrics",
            "/api/v1/resolve_model",
            "/api/v1/list_models",
            "/api/v1/detect_engines",
            "/api/v1/hardware_profile",
            "/api/v1/recommend_model",
            "/api/v1/scan_codebase",
            "/api/v1/compress_prompt",
            # Settlement status — operator-only, not gated by x402
            "/api/v1/settlement_status",
            # Cloud-proxied tools — gated by OCTOMIL_API_KEY, not x402
            "/api/v1/deploy_model",
            "/api/v1/optimize_model",
            "/api/v1/plan_deployment",
            "/api/v1/embed",
            "/api/v1/convert_model",
        ]
    )
    expiry_seconds: int = _DEFAULT_EXPIRY_SECONDS

    def resolved_token_contract(self) -> str:
        """Return token contract, auto-resolving from network if not set."""
        if self.token_contract:
            return self.token_contract
        return USDC_CONTRACTS.get(self.network, "")

    def resolved_chain_id(self) -> int:
        """Return chain ID for the configured network."""
        return CHAIN_IDS.get(self.network, 8453)


# ---------------------------------------------------------------------------
# Payment requirements (402 response)
# ---------------------------------------------------------------------------


def build_payment_requirements(config: X402Config, resource: str = "") -> dict[str, Any]:
    """Build x402 v1 spec-compliant payment requirements."""
    return {
        "x402Version": 1,
        "accepts": [
            {
                "scheme": "exact",
                "network": config.network,
                "maxAmountRequired": config.price_per_call,
                "resource": resource,
                "payTo": config.payment_address,
                "asset": config.resolved_token_contract(),
                "maxTimeoutSeconds": config.expiry_seconds,
            }
        ],
        "error": "X-PAYMENT header is required",
    }


def encode_payment_requirements(requirements: dict[str, Any]) -> str:
    """Base64-encode payment requirements for the header."""
    return base64.b64encode(json.dumps(requirements).encode()).decode()


# ---------------------------------------------------------------------------
# Payment decoding
# ---------------------------------------------------------------------------


def decode_x402_payment(header_value: str) -> Optional[dict[str, Any]]:
    """Decode an x-payment header (x402 spec format).

    The payload is a base64-encoded JSON object with EIP-3009
    TransferWithAuthorization fields.
    """
    try:
        decoded = base64.b64decode(header_value)
        data = json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return None

    # Minimal structure check for x402 payload
    if not isinstance(data, dict):
        return None

    # Accept either nested authorization or flat structure
    return data


def decode_payment_signature(header_value: str) -> Optional[dict[str, Any]]:
    """Decode and validate structure of a legacy payment-signature header.

    Returns the parsed JSON if it has required fields, None otherwise.
    """
    try:
        decoded = base64.b64decode(header_value)
        data = json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return None

    required_fields = {"paymentId", "payer", "signature"}
    if not required_fields.issubset(data.keys()):
        return None

    return data


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def check_authorization_expiry(authorization: dict[str, Any]) -> tuple[bool, str]:
    """Check if an authorization is within its validity window.

    Returns (ok, error_message).
    """
    now = int(time.time())

    valid_after = authorization.get("validAfter", 0)
    valid_before = authorization.get("validBefore", 0)

    if isinstance(valid_after, str):
        valid_after = int(valid_after)
    if isinstance(valid_before, str):
        valid_before = int(valid_before)

    if valid_after and now < valid_after:
        return False, f"Authorization not yet valid (validAfter={valid_after}, now={now})"

    if valid_before and now > valid_before:
        return False, f"Authorization expired (validBefore={valid_before}, now={now})"

    return True, ""


def check_payment_amount(authorization: dict[str, Any], required_amount: str) -> tuple[bool, str]:
    """Check if payment amount meets the required minimum.

    Returns (ok, error_message).
    """
    value = authorization.get("value", "0")
    try:
        paid = int(value)
        required = int(required_amount)
    except (ValueError, TypeError):
        return False, f"Invalid amount values: paid={value}, required={required_amount}"

    if paid < required:
        return False, f"Insufficient payment: {paid} < {required}"

    return True, ""


def verify_eip712_signature(
    authorization: dict[str, Any],
    signature: str,
    token_contract: str,
    chain_id: int,
) -> tuple[bool, str]:
    """Verify an EIP-712 TransferWithAuthorization signature.

    Uses eth_account for recovery. Gracefully degrades if eth-account
    is not installed (returns True).

    Returns (ok, error_message).
    """
    try:
        from eth_account import Account
        from eth_account.messages import encode_typed_data
    except ImportError:
        logger.warning("eth-account not installed — skipping EIP-712 verification")
        return True, ""

    try:
        # EIP-3009 TransferWithAuthorization typed data
        domain = {
            "name": "USD Coin",
            "version": "2",
            "chainId": chain_id,
            "verifyingContract": token_contract,
        }

        types = {
            "TransferWithAuthorization": [
                {"name": "from", "type": "address"},
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "validAfter", "type": "uint256"},
                {"name": "validBefore", "type": "uint256"},
                {"name": "nonce", "type": "bytes32"},
            ],
        }

        message = {
            "from": authorization.get("from", ""),
            "to": authorization.get("to", ""),
            "value": int(authorization.get("value", 0)),
            "validAfter": int(authorization.get("validAfter", 0)),
            "validBefore": int(authorization.get("validBefore", 0)),
            "nonce": authorization.get("nonce", b"\x00" * 32),
        }

        signable = encode_typed_data(
            domain_data=domain,
            message_types=types,
            message_data=message,
        )

        recovered = Account.recover_message(signable, signature=signature)
        expected_from = authorization.get("from", "")

        if recovered.lower() != expected_from.lower():
            return False, f"Signer mismatch: recovered={recovered}, expected={expected_from}"

        return True, ""

    except Exception as exc:
        return False, f"EIP-712 verification failed: {exc}"


# ---------------------------------------------------------------------------
# Replay protection
# ---------------------------------------------------------------------------


class NonceTracker:
    """Thread-safe in-memory nonce tracker for replay protection."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self._max_entries = 100_000
        self._ttl_seconds = 3600  # 1 hour

    def check_and_mark(self, nonce: str) -> bool:
        """Check if nonce is fresh and mark it as used.

        Returns True if fresh (not seen before), False if replay.
        """
        now = time.time()
        with self._lock:
            # Cleanup if over capacity
            if len(self._seen) > self._max_entries:
                cutoff = now - self._ttl_seconds
                self._seen = {k: v for k, v in self._seen.items() if v > cutoff}

            if nonce in self._seen:
                return False

            self._seen[nonce] = now
            return True


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _is_exempt(path: str, config: X402Config) -> bool:
    """Check if a request path is exempt from payment."""
    for exempt in config.exempt_paths:
        if path == exempt or path.rstrip("/") == exempt.rstrip("/"):
            return True
    return False


def _is_protected(path: str, config: X402Config) -> bool:
    """Check if a request path requires payment."""
    if _is_exempt(path, config):
        return False
    for prefix in config.protected_prefixes:
        if path.startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class X402Middleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing x402 payment on protected routes.

    Verification chain:
    1. Check if path is protected
    2. Read x-payment or payment-signature header
    3. Decode payment payload
    4. Expiry check
    5. Replay check (nonce)
    6. Amount check
    7. EIP-712 signature verification (if verify_signatures=True)
    8. Pass through to handler
    """

    def __init__(self, app: Any, config: X402Config, settlement_store: Optional[SettlementStore] = None) -> None:
        super().__init__(app)
        self.config = config
        self.nonce_tracker = NonceTracker()
        if settlement_store is not None:
            self.settlement_store: Optional[SettlementStore] = settlement_store
        elif config.enable_settlement:
            from .x402_settlement import SettlementStore as _SS

            self.settlement_store = _SS(threshold=config.settlement_threshold)
        else:
            self.settlement_store = None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        if not _is_protected(path, self.config):
            return await call_next(request)

        # Read payment header — try x402 spec first, then legacy
        x402_header = request.headers.get("x-payment", "")
        legacy_header = request.headers.get("payment-signature", "")

        if not x402_header and not legacy_header:
            return self._payment_required_response(path)

        # Decode payment
        if x402_header:
            payment_data = decode_x402_payment(x402_header)
            if payment_data is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_payment",
                        "message": "x-payment header must be base64-encoded JSON.",
                    },
                )
        else:
            payment_data = decode_payment_signature(legacy_header)
            if payment_data is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_payment",
                        "message": "PAYMENT-SIGNATURE header must be base64-encoded JSON with paymentId, payer, and signature fields.",
                    },
                )

        # Extract authorization (may be nested or flat)
        authorization = payment_data.get("authorization", payment_data)
        signature = payment_data.get("signature", authorization.get("signature", ""))

        # 1. Expiry check
        ok, err = check_authorization_expiry(authorization)
        if not ok:
            return JSONResponse(
                status_code=400,
                content={"error": "payment_expired", "message": err},
            )

        # 2. Replay check
        nonce = authorization.get("nonce", payment_data.get("paymentId", ""))
        if nonce:
            nonce_str = nonce if isinstance(nonce, str) else str(nonce)
            if not self.nonce_tracker.check_and_mark(nonce_str):
                return JSONResponse(
                    status_code=400,
                    content={"error": "payment_replay", "message": "This payment nonce has already been used."},
                )

        # 3. Amount check
        if "value" in authorization:
            ok, err = check_payment_amount(authorization, self.config.price_per_call)
            if not ok:
                return JSONResponse(
                    status_code=400,
                    content={"error": "insufficient_payment", "message": err},
                )

        # 4. EIP-712 signature verification
        if self.config.verify_signatures and signature and "from" in authorization:
            token_contract = self.config.resolved_token_contract()
            chain_id = self.config.resolved_chain_id()
            ok, err = verify_eip712_signature(authorization, signature, token_contract, chain_id)
            if not ok:
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid_signature", "message": err},
                )

        # Payment accepted
        payer = authorization.get("from", payment_data.get("payer", "?"))
        paid_amount = int(authorization.get("value", 0))
        logger.info("x402: payment accepted for %s from %s (amount=%d)", path, payer, paid_amount)

        # Store paid amount on request state so endpoints can check pricing tiers
        request.state.x402_paid_amount = paid_amount
        request.state.x402_cloud_price = int(self.config.cloud_price_per_call)

        response = await call_next(request)

        # Only settle payment on successful (2xx) responses.
        # If the service fails (5xx) or returns client error (4xx), the agent
        # shouldn't be charged — their payment is accepted but not settled.
        if 200 <= response.status_code < 300:
            if self.settlement_store is not None:
                from .x402_settlement import PendingAuthorization, settle_batch

                amount = int(authorization.get("value", 0))
                nonce_key = nonce_str if nonce else ""
                pending = PendingAuthorization(
                    authorization=authorization,
                    signature=signature,
                    payer=payer,
                    amount=amount,
                    request_path=path,
                )
                ready = self.settlement_store.add(nonce_key, pending)
                if ready:
                    asyncio.ensure_future(settle_batch(self.settlement_store, self.config.facilitator_url, self.config))
                response.headers["x-payment-response"] = json.dumps({"status": "pending_settlement"})
            else:
                response.headers["x-payment-response"] = json.dumps({"status": "settled"})
        elif response.status_code >= 400:
            if self.settlement_store is not None:
                nonce_key = nonce_str if nonce else ""
                self.settlement_store.discard(nonce_key)
            response.headers["x-payment-response"] = json.dumps(
                {"status": "refunded", "reason": f"Service returned {response.status_code}"}
            )

        return response

    def _payment_required_response(self, resource: str = "") -> JSONResponse:
        """Build a 402 Payment Required response."""
        requirements = build_payment_requirements(self.config, resource=resource)
        encoded = encode_payment_requirements(requirements)
        return JSONResponse(
            status_code=402,
            content={
                "error": "payment_required",
                "message": "This endpoint requires payment. Include an x-payment or payment-signature header.",
                "requirements": requirements,
            },
            headers={"payment-required": encoded},
        )
