"""x402 payment gating middleware for the Octomil HTTP agent server.

Implements the HTTP 402 payment protocol surface:
- Returns 402 + PAYMENT-REQUIRED header when payment is needed
- Accepts PAYMENT-SIGNATURE header and validates structure
- Exempts discovery and health endpoints

Phase 1: protocol surface only. Defers actual cryptographic verification,
blockchain integration, and payment settlement to later phases.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Payment requirement validity window
_DEFAULT_EXPIRY_SECONDS = 300  # 5 minutes


@dataclass
class X402Config:
    """Configuration for x402 payment gating."""

    price_per_call: str = "0.001"
    currency: str = "USDC"
    network: str = "base"
    payment_address: str = ""
    protected_prefixes: list[str] = field(default_factory=lambda: ["/api/v1/"])
    exempt_paths: list[str] = field(
        default_factory=lambda: [
            "/.well-known/agent-card.json",
            "/health",
            "/api/v1/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
    )
    expiry_seconds: int = _DEFAULT_EXPIRY_SECONDS


def build_payment_requirements(config: X402Config) -> dict[str, Any]:
    """Build the PaymentRequirements JSON for a 402 response."""
    payment_id = str(uuid.uuid4())
    expires_at = int(time.time()) + config.expiry_seconds
    return {
        "scheme": "x402",
        "version": "1",
        "paymentId": payment_id,
        "price": config.price_per_call,
        "currency": config.currency,
        "network": config.network,
        "payTo": config.payment_address,
        "expiresAt": expires_at,
    }


def encode_payment_requirements(requirements: dict[str, Any]) -> str:
    """Base64-encode payment requirements for the PAYMENT-REQUIRED header."""
    return base64.b64encode(json.dumps(requirements).encode()).decode()


def decode_payment_signature(header_value: str) -> dict[str, Any] | None:
    """Decode and validate structure of a PAYMENT-SIGNATURE header.

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


class X402Middleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces x402 payment on protected routes.

    When a request hits a protected path without a valid PAYMENT-SIGNATURE
    header, responds with 402 and a PAYMENT-REQUIRED header containing
    base64-encoded payment requirements.

    Phase 1: validates signature structure only (required fields present).
    Actual cryptographic verification deferred to later phases.
    """

    def __init__(self, app: Any, config: X402Config) -> None:
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        if not _is_protected(path, self.config):
            return await call_next(request)

        # Check for payment signature
        payment_header = request.headers.get("payment-signature", "")
        if not payment_header:
            return self._payment_required_response()

        # Validate signature structure
        payment_data = decode_payment_signature(payment_header)
        if payment_data is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_payment",
                    "message": "PAYMENT-SIGNATURE header must be base64-encoded JSON with paymentId, payer, and signature fields.",
                },
            )

        # Phase 1: structure is valid — allow through
        # TODO: Phase 2+ will verify cryptographic signature, check expiry, settle payment
        logger.info("x402: payment accepted (structure-only) for %s from %s", path, payment_data.get("payer", "?"))
        return await call_next(request)

    def _payment_required_response(self) -> JSONResponse:
        """Build a 402 Payment Required response."""
        requirements = build_payment_requirements(self.config)
        encoded = encode_payment_requirements(requirements)
        return JSONResponse(
            status_code=402,
            content={
                "error": "payment_required",
                "message": "This endpoint requires payment. Include a PAYMENT-SIGNATURE header.",
                "requirements": requirements,
            },
            headers={"payment-required": encoded},
        )
