"""Tests for x402 payment gating middleware."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
import pytest_asyncio

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")


@pytest_asyncio.fixture
async def x402_client():
    """Create a test client with x402 enabled."""
    from httpx import ASGITransport, AsyncClient

    from octomil.mcp.http_server import HTTPServerConfig, create_http_app

    config = HTTPServerConfig(
        host="127.0.0.1",
        port=8402,
        enable_x402=True,
        x402_address="0xTEST_ADDRESS",
        x402_price="0.01",
    )
    app = create_http_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def no_x402_client():
    """Create a test client without x402."""
    from httpx import ASGITransport, AsyncClient

    from octomil.mcp.http_server import HTTPServerConfig, create_http_app

    config = HTTPServerConfig(host="127.0.0.1", port=8402, enable_x402=False)
    app = create_http_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _make_payment_signature(payment_id: str = "test-id", payer: str = "0xPAYER", signature: str = "0xSIG") -> str:
    """Create a valid PAYMENT-SIGNATURE header value."""
    data = {"paymentId": payment_id, "payer": payer, "signature": signature}
    return base64.b64encode(json.dumps(data).encode()).decode()


# ---------------------------------------------------------------------------
# 402 response
# ---------------------------------------------------------------------------


class TestPaymentRequired:
    @pytest.mark.asyncio
    async def test_protected_endpoint_returns_402(self, x402_client: Any) -> None:
        """Protected endpoints return 402 without payment."""
        resp = await x402_client.post("/api/v1/run_inference", json={"prompt": "hello"})
        assert resp.status_code == 402
        data = resp.json()
        assert data["error"] == "payment_required"
        assert "requirements" in data
        assert data["requirements"]["price"] == "0.01"
        assert data["requirements"]["payTo"] == "0xTEST_ADDRESS"

    @pytest.mark.asyncio
    async def test_402_has_payment_required_header(self, x402_client: Any) -> None:
        """402 response includes base64-encoded PAYMENT-REQUIRED header."""
        resp = await x402_client.post("/api/v1/list_models", json={})
        assert resp.status_code == 402
        header = resp.headers.get("payment-required")
        assert header is not None
        # Decode and verify structure
        decoded = json.loads(base64.b64decode(header))
        assert "paymentId" in decoded
        assert "price" in decoded
        assert "expiresAt" in decoded
        assert decoded["scheme"] == "x402"

    @pytest.mark.asyncio
    async def test_402_contains_payment_id(self, x402_client: Any) -> None:
        """Each 402 response has a unique paymentId."""
        resp1 = await x402_client.post("/api/v1/list_models", json={})
        resp2 = await x402_client.post("/api/v1/list_models", json={})
        id1 = resp1.json()["requirements"]["paymentId"]
        id2 = resp2.json()["requirements"]["paymentId"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# Payment acceptance
# ---------------------------------------------------------------------------


class TestPaymentAcceptance:
    @pytest.mark.asyncio
    async def test_valid_payment_passes_through(self, x402_client: Any) -> None:
        """Valid PAYMENT-SIGNATURE header allows request through."""
        from unittest.mock import patch

        sig = _make_payment_signature()
        with patch("octomil.models.catalog.CATALOG", {}):
            resp = await x402_client.post(
                "/api/v1/list_models",
                json={},
                headers={"payment-signature": sig},
            )
        # Should pass through to the actual handler
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_400(self, x402_client: Any) -> None:
        """Malformed base64 in PAYMENT-SIGNATURE returns 400."""
        resp = await x402_client.post(
            "/api/v1/list_models",
            json={},
            headers={"payment-signature": "not-valid-base64!!!"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_payment"

    @pytest.mark.asyncio
    async def test_missing_fields_returns_400(self, x402_client: Any) -> None:
        """PAYMENT-SIGNATURE missing required fields returns 400."""
        incomplete = base64.b64encode(json.dumps({"paymentId": "x"}).encode()).decode()
        resp = await x402_client.post(
            "/api/v1/list_models",
            json={},
            headers={"payment-signature": incomplete},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Exempt paths
# ---------------------------------------------------------------------------


class TestExemptPaths:
    @pytest.mark.asyncio
    async def test_agent_card_exempt(self, x402_client: Any) -> None:
        """Agent card endpoint is exempt from payment."""
        resp = await x402_client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_exempt(self, x402_client: Any) -> None:
        """Health endpoint is exempt from payment."""
        resp = await x402_client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_exempt(self, x402_client: Any) -> None:
        """Metrics endpoint is exempt from payment."""
        resp = await x402_client.get("/api/v1/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_docs_exempt(self, x402_client: Any) -> None:
        """OpenAPI docs are exempt from payment."""
        resp = await x402_client.get("/docs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# x402 disabled
# ---------------------------------------------------------------------------


class TestX402Disabled:
    @pytest.mark.asyncio
    async def test_no_402_when_disabled(self, no_x402_client: Any) -> None:
        """Without x402, protected endpoints work normally."""
        from unittest.mock import patch

        with patch("octomil.models.catalog.CATALOG", {}):
            resp = await no_x402_client.post("/api/v1/list_models", json={})
        # Should work without payment
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_build_payment_requirements(self) -> None:
        from octomil.mcp.x402 import X402Config, build_payment_requirements

        config = X402Config(price_per_call="0.05", currency="ETH", network="mainnet", payment_address="0xABC")
        req = build_payment_requirements(config)
        assert req["price"] == "0.05"
        assert req["currency"] == "ETH"
        assert req["network"] == "mainnet"
        assert req["payTo"] == "0xABC"
        assert "paymentId" in req
        assert "expiresAt" in req

    def test_encode_decode_roundtrip(self) -> None:
        from octomil.mcp.x402 import (
            X402Config,
            build_payment_requirements,
            encode_payment_requirements,
        )

        config = X402Config(payment_address="0xTEST")
        req = build_payment_requirements(config)
        encoded = encode_payment_requirements(req)
        # This is a requirements header, not a signature — but verify encoding works
        decoded = json.loads(base64.b64decode(encoded))
        assert decoded["payTo"] == "0xTEST"

    def test_decode_valid_signature(self) -> None:
        from octomil.mcp.x402 import decode_payment_signature

        sig = _make_payment_signature("id-1", "0xALICE", "0xSIG123")
        result = decode_payment_signature(sig)
        assert result is not None
        assert result["paymentId"] == "id-1"
        assert result["payer"] == "0xALICE"
        assert result["signature"] == "0xSIG123"

    def test_decode_invalid_base64(self) -> None:
        from octomil.mcp.x402 import decode_payment_signature

        assert decode_payment_signature("not!base64!!") is None

    def test_decode_missing_fields(self) -> None:
        from octomil.mcp.x402 import decode_payment_signature

        partial = base64.b64encode(json.dumps({"paymentId": "x"}).encode()).decode()
        assert decode_payment_signature(partial) is None
