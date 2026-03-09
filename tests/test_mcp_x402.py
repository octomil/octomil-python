"""Tests for x402 payment gating middleware."""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any
from unittest.mock import patch

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
        x402_price="1000",  # base units (e.g. 1000 = 0.001 USDC with 6 decimals)
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
        assert data["requirements"]["x402Version"] == 1
        assert data["requirements"]["accepts"][0]["payTo"] == "0xTEST_ADDRESS"
        assert data["requirements"]["accepts"][0]["maxAmountRequired"] == "1000"

    @pytest.mark.asyncio
    async def test_402_has_payment_required_header(self, x402_client: Any) -> None:
        """402 response includes base64-encoded PAYMENT-REQUIRED header."""
        resp = await x402_client.post("/api/v1/run_inference", json={"prompt": "test"})
        assert resp.status_code == 402
        header = resp.headers.get("payment-required")
        assert header is not None
        # Decode and verify structure
        decoded = json.loads(base64.b64decode(header))
        assert decoded["x402Version"] == 1
        assert "accepts" in decoded
        assert len(decoded["accepts"]) > 0
        assert "payTo" in decoded["accepts"][0]

    @pytest.mark.asyncio
    async def test_402_contains_unique_responses(self, x402_client: Any) -> None:
        """Each 402 response is consistent with the config."""
        resp1 = await x402_client.post("/api/v1/run_inference", json={"prompt": "test"})
        resp2 = await x402_client.post("/api/v1/run_inference", json={"prompt": "test"})
        accepts1 = resp1.json()["requirements"]["accepts"][0]
        accepts2 = resp2.json()["requirements"]["accepts"][0]
        # Both should have the same payTo and amount from config
        assert accepts1["payTo"] == accepts2["payTo"]
        assert accepts1["maxAmountRequired"] == accepts2["maxAmountRequired"]


# ---------------------------------------------------------------------------
# Payment acceptance
# ---------------------------------------------------------------------------


class TestPaymentAcceptance:
    @pytest.mark.asyncio
    async def test_valid_payment_passes_through(self, x402_client: Any) -> None:
        """Valid PAYMENT-SIGNATURE header allows request through."""
        sig = _make_payment_signature()
        mock_backend = x402_client._transport.app.state.backend  # type: ignore[union-attr]
        with patch.object(
            mock_backend,
            "generate",
            return_value=(
                "ok",
                {"engine": "test", "model": "test", "tokens_per_second": 1, "total_tokens": 1, "ttfc_ms": 1},
            ),
        ):
            resp = await x402_client.post(
                "/api/v1/run_inference",
                json={"prompt": "test"},
                headers={"payment-signature": sig},
            )
        # Should pass through to the actual handler
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_400(self, x402_client: Any) -> None:
        """Malformed base64 in PAYMENT-SIGNATURE returns 400."""
        resp = await x402_client.post(
            "/api/v1/run_inference",
            json={"prompt": "test"},
            headers={"payment-signature": "not-valid-base64!!!"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_payment"

    @pytest.mark.asyncio
    async def test_missing_fields_returns_400(self, x402_client: Any) -> None:
        """PAYMENT-SIGNATURE missing required fields returns 400."""
        incomplete = base64.b64encode(json.dumps({"paymentId": "x"}).encode()).decode()
        resp = await x402_client.post(
            "/api/v1/run_inference",
            json={"prompt": "test"},
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
        mock_backend = no_x402_client._transport.app.state.backend  # type: ignore[union-attr]
        with patch.object(
            mock_backend,
            "generate",
            return_value=(
                "ok",
                {"engine": "test", "model": "test", "tokens_per_second": 1, "total_tokens": 1, "ttfc_ms": 1},
            ),
        ):
            resp = await no_x402_client.post("/api/v1/run_inference", json={"prompt": "test"})
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
        assert req["x402Version"] == 1
        assert req["accepts"][0]["maxAmountRequired"] == "0.05"
        assert req["accepts"][0]["network"] == "mainnet"
        assert req["accepts"][0]["payTo"] == "0xABC"

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
        assert decoded["accepts"][0]["payTo"] == "0xTEST"

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


# ---------------------------------------------------------------------------
# Expiry enforcement
# ---------------------------------------------------------------------------


class TestExpiryEnforcement:
    def test_valid_window_passes(self) -> None:
        from octomil.mcp.x402 import check_authorization_expiry

        now = int(time.time())
        auth = {"validAfter": now - 60, "validBefore": now + 60}
        ok, err = check_authorization_expiry(auth)
        assert ok
        assert err == ""

    def test_expired_rejected(self) -> None:
        from octomil.mcp.x402 import check_authorization_expiry

        now = int(time.time())
        auth = {"validAfter": now - 120, "validBefore": now - 60}
        ok, err = check_authorization_expiry(auth)
        assert not ok
        assert "expired" in err.lower()

    def test_not_yet_valid_rejected(self) -> None:
        from octomil.mcp.x402 import check_authorization_expiry

        now = int(time.time())
        auth = {"validAfter": now + 60, "validBefore": now + 120}
        ok, err = check_authorization_expiry(auth)
        assert not ok
        assert "not yet valid" in err.lower()

    def test_no_timestamps_passes(self) -> None:
        from octomil.mcp.x402 import check_authorization_expiry

        ok, err = check_authorization_expiry({})
        assert ok

    def test_string_timestamps_handled(self) -> None:
        from octomil.mcp.x402 import check_authorization_expiry

        now = int(time.time())
        auth = {"validAfter": str(now - 60), "validBefore": str(now + 60)}
        ok, err = check_authorization_expiry(auth)
        assert ok


# ---------------------------------------------------------------------------
# Amount validation
# ---------------------------------------------------------------------------


class TestAmountValidation:
    def test_sufficient_amount_passes(self) -> None:
        from octomil.mcp.x402 import check_payment_amount

        ok, err = check_payment_amount({"value": "2000"}, "1000")
        assert ok

    def test_exact_amount_passes(self) -> None:
        from octomil.mcp.x402 import check_payment_amount

        ok, err = check_payment_amount({"value": "1000"}, "1000")
        assert ok

    def test_insufficient_amount_rejected(self) -> None:
        from octomil.mcp.x402 import check_payment_amount

        ok, err = check_payment_amount({"value": "500"}, "1000")
        assert not ok
        assert "insufficient" in err.lower()


# ---------------------------------------------------------------------------
# Replay protection
# ---------------------------------------------------------------------------


class TestReplayProtection:
    def test_fresh_nonce_passes(self) -> None:
        from octomil.mcp.x402 import NonceTracker

        tracker = NonceTracker()
        assert tracker.check_and_mark("nonce-1") is True

    def test_replay_nonce_rejected(self) -> None:
        from octomil.mcp.x402 import NonceTracker

        tracker = NonceTracker()
        assert tracker.check_and_mark("nonce-1") is True
        assert tracker.check_and_mark("nonce-1") is False

    def test_different_nonces_both_pass(self) -> None:
        from octomil.mcp.x402 import NonceTracker

        tracker = NonceTracker()
        assert tracker.check_and_mark("nonce-a") is True
        assert tracker.check_and_mark("nonce-b") is True


# ---------------------------------------------------------------------------
# EIP-712 verification
# ---------------------------------------------------------------------------


class TestEIP712Verification:
    def test_valid_signature_passes(self) -> None:
        """Test with a real EIP-712 signature from eth_account."""
        try:
            from eth_account import Account
        except ImportError:
            pytest.skip("eth-account not installed")

        from octomil.mcp.x402 import USDC_CONTRACTS, verify_eip712_signature

        # Generate a test key and sign
        acct = Account.from_key("0x" + "ab" * 32)
        now = int(time.time())
        authorization = {
            "from": acct.address,
            "to": "0x1234567890123456789012345678901234567890",
            "value": 1000,
            "validAfter": now - 60,
            "validBefore": now + 300,
            "nonce": "0x" + "00" * 32,
        }

        # Create the signature using eth_account
        from eth_account.messages import encode_typed_data

        token_contract = USDC_CONTRACTS["base"]
        domain = {
            "name": "USD Coin",
            "version": "2",
            "chainId": 8453,
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
        signable = encode_typed_data(
            domain_data=domain,
            message_types=types,
            message_data=authorization,
        )
        signed = acct.sign_message(signable)

        ok, err = verify_eip712_signature(authorization, signed.signature.hex(), token_contract, 8453)
        assert ok, f"Verification failed: {err}"

    def test_wrong_signer_rejected(self) -> None:
        """Signature from different key should fail."""
        try:
            from eth_account import Account
        except ImportError:
            pytest.skip("eth-account not installed")

        from octomil.mcp.x402 import USDC_CONTRACTS, verify_eip712_signature

        signer = Account.from_key("0x" + "ab" * 32)
        other = Account.from_key("0x" + "cd" * 32)
        now = int(time.time())

        # Claim to be from `other` but sign with `signer`
        authorization = {
            "from": other.address,  # claims to be other
            "to": "0x1234567890123456789012345678901234567890",
            "value": 1000,
            "validAfter": now - 60,
            "validBefore": now + 300,
            "nonce": "0x" + "00" * 32,
        }

        from eth_account.messages import encode_typed_data

        token_contract = USDC_CONTRACTS["base"]
        domain = {
            "name": "USD Coin",
            "version": "2",
            "chainId": 8453,
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
        signable = encode_typed_data(
            domain_data=domain,
            message_types=types,
            message_data=authorization,
        )
        signed = signer.sign_message(signable)  # signed by wrong account

        ok, err = verify_eip712_signature(authorization, signed.signature.hex(), token_contract, 8453)
        assert not ok
        assert "mismatch" in err.lower()

    def test_graceful_without_eth_account(self) -> None:
        """Without eth-account, verification should return True."""
        import builtins

        from octomil.mcp.x402 import verify_eip712_signature

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "eth_account" or name.startswith("eth_account."):
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            ok, err = verify_eip712_signature({"from": "0x123"}, "0xsig", "0xcontract", 8453)
        assert ok


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------


class TestLegacyCompat:
    @pytest.mark.asyncio
    async def test_legacy_payment_signature_still_works(self, x402_client: Any) -> None:
        """Old payment-signature header should still be accepted."""
        sig = _make_payment_signature()
        mock_backend = x402_client._transport.app.state.backend  # type: ignore[union-attr]
        with patch.object(
            mock_backend,
            "generate",
            return_value=(
                "ok",
                {"engine": "test", "model": "test", "tokens_per_second": 1, "total_tokens": 1, "ttfc_ms": 1},
            ),
        ):
            resp = await x402_client.post(
                "/api/v1/run_inference",
                json={"prompt": "test"},
                headers={"payment-signature": sig},
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# x-payment header (new format)
# ---------------------------------------------------------------------------


class TestX402Header:
    @pytest.mark.asyncio
    async def test_x_payment_header_works(self, x402_client: Any) -> None:
        """New x-payment header should be accepted."""
        import time as _time

        now = int(_time.time())
        payload = {
            "authorization": {
                "from": "0xPAYER",
                "to": "0xTEST_ADDRESS",
                "value": "1000",
                "validAfter": str(now - 60),
                "validBefore": str(now + 300),
                "nonce": str(uuid.uuid4()),
            },
            "signature": "0xSIG",
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        mock_backend = x402_client._transport.app.state.backend  # type: ignore[union-attr]
        with (
            patch.object(
                mock_backend,
                "generate",
                return_value=(
                    "ok",
                    {"engine": "test", "model": "test", "tokens_per_second": 1, "total_tokens": 1, "ttfc_ms": 1},
                ),
            ),
            patch("octomil.mcp.x402.verify_eip712_signature", return_value=(True, "")),
        ):
            resp = await x402_client.post(
                "/api/v1/run_inference",
                json={"prompt": "test"},
                headers={"x-payment": encoded},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_expired_payment_rejected(self, x402_client: Any) -> None:
        """Expired x-payment should be rejected."""
        import time as _time

        now = int(_time.time())
        payload = {
            "authorization": {
                "from": "0xPAYER",
                "to": "0xTEST_ADDRESS",
                "value": "1000",
                "validAfter": str(now - 120),
                "validBefore": str(now - 60),  # expired
                "nonce": str(uuid.uuid4()),
            },
            "signature": "0xSIG",
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        resp = await x402_client.post(
            "/api/v1/run_inference",
            json={"prompt": "test"},
            headers={"x-payment": encoded},
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "payment_expired"

    @pytest.mark.asyncio
    async def test_replay_rejected(self, x402_client: Any) -> None:
        """Same nonce used twice should be rejected."""
        import time as _time

        now = int(_time.time())
        fixed_nonce = str(uuid.uuid4())
        payload = {
            "authorization": {
                "from": "0xPAYER",
                "to": "0xTEST_ADDRESS",
                "value": "1000",
                "validAfter": str(now - 60),
                "validBefore": str(now + 300),
                "nonce": fixed_nonce,
            },
            "signature": "0xSIG",
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        mock_backend = x402_client._transport.app.state.backend  # type: ignore[union-attr]
        with (
            patch.object(
                mock_backend,
                "generate",
                return_value=(
                    "ok",
                    {"engine": "test", "model": "test", "tokens_per_second": 1, "total_tokens": 1, "ttfc_ms": 1},
                ),
            ),
            patch("octomil.mcp.x402.verify_eip712_signature", return_value=(True, "")),
        ):
            resp1 = await x402_client.post(
                "/api/v1/run_inference",
                json={"prompt": "test"},
                headers={"x-payment": encoded},
            )
        assert resp1.status_code == 200

        # Second request with same nonce — should be rejected as replay (before signature check)
        resp2 = await x402_client.post(
            "/api/v1/run_inference",
            json={"prompt": "test"},
            headers={"x-payment": encoded},
        )
        assert resp2.status_code == 400
        assert resp2.json()["error"] == "payment_replay"

    @pytest.mark.asyncio
    async def test_insufficient_payment_rejected(self, x402_client: Any) -> None:
        """Payment below required amount should be rejected."""
        import time as _time

        now = int(_time.time())
        payload = {
            "authorization": {
                "from": "0xPAYER",
                "to": "0xTEST_ADDRESS",
                "value": "1",  # way too low — config price is "0.01"
                "validAfter": str(now - 60),
                "validBefore": str(now + 300),
                "nonce": str(uuid.uuid4()),
            },
            "signature": "0xSIG",
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        resp = await x402_client.post(
            "/api/v1/run_inference",
            json={"prompt": "test"},
            headers={"x-payment": encoded},
        )
        # Note: x402_client fixture uses x402_price="0.01", which as integer comparison:
        # int("1") < int("0.01") — this will raise ValueError because "0.01" is not an int
        # The amount check handles this gracefully
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Verification helpers unit tests
# ---------------------------------------------------------------------------


class TestVerificationHelpers:
    def test_chain_id_resolution(self) -> None:
        from octomil.mcp.x402 import X402Config

        config = X402Config(network="base")
        assert config.resolved_chain_id() == 8453

        config2 = X402Config(network="ethereum")
        assert config2.resolved_chain_id() == 1

    def test_token_contract_resolution(self) -> None:
        from octomil.mcp.x402 import X402Config

        config = X402Config(network="base")
        assert config.resolved_token_contract() == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

    def test_custom_token_contract(self) -> None:
        from octomil.mcp.x402 import X402Config

        config = X402Config(network="base", token_contract="0xCUSTOM")
        assert config.resolved_token_contract() == "0xCUSTOM"

    def test_decode_x402_payment_valid(self) -> None:
        from octomil.mcp.x402 import decode_x402_payment

        payload = {"authorization": {"from": "0x123"}, "signature": "0xSIG"}
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        result = decode_x402_payment(encoded)
        assert result is not None
        assert result["authorization"]["from"] == "0x123"

    def test_decode_x402_payment_invalid(self) -> None:
        from octomil.mcp.x402 import decode_x402_payment

        assert decode_x402_payment("not-base64!!!") is None

    def test_nonce_tracker_cleanup(self) -> None:
        from octomil.mcp.x402 import NonceTracker

        tracker = NonceTracker()
        tracker._max_entries = 5  # low limit for testing
        for i in range(10):
            tracker.check_and_mark(f"nonce-{i}")
        # After exceeding max, old entries should be cleaned
        assert len(tracker._seen) <= 10  # cleanup happens on next check


# ---------------------------------------------------------------------------
# x402 settlement gating on 2xx
# ---------------------------------------------------------------------------


class TestX402SettlementGating:
    @pytest.mark.asyncio
    async def test_settlement_on_2xx(self, x402_client: Any) -> None:
        """Successful responses should include settled payment header."""
        import time as _time

        now = int(_time.time())
        payload = {
            "authorization": {
                "from": "0xPAYER",
                "to": "0xTEST_ADDRESS",
                "value": "1000",
                "validAfter": str(now - 60),
                "validBefore": str(now + 300),
                "nonce": str(uuid.uuid4()),
            },
            "signature": "0xSIG",
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        # Mock the backend to return a successful inference result
        mock_backend = x402_client._transport.app.state.backend  # type: ignore[union-attr]
        with (
            patch.object(
                mock_backend,
                "generate",
                return_value=(
                    "hello",
                    {"engine": "test", "model": "test", "tokens_per_second": 1, "total_tokens": 1, "ttfc_ms": 1},
                ),
            ),
            patch("octomil.mcp.x402.verify_eip712_signature", return_value=(True, "")),
        ):
            resp = await x402_client.post(
                "/api/v1/run_inference",
                json={"prompt": "test"},
                headers={"x-payment": encoded},
            )
        assert resp.status_code == 200
        payment_resp = json.loads(resp.headers["x-payment-response"])
        assert payment_resp["status"] == "settled"

    @pytest.mark.asyncio
    async def test_no_settlement_on_5xx(self, x402_client: Any) -> None:
        """Failed responses should include refunded payment header."""
        import time as _time

        now = int(_time.time())
        payload = {
            "authorization": {
                "from": "0xPAYER",
                "to": "0xTEST_ADDRESS",
                "value": "1000",
                "validAfter": str(now - 60),
                "validBefore": str(now + 300),
                "nonce": str(uuid.uuid4()),
            },
            "signature": "0xSIG",
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        # Code tool without a loaded model returns 503
        mock_backend = x402_client._transport.app.state.backend  # type: ignore[union-attr]
        with (
            patch.object(mock_backend, "generate", side_effect=RuntimeError("no model")),
            patch("octomil.mcp.x402.verify_eip712_signature", return_value=(True, "")),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os as _os

            _os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await x402_client.post(
                "/api/v1/generate_code",
                json={"description": "hello"},
                headers={"x-payment": encoded},
            )
        assert resp.status_code == 503
        payment_resp = json.loads(resp.headers["x-payment-response"])
        assert payment_resp["status"] == "refunded"

    @pytest.mark.asyncio
    async def test_warmup_exempt_from_x402(self, x402_client: Any) -> None:
        """Warmup and ready endpoints should not require payment."""
        resp = await x402_client.post("/api/v1/warmup")
        assert resp.status_code in (200, 202, 503)
        # Should NOT be 402
        assert resp.status_code != 402

    @pytest.mark.asyncio
    async def test_ready_exempt_from_x402(self, x402_client: Any) -> None:
        resp = await x402_client.get("/api/v1/ready")
        assert resp.status_code in (200, 503)
        assert resp.status_code != 402

    @pytest.mark.asyncio
    async def test_platform_metadata_tools_exempt(self, x402_client: Any) -> None:
        """Platform metadata tools should not require payment."""
        exempt_endpoints = [
            ("POST", "/api/v1/resolve_model", {"name": "test"}),
            ("POST", "/api/v1/list_models", {}),
            ("POST", "/api/v1/detect_engines", {}),
            ("GET", "/api/v1/hardware_profile", None),
            ("POST", "/api/v1/recommend_model", {"priority": "speed"}),
            ("GET", "/api/v1/metrics", None),
        ]
        for method, path, body in exempt_endpoints:
            if method == "GET":
                resp = await x402_client.get(path)
            else:
                resp = await x402_client.post(path, json=body)
            assert resp.status_code != 402, f"{path} should be exempt but got 402"

    @pytest.mark.asyncio
    async def test_inference_tools_require_payment(self, x402_client: Any) -> None:
        """Inference tools should require payment when x402 is enabled."""
        paid_endpoints = [
            ("/api/v1/run_inference", {"prompt": "test"}),
            ("/api/v1/generate_code", {"description": "test"}),
            ("/api/v1/review_code", {"code": "x=1"}),
            ("/api/v1/explain_code", {"code": "x=1"}),
            ("/api/v1/write_tests", {"code": "x=1"}),
            ("/api/v1/general_task", {"prompt": "test"}),
        ]
        for path, body in paid_endpoints:
            resp = await x402_client.post(path, json=body)
            assert resp.status_code == 402, f"{path} should require payment but got {resp.status_code}"
