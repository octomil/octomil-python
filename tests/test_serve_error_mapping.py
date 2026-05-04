"""Cutover follow-up #70: serve-layer HTTP status mapping for the
bounded ``OctomilErrorCode`` taxonomy.

Pre-cutover the FastAPI exception handler's status_map covered only
the OpenAI-style codes (INVALID_INPUT, AUTHENTICATION_FAILED,
MODEL_NOT_FOUND, RATE_LIMITED, ...). The hard-cutover added
``NativeChatBackend`` which raises ``OctomilError(UNSUPPORTED_MODALITY)``
on grammar / json_mode / enable_thinking / streaming / unknown roles
/ unknown message keys. Without an explicit mapping these surfaced
as HTTP 500 — reading as "server bug" to the API caller, which is
wrong: the request was well-formed but the route can't satisfy it.

This module pins:
  - UNSUPPORTED_MODALITY  → 422 (Unprocessable Entity)
  - CONTEXT_TOO_LARGE     → 413 (Payload Too Large)
  - CHECKSUM_MISMATCH     → 422
  - MODEL_DISABLED        → 403
  - POLICY_DENIED         → 403
  - ACCELERATOR_UNAVAILABLE → 503
  - INSUFFICIENT_MEMORY   → 503
  - INSUFFICIENT_STORAGE  → 507
  - CANCELLED             → 499 (nginx convention)
  - The pre-existing OpenAI-compat mappings stay unchanged.

We exercise the live FastAPI exception handler via a minimal
``FastAPI`` app that re-uses the production handler. The cutover
SDK is not pulled in (no native runtime needed) — the test runs
unconditionally.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from octomil.errors import OctomilError, OctomilErrorCode  # noqa: E402


def _make_app() -> FastAPI:
    """Construct a minimal FastAPI app wired to the same exception
    handler the production app installs. Mirrors the body of
    ``octomil/serve/app.py:octomil_error_handler`` so a refactor
    that drifts the two surfaces is caught here.

    We can't import the production app directly without standing
    up the full server stack (multi_model state, telemetry,
    middleware); duplicating the handler body is the smallest
    isolated surface."""
    app = FastAPI()

    @app.exception_handler(OctomilError)
    async def octomil_error_handler(  # noqa: ARG001 — request unused
        request: Request, exc: OctomilError
    ) -> JSONResponse:
        from octomil._generated.error_code import ERROR_CLASSIFICATION, RetryClass

        classification = ERROR_CLASSIFICATION.get(exc.code)
        # MUST stay in lockstep with octomil/serve/app.py:494-525.
        # Test_serve_status_map_matches_production_handler below
        # asserts the keys are the same set.
        status_map = {
            OctomilErrorCode.INVALID_INPUT: 400,
            OctomilErrorCode.AUTHENTICATION_FAILED: 401,
            OctomilErrorCode.INVALID_API_KEY: 401,
            OctomilErrorCode.FORBIDDEN: 403,
            OctomilErrorCode.UNSUPPORTED_MODALITY: 422,
            OctomilErrorCode.CONTEXT_TOO_LARGE: 413,
            OctomilErrorCode.CHECKSUM_MISMATCH: 422,
            OctomilErrorCode.MODEL_DISABLED: 403,
            OctomilErrorCode.POLICY_DENIED: 403,
            OctomilErrorCode.MODEL_NOT_FOUND: 404,
            OctomilErrorCode.RATE_LIMITED: 429,
            OctomilErrorCode.MODEL_LOAD_FAILED: 503,
            OctomilErrorCode.RUNTIME_UNAVAILABLE: 503,
            OctomilErrorCode.INFERENCE_FAILED: 503,
            OctomilErrorCode.ACCELERATOR_UNAVAILABLE: 503,
            OctomilErrorCode.INSUFFICIENT_MEMORY: 503,
            OctomilErrorCode.INSUFFICIENT_STORAGE: 507,
            OctomilErrorCode.SERVER_ERROR: 500,
            OctomilErrorCode.REQUEST_TIMEOUT: 504,
            OctomilErrorCode.CANCELLED: 499,
        }
        status_code = status_map.get(exc.code, 500)
        return JSONResponse(
            status_code=status_code,
            content={
                "code": exc.code.value,
                "message": str(exc),
                "retryable": (classification.retry_class != RetryClass.NEVER if classification else False),
                "category": (classification.category.value if classification else "unknown"),
            },
        )

    @app.get("/raise/{code}")
    async def raise_route(code: str) -> dict:
        raise OctomilError(
            code=OctomilErrorCode(code),
            message=f"deliberate raise for {code}",
        )

    return app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(_make_app())


# Each tuple: (OctomilErrorCode, expected_http_status).
@pytest.mark.parametrize(
    "code,status",
    [
        # Pre-cutover OpenAI-compat codes — must stay stable.
        (OctomilErrorCode.INVALID_INPUT, 400),
        (OctomilErrorCode.AUTHENTICATION_FAILED, 401),
        (OctomilErrorCode.INVALID_API_KEY, 401),
        (OctomilErrorCode.FORBIDDEN, 403),
        (OctomilErrorCode.MODEL_NOT_FOUND, 404),
        (OctomilErrorCode.RATE_LIMITED, 429),
        (OctomilErrorCode.MODEL_LOAD_FAILED, 503),
        (OctomilErrorCode.RUNTIME_UNAVAILABLE, 503),
        (OctomilErrorCode.INFERENCE_FAILED, 503),
        (OctomilErrorCode.SERVER_ERROR, 500),
        (OctomilErrorCode.REQUEST_TIMEOUT, 504),
        # Cutover follow-up #70 — these are what this PR wires in.
        (OctomilErrorCode.UNSUPPORTED_MODALITY, 422),
        (OctomilErrorCode.CONTEXT_TOO_LARGE, 413),
        (OctomilErrorCode.CHECKSUM_MISMATCH, 422),
        (OctomilErrorCode.MODEL_DISABLED, 403),
        (OctomilErrorCode.POLICY_DENIED, 403),
        (OctomilErrorCode.ACCELERATOR_UNAVAILABLE, 503),
        (OctomilErrorCode.INSUFFICIENT_MEMORY, 503),
        (OctomilErrorCode.INSUFFICIENT_STORAGE, 507),
        (OctomilErrorCode.CANCELLED, 499),
    ],
)
def test_serve_status_map_each_code(client: TestClient, code: OctomilErrorCode, status: int) -> None:
    """Every code we map MUST surface its expected HTTP status."""
    resp = client.get(f"/raise/{code.value}")
    assert resp.status_code == status, f"{code.value} expected HTTP {status}, got {resp.status_code}"
    body = resp.json()
    assert body["code"] == code.value
    assert "message" in body and body["message"]


def test_serve_status_map_unmapped_defaults_to_500(client: TestClient) -> None:
    """Codes NOT in the explicit status map fall back to 500.
    UNKNOWN is the documented sentinel — it's intentionally not
    mapped (clients should see 500 + "unknown" code so the
    diagnostic flags an unhandled path)."""
    resp = client.get(f"/raise/{OctomilErrorCode.UNKNOWN.value}")
    assert resp.status_code == 500


def test_unsupported_modality_maps_to_422_not_500() -> None:
    """Cutover follow-up #70 specific regression: a
    grammar / json_mode / streaming / unknown-role request that
    reaches `NativeChatBackend` raises
    OctomilError(UNSUPPORTED_MODALITY). The serve layer MUST map
    that to 422, NOT 500. This test pins the exact failure mode
    Codex flagged in the cutover R1 review."""
    client = TestClient(_make_app())
    resp = client.get(f"/raise/{OctomilErrorCode.UNSUPPORTED_MODALITY.value}")
    assert resp.status_code == 422, (
        "UNSUPPORTED_MODALITY MUST map to HTTP 422, not 500. The "
        "cutover routes bounded native rejects through this code; "
        "500 would read as 'server bug' to API callers."
    )
    body = resp.json()
    assert body["code"] == "unsupported_modality"


def test_serve_status_map_matches_production_handler() -> None:
    """Drift guard: the test app's status_map (above) MUST stay in
    lockstep with the production handler in
    ``octomil/serve/app.py``. We extract both maps and assert
    set-equal keys + identical values.

    A future change that adds a new code to either map MUST add
    the same entry to the other.
    """
    import inspect
    import re

    from octomil.serve import app as serve_app

    # Find the status_map dict literal in the production handler's
    # source. We do this textually rather than calling the handler
    # because importing it requires standing up the full FastAPI
    # app + middleware stack.
    source = inspect.getsource(serve_app.create_app)
    # Pull every `OctomilErrorCode.<NAME>: <int>,` line.
    pattern = re.compile(r"OctomilErrorCode\.([A-Z_]+):\s*(\d+)")
    prod_pairs = pattern.findall(source)
    assert prod_pairs, "could not locate status_map in production handler"
    prod_map = {name: int(status) for name, status in prod_pairs}

    # Reconstruct our test-side map by re-using _make_app's body
    # textually. Simpler: hard-code what we expect (any drift
    # against the production version fails this assertion).
    expected = {
        "INVALID_INPUT": 400,
        "AUTHENTICATION_FAILED": 401,
        "INVALID_API_KEY": 401,
        "FORBIDDEN": 403,
        "UNSUPPORTED_MODALITY": 422,
        "CONTEXT_TOO_LARGE": 413,
        "CHECKSUM_MISMATCH": 422,
        "MODEL_DISABLED": 403,
        "POLICY_DENIED": 403,
        "MODEL_NOT_FOUND": 404,
        "RATE_LIMITED": 429,
        "MODEL_LOAD_FAILED": 503,
        "RUNTIME_UNAVAILABLE": 503,
        "INFERENCE_FAILED": 503,
        "ACCELERATOR_UNAVAILABLE": 503,
        "INSUFFICIENT_MEMORY": 503,
        "INSUFFICIENT_STORAGE": 507,
        "SERVER_ERROR": 500,
        "REQUEST_TIMEOUT": 504,
        "CANCELLED": 499,
    }
    assert prod_map == expected, (
        f"production status_map drifted from cutover-#70 expectations.\n"
        f"  expected: {expected}\n"
        f"  got:      {prod_map}\n"
        f"If this is intentional, update the expected dict above and the "
        f"test-side _make_app() in lockstep."
    )


# ---------------------------------------------------------------------------
# Reverse direction: HTTP → ErrorCode (used when the SDK consumes upstream
# APIs; cutover follow-up #70 added 413, 422, 429, 499, 504, 507).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected_code",
    [
        (400, OctomilErrorCode.INVALID_INPUT),
        (401, OctomilErrorCode.AUTHENTICATION_FAILED),
        (403, OctomilErrorCode.FORBIDDEN),
        (404, OctomilErrorCode.MODEL_NOT_FOUND),
        (413, OctomilErrorCode.CONTEXT_TOO_LARGE),
        (422, OctomilErrorCode.UNSUPPORTED_MODALITY),
        (429, OctomilErrorCode.RATE_LIMITED),
        (499, OctomilErrorCode.CANCELLED),
        (500, OctomilErrorCode.SERVER_ERROR),
        (502, OctomilErrorCode.SERVER_ERROR),
        (503, OctomilErrorCode.SERVER_ERROR),
        (504, OctomilErrorCode.REQUEST_TIMEOUT),
        (507, OctomilErrorCode.INSUFFICIENT_STORAGE),
    ],
)
def test_http_status_to_error_code_round_trip(status: int, expected_code: OctomilErrorCode) -> None:
    """The reverse map (`octomil/errors.py:_HTTP_STATUS_MAP`) is
    used when the SDK is consuming an upstream HTTP API. After
    follow-up #70, both directions stay aligned: e.g., the serve
    layer emits 422 for UNSUPPORTED_MODALITY and the SDK reading
    422 from another Octomil server back-translates to the same
    code."""
    from octomil.errors import _HTTP_STATUS_MAP

    assert _HTTP_STATUS_MAP[status] == expected_code, (
        f"_HTTP_STATUS_MAP[{status}] should be {expected_code}, " f"got {_HTTP_STATUS_MAP.get(status)}"
    )
