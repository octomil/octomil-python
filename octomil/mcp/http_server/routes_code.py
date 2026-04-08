"""Code tool endpoints with cloud fallback and 503 next actions."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from ..auth import require_auth
from ..backend import OctomilMCPBackend
from ..prompts import build_messages
from .config import HTTPServerConfig
from .models import (
    ExplainCodeRequest,
    GeneralTaskRequest,
    GenerateCodeRequest,
    ReviewCodeRequest,
    RunInferenceRequest,
    WriteTestsRequest,
)

logger = logging.getLogger(__name__)

_VALID_PREFERENCES = {"auto", "local-only", "cloud-only"}


def _x402_pricing(request: Request) -> tuple[int, int]:
    """Extract x402 pricing from request state (set by middleware)."""
    return (
        getattr(request.state, "x402_paid_amount", 0),
        getattr(request.state, "x402_cloud_price", 0),
    )


def _inference_preference(request: Request) -> str:
    """Read X-Inference-Preference header (auto | local-only | cloud-only)."""
    pref = request.headers.get("x-inference-preference", "auto").lower().strip()
    return pref if pref in _VALID_PREFERENCES else "auto"


def register_code_routes(
    app: FastAPI,
    backend: OctomilMCPBackend,
    config: HTTPServerConfig,
    base_url: str,
) -> None:
    """Register inference and code tool routes on *app*."""

    def _code_tool_response(
        tool_name: str,
        messages: list[dict[str, str]],
        paid_amount: int = 0,
        cloud_price: int = 0,
        preference: str = "auto",
    ) -> JSONResponse:
        """Run a code tool through local model with cloud fallback.

        1. Try local model (unless preference is ``cloud-only``)
        2. On failure, try cloud fallback if the agent paid enough (unless ``local-only``)
        3. If both fail, return 503 with machine-readable next actions

        The ``preference`` parameter is read from the ``X-Inference-Preference``
        header and can be ``auto`` (default), ``local-only``, or ``cloud-only``.
        """
        # 1. Try local model
        if preference != "cloud-only":
            try:
                text, metrics = backend.generate(messages)
                return JSONResponse(content={"text": text, "metrics": metrics})
            except Exception as local_exc:
                logger.warning("%s: local model failed: %s", tool_name, local_exc)

        # 2. Try cloud fallback
        if preference == "local-only":
            return JSONResponse(
                status_code=503,
                content={
                    "error": "local_only",
                    "message": f"Local model unavailable for {tool_name} and preference is local-only.",
                    "retryable": True,
                    "retryAfterSeconds": 30,
                },
            )

        allow_cloud = True
        if config.enable_x402 and cloud_price > 0 and paid_amount < cloud_price:
            allow_cloud = False

        api_key = os.environ.get("OCTOMIL_API_KEY")
        if allow_cloud and api_key:
            try:
                from octomil.auth import OrgApiKeyAuth
                from octomil.client import OctomilClient

                client = OctomilClient(
                    auth=OrgApiKeyAuth(api_key=api_key, org_id=os.getenv("OCTOMIL_ORG_ID", "default"))
                )
                completion = client.chat.create(backend.model_name, messages)
                text = completion.message.get("content", "")
                return JSONResponse(
                    content={
                        "text": text,
                        "metrics": {"engine": "cloud", "model": backend.model_name, "fallback": True},
                    }
                )
            except Exception as cloud_exc:
                logger.warning("%s: cloud fallback also failed: %s", tool_name, cloud_exc)

        # 3. Return 503 with next actions
        error_content: dict[str, Any] = {
            "error": "model_not_ready",
            "message": f"No local model loaded for {tool_name}.",
            "retryable": True,
            "retryAfterSeconds": 30,
            "actions": {
                "warmup": f"{base_url}/api/v1/warmup",
                "ready": f"{base_url}/api/v1/ready",
            },
        }
        # Tell the agent the cloud fallback price so they can resubmit
        if config.enable_x402 and cloud_price > 0 and paid_amount < cloud_price:
            error_content["cloud_fallback"] = {
                "available": True,
                "price": str(cloud_price),
                "currency": "USDC",
                "message": f"Resubmit with payment >= {cloud_price} base units for cloud inference.",
            }
        return JSONResponse(status_code=503, content=error_content)

    @app.post("/api/v1/run_inference", tags=["inference"], dependencies=[Depends(require_auth)])
    async def api_run_inference(req: RunInferenceRequest, request: Request) -> JSONResponse:
        """Run inference through the local on-device model."""
        messages = [{"role": "user", "content": req.prompt}]
        paid, cloud = _x402_pricing(request)
        pref = _inference_preference(request)
        return _code_tool_response("run_inference", messages, paid_amount=paid, cloud_price=cloud, preference=pref)

    @app.post("/api/v1/generate_code", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_generate_code(req: GenerateCodeRequest, request: Request) -> JSONResponse:
        """Generate code from a natural language description."""
        parts = [f"Generate {req.language + ' ' if req.language else ''}code: {req.description}"]
        if req.context:
            parts.append(f"\nContext:\n{req.context}")
        messages = build_messages("generate_code", "\n".join(parts))
        paid, cloud = _x402_pricing(request)
        pref = _inference_preference(request)
        return _code_tool_response("generate_code", messages, paid_amount=paid, cloud_price=cloud, preference=pref)

    @app.post("/api/v1/review_code", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_review_code(req: ReviewCodeRequest, request: Request) -> JSONResponse:
        """Review code for bugs, security issues, and improvements."""
        parts = [f"Review this {req.language + ' ' if req.language else ''}code:"]
        if req.focus:
            parts.append(f"Focus on: {req.focus}")
        parts.append(f"\n```{req.language}\n{req.code}\n```")
        messages = build_messages("review_code", "\n".join(parts))
        paid, cloud = _x402_pricing(request)
        pref = _inference_preference(request)
        return _code_tool_response("review_code", messages, paid_amount=paid, cloud_price=cloud, preference=pref)

    @app.post("/api/v1/explain_code", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_explain_code(req: ExplainCodeRequest, request: Request) -> JSONResponse:
        """Explain code in plain English."""
        parts = [f"Explain this {req.language + ' ' if req.language else ''}code ({req.detail_level} detail):"]
        parts.append(f"\n```{req.language}\n{req.code}\n```")
        messages = build_messages("explain_code", "\n".join(parts))
        paid, cloud = _x402_pricing(request)
        pref = _inference_preference(request)
        return _code_tool_response("explain_code", messages, paid_amount=paid, cloud_price=cloud, preference=pref)

    @app.post("/api/v1/write_tests", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_write_tests(req: WriteTestsRequest, request: Request) -> JSONResponse:
        """Generate unit tests for code."""
        parts = [
            f"Write {req.framework + ' ' if req.framework else ''}tests for this {req.language + ' ' if req.language else ''}code:"
        ]
        if req.focus:
            parts.append(f"Focus on: {req.focus}")
        parts.append(f"\n```{req.language}\n{req.code}\n```")
        messages = build_messages("write_tests", "\n".join(parts))
        paid, cloud = _x402_pricing(request)
        pref = _inference_preference(request)
        return _code_tool_response("write_tests", messages, paid_amount=paid, cloud_price=cloud, preference=pref)

    @app.post("/api/v1/general_task", tags=["code"], dependencies=[Depends(require_auth)])
    async def api_general_task(req: GeneralTaskRequest, request: Request) -> JSONResponse:
        """Run a free-form prompt through the local model."""
        content = req.prompt
        if req.context:
            content = f"{req.prompt}\n\nContext:\n{req.context}"
        messages = build_messages("general_task", content)
        paid, cloud = _x402_pricing(request)
        pref = _inference_preference(request)
        return _code_tool_response("general_task", messages, paid_amount=paid, cloud_price=cloud, preference=pref)
