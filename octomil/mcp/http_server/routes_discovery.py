"""Discovery, readiness, and health endpoints (no auth required)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..a2a import AgentCardConfig, build_agent_card
from ..backend import OctomilMCPBackend


def register_discovery_routes(
    app: FastAPI,
    backend: OctomilMCPBackend,
    card_config: AgentCardConfig,
    tool_defs: list[dict[str, Any]],
    base_url: str,
    settlement_store: Any,
) -> None:
    """Register discovery, health, and readiness routes on *app*."""

    @app.get("/.well-known/agent-card.json", tags=["discovery"])
    async def get_agent_card() -> JSONResponse:
        """A2A agent card for discovery by other agents.

        Rebuilt per-request so skill readiness reflects current model state.
        """
        card = build_agent_card(tool_defs, card_config, model_ready=backend.is_loaded)
        return JSONResponse(content=card)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, Any]:
        """Health check."""
        return {
            "status": "ok",
            "model": backend.model_name,
            "loaded": backend.is_loaded,
        }

    @app.get("/api/v1/settlement_status", tags=["x402"])
    async def api_settlement_status() -> JSONResponse:
        """Return x402 batch settlement statistics."""
        if settlement_store is None:
            return JSONResponse(
                status_code=404,
                content={"error": "settlement_disabled", "message": "x402 settlement is not enabled."},
            )
        return JSONResponse(content=settlement_store.stats())

    @app.get("/api/v1/ready", tags=["readiness"])
    async def api_ready() -> JSONResponse:
        """Lightweight readiness probe. Returns model load status."""
        if backend.is_loaded:
            return JSONResponse(
                content={
                    "ready": True,
                    "model": backend.model_name,
                    "engine": backend._engine_name,
                },
            )
        content: dict[str, Any] = {
            "ready": False,
            "model": backend.model_name,
            "engine": backend._engine_name,
        }
        if backend._loading:
            content["status"] = "loading"
            content["message"] = "Model is being downloaded and loaded."
        else:
            content["message"] = "Model not loaded. Call POST /api/v1/warmup to start."
            content["actions"] = {
                "warmup": f"{base_url}/api/v1/warmup",
            }
        return JSONResponse(status_code=503, content=content)

    @app.post("/api/v1/warmup", tags=["readiness"])
    async def api_warmup() -> JSONResponse:
        """Trigger model loading (with auto-download if needed).

        Call this before invoking code tools. If the model is already loaded,
        returns immediately. Otherwise kicks off background loading and returns
        202 with a loading status -- poll GET /api/v1/ready to check progress.
        No authentication required.
        """
        import asyncio

        if backend.is_loaded:
            return JSONResponse(content=backend.warmup())

        if backend._loading:
            return JSONResponse(
                status_code=202,
                content={
                    "status": "loading",
                    "model": backend.model_name,
                    "message": "Model is loading. Poll GET /api/v1/ready to check.",
                },
            )

        # Run model loading in background thread so we don't block the server
        async def _load_in_background() -> None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, backend.warmup)

        asyncio.ensure_future(_load_in_background())

        return JSONResponse(
            status_code=202,
            content={
                "status": "loading",
                "model": backend.model_name,
                "message": "Model download and loading started. Poll GET /api/v1/ready to check.",
            },
        )
