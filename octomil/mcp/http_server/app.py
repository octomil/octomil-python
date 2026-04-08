"""FastAPI application factory for the Octomil HTTP agent server.

Assembles middleware, routes, and MCP transport into a single app.
Start via: ``octomil mcp serve --port 8402``
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..a2a import AgentCardConfig
from ..backend import OctomilMCPBackend
from .config import HTTPServerConfig, _get_tool_definitions
from .routes_code import register_code_routes
from .routes_discovery import register_discovery_routes
from .routes_platform import register_platform_routes

logger = logging.getLogger(__name__)


def create_http_app(config: Optional[HTTPServerConfig] = None) -> FastAPI:
    """Create the FastAPI application for the Octomil agent HTTP server.

    Parameters
    ----------
    config:
        Server configuration. Uses defaults if not provided.

    Returns
    -------
    FastAPI
        Configured app with all routes, middleware, and agent card.
    """
    import contextlib
    from collections.abc import AsyncIterator

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    from ..server import create_mcp_server

    if config is None:
        config = HTTPServerConfig()

    # Create MCP server + session manager before FastAPI so lifespan can manage it
    mcp_server = create_mcp_server(model=config.model)
    mcp_session_manager = StreamableHTTPSessionManager(
        app=mcp_server._mcp_server,
        json_response=False,
        stateless=True,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with mcp_session_manager.run():
            yield

    app = FastAPI(
        title="Octomil Agent",
        description="On-device ML inference, model resolution, and deployment — agent-callable via REST, A2A, and MCP",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — permissive for agent-to-agent communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # x402 middleware (opt-in)
    settlement_store = None
    if config.enable_x402:
        from ..x402 import X402Config, X402Middleware
        from ..x402_settlement import SettlementStore

        x402_config = X402Config(
            price_per_call=config.x402_price or os.environ.get("OCTOMIL_X402_PRICE", "1000"),
            currency=config.x402_currency or os.environ.get("OCTOMIL_X402_CURRENCY", "USDC"),
            network=config.x402_network or os.environ.get("OCTOMIL_X402_NETWORK", "base"),
            payment_address=config.x402_address or os.environ.get("OCTOMIL_X402_ADDRESS", ""),
            settlement_threshold=config.x402_threshold,
            facilitator_url=config.settler_url or os.environ.get("OCTOMIL_SETTLER_URL", "https://api.settle402.dev"),
            settler_token=config.settler_token or os.environ.get("OCTOMIL_SETTLER_TOKEN", ""),
        )
        if x402_config.enable_settlement:
            settlement_store = SettlementStore(threshold=x402_config.settlement_threshold)
        app.add_middleware(X402Middleware, config=x402_config, settlement_store=settlement_store)
        logger.info("x402 payment gating enabled (address=%s)", x402_config.payment_address)

    # Shared backend instance
    backend = OctomilMCPBackend(model=config.model)

    # Agent card config — card is rebuilt per-request for live readiness
    base_url = config.base_url or f"http://{config.host}:{config.port}"
    card_config = AgentCardConfig(url=base_url)
    tool_defs = _get_tool_definitions()

    # Store on app state for testing access
    app.state.backend = backend
    app.state.settlement_store = settlement_store

    # ------------------------------------------------------------------
    # Register route groups
    # ------------------------------------------------------------------

    register_discovery_routes(app, backend, card_config, tool_defs, base_url, settlement_store)
    register_platform_routes(app, backend)
    register_code_routes(app, backend, config, base_url)

    # ------------------------------------------------------------------
    # MCP Streamable HTTP transport at /mcp
    # ------------------------------------------------------------------

    from starlette.routing import Route

    class _MCPTransport:
        """Thin ASGI wrapper so Starlette treats this as an app, not a request handler."""

        def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
            self._sm = session_manager

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            await self._sm.handle_request(scope, receive, send)

    app.router.routes.append(
        Route("/mcp", endpoint=_MCPTransport(mcp_session_manager), methods=["GET", "POST", "DELETE"])
    )

    return app
