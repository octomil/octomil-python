"""HTTP server exposing Octomil tools via REST + A2A agent card + OpenAPI.

This package is a refactored version of the original ``http_server.py`` monolith.
Sub-modules: app, config, models, routes, tools.
"""

from .app import create_http_app
from .config import HTTPServerConfig

__all__ = [
    "HTTPServerConfig",
    "create_http_app",
]
