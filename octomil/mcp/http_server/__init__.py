"""HTTP server exposing Octomil tools via REST + A2A agent card + OpenAPI.

This package is a refactored version of the original ``http_server.py`` monolith.
All public symbols are re-exported here for backward compatibility.
"""

from .app import create_http_app  # noqa: F401
from .config import HTTPServerConfig  # noqa: F401

__all__ = ["HTTPServerConfig", "create_http_app"]
