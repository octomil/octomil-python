"""Entry point for ``python -m octomil.mcp`` — runs the MCP stdio server."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError:
        print(
            "Error: the 'mcp' package is required.\n" "Install with: pip install 'mcp[cli]>=1.2.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    from .server import create_mcp_server

    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
