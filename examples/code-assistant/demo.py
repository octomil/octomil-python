#!/usr/bin/env python3
"""Thin wrapper — run the code assistant demo directly.

Usage::

    python examples/code-assistant/demo.py
    python examples/code-assistant/demo.py --model phi-3-mini
    python examples/code-assistant/demo.py --url http://localhost:8080
"""

from edgeml.demos.code_assistant import main

if __name__ == "__main__":
    main()
