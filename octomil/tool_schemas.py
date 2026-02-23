"""Pre-built JSON schemas for coding agent tool-use enforcement.

When ``octomil serve`` is started with ``--tool-use``, these schemas are
advertised at ``/v1/tool-schemas`` so coding agents (Aider, Goose, OpenCode)
can discover available tools and enforce structured JSON output for each.
"""

from __future__ import annotations

CODING_TOOL_SCHEMAS: dict[str, dict] = {
    "read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
        },
        "required": ["path"],
    },
    "write_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "File content"},
        },
        "required": ["path", "content"],
    },
    "edit_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to edit"},
            "old_text": {"type": "string", "description": "Text to find and replace"},
            "new_text": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_text", "new_text"],
    },
    "run_command": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "working_dir": {
                "type": "string",
                "description": "Working directory for the command",
            },
        },
        "required": ["command"],
    },
    "search_files": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (regex)"},
            "path": {
                "type": "string",
                "description": "Directory to search in",
            },
        },
        "required": ["pattern"],
    },
}


def get_tool_use_tools() -> list[dict]:
    """Return OpenAI-compatible tool definitions for coding agents.

    Each tool definition follows the OpenAI function-calling schema so
    that agents can discover and call tools via the standard
    ``tools`` parameter in chat completion requests.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Coding tool: {name.replace('_', ' ')}",
                "parameters": schema,
            },
        }
        for name, schema in CODING_TOOL_SCHEMAS.items()
    ]
