"""System prompts for each MCP tool."""

from __future__ import annotations

# Platform tool prompts are not used for inference (they wrap SDK calls),
# but are defined here for consistency and potential future use.
PLATFORM_TOOL_DESCRIPTIONS: dict[str, str] = {
    "resolve_model": "Resolve a model name to engine-specific artifacts (repo, filename, engine, quant).",
    "list_models": "List all models in the Octomil catalog with metadata.",
    "detect_engines": "Detect which inference engines are available on this machine.",
    "run_inference": "Run raw inference through the local on-device model.",
    "get_metrics": "Get current model/engine status and readiness.",
    "deploy_model": "Deploy a model to edge devices via the Octomil platform.",
}

SYSTEM_PROMPTS: dict[str, str] = {
    "generate_code": (
        "You are an expert programmer. Generate clean, idiomatic, "
        "production-ready code. Include brief inline comments only where "
        "the logic is non-obvious. Do not add tests unless asked."
    ),
    "review_code": (
        "You are a senior code reviewer. Identify bugs, security issues, "
        "performance problems, and style violations. Be specific: cite line "
        "numbers and suggest concrete fixes. Prioritise severity."
    ),
    "explain_code": (
        "You are a patient technical educator. Explain what the code does, "
        "how it works, and why it was written this way. Adjust depth to the "
        "requested detail level."
    ),
    "write_tests": (
        "You are a test engineer. Write thorough, readable unit tests that "
        "cover happy paths, edge cases, and error handling. Use the specified "
        "framework and follow its conventions."
    ),
    "general_task": (
        "You are a helpful AI assistant specialising in software engineering. " "Answer clearly and concisely."
    ),
    "review_file": (
        "You are a senior code reviewer. The user has provided the full "
        "contents of a source file. Identify bugs, security issues, "
        "performance problems, and style violations. Be specific."
    ),
    "analyze_files": (
        "You are a senior software engineer. The user has provided the "
        "contents of one or more source files and a question. Analyse the "
        "code and answer the question thoroughly."
    ),
}


def build_messages(tool_name: str, user_content: str) -> list[dict[str, str]]:
    """Build a [system, user] message pair for the given tool."""
    system = SYSTEM_PROMPTS.get(tool_name, SYSTEM_PROMPTS["general_task"])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
