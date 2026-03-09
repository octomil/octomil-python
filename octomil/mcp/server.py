"""FastMCP server with 7 tools for local inference via Octomil."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# File size limits
_MAX_SINGLE_FILE_CHARS = 50_000
_MAX_MULTI_FILE_CHARS = 30_000
_MAX_TOTAL_CHARS = 100_000

# Sensitive path markers — never read these
_SENSITIVE_PATH_MARKERS = frozenset(
    {
        ".ssh",
        ".gnupg",
        ".gpg",
        ".aws",
        ".azure",
        ".gcloud",
        ".config/gcloud",
        ".kube",
        ".docker/config.json",
        ".npmrc",
        ".pypirc",
        ".netrc",
        ".env",
        ".secrets",
        "id_rsa",
        "id_ed25519",
        "credentials.json",
    }
)

# Language detection from file extension
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sh": "bash",
    ".sql": "sql",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
}


def _allowed_roots() -> list[Path]:
    """Return allowed root directories from env, or [cwd, home]."""
    env_val = os.environ.get("OCTOMIL_MCP_ALLOWED_ROOTS", "")
    if env_val.strip():
        return [Path(p).resolve() for p in env_val.split(":") if p.strip()]
    return [Path.cwd().resolve(), Path.home().resolve()]


def _is_sensitive_path(path: Path) -> bool:
    """Check if a path touches sensitive files."""
    parts_str = str(path)
    for marker in _SENSITIVE_PATH_MARKERS:
        if marker in parts_str:
            return True
    return False


def _is_within_any_root(path: Path, roots: list[Path]) -> bool:
    """Check if path is within any allowed root."""
    resolved = path.resolve()
    return any(resolved == root or str(resolved).startswith(str(root) + os.sep) for root in roots)


def _resolve_and_validate_path(file_path: str) -> Path:
    """Resolve a file path, validate it's safe to read."""
    expanded = os.path.expanduser(file_path)
    path = Path(expanded).resolve()

    if _is_sensitive_path(path):
        raise ValueError(f"Access denied: '{file_path}' matches a sensitive path pattern")

    roots = _allowed_roots()
    if not _is_within_any_root(path, roots):
        raise ValueError(f"Access denied: '{file_path}' is outside allowed directories")

    return path


def _read_file(file_path: str, max_chars: int = _MAX_SINGLE_FILE_CHARS) -> str:
    """Read a file with path validation and size truncation."""
    path = _resolve_and_validate_path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text(errors="replace")
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n... [truncated at {max_chars:,} chars]"
    return content


def _detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    return _EXT_TO_LANG.get(ext, "")


def create_mcp_server(model: str | None = None) -> Any:
    """Create and configure the FastMCP server with all tools."""
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]

    from .backend import OctomilMCPBackend
    from .platform_tools import register_platform_tools
    from .prompts import build_messages

    mcp = FastMCP("octomil", description="Octomil on-device ML inference, model resolution, and deployment")
    backend = OctomilMCPBackend(model=model)

    @mcp.tool()
    def generate_code(description: str, language: str = "", context: str = "") -> str:
        """Generate code from a natural language description.

        Args:
            description: What the code should do
            language: Target programming language (e.g. python, typescript)
            context: Additional context (existing code, constraints, etc.)
        """
        parts = [f"Generate {language + ' ' if language else ''}code: {description}"]
        if context:
            parts.append(f"\nContext:\n{context}")
        messages = build_messages("generate_code", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool()
    def review_code(code: str, language: str = "", focus: str = "") -> str:
        """Review code for bugs, security issues, and improvements.

        Args:
            code: The code to review
            language: Programming language
            focus: Specific focus area (e.g. security, performance, style)
        """
        parts = [f"Review this {language + ' ' if language else ''}code:"]
        if focus:
            parts.append(f"Focus on: {focus}")
        parts.append(f"\n```{language}\n{code}\n```")
        messages = build_messages("review_code", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool()
    def explain_code(code: str, language: str = "", detail_level: str = "medium") -> str:
        """Explain what code does in plain English.

        Args:
            code: The code to explain
            language: Programming language
            detail_level: How detailed (brief, medium, thorough)
        """
        parts = [f"Explain this {language + ' ' if language else ''}code ({detail_level} detail):"]
        parts.append(f"\n```{language}\n{code}\n```")
        messages = build_messages("explain_code", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool()
    def write_tests(code: str, language: str = "", framework: str = "", focus: str = "") -> str:
        """Generate unit tests for the given code.

        Args:
            code: The code to test
            language: Programming language
            framework: Test framework (e.g. pytest, jest, go test)
            focus: Specific areas to test (e.g. edge cases, error handling)
        """
        parts = [
            f"Write {framework + ' ' if framework else ''}tests for this {language + ' ' if language else ''}code:"
        ]
        if focus:
            parts.append(f"Focus on: {focus}")
        parts.append(f"\n```{language}\n{code}\n```")
        messages = build_messages("write_tests", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool()
    def general_task(prompt: str, context: str = "") -> str:
        """Run a free-form prompt through the local model.

        Args:
            prompt: The prompt or question
            context: Additional context
        """
        content = prompt
        if context:
            content = f"{prompt}\n\nContext:\n{context}"
        messages = build_messages("general_task", content)
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool()
    def review_file(file_path: str, focus: str = "") -> str:
        """Read a file from disk and review it locally using the on-device model.

        This saves ~98% of API tokens by running the review on the local model
        instead of sending the full file contents through the Anthropic API.

        Args:
            file_path: Path to the file to review
            focus: Specific focus area (e.g. security, performance, bugs)
        """
        try:
            content = _read_file(file_path)
        except (FileNotFoundError, ValueError) as exc:
            return f"Error: {exc}"

        language = _detect_language(file_path)
        parts = [f"Review this file ({os.path.basename(file_path)}):"]
        if focus:
            parts.append(f"Focus on: {focus}")
        parts.append(f"\n```{language}\n{content}\n```")
        messages = build_messages("review_file", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool()
    def analyze_files(file_paths: list[str], question: str) -> str:
        """Read multiple files and answer a question about them.

        Args:
            file_paths: List of file paths to read
            question: Question to answer about the files
        """
        file_contents: list[str] = []
        total_chars = 0

        for fp in file_paths:
            try:
                content = _read_file(fp, max_chars=_MAX_MULTI_FILE_CHARS)
                total_chars += len(content)
                if total_chars > _MAX_TOTAL_CHARS:
                    file_contents.append(f"--- {fp} ---\n[Skipped: total size limit reached]")
                    continue
                language = _detect_language(fp)
                file_contents.append(f"--- {fp} ---\n```{language}\n{content}\n```")
            except (FileNotFoundError, ValueError) as exc:
                file_contents.append(f"--- {fp} ---\nError: {exc}")

        parts = [question, "\nFiles:\n"]
        parts.extend(file_contents)
        messages = build_messages("analyze_files", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    # Register platform-level tools (model resolution, inference, deployment)
    register_platform_tools(mcp, backend)

    return mcp
