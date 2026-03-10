"""FastMCP server with 7 tools for local inference via Octomil."""

import logging
import os
from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import Field

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


def create_mcp_server(model: Optional[str] = None, **fastmcp_kwargs: Any) -> Any:
    """Create and configure the FastMCP server with all tools."""
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]
    from mcp.types import ToolAnnotations

    from .backend import OctomilMCPBackend
    from .platform_tools import register_platform_tools
    from .prompts import build_messages

    mcp = FastMCP(
        "octomil",
        instructions="Octomil on-device ML inference, model resolution, and deployment",
        **fastmcp_kwargs,
    )
    backend = OctomilMCPBackend(model=model)

    @mcp.tool(
        annotations=ToolAnnotations(title="Generate Code", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def generate_code(
        description: Annotated[str, Field(description="What the code should do")],
        language: Annotated[str, Field(description="Target programming language (e.g. python, typescript)")] = "",
        context: Annotated[str, Field(description="Additional context (existing code, constraints, etc.)")] = "",
    ) -> str:
        """Generate code from a natural language description."""
        parts = [f"Generate {language + ' ' if language else ''}code: {description}"]
        if context:
            parts.append(f"\nContext:\n{context}")
        messages = build_messages("generate_code", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool(
        annotations=ToolAnnotations(title="Review Code", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def review_code(
        code: Annotated[str, Field(description="The code to review")],
        language: Annotated[str, Field(description="Programming language")] = "",
        focus: Annotated[str, Field(description="Specific focus area (e.g. security, performance, style)")] = "",
    ) -> str:
        """Review code for bugs, security issues, and improvements."""
        parts = [f"Review this {language + ' ' if language else ''}code:"]
        if focus:
            parts.append(f"Focus on: {focus}")
        parts.append(f"\n```{language}\n{code}\n```")
        messages = build_messages("review_code", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool(
        annotations=ToolAnnotations(title="Explain Code", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def explain_code(
        code: Annotated[str, Field(description="The code to explain")],
        language: Annotated[str, Field(description="Programming language")] = "",
        detail_level: Annotated[str, Field(description="How detailed: brief, medium, or thorough")] = "medium",
    ) -> str:
        """Explain what code does in plain English."""
        parts = [f"Explain this {language + ' ' if language else ''}code ({detail_level} detail):"]
        parts.append(f"\n```{language}\n{code}\n```")
        messages = build_messages("explain_code", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool(
        annotations=ToolAnnotations(title="Write Tests", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def write_tests(
        code: Annotated[str, Field(description="The code to test")],
        language: Annotated[str, Field(description="Programming language")] = "",
        framework: Annotated[str, Field(description="Test framework (e.g. pytest, jest, go test)")] = "",
        focus: Annotated[str, Field(description="Specific areas to test (e.g. edge cases, error handling)")] = "",
    ) -> str:
        """Generate unit tests for the given code."""
        parts = [
            f"Write {framework + ' ' if framework else ''}tests for this {language + ' ' if language else ''}code:"
        ]
        if focus:
            parts.append(f"Focus on: {focus}")
        parts.append(f"\n```{language}\n{code}\n```")
        messages = build_messages("write_tests", "\n".join(parts))
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool(
        annotations=ToolAnnotations(title="Run Prompt", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def run_prompt(
        prompt: Annotated[str, Field(description="The prompt or question to send to the local model")],
        context: Annotated[str, Field(description="Additional context to include with the prompt")] = "",
    ) -> str:
        """Run a free-form prompt through the local on-device model."""
        content = prompt
        if context:
            content = f"{prompt}\n\nContext:\n{context}"
        messages = build_messages("general_task", content)
        text, metrics = backend.generate(messages)
        return f"{text}\n\n{backend.format_metrics(metrics)}"

    @mcp.tool(
        annotations=ToolAnnotations(title="Review File", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def review_file(
        file_path: Annotated[str, Field(description="Path to the file to review")],
        focus: Annotated[str, Field(description="Specific focus area (e.g. security, performance, bugs)")] = "",
    ) -> str:
        """Read a file from disk and review it locally using the on-device model."""
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

    @mcp.tool(
        annotations=ToolAnnotations(title="Analyze Files", readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    )
    def analyze_files(
        file_paths: Annotated[list[str], Field(description="List of file paths to read")],
        question: Annotated[str, Field(description="Question to answer about the files")],
    ) -> str:
        """Read multiple files and answer a question about them."""
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

    # MCP prompts
    @mcp.prompt()
    def deploy_guide(model_name: Annotated[str, Field(description="Model to deploy")] = "phi-mini") -> str:
        """Step-by-step guide for deploying a model to edge devices."""
        return f"""Help me deploy the model '{model_name}' to edge devices using Octomil.

Steps:
1. First, resolve the model to check availability: use resolve_model with name="{model_name}"
2. Check hardware capabilities: use detect_hardware_profile
3. Get optimization recommendations: use recommend_model
4. Plan the deployment: use plan_deployment with name="{model_name}"
5. Execute the deployment: use deploy_model with name="{model_name}"

Start with step 1."""

    @mcp.prompt()
    def code_review_guide(language: Annotated[str, Field(description="Programming language")] = "python") -> str:
        """Guide for reviewing code with local inference."""
        return f"""Review my {language} code using Octomil's local inference.

I'll provide code and you should:
1. Use review_code to check for bugs and security issues
2. Use write_tests to generate test cases
3. Use explain_code if any sections are unclear

This runs entirely on-device — no code leaves the machine."""

    # MCP resources
    @mcp.resource("octomil://models")
    def available_models() -> str:
        """List of available models in the Octomil catalog."""
        try:
            import json

            from octomil.models.catalog import CATALOG

            models = [{"name": n, "params": e.params, "publisher": e.publisher} for n, e in sorted(CATALOG.items())]
            return json.dumps({"models": models})
        except Exception:
            return '{"models": []}'

    @mcp.resource("octomil://status")
    def server_status() -> str:
        """Current server status including model and engine info."""
        import json

        return json.dumps(
            {
                "model": backend.model_name,
                "loaded": backend.is_loaded,
                "engine": backend._engine_name,
            }
        )

    return mcp
