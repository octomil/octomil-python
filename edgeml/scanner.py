"""Scan a codebase for ML inference points.

Walks a directory tree, matches known ML/inference patterns in source files,
and returns structured ``InferencePoint`` results that tell the developer
where they can instrument with ``EdgeML.wrap()``.

Usage (programmatic)::

    from edgeml.scanner import scan_directory
    points = scan_directory("./MyApp", platform="ios")

Usage (CLI)::

    edgeml scan ./MyApp
    edgeml scan ./MyApp --format json --platform ios
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class InferencePoint:
    """A location in the scanned codebase where ML inference happens."""

    file: str
    line: int
    pattern: str
    type: str  # "CoreML", "TFLite", "PyTorch", "OpenAI", etc.
    platform: str  # "ios", "android", "python"
    suggestion: str
    context: str  # the actual source line


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------


@dataclass
class _Pattern:
    """A compiled regex + metadata used to detect an inference point."""

    regex: re.Pattern[str]
    display: str  # human-readable pattern name
    type: str
    platform: str
    suggestion: str


def _build_patterns() -> list[_Pattern]:
    """Build the full list of patterns, grouped by platform."""
    patterns: list[_Pattern] = []

    def _add(
        regex: str,
        display: str,
        type_: str,
        platform: str,
        suggestion: str,
    ) -> None:
        patterns.append(
            _Pattern(
                regex=re.compile(regex),
                display=display,
                type=type_,
                platform=platform,
                suggestion=suggestion,
            )
        )

    # ----- iOS / Swift / CoreML -----
    _add(
        r"\bMLModel\s*\(\s*contentsOf\s*:",
        "MLModel(contentsOf:)",
        "CoreML model loading",
        "ios",
        "Wrap with EdgeML.wrap(model) for telemetry",
    )
    _add(
        r"\bMLModel\.load\b",
        "MLModel.load()",
        "CoreML model loading",
        "ios",
        "Wrap with EdgeML.wrap(model) for telemetry",
    )
    _add(
        r"\bVNCoreMLModel\b",
        "VNCoreMLModel",
        "CoreML Vision model",
        "ios",
        "Wrap with EdgeML.wrap(model) for telemetry",
    )
    _add(
        r"\bVNCoreMLRequest\b",
        "VNCoreMLRequest",
        "CoreML Vision request",
        "ios",
        "Wrap with EdgeML.wrap(request) for telemetry",
    )
    _add(
        r"\bNLModel\s*\(\s*mlModel\s*:",
        "NLModel(mlModel:)",
        "CoreML NLP model",
        "ios",
        "Wrap with EdgeML.wrap(model) for telemetry",
    )
    _add(
        r"\bMLMultiArray\b",
        "MLMultiArray",
        "CoreML tensor",
        "ios",
        "Wrap model input/output with EdgeML for telemetry",
    )
    _add(
        r"\bMLFeatureProvider\b",
        "MLFeatureProvider",
        "CoreML feature provider",
        "ios",
        "Wrap model input/output with EdgeML for telemetry",
    )
    _add(
        r"^\s*import\s+CoreML\b",
        "import CoreML",
        "CoreML import",
        "ios",
        "Add EdgeML.wrap() around model usage in this file",
    )
    _add(
        r"^\s*import\s+Vision\b",
        "import Vision",
        "Vision framework import",
        "ios",
        "Add EdgeML.wrap() around Vision model usage in this file",
    )

    # ----- Android / Kotlin / TFLite -----
    _add(
        r"\bInterpreter\s*\(",
        "Interpreter(",
        "TFLite model loading",
        "android",
        "Wrap with EdgeML.wrap(interpreter) for telemetry",
    )
    _add(
        r"\bInterpreter\.Options\b",
        "Interpreter.Options",
        "TFLite interpreter options",
        "android",
        "Wrap with EdgeML.wrap(interpreter) for telemetry",
    )
    _add(
        r"\bGpuDelegate\b",
        "GpuDelegate",
        "TFLite GPU delegate",
        "android",
        "Wrap with EdgeML.wrap(interpreter) for telemetry",
    )
    _add(
        r"\bNnApiDelegate\b",
        "NnApiDelegate",
        "TFLite NNAPI delegate",
        "android",
        "Wrap with EdgeML.wrap(interpreter) for telemetry",
    )
    _add(
        r"\bloadModelFile\b",
        "loadModelFile",
        "TFLite model file loading",
        "android",
        "Wrap with EdgeML.wrap(interpreter) for telemetry",
    )
    _add(
        r"\bMappedByteBuffer\b",
        "MappedByteBuffer",
        "TFLite model buffer",
        "android",
        "Wrap with EdgeML.wrap(interpreter) for telemetry",
    )
    _add(
        r"import\s+org\.tensorflow\.lite\b",
        "import org.tensorflow.lite",
        "TFLite import",
        "android",
        "Add EdgeML.wrap() around interpreter usage in this file",
    )

    # ----- Python / PyTorch -----
    _add(
        r"\btorch\.load\s*\(",
        "torch.load(",
        "PyTorch model loading",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )
    _add(
        r"\bmodel\.eval\s*\(",
        "model.eval()",
        "PyTorch eval mode",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )
    _add(
        r"\bmodel\.forward\s*\(",
        "model.forward(",
        "PyTorch forward pass",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )

    # ----- Python / ONNX Runtime -----
    _add(
        r"\bonnxruntime\.InferenceSession\b",
        "onnxruntime.InferenceSession",
        "ONNX Runtime inference",
        "python",
        "Wrap with edgeml.wrap(session) for telemetry",
    )

    # ----- Python / TFLite (Python binding) -----
    _add(
        r"\btf\.lite\.Interpreter\b",
        "tf.lite.Interpreter",
        "TFLite Python inference",
        "python",
        "Wrap with edgeml.wrap(interpreter) for telemetry",
    )

    # ----- Python / OpenAI -----
    _add(
        r"\bopenai\.OpenAI\s*\(",
        "openai.OpenAI(",
        "OpenAI client",
        "python",
        "Wrap with edgeml.wrap(client) for telemetry",
    )
    _add(
        r"\bclient\.chat\.completions\b",
        "client.chat.completions",
        "OpenAI chat completions",
        "python",
        "Wrap with edgeml.wrap(client) for telemetry",
    )

    # ----- Python / MLX -----
    _add(
        r"\bmlx\.core\b",
        "mlx.core",
        "MLX framework",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )
    _add(
        r"\bmlx_lm\b",
        "mlx_lm",
        "MLX language model",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )

    # ----- Python / HuggingFace Transformers -----
    _add(
        r"\btransformers\.pipeline\s*\(",
        "transformers.pipeline(",
        "HuggingFace pipeline",
        "python",
        "Wrap with edgeml.wrap(pipeline) for telemetry",
    )
    _add(
        r"\bAutoModelFor\w+",
        "AutoModelFor*",
        "HuggingFace AutoModel",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )

    # ----- General / Cross-platform -----
    _add(
        r"\bpredict\s*\(",
        "predict(",
        "Inference call",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )
    _add(
        r"\binference\s*\(",
        "inference(",
        "Inference call",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )
    _add(
        r"\bclassify\s*\(",
        "classify(",
        "Classification call",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )
    _add(
        r"\bdetect\s*\(",
        "detect(",
        "Detection call",
        "python",
        "Wrap with edgeml.wrap(model) for telemetry",
    )

    return patterns


# Singleton pattern list, built once.
_PATTERNS: list[_Pattern] = _build_patterns()

# File extensions worth scanning, mapped to probable platform.
_EXT_PLATFORM: dict[str, str] = {
    ".swift": "ios",
    ".m": "ios",
    ".mm": "ios",
    ".kt": "android",
    ".java": "android",
    ".py": "python",
    ".pyx": "python",
}

# Model file extensions (reported as standalone inference points).
_MODEL_FILE_EXTS: dict[str, tuple[str, str]] = {
    ".mlmodel": ("CoreML model file", "ios"),
    ".mlpackage": ("CoreML model package", "ios"),
    ".tflite": ("TFLite model file", "android"),
    ".onnx": ("ONNX model file", "python"),
    ".pt": ("PyTorch model file", "python"),
    ".pth": ("PyTorch model file", "python"),
    ".safetensors": ("Safetensors model file", "python"),
}

# Directories to always skip.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".build",
        "build",
        "DerivedData",
        "Pods",
        ".venv",
        "venv",
        "env",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "egg-info",
    }
)


# ---------------------------------------------------------------------------
# Comment detection (best effort)
# ---------------------------------------------------------------------------

# Single-line comment prefixes by file extension.
_COMMENT_PREFIXES: dict[str, list[str]] = {
    ".swift": ["//"],
    ".m": ["//"],
    ".mm": ["//"],
    ".kt": ["//"],
    ".java": ["//"],
    ".py": ["#"],
    ".pyx": ["#"],
}


def _is_comment_line(line: str, ext: str) -> bool:
    """Return True if the line is a single-line comment."""
    stripped = line.lstrip()
    for prefix in _COMMENT_PREFIXES.get(ext, []):
        if stripped.startswith(prefix):
            return True
    return False


# Multi-line comment/string delimiters by extension.
_MULTILINE_DELIMITERS: dict[str, list[str]] = {
    ".py": ['"""', "'''"],
    ".pyx": ['"""', "'''"],
    ".swift": ["/*"],
    ".m": ["/*"],
    ".mm": ["/*"],
    ".kt": ["/*"],
    ".java": ["/*"],
}

_MULTILINE_CLOSE: dict[str, str] = {
    '"""': '"""',
    "'''": "'''",
    "/*": "*/",
}


def _scan_lines(
    lines: list[str],
    ext: str,
    file_patterns: list[_Pattern],
    rel_path: str,
) -> list[InferencePoint]:
    """Scan lines of a single file, tracking multi-line string/comment state."""
    results: list[InferencePoint] = []
    in_block: str | None = None  # The closing delimiter we're looking for
    delimiters = _MULTILINE_DELIMITERS.get(ext, [])

    for lineno, line in enumerate(lines, start=1):
        stripped = line.lstrip()

        # If we're inside a multi-line block, look for the close.
        if in_block is not None:
            if in_block in line:
                in_block = None
            continue

        # Check if this line opens a multi-line block.
        opened_block = False
        for delim in delimiters:
            if delim in stripped:
                count = stripped.count(delim)
                if count == 1:
                    # Opens a block that isn't closed on the same line.
                    close = _MULTILINE_CLOSE[delim]
                    # For triple-quote, the open and close are the same string,
                    # so a single occurrence means it opens and doesn't close.
                    in_block = close
                    opened_block = True
                    break
                # count >= 2 means it opens and closes on the same line (e.g.
                # '''docstring on one line''').  We skip the line but don't
                # enter a block.
                if count >= 2:
                    opened_block = True
                    break

        if opened_block:
            continue

        # Skip single-line comments.
        if _is_comment_line(line, ext):
            continue

        # Match patterns.
        for pat in file_patterns:
            if pat.regex.search(line):
                results.append(
                    InferencePoint(
                        file=rel_path,
                        line=lineno,
                        pattern=pat.display,
                        type=pat.type,
                        platform=pat.platform,
                        suggestion=pat.suggestion,
                        context=line.rstrip(),
                    )
                )

    return results


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------


def scan_directory(
    path: str,
    platform: Optional[str] = None,
) -> list[InferencePoint]:
    """Walk *path*, match patterns, and return a list of inference points.

    Parameters
    ----------
    path:
        Root directory to scan.
    platform:
        Optional filter — ``"ios"``, ``"android"``, or ``"python"``.
        When ``None``, all platforms are scanned.
    """
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    # Pre-filter patterns by platform when requested.
    active_patterns = _PATTERNS
    if platform:
        active_patterns = [p for p in _PATTERNS if p.platform == platform]

    results: list[InferencePoint] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place so os.walk does not descend.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()

            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root)

            # Check model file extensions.
            if ext in _MODEL_FILE_EXTS:
                type_label, plat = _MODEL_FILE_EXTS[ext]
                if platform and plat != platform:
                    continue
                results.append(
                    InferencePoint(
                        file=rel,
                        line=0,
                        pattern=fname,
                        type=type_label,
                        platform=plat,
                        suggestion="Register with EdgeML for deployment tracking",
                        context=f"[model file] {fname}",
                    )
                )
                continue

            # Only scan known source extensions.
            if ext not in _EXT_PLATFORM:
                continue

            file_platform = _EXT_PLATFORM[ext]
            if platform and file_platform != platform:
                continue

            # Further narrow patterns to those matching this file's platform.
            file_patterns = [p for p in active_patterns if p.platform == file_platform]
            if not file_patterns:
                continue

            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    file_lines = fh.readlines()
                results.extend(_scan_lines(file_lines, ext, file_patterns, rel))
            except (OSError, UnicodeDecodeError):
                # Skip files that can't be read.
                continue

    return results


# ---------------------------------------------------------------------------
# Formatting helpers (used by CLI)
# ---------------------------------------------------------------------------


def format_text(points: list[InferencePoint]) -> str:
    """Format inference points as human-readable coloured text."""
    if not points:
        return "No inference points found."

    lines: list[str] = []
    lines.append(f"Found {len(points)} inference point(s):\n")

    for pt in points:
        loc = f"{pt.file}:{pt.line}" if pt.line > 0 else pt.file
        lines.append(f"  {loc}")
        lines.append(f"    Pattern: {pt.pattern}")
        lines.append(f"    Type: {pt.type}")
        lines.append(f"    Suggestion: {pt.suggestion}")
        lines.append("")

    # Summary
    files = {pt.file for pt in points}
    lines.append(
        f"Summary: {len(points)} inference point(s) found across {len(files)} file(s)"
    )

    # Per-platform breakdown
    platform_counts: dict[str, int] = {}
    platform_labels: dict[str, str] = {
        "ios": "iOS (CoreML)",
        "android": "Android (TFLite)",
        "python": "Python",
    }
    for pt in points:
        platform_counts[pt.platform] = platform_counts.get(pt.platform, 0) + 1
    for plat, count in sorted(platform_counts.items()):
        label = platform_labels.get(plat, plat)
        lines.append(f"  {label}: {count}")

    return "\n".join(lines)


def format_json(points: list[InferencePoint]) -> str:
    """Format inference points as a JSON string."""
    import json

    data = {
        "total": len(points),
        "files": len({pt.file for pt in points}),
        "points": [asdict(pt) for pt in points],
    }
    # Per-platform summary
    platform_counts: dict[str, int] = {}
    for pt in points:
        platform_counts[pt.platform] = platform_counts.get(pt.platform, 0) + 1
    data["platforms"] = platform_counts
    return json.dumps(data, indent=2)
