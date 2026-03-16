"""HuggingFace checkpoint resolver.

Resolves a repo-level hf:// URI to the right download strategy:
- Single GGUF file -> hf_hub_download(repo, filename)
- Sharded GGUF -> snapshot_download(repo, allow_patterns=[shard_names])
- Directory (MLX/safetensors) -> snapshot_download(repo, allow_patterns=[known_runtime_files])

Uses list_repo_tree() for file metadata (size, LFS sha256).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResolvedFile:
    """A single file within a resolved checkpoint."""

    path: str  # relative path in repo
    size_bytes: int
    sha256: Optional[str] = None
    local_path: Optional[str] = None  # set after download


@dataclass
class ResolvedCheckpoint:
    """Result of resolving a HF repo to downloadable files."""

    kind: str  # "single_file", "sharded", "directory"
    path: str  # local path to entrypoint (first shard, or directory root)
    files: list[ResolvedFile] = field(default_factory=list)
    total_size_bytes: int = 0


# Shard naming pattern: *-00001-of-00002.gguf
_SHARD_RE = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf$")

# Known runtime artifact patterns for directory-shaped models (MLX, safetensors).
# Intentionally tight — no *.py (untrusted remote code), no docs.
_DIRECTORY_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.model",
    "*.tiktoken",
    "tokenizer.model",
    "*.vocab",
]

# File extensions matched by _DIRECTORY_ALLOW_PATTERNS (for filtering tree results).
_DIRECTORY_EXTENSIONS = frozenset({".safetensors", ".json", ".model", ".tiktoken", ".vocab"})


def _lfs_sha256(repo_file: object) -> Optional[str]:
    """Extract sha256 from a RepoFile's lfs metadata, if present."""
    lfs = getattr(repo_file, "lfs", None)
    if lfs and isinstance(lfs, dict):
        return lfs.get("sha256")
    return None


def _list_repo_files(repo_id: str, revision: Optional[str]) -> list:
    """Fetch repo tree and return file entries (patchable in tests)."""
    from huggingface_hub import HfApi

    api = HfApi()
    tree = list(api.list_repo_tree(repo_id, revision=revision, recursive=True))
    # Duck typing: files have .path and .size, folders only have .path
    return [item for item in tree if hasattr(item, "size") and hasattr(item, "path")]


def resolve_hf_checkpoint(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    quantization_hint: Optional[str] = None,
    artifact_format: str = "gguf",
) -> ResolvedCheckpoint:
    """Resolve a HF repo to downloadable files and download them.

    Parameters
    ----------
    repo_id:
        HuggingFace repo ID (e.g. "Qwen/Qwen2.5-7B-Instruct-GGUF").
    revision:
        Pinned commit SHA or branch. Defaults to "main".
    quantization_hint:
        Quantization to filter by (e.g. "q4_k_m"). Required for GGUF repos
        with multiple quantizations.
    artifact_format:
        "gguf" or "directory" (for MLX/safetensors repos).

    Raises
    ------
    RuntimeError
        If no matching files are found. Fails loudly — no silent fallback.
    """
    files = _list_repo_files(repo_id, revision)

    if artifact_format == "gguf":
        return _resolve_gguf(repo_id, revision, files, quantization_hint)
    elif artifact_format in ("directory", "mlx", "safetensors"):
        return _resolve_directory(repo_id, revision, files)
    else:
        raise RuntimeError(
            f"Unknown artifact_format '{artifact_format}' for {repo_id}. Expected 'gguf' or 'directory'."
        )


def _hf_hub_download(repo_id: str, filename: str, revision: Optional[str] = None) -> str:
    """Lazy wrapper for huggingface_hub.hf_hub_download (patchable in tests)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _snapshot_download(
    repo_id: str,
    revision: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
) -> str:
    """Lazy wrapper for huggingface_hub.snapshot_download (patchable in tests)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)


def _resolve_gguf(
    repo_id: str,
    revision: Optional[str],
    files: list,
    quant_hint: Optional[str],
) -> ResolvedCheckpoint:
    """Resolve GGUF files from a repo, handling single and sharded cases."""

    gguf_files = sorted(
        [f for f in files if f.path.endswith(".gguf")],
        key=lambda f: f.path,
    )

    if not gguf_files:
        raise RuntimeError(f"No GGUF files found in {repo_id}")

    # Filter by quantization hint
    if quant_hint:
        q = quant_hint.lower().replace("-", "_")
        matching = [f for f in gguf_files if q in f.path.lower().replace("-", "_")]
        if not matching:
            available = [f.path for f in gguf_files]
            raise RuntimeError(
                f"No GGUF files matching quantization '{quant_hint}' in {repo_id}. Available: {available}"
            )
        gguf_files = matching

    # Separate single files from shards
    shards = [f for f in gguf_files if _SHARD_RE.match(f.path)]
    singles = [f for f in gguf_files if not _SHARD_RE.match(f.path)]

    # Prefer single file over shards (simpler)
    if singles:
        target = singles[0]
        local_path = _hf_hub_download(repo_id, filename=target.path, revision=revision)
        return ResolvedCheckpoint(
            kind="single_file",
            path=local_path,
            files=[
                ResolvedFile(
                    path=target.path,
                    size_bytes=target.size,
                    sha256=_lfs_sha256(target),
                    local_path=local_path,
                )
            ],
            total_size_bytes=target.size,
        )

    # Sharded: group by stem (everything before -NNNNN-of-NNNNN.gguf)
    if shards:
        first_match = _SHARD_RE.match(shards[0].path)
        if not first_match:
            raise RuntimeError(f"Shard regex failed on {shards[0].path}")

        base_stem = first_match.group(1)
        total_count = first_match.group(3)

        # Collect all shards with the same stem
        shard_group = sorted(
            [
                f
                for f in shards
                if (m := _SHARD_RE.match(f.path)) and m.group(1) == base_stem and m.group(3) == total_count
            ],
            key=lambda f: f.path,
        )

        if len(shard_group) != int(total_count):
            logger.warning(
                "Expected %s shards for %s but found %d in %s",
                total_count,
                base_stem,
                len(shard_group),
                repo_id,
            )

        shard_names = [f.path for f in shard_group]
        local_dir = _snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=shard_names,
        )

        resolved_files = []
        total_size = 0
        for f in shard_group:
            resolved_files.append(
                ResolvedFile(
                    path=f.path,
                    size_bytes=f.size,
                    sha256=_lfs_sha256(f),
                    local_path=os.path.join(local_dir, f.path),
                )
            )
            total_size += f.size

        return ResolvedCheckpoint(
            kind="sharded",
            path=os.path.join(local_dir, shard_names[0]),
            files=resolved_files,
            total_size_bytes=total_size,
        )

    raise RuntimeError(f"No GGUF files found in {repo_id} after filtering")


def _resolve_directory(
    repo_id: str,
    revision: Optional[str],
    files: list,
) -> ResolvedCheckpoint:
    """Resolve a directory-shaped model (MLX, safetensors)."""
    local_dir = _snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=_DIRECTORY_ALLOW_PATTERNS,
    )

    resolved_files = []
    total_size = 0
    for f in files:
        ext = os.path.splitext(f.path)[1]
        if ext in _DIRECTORY_EXTENSIONS or f.path == "tokenizer.model":
            resolved_files.append(
                ResolvedFile(
                    path=f.path,
                    size_bytes=f.size,
                    sha256=_lfs_sha256(f),
                    local_path=os.path.join(local_dir, f.path),
                )
            )
            total_size += f.size

    return ResolvedCheckpoint(
        kind="directory",
        path=local_dir,
        files=resolved_files,
        total_size_bytes=total_size,
    )
