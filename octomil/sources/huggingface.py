"""HuggingFace Hub source backend.

Wraps ``huggingface_hub.hf_hub_download()`` for downloading GGUF files
and ``huggingface_hub.snapshot_download()`` for full model repos.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import SourceBackend, SourceResult

logger = logging.getLogger(__name__)


class HuggingFaceSource(SourceBackend):
    """Download models from HuggingFace Hub."""

    name = "huggingface"

    def is_available(self) -> bool:
        """Check if huggingface_hub is installed."""
        try:
            import huggingface_hub  # noqa: F401

            return True
        except ImportError:
            return False

    def check_cache(self, ref: str, filename: Optional[str] = None) -> Optional[str]:
        """Check if a model file is already in the HuggingFace cache.

        Returns the cached path if found, None otherwise.
        """
        try:
            from huggingface_hub import try_to_load_from_cache

            if filename:
                result = try_to_load_from_cache(ref, filename)
                if isinstance(result, str) and os.path.exists(result):
                    return result
            else:
                # For full repos, check the snapshot cache
                from huggingface_hub import scan_cache_dir

                cache_info = scan_cache_dir()
                for repo_info in cache_info.repos:
                    if repo_info.repo_id == ref:
                        # Return the most recent revision's path
                        for revision in sorted(
                            repo_info.revisions,
                            key=lambda r: r.last_modified,
                            reverse=True,
                        ):
                            snapshot_path = str(revision.snapshot_path)
                            if os.path.isdir(snapshot_path):
                                return snapshot_path
                        break
        except (ImportError, Exception) as exc:
            logger.debug("HuggingFace cache check failed: %s", exc)

        return None

    def resolve(
        self,
        ref: str,
        filename: Optional[str] = None,
        *,
        revision: Optional[str] = None,
        quantization_hint: Optional[str] = None,
        artifact_format: Optional[str] = None,
    ) -> SourceResult:
        """Download a model from HuggingFace Hub.

        Parameters
        ----------
        ref:
            HuggingFace repo ID (e.g. ``"Qwen/Qwen2.5-7B-Instruct-GGUF"``).
        filename:
            Specific file to download (e.g. ``"model-Q4_K_M.gguf"``).
            If None and artifact_format is set, uses the checkpoint resolver.
        revision:
            Pinned commit SHA or branch name.
        quantization_hint:
            Quantization to filter by (e.g. "q4_k_m") for multi-quant repos.
        artifact_format:
            "gguf" or "directory" — determines resolution strategy for
            repo-level URIs.
        """
        # Check cache first (single-file only — repo-level caching is handled
        # by the resolver via huggingface_hub's built-in cache)
        cached_path = self.check_cache(ref, filename)
        if cached_path:
            logger.info("HuggingFace cache hit: %s", cached_path)
            return SourceResult(
                path=cached_path,
                source_type="huggingface",
                cached=True,
                repo_id=ref,
                filename=filename,
            )

        from huggingface_hub import hf_hub_download

        if filename:
            logger.info("Downloading %s/%s from HuggingFace...", ref, filename)
            path = hf_hub_download(repo_id=ref, filename=filename, revision=revision)
            return SourceResult(
                path=path,
                source_type="huggingface",
                cached=False,
                repo_id=ref,
                filename=filename,
            )

        # Repo-level: use checkpoint resolver when artifact_format is known
        if artifact_format:
            from .hf_resolver import resolve_hf_checkpoint

            logger.info(
                "Resolving %s (format=%s, quant=%s) via checkpoint resolver...",
                ref,
                artifact_format,
                quantization_hint,
            )
            result = resolve_hf_checkpoint(
                repo_id=ref,
                revision=revision,
                quantization_hint=quantization_hint,
                artifact_format=artifact_format,
            )
            return SourceResult(
                path=result.path,
                source_type="huggingface",
                cached=False,
                repo_id=ref,
            )

        # No filename AND no artifact_format — fall back to full snapshot
        # (backward compat for direct hf:org/repo usage)
        logger.info("Downloading full repo %s from HuggingFace...", ref)
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=ref, revision=revision)
        return SourceResult(
            path=path,
            source_type="huggingface",
            cached=False,
            repo_id=ref,
        )
