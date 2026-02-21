"""Kaggle source backend (stub).

Detects the ``kaggle`` CLI and provides the download command.
Full download integration is deferred to Phase 2.
"""

from __future__ import annotations

import logging
import shutil
from typing import Optional

from .base import SourceBackend, SourceResult

logger = logging.getLogger(__name__)


class KaggleSource(SourceBackend):
    """Download models via the Kaggle CLI.

    This is a stub implementation for Phase 1. It detects whether the
    ``kaggle`` CLI is installed and provides the download command, but
    does not yet automate the full download workflow.
    """

    name = "kaggle"

    def is_available(self) -> bool:
        """Check if the Kaggle CLI is installed."""
        return shutil.which("kaggle") is not None

    def check_cache(self, ref: str, filename: Optional[str] = None) -> Optional[str]:
        """Kaggle cache checking is not implemented in Phase 1."""
        return None

    def resolve(
        self,
        ref: str,
        filename: Optional[str] = None,
    ) -> SourceResult:
        """Resolve a Kaggle model reference.

        In Phase 1, this raises an error with the manual download command.
        Full automation is planned for Phase 2.

        Parameters
        ----------
        ref:
            Kaggle model path (e.g. ``"google/gemma/pyTorch/gemma-2b"``).
        """
        if not self.is_available():
            raise RuntimeError(
                f"Kaggle CLI is not installed. Install it with: pip install kaggle\n"
                f"Then download manually: kaggle models instances versions download {ref}"
            )

        raise RuntimeError(
            f"Kaggle automated download is not yet implemented (Phase 2).\n"
            f"Download manually: kaggle models instances versions download {ref}\n"
            f"Then pass the local file path to edgeml serve."
        )
