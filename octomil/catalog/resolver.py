"""Catalog alias resolver — placeholder for future catalog-driven resolution.

This module will replace per-engine model alias tables with a single
catalog-driven resolver backed by server manifest data or offline snapshot.
"""

from __future__ import annotations


class CatalogResolver:
    """Resolves model names/aliases to canonical model identifiers.

    Currently a placeholder. Will be wired to server catalog manifest
    and offline snapshot in a future PR.
    """

    def resolve(self, model_name: str) -> str | None:
        """Resolve a model name or alias to its canonical identifier.

        Returns None if the model is not found in the catalog.
        """
        # TODO: implement catalog-driven resolution
        return None
