"""Tests for the server-side model routing client (catalog_client.py).

Covers:
- CachedData TTL and serialization
- _ServerFetcher fallback behavior
- CatalogClient, EnginePriorityClient, ModelFamiliesClient
- Disk cache persistence and ETag-based conditional requests
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octomil.models.catalog_client import (
    CachedData,
    CatalogClient,
    EnginePriorityClient,
    ModelFamiliesClient,
    SourceAliasesClient,
    _ServerFetcher,
)

# =====================================================================
# CachedData tests
# =====================================================================


class TestCachedData:
    """Tests for the CachedData container."""

    def test_expired_when_never_fetched(self) -> None:
        """Data with fetched_at=0 should be expired."""
        cd = CachedData(data={}, fetched_at=0.0, ttl_seconds=3600)
        assert cd.is_expired is True

    def test_not_expired_within_ttl(self) -> None:
        """Data fetched recently should not be expired."""
        cd = CachedData(data={}, fetched_at=time.time(), ttl_seconds=3600)
        assert cd.is_expired is False

    def test_expired_after_ttl(self) -> None:
        """Data fetched beyond TTL should be expired."""
        cd = CachedData(data={}, fetched_at=time.time() - 7200, ttl_seconds=3600)
        assert cd.is_expired is True

    def test_to_dict_roundtrip(self) -> None:
        """to_dict/from_dict should roundtrip correctly."""
        original = CachedData(
            data={"key": "value"},
            fetched_at=1234567890.0,
            etag='"abc123"',
            ttl_seconds=1800,
        )
        d = original.to_dict()
        restored = CachedData.from_dict(d)
        assert restored.data == original.data
        assert restored.fetched_at == original.fetched_at
        assert restored.etag == original.etag
        assert restored.ttl_seconds == original.ttl_seconds


# =====================================================================
# _ServerFetcher tests
# =====================================================================


class TestServerFetcher:
    """Tests for the generic fetch + cache helper."""

    def test_returns_default_when_no_server_no_cache(self) -> None:
        """When server is unreachable and no disk cache, return default."""
        fetcher = _ServerFetcher(
            endpoint="test/endpoint",
            cache_filename="test_cache.json",
            default_data={"fallback": True},
            api_base="https://api.example.com",
        )

        # Patch httpx to fail
        with patch.dict("sys.modules", {"httpx": None}):
            result = fetcher.get()

        assert result == {"fallback": True}

    def test_disk_cache_persistence(self, tmp_path: Path) -> None:
        """Data should be saved to and loaded from disk cache."""
        with patch("octomil.models.catalog_client._CACHE_DIR", tmp_path):
            fetcher = _ServerFetcher(
                endpoint="test/endpoint",
                cache_filename="test_persist.json",
                default_data={},
                api_base="https://api.example.com",
            )

            # Manually save data
            cached = CachedData(
                data={"persisted": True},
                fetched_at=time.time(),
                etag='"persist-etag"',
                ttl_seconds=3600,
            )
            fetcher._save_to_disk(cached)

            # Verify file exists
            cache_file = tmp_path / "test_persist.json"
            assert cache_file.exists()

            # Load it back
            loaded = fetcher._load_from_disk()
            assert loaded is not None
            assert loaded.data == {"persisted": True}
            assert loaded.etag == '"persist-etag"'

    def test_in_memory_cache_avoids_refetch(self) -> None:
        """Second call should use in-memory cache, not re-fetch."""
        fetcher = _ServerFetcher(
            endpoint="test/endpoint",
            cache_filename="test_inmem.json",
            default_data={"default": True},
            api_base="https://api.example.com",
        )

        # Manually populate in-memory cache
        fetcher._cached = CachedData(
            data={"cached": True},
            fetched_at=time.time(),
            ttl_seconds=3600,
        )

        result = fetcher.get()
        assert result == {"cached": True}


# =====================================================================
# CatalogClient tests
# =====================================================================


class TestCatalogClient:
    """Tests for the CatalogClient."""

    def test_fallback_catalog_has_minimal_models(self) -> None:
        """Fallback catalog should have a few generic model IDs."""
        assert "gemma-1b" in CatalogClient._FALLBACK_CATALOG
        assert "llama-1b" in CatalogClient._FALLBACK_CATALOG
        assert "phi-mini" in CatalogClient._FALLBACK_CATALOG
        # Should NOT have engine-specific variants
        for entry in CatalogClient._FALLBACK_CATALOG.values():
            assert entry["variants"] == {}

    def test_fallback_aliases_empty(self) -> None:
        """Fallback aliases should be empty."""
        assert CatalogClient._FALLBACK_ALIASES == {}


class TestEnginePriorityClient:
    """Tests for the EnginePriorityClient."""

    def test_fallback_priority_is_auto(self) -> None:
        """Fallback engine priority should be ["auto"]."""
        assert EnginePriorityClient._FALLBACK_PRIORITY == ["auto"]


class TestModelFamiliesClient:
    """Tests for the ModelFamiliesClient."""

    def test_fallback_families_empty(self) -> None:
        """Fallback model families should be empty."""
        assert ModelFamiliesClient._FALLBACK_FAMILIES == {}


class TestSourceAliasesClient:
    """Tests for the SourceAliasesClient."""

    def test_fallback_aliases_empty(self) -> None:
        """Fallback source aliases should be empty."""
        assert SourceAliasesClient._FALLBACK_ALIASES == {}
