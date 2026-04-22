"""Tests for planner source normalization to canonical enum values.

Canonical planner sources: "server", "cache", "offline".
All deprecated aliases must map to a canonical value.
"""

from __future__ import annotations

from dataclasses import asdict

from octomil.execution.kernel import (
    _route_metadata_from_selection,
)
from octomil.runtime.planner.schemas import (
    CANONICAL_PLANNER_SOURCES,
    RuntimeSelection,
    _PLANNER_SOURCE_ALIASES,
    normalize_planner_source,
)


class TestNormalizePlannerSource:
    """normalize_planner_source() unit tests."""

    def test_canonical_values_pass_through(self) -> None:
        """Canonical values must be returned unchanged."""
        for canonical in ("server", "cache", "offline"):
            assert normalize_planner_source(canonical) == canonical

    def test_deprecated_aliases_map_correctly(self) -> None:
        """Every documented alias must map to the right canonical value."""
        expected = {
            "local_default": "offline",
            "server_plan": "server",
            "cached": "cache",
            "fallback": "offline",
            "none": "offline",
            "local_benchmark": "offline",
        }
        for alias, canonical in expected.items():
            result = normalize_planner_source(alias)
            assert result == canonical, f"Expected {alias!r} -> {canonical!r}, got {result!r}"

    def test_unknown_values_pass_through(self) -> None:
        """Unknown source strings pass through unchanged."""
        assert normalize_planner_source("future_value") == "future_value"

    def test_empty_string_passes_through(self) -> None:
        """Empty string is not aliased -- passes through."""
        assert normalize_planner_source("") == ""

    def test_alias_map_covers_all_known_aliases(self) -> None:
        """The alias map must include at least the 6 documented aliases."""
        documented = {"local_default", "server_plan", "cached", "fallback", "none", "local_benchmark"}
        assert documented.issubset(set(_PLANNER_SOURCE_ALIASES.keys()))

    def test_all_alias_targets_are_canonical(self) -> None:
        """Every alias must map to a value in CANONICAL_PLANNER_SOURCES."""
        for alias, target in _PLANNER_SOURCE_ALIASES.items():
            assert target in CANONICAL_PLANNER_SOURCES, (
                f"Alias {alias!r} maps to {target!r} which is not canonical"
            )

    def test_canonical_set_contains_exactly_three(self) -> None:
        """The canonical set must be exactly server, cache, offline."""
        assert CANONICAL_PLANNER_SOURCES == frozenset({"server", "cache", "offline"})


class TestRouteMetadataPlannerSource:
    """RouteMetadata must always contain a canonical planner.source."""

    def test_normalizes_server_plan(self) -> None:
        selection = RuntimeSelection(locality="local", source="server_plan")
        route = _route_metadata_from_selection(selection, "local", False, model_name="test-model")
        assert route.planner.source == "server"

    def test_normalizes_fallback(self) -> None:
        selection = RuntimeSelection(locality="cloud", source="fallback")
        route = _route_metadata_from_selection(selection, "cloud", True, model_name="test-model")
        assert route.planner.source == "offline"

    def test_keeps_cache(self) -> None:
        selection = RuntimeSelection(locality="local", source="cache")
        route = _route_metadata_from_selection(selection, "local", False, model_name="test-model")
        assert route.planner.source == "cache"

    def test_normalizes_local_benchmark(self) -> None:
        selection = RuntimeSelection(locality="local", source="local_benchmark")
        route = _route_metadata_from_selection(selection, "local", False, model_name="test-model")
        assert route.planner.source == "offline"

    def test_no_selection_uses_offline(self) -> None:
        route = _route_metadata_from_selection(None, "cloud", False, model_name="test-model")
        assert route.planner.source == "offline"


class TestRouteMetadataSerializationShape:
    """RouteMetadata serialization must match the contract fixture shape."""

    def test_serialized_has_planner_source(self) -> None:
        selection = RuntimeSelection(locality="local", engine="llama.cpp", source="server_plan")
        route = _route_metadata_from_selection(selection, "local", False, model_name="gemma-2b")
        data = asdict(route)

        assert "planner" in data
        assert "source" in data["planner"]
        assert data["planner"]["source"] in CANONICAL_PLANNER_SOURCES
        assert "status" in data
        assert "execution" in data
        assert "model" in data
        assert "fallback" in data
        assert "reason" in data

    def test_serialized_execution_shape(self) -> None:
        selection = RuntimeSelection(locality="local", engine="coreml", source="cache")
        route = _route_metadata_from_selection(selection, "local", False, model_name="phi-4")
        data = asdict(route)

        execution = data["execution"]
        assert execution["locality"] == "local"
        assert execution["mode"] == "sdk_runtime"
        assert execution["engine"] == "coreml"

    def test_serialized_planner_source_never_internal(self) -> None:
        internal_sources = ["server_plan", "local_default", "fallback", "cached", "none", "local_benchmark"]
        for src in internal_sources:
            selection = RuntimeSelection(locality="local", source=src)
            route = _route_metadata_from_selection(selection, "local", False, model_name="test")
            data = asdict(route)
            assert data["planner"]["source"] in CANONICAL_PLANNER_SOURCES, (
                f"Internal source {src!r} leaked as {data['planner']['source']!r}"
            )
