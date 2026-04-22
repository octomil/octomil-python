"""Smoke tests verifying backward-compatible imports after kernel.py split.

Every public symbol that was importable from ``octomil.execution.kernel``
before the split MUST still be importable from the same path.  This file
asserts that the re-exports in the slimmed-down kernel.py work.
"""

from __future__ import annotations


def test_kernel_reexports_route_metadata_types() -> None:
    """RouteMetadata types are importable from kernel (re-exported)."""
    from octomil.execution.kernel import (
        ArtifactCache,
        FallbackInfo,
        PlannerInfo,
        RouteArtifact,
        RouteExecution,
        RouteMetadata,
        RouteModel,
        RouteModelRequested,
        RouteModelResolved,
        RouteReason,
    )

    # Verify they are real types
    assert RouteMetadata is not None
    assert RouteExecution is not None
    assert RouteModel is not None
    assert RouteModelRequested is not None
    assert RouteModelResolved is not None
    assert RouteArtifact is not None
    assert ArtifactCache is not None
    assert PlannerInfo is not None
    assert FallbackInfo is not None
    assert RouteReason is not None


def test_kernel_reexports_policy_helpers() -> None:
    """Policy conversion functions importable from kernel."""
    from octomil.execution.kernel import (
        _cloud_available,
        _inline_to_routing_policy,
        _resolve_localities,
        _resolve_routing_policy,
        _select_locality_for_capability,
    )

    assert callable(_resolve_routing_policy)
    assert callable(_inline_to_routing_policy)
    assert callable(_cloud_available)
    assert callable(_resolve_localities)
    assert callable(_select_locality_for_capability)


def test_kernel_reexports_cloud_helpers() -> None:
    """Cloud dispatch functions importable from kernel."""
    from octomil.execution.kernel import (
        _cloud_api_key,
        _openai_base_url,
        _platform_api_base_url,
    )

    assert callable(_openai_base_url)
    assert callable(_platform_api_base_url)
    assert callable(_cloud_api_key)


def test_kernel_reexports_planner_helpers() -> None:
    """Planner resolution functions importable from kernel."""
    from octomil.execution.kernel import (
        _PLANNER_CAPABILITY_MAP,
        _candidate_fallback_allowed,
        _candidate_to_selection,
        _is_synthetic_cloud_fallback,
        _resolve_planner_selection,
        _routing_policy_for_candidate,
        _selection_candidate_dicts,
    )

    assert isinstance(_PLANNER_CAPABILITY_MAP, dict)
    assert callable(_resolve_planner_selection)
    assert callable(_is_synthetic_cloud_fallback)
    assert callable(_selection_candidate_dicts)
    assert callable(_candidate_fallback_allowed)
    assert callable(_candidate_to_selection)
    assert callable(_routing_policy_for_candidate)


def test_kernel_reexports_benchmark_helpers() -> None:
    """Benchmark upload functions importable from kernel."""
    from octomil.execution.kernel import (
        _sanitize_benchmark_payload,
        _upload_benchmark_async,
    )

    assert callable(_sanitize_benchmark_payload)
    assert callable(_upload_benchmark_async)


def test_kernel_reexports_route_metadata_builder() -> None:
    """The _route_metadata_from_selection builder is importable from kernel."""
    from octomil.execution.kernel import _route_metadata_from_selection

    assert callable(_route_metadata_from_selection)


def test_new_module_direct_imports() -> None:
    """Extracted modules are importable directly."""
    from octomil.execution.attempt_execution import _resolve_routing_policy
    from octomil.execution.benchmark_upload import _upload_benchmark_async
    from octomil.execution.cloud_dispatch import _openai_base_url
    from octomil.execution.planner_resolution import _resolve_planner_selection
    from octomil.execution.route_metadata_mapper import RouteMetadata

    assert callable(_resolve_routing_policy)
    assert callable(_upload_benchmark_async)
    assert callable(_openai_base_url)
    assert callable(_resolve_planner_selection)
    assert RouteMetadata is not None


def test_types_module_imports() -> None:
    """octomil.types imports route metadata from the new module."""
    # They should be the same objects regardless of import path
    from octomil.execution.route_metadata_mapper import RouteMetadata as DirectRouteMetadata
    from octomil.types import (
        ArtifactCache,
        FallbackInfo,
        PlannerInfo,
        RouteArtifact,
        RouteExecution,
        RouteMetadata,
        RouteModel,
        RouteModelRequested,
        RouteModelResolved,
        RouteReason,
    )

    assert RouteMetadata is DirectRouteMetadata
    assert RouteExecution is not None
    assert RouteModel is not None
    assert RouteModelRequested is not None
    assert RouteModelResolved is not None
    assert RouteArtifact is not None
    assert ArtifactCache is not None
    assert PlannerInfo is not None
    assert FallbackInfo is not None
    assert RouteReason is not None


def test_execution_kernel_still_importable() -> None:
    """ExecutionKernel, ExecutionResult, StreamChunk still in kernel."""
    from octomil.execution.kernel import (
        ChatRoutingDecision,
        ExecutionKernel,
        ExecutionResult,
        StreamChunk,
    )

    assert ExecutionKernel is not None
    assert ExecutionResult is not None
    assert StreamChunk is not None
    assert ChatRoutingDecision is not None


def test_execution_init_exports() -> None:
    """execution package __init__ still exports the main types."""
    from octomil.execution import (
        ChatRoutingDecision,
        ExecutionKernel,
        ExecutionResult,
    )

    assert ExecutionKernel is not None
    assert ExecutionResult is not None
    assert ChatRoutingDecision is not None
