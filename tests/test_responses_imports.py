"""Smoke tests for responses module split.

Verifies that all public and commonly-used import paths continue to work
after the responses.py split into focused modules.
"""

from __future__ import annotations


class TestPublicAPIImports:
    """All previously public symbols remain importable from octomil.responses."""

    def test_octomil_responses_importable(self) -> None:
        from octomil.responses import OctomilResponses

        assert OctomilResponses is not None

    def test_response_request_importable(self) -> None:
        from octomil.responses import ResponseRequest

        assert ResponseRequest is not None

    def test_response_importable(self) -> None:
        from octomil.responses import Response

        assert Response is not None

    def test_all_public_symbols(self) -> None:
        """All symbols in __all__ are importable."""
        import octomil.responses as mod

        for name in [
            "OctomilResponses",
            "ResponseRequest",
            "Response",
            "ResponseFormat",
            "ResponseStreamEvent",
            "ResponseToolCall",
            "ResponseUsage",
            "ContentPart",
            "InputItem",
            "OutputItem",
            "ToolChoice",
            "runtime",
            "tools",
        ]:
            assert hasattr(mod, name), f"Missing public symbol: {name}"


class TestDirectModuleImports:
    """Import paths used by internal consumers (e.g. from octomil.responses.responses)."""

    def test_direct_responses_import(self) -> None:
        from octomil.responses.responses import OctomilResponses

        assert OctomilResponses is not None

    def test_backward_compat_private_names(self) -> None:
        """Private module-level names that were previously in responses.py."""
        from octomil.responses.responses import (
            _AttemptRunnerResult,
            _determine_locality,
            _generate_id,
            _model_id_str,
            _selection_to_candidate_dicts,
            _ToolCallBuffer,
            _try_resolve_planner_selection,
        )

        assert _model_id_str is not None
        assert _generate_id is not None
        assert _ToolCallBuffer is not None
        assert _determine_locality is not None
        assert _try_resolve_planner_selection is not None
        assert _selection_to_candidate_dicts is not None
        assert _AttemptRunnerResult is not None


class TestExtractedModuleImports:
    """Verify the new extracted modules are importable."""

    def test_request_normalization(self) -> None:
        from octomil.responses.request_normalization import (
            _model_id_str,
            apply_previous_response,
            build_runtime_request,
        )

        assert _model_id_str is not None
        assert apply_previous_response is not None
        assert build_runtime_request is not None

    def test_response_builder(self) -> None:
        from octomil.responses.response_builder import (
            ToolCallBuffer,
            build_response,
            generate_id,
        )

        assert build_response is not None
        assert generate_id is not None
        assert ToolCallBuffer is not None

    def test_streaming(self) -> None:
        from octomil.responses.streaming import (
            stream_direct,
            stream_with_attempt_runner,
        )

        assert stream_direct is not None
        assert stream_with_attempt_runner is not None

    def test_route_attachment(self) -> None:
        from octomil.responses.route_attachment import (
            _locality_for_candidate,
            determine_locality,
            report_fallback_if_needed,
        )

        assert determine_locality is not None
        assert report_fallback_if_needed is not None
        assert _locality_for_candidate is not None

    def test_dispatch(self) -> None:
        from octomil.responses.dispatch import (
            AttemptRunnerResult,
            is_synthetic_cloud_fallback,
            resolve_runtime,
            resolve_runtime_for_candidate,
            resolve_via_attempt_runner,
            selection_to_candidate_dicts,
            try_resolve_planner_selection,
        )

        assert AttemptRunnerResult is not None
        assert try_resolve_planner_selection is not None
        assert selection_to_candidate_dicts is not None
        assert is_synthetic_cloud_fallback is not None
        assert resolve_runtime is not None
        assert resolve_runtime_for_candidate is not None
        assert resolve_via_attempt_runner is not None
