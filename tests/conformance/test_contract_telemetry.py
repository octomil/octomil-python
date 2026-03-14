"""Conformance tests: telemetry events and resource attributes match contract.

Validates that the SDK's TelemetryReporter emits events using the correct
event names and includes the required attributes defined in the contract.
"""

from __future__ import annotations

from octomil._generated import otlp_resource_attributes as contract_attrs
from octomil._generated import telemetry_events as contract_events
from octomil.telemetry import TelemetryReporter, _scope_for_event

# ---------------------------------------------------------------------------
# Telemetry event name constants
# ---------------------------------------------------------------------------


class TestTelemetryEventConstants:
    """Contract event name constants must match expected string values."""

    def test_inference_started(self) -> None:
        assert contract_events.INFERENCE_STARTED == "inference.started"

    def test_inference_completed(self) -> None:
        assert contract_events.INFERENCE_COMPLETED == "inference.completed"

    def test_inference_failed(self) -> None:
        assert contract_events.INFERENCE_FAILED == "inference.failed"

    def test_inference_chunk_produced(self) -> None:
        assert contract_events.INFERENCE_CHUNK_PRODUCED == "inference.chunk_produced"

    def test_deploy_started(self) -> None:
        assert contract_events.DEPLOY_STARTED == "deploy.started"

    def test_deploy_completed(self) -> None:
        assert contract_events.DEPLOY_COMPLETED == "deploy.completed"


class TestTelemetryEventRequiredAttributes:
    """Each contract event must define required attributes that match the spec."""

    def test_inference_started_requires_model_id(self) -> None:
        attrs = contract_events.EVENT_REQUIRED_ATTRIBUTES["inference.started"]
        assert "model.id" in attrs

    def test_inference_completed_requires_model_id_and_duration(self) -> None:
        attrs = contract_events.EVENT_REQUIRED_ATTRIBUTES["inference.completed"]
        assert "model.id" in attrs
        assert "inference.duration_ms" in attrs

    def test_inference_failed_requires_error_fields(self) -> None:
        attrs = contract_events.EVENT_REQUIRED_ATTRIBUTES["inference.failed"]
        assert "model.id" in attrs
        assert "error.type" in attrs
        assert "error.message" in attrs

    def test_inference_chunk_requires_chunk_index(self) -> None:
        attrs = contract_events.EVENT_REQUIRED_ATTRIBUTES["inference.chunk_produced"]
        assert "model.id" in attrs
        assert "inference.chunk_index" in attrs

    def test_deploy_started_requires_model_and_version(self) -> None:
        attrs = contract_events.EVENT_REQUIRED_ATTRIBUTES["deploy.started"]
        assert "model.id" in attrs
        assert "model.version" in attrs

    def test_deploy_completed_requires_duration(self) -> None:
        attrs = contract_events.EVENT_REQUIRED_ATTRIBUTES["deploy.completed"]
        assert "model.id" in attrs
        assert "deploy.duration_ms" in attrs


# ---------------------------------------------------------------------------
# OTLP resource attribute keys
# ---------------------------------------------------------------------------


class TestOtlpResourceAttributes:
    """Contract resource attribute keys must match expected OTLP keys."""

    def test_service_name(self) -> None:
        assert contract_attrs.SERVICE_NAME == "service.name"

    def test_service_version(self) -> None:
        assert contract_attrs.SERVICE_VERSION == "service.version"

    def test_telemetry_sdk_name(self) -> None:
        assert contract_attrs.TELEMETRY_SDK_NAME == "telemetry.sdk.name"

    def test_telemetry_sdk_language(self) -> None:
        assert contract_attrs.TELEMETRY_SDK_LANGUAGE == "telemetry.sdk.language"

    def test_telemetry_sdk_version(self) -> None:
        assert contract_attrs.TELEMETRY_SDK_VERSION == "telemetry.sdk.version"

    def test_octomil_org_id(self) -> None:
        assert contract_attrs.OCTOMIL_ORG_ID == "octomil.org.id"

    def test_octomil_device_id(self) -> None:
        assert contract_attrs.OCTOMIL_DEVICE_ID == "octomil.device.id"

    def test_octomil_platform(self) -> None:
        assert contract_attrs.OCTOMIL_PLATFORM == "octomil.platform"

    def test_octomil_sdk_surface(self) -> None:
        assert contract_attrs.OCTOMIL_SDK_SURFACE == "octomil.sdk.surface"

    def test_required_keys_list(self) -> None:
        assert set(contract_attrs.REQUIRED_KEYS) == {
            "service.name",
            "service.version",
            "telemetry.sdk.name",
            "telemetry.sdk.language",
            "telemetry.sdk.version",
            "octomil.org.id",
            "octomil.device.id",
            "octomil.platform",
            "octomil.sdk.surface",
        }


# ---------------------------------------------------------------------------
# Scope routing
# ---------------------------------------------------------------------------


class TestScopeRouting:
    """SDK scope routing must match the contract's scope definitions."""

    def test_inference_events_route_to_inference_scope(self) -> None:
        assert _scope_for_event("inference.started") == "octomil.inference"
        assert _scope_for_event("inference.completed") == "octomil.inference"
        assert _scope_for_event("inference.failed") == "octomil.inference"
        assert _scope_for_event("inference.chunk_produced") == "octomil.inference"

    def test_deploy_events_route_to_deploy_scope(self) -> None:
        assert _scope_for_event("deploy.started") == "octomil.deploy"
        assert _scope_for_event("deploy.completed") == "octomil.deploy"


# ---------------------------------------------------------------------------
# TelemetryReporter method coverage
# ---------------------------------------------------------------------------


class TestReporterMethodCoverage:
    """TelemetryReporter must have a public method for each contract event type."""

    EXPECTED_METHODS = [
        "report_inference_started",
        "report_inference_completed",
        "report_inference_failed",
        "report_inference_chunk",
        "report_deploy_started",
        "report_deploy_completed",
    ]

    def test_all_contract_events_have_reporter_methods(self) -> None:
        for method_name in self.EXPECTED_METHODS:
            assert hasattr(TelemetryReporter, method_name), f"TelemetryReporter missing method: {method_name}"
            assert callable(getattr(TelemetryReporter, method_name))
