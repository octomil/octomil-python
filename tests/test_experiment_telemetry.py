"""Tests for experiment telemetry wiring.

Covers:
- ExperimentsAPI.create() emitting report_experiment_assigned for each variant
- Model.predict() emitting report_experiment_metric when experiment_id is set
- Model.predict_stream() emitting report_experiment_metric when experiment_id is set
- Telemetry exceptions are swallowed (never crash)
- No experiment metrics when experiment_id is None
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import MagicMock, patch

import pytest

from octomil.model import Model, ModelMetadata, Prediction
from octomil.serve import GenerationChunk, GenerationRequest, InferenceMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _make_metadata(
    model_id: str = "model-abc",
    name: str = "test-model",
    version: str = "1.0.0",
    experiment_id: str | None = None,
) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id, name=name, version=version, experiment_id=experiment_id
    )


def _make_engine(backend: MagicMock | None = None) -> MagicMock:
    engine = MagicMock()
    engine.create_backend.return_value = backend or MagicMock()
    return engine


def _make_metrics(**overrides) -> InferenceMetrics:
    defaults = dict(
        ttfc_ms=50.0,
        total_tokens=10,
        tokens_per_second=100.0,
        total_duration_ms=100.0,
    )
    defaults.update(overrides)
    return InferenceMetrics(**defaults)


def _make_request() -> GenerationRequest:
    return GenerationRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )


async def _async_chunks(*chunks: GenerationChunk):
    """Helper async generator yielding given chunks."""
    for c in chunks:
        yield c


class _StubApi:
    """Minimal API stub for ExperimentsAPI tests."""

    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append(("get", path, params))
        return {"path": path, "params": params}

    def post(self, path, payload=None):
        self.calls.append(("post", path, payload))
        return {"path": path, "payload": payload, "id": "exp_123", "version": "1.0.0"}

    def patch(self, path, payload=None):
        self.calls.append(("patch", path, payload))
        return {"path": path, "payload": payload}

    def put(self, path, payload=None):
        self.calls.append(("put", path, payload))
        return {"path": path, "payload": payload}

    def delete(self, path, params=None):
        self.calls.append(("delete", path, params))
        return {"path": path, "params": params}


# ---------------------------------------------------------------------------
# ExperimentsAPI.create() -> report_experiment_assigned
# ---------------------------------------------------------------------------


class TestExperimentsAPICreateTelemetry:
    def test_create_emits_assigned_for_each_variant(self):
        from octomil.control_plane import ExperimentsAPI

        api = _StubApi()
        reporter = MagicMock()
        experiments = ExperimentsAPI(api, org_id="org_1", reporter=reporter)

        result = experiments.create(
            name="ab_test",
            model_id="model_1",
            control_version="1.0.0",
            treatment_version="1.1.0",
        )

        # Verify the API call still works
        assert result["id"] == "exp_123"

        # Verify two report_experiment_assigned calls (control + treatment)
        assert reporter.report_experiment_assigned.call_count == 2

        control_call = reporter.report_experiment_assigned.call_args_list[0]
        assert control_call.kwargs["model_id"] == "model_1"
        assert control_call.kwargs["experiment_id"] == "exp_123"
        assert control_call.kwargs["variant"] == "control"

        treatment_call = reporter.report_experiment_assigned.call_args_list[1]
        assert treatment_call.kwargs["model_id"] == "model_1"
        assert treatment_call.kwargs["experiment_id"] == "exp_123"
        assert treatment_call.kwargs["variant"] == "treatment"

    def test_create_works_without_reporter(self):
        from octomil.control_plane import ExperimentsAPI

        api = _StubApi()
        experiments = ExperimentsAPI(api, org_id="org_1")

        # Patch get_reporter to return None (no global reporter)
        with patch("octomil.get_reporter", return_value=None):
            result = experiments.create(
                name="ab_test",
                model_id="model_1",
                control_version="1.0.0",
                treatment_version="1.1.0",
            )
        assert result["id"] == "exp_123"

    def test_create_swallows_telemetry_exception(self):
        from octomil.control_plane import ExperimentsAPI

        api = _StubApi()
        reporter = MagicMock()
        reporter.report_experiment_assigned.side_effect = RuntimeError("boom")
        experiments = ExperimentsAPI(api, org_id="org_1", reporter=reporter)

        # Should not raise despite telemetry failure
        result = experiments.create(
            name="ab_test",
            model_id="model_1",
            control_version="1.0.0",
            treatment_version="1.1.0",
        )
        assert result["id"] == "exp_123"
        # Both calls attempted despite the exception
        assert reporter.report_experiment_assigned.call_count == 2

    def test_create_uses_global_reporter_fallback(self):
        from octomil.control_plane import ExperimentsAPI

        api = _StubApi()
        global_reporter = MagicMock()
        experiments = ExperimentsAPI(api, org_id="org_1")

        with patch("octomil.get_reporter", return_value=global_reporter):
            experiments.create(
                name="ab_test",
                model_id="model_1",
                control_version="1.0.0",
                treatment_version="1.1.0",
            )

        assert global_reporter.report_experiment_assigned.call_count == 2

    def test_override_reporter_takes_precedence_over_global(self):
        from octomil.control_plane import ExperimentsAPI

        api = _StubApi()
        override_reporter = MagicMock()
        global_reporter = MagicMock()
        experiments = ExperimentsAPI(api, org_id="org_1", reporter=override_reporter)

        with patch("octomil.get_reporter", return_value=global_reporter):
            experiments.create(
                name="ab_test",
                model_id="model_1",
                control_version="1.0.0",
                treatment_version="1.1.0",
            )

        assert override_reporter.report_experiment_assigned.call_count == 2
        global_reporter.report_experiment_assigned.assert_not_called()


# ---------------------------------------------------------------------------
# Model.predict() -> report_experiment_metric
# ---------------------------------------------------------------------------


class TestPredictExperimentMetrics:
    def test_predict_emits_experiment_metrics_when_experiment_id_set(self):
        reporter = MagicMock()
        backend = MagicMock()
        backend.generate.return_value = ("hello world", _make_metrics())

        meta = _make_metadata(experiment_id="exp_456")
        model = Model(meta, _make_engine(backend), _reporter=reporter)
        result = model.predict(_make_request())

        assert isinstance(result, Prediction)
        assert result.text == "hello world"

        # Verify 3 experiment metric calls
        assert reporter.report_experiment_metric.call_count == 3

        metric_calls = reporter.report_experiment_metric.call_args_list
        metric_names = [c.kwargs["metric_name"] for c in metric_calls]
        assert "inference.duration_ms" in metric_names
        assert "inference.ttfc_ms" in metric_names
        assert "inference.throughput_tps" in metric_names

        # All calls should reference the same experiment_id
        for call in metric_calls:
            assert call.kwargs["experiment_id"] == "exp_456"

        # duration_ms and ttfc_ms values should be positive
        duration_call = next(
            c
            for c in metric_calls
            if c.kwargs["metric_name"] == "inference.duration_ms"
        )
        assert duration_call.kwargs["metric_value"] > 0

        ttfc_call = next(
            c for c in metric_calls if c.kwargs["metric_name"] == "inference.ttfc_ms"
        )
        assert ttfc_call.kwargs["metric_value"] == 50.0

    def test_predict_skips_experiment_metrics_when_no_experiment_id(self):
        reporter = MagicMock()
        backend = MagicMock()
        backend.generate.return_value = ("hello", _make_metrics())

        meta = _make_metadata()  # no experiment_id
        model = Model(meta, _make_engine(backend), _reporter=reporter)
        model.predict(_make_request())

        reporter.report_experiment_metric.assert_not_called()

    def test_predict_experiment_metric_exception_swallowed(self):
        reporter = MagicMock()
        reporter.report_experiment_metric.side_effect = RuntimeError("telemetry down")
        backend = MagicMock()
        backend.generate.return_value = ("ok", _make_metrics())

        meta = _make_metadata(experiment_id="exp_789")
        model = Model(meta, _make_engine(backend), _reporter=reporter)

        # Should not raise
        result = model.predict(_make_request())
        assert result.text == "ok"

    def test_predict_no_experiment_metrics_on_failure(self):
        """When generation fails, no experiment metrics should be emitted."""
        reporter = MagicMock()
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("boom")

        meta = _make_metadata(experiment_id="exp_fail")
        model = Model(meta, _make_engine(backend), _reporter=reporter)

        with pytest.raises(RuntimeError, match="boom"):
            model.predict(_make_request())

        reporter.report_experiment_metric.assert_not_called()
        reporter.report_inference_failed.assert_called_once()


# ---------------------------------------------------------------------------
# Model.predict_stream() -> report_experiment_metric
# ---------------------------------------------------------------------------


class TestPredictStreamExperimentMetrics:
    def test_stream_emits_experiment_metrics_when_experiment_id_set(self):
        reporter = MagicMock()
        backend = MagicMock()
        chunks = [
            GenerationChunk(text="Hello", token_count=1),
            GenerationChunk(text=" world", token_count=1),
            GenerationChunk(text="", finish_reason="stop"),
        ]
        backend.generate_stream = MagicMock(return_value=_async_chunks(*chunks))

        meta = _make_metadata(experiment_id="exp_stream")
        model = Model(meta, _make_engine(backend), _reporter=reporter)

        collected = []

        async def _run():
            async for chunk in model.predict_stream(_make_request()):
                collected.append(chunk)

        asyncio.run(_run())

        assert len(collected) == 3

        # Verify 3 experiment metric calls
        assert reporter.report_experiment_metric.call_count == 3

        metric_calls = reporter.report_experiment_metric.call_args_list
        metric_names = [c.kwargs["metric_name"] for c in metric_calls]
        assert "inference.duration_ms" in metric_names
        assert "inference.ttfc_ms" in metric_names
        assert "inference.throughput_tps" in metric_names

        for call in metric_calls:
            assert call.kwargs["experiment_id"] == "exp_stream"

    def test_stream_skips_experiment_metrics_when_no_experiment_id(self):
        reporter = MagicMock()
        backend = MagicMock()
        chunks = [GenerationChunk(text="ok", finish_reason="stop")]
        backend.generate_stream = MagicMock(return_value=_async_chunks(*chunks))

        meta = _make_metadata()  # no experiment_id
        model = Model(meta, _make_engine(backend), _reporter=reporter)

        async def _run():
            async for _ in model.predict_stream(_make_request()):
                pass

        asyncio.run(_run())

        reporter.report_experiment_metric.assert_not_called()

    def test_stream_experiment_metric_exception_swallowed(self):
        reporter = MagicMock()
        reporter.report_experiment_metric.side_effect = RuntimeError("boom")
        backend = MagicMock()
        chunks = [GenerationChunk(text="ok", finish_reason="stop")]
        backend.generate_stream = MagicMock(return_value=_async_chunks(*chunks))

        meta = _make_metadata(experiment_id="exp_err")
        model = Model(meta, _make_engine(backend), _reporter=reporter)

        collected = []

        async def _run():
            async for chunk in model.predict_stream(_make_request()):
                collected.append(chunk)

        # Should not raise
        asyncio.run(_run())
        assert len(collected) == 1


# ---------------------------------------------------------------------------
# ModelMetadata.experiment_id field
# ---------------------------------------------------------------------------


class TestModelMetadataExperimentId:
    def test_experiment_id_defaults_to_none(self):
        meta = ModelMetadata(model_id="m1", name="n1", version="1.0")
        assert meta.experiment_id is None

    def test_experiment_id_can_be_set(self):
        meta = ModelMetadata(
            model_id="m1", name="n1", version="1.0", experiment_id="exp_1"
        )
        assert meta.experiment_id == "exp_1"
