"""Tests for octomil.model — Model.predict() and predict_stream() telemetry wiring."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

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
) -> ModelMetadata:
    return ModelMetadata(model_id=model_id, name=name, version=version)


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


async def _async_chunks_with_error(*chunks: GenerationChunk):
    """Yield some chunks then raise."""
    for c in chunks:
        yield c
    raise RuntimeError("stream exploded")


# ---------------------------------------------------------------------------
# predict() telemetry
# ---------------------------------------------------------------------------


class TestPredictTelemetry:
    def test_predict_reports_started_and_completed(self):
        reporter = MagicMock()
        backend = MagicMock()
        backend.generate.return_value = ("hello world", _make_metrics())

        meta = _make_metadata()
        model = Model(meta, _make_engine(backend), _reporter=reporter)
        result = model.predict(_make_request())

        assert isinstance(result, Prediction)
        assert result.text == "hello world"

        reporter.report_generation_started.assert_called_once()
        call_kwargs = reporter.report_generation_started.call_args.kwargs
        assert call_kwargs["model_id"] == "model-abc"
        assert call_kwargs["version"] == "1.0.0"
        assert _UUID_HEX_RE.match(call_kwargs["session_id"])

        reporter.report_generation_completed.assert_called_once()
        comp_kwargs = reporter.report_generation_completed.call_args.kwargs
        assert comp_kwargs["model_id"] == "model-abc"
        assert comp_kwargs["total_chunks"] == 10
        assert comp_kwargs["ttfc_ms"] == 50.0
        assert comp_kwargs["total_duration_ms"] > 0

        reporter.report_generation_failed.assert_not_called()

    def test_predict_reports_failed_on_backend_error(self):
        reporter = MagicMock()
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("boom")

        model = Model(_make_metadata(), _make_engine(backend), _reporter=reporter)

        with pytest.raises(RuntimeError, match="boom"):
            model.predict(_make_request())

        reporter.report_generation_started.assert_called_once()
        reporter.report_generation_failed.assert_called_once()
        fail_kwargs = reporter.report_generation_failed.call_args.kwargs
        assert fail_kwargs["model_id"] == "model-abc"
        assert _UUID_HEX_RE.match(fail_kwargs["session_id"])

        reporter.report_generation_completed.assert_not_called()

    def test_predict_unique_session_ids(self):
        reporter = MagicMock()
        backend = MagicMock()
        backend.generate.return_value = ("ok", _make_metrics())

        model = Model(_make_metadata(), _make_engine(backend), _reporter=reporter)
        model.predict(_make_request())
        model.predict(_make_request())

        calls = reporter.report_generation_started.call_args_list
        session_ids = [c.kwargs["session_id"] for c in calls]
        assert len(session_ids) == 2
        assert session_ids[0] != session_ids[1]
        for sid in session_ids:
            assert _UUID_HEX_RE.match(sid)


# ---------------------------------------------------------------------------
# predict_stream() telemetry
# ---------------------------------------------------------------------------


class TestPredictStreamTelemetry:
    def test_stream_reports_chunks_and_completed(self):
        reporter = MagicMock()
        backend = MagicMock()
        chunks = [
            GenerationChunk(text="Hello", token_count=1),
            GenerationChunk(text=" world", token_count=1),
            GenerationChunk(text="", finish_reason="stop"),
        ]
        backend.generate_stream = MagicMock(
            return_value=_async_chunks(*chunks)
        )

        meta = _make_metadata()
        model = Model(meta, _make_engine(backend), _reporter=reporter)

        collected = []

        async def _run():
            async for chunk in model.predict_stream(_make_request()):
                collected.append(chunk)

        asyncio.run(_run())

        assert len(collected) == 3

        reporter.report_generation_started.assert_called_once()

        # Two text chunks should produce two chunk_produced calls
        assert reporter.report_chunk_produced.call_count == 2
        first_chunk_call = reporter.report_chunk_produced.call_args_list[0].kwargs
        assert first_chunk_call["chunk_index"] == 0
        assert first_chunk_call["ttfc_ms"] is not None  # first chunk has ttfc

        second_chunk_call = reporter.report_chunk_produced.call_args_list[1].kwargs
        assert second_chunk_call["chunk_index"] == 1
        assert second_chunk_call["ttfc_ms"] is None  # only first has ttfc

        reporter.report_generation_completed.assert_called_once()
        comp_kwargs = reporter.report_generation_completed.call_args.kwargs
        assert comp_kwargs["total_chunks"] == 2
        assert comp_kwargs["total_duration_ms"] > 0

        reporter.report_generation_failed.assert_not_called()

    def test_stream_reports_failed_on_mid_stream_error(self):
        reporter = MagicMock()
        backend = MagicMock()
        chunks = [GenerationChunk(text="partial")]
        backend.generate_stream = MagicMock(
            return_value=_async_chunks_with_error(*chunks)
        )

        model = Model(_make_metadata(), _make_engine(backend), _reporter=reporter)

        async def _run():
            async for _ in model.predict_stream(_make_request()):
                pass

        with pytest.raises(RuntimeError, match="stream exploded"):
            asyncio.run(_run())

        reporter.report_generation_started.assert_called_once()
        reporter.report_generation_failed.assert_called_once()
        reporter.report_generation_completed.assert_not_called()


# ---------------------------------------------------------------------------
# No reporter — inference works, no crash
# ---------------------------------------------------------------------------


class TestNoReporter:
    @patch("octomil.get_reporter", return_value=None)
    def test_predict_works_without_reporter(self, _mock_get):
        backend = MagicMock()
        backend.generate.return_value = ("result", _make_metrics())

        model = Model(_make_metadata(), _make_engine(backend))
        result = model.predict(_make_request())

        assert result.text == "result"

    @patch("octomil.get_reporter", return_value=None)
    def test_predict_stream_works_without_reporter(self, _mock_get):
        backend = MagicMock()
        chunks = [GenerationChunk(text="ok", finish_reason="stop")]
        backend.generate_stream = MagicMock(
            return_value=_async_chunks(*chunks)
        )

        model = Model(_make_metadata(), _make_engine(backend))

        collected = []

        async def _run():
            async for chunk in model.predict_stream(_make_request()):
                collected.append(chunk)

        asyncio.run(_run())
        assert len(collected) == 1


# ---------------------------------------------------------------------------
# Reporter raises — exception swallowed, predict still returns
# ---------------------------------------------------------------------------


class TestReporterExceptionSwallowed:
    def test_predict_succeeds_when_reporter_raises(self):
        reporter = MagicMock()
        reporter.report_generation_started.side_effect = Exception("telemetry down")
        reporter.report_generation_completed.side_effect = Exception("telemetry down")

        backend = MagicMock()
        backend.generate.return_value = ("ok", _make_metrics())

        model = Model(_make_metadata(), _make_engine(backend), _reporter=reporter)
        result = model.predict(_make_request())

        assert result.text == "ok"

    def test_predict_stream_succeeds_when_reporter_raises(self):
        reporter = MagicMock()
        reporter.report_generation_started.side_effect = Exception("telemetry down")
        reporter.report_chunk_produced.side_effect = Exception("telemetry down")
        reporter.report_generation_completed.side_effect = Exception("telemetry down")

        backend = MagicMock()
        chunks = [GenerationChunk(text="ok", finish_reason="stop")]
        backend.generate_stream = MagicMock(
            return_value=_async_chunks(*chunks)
        )

        model = Model(_make_metadata(), _make_engine(backend), _reporter=reporter)

        collected = []

        async def _run():
            async for chunk in model.predict_stream(_make_request()):
                collected.append(chunk)

        asyncio.run(_run())
        assert len(collected) == 1


# ---------------------------------------------------------------------------
# Fallback to global reporter
# ---------------------------------------------------------------------------


class TestGlobalReporterFallback:
    def test_uses_global_reporter_when_no_override(self):
        global_reporter = MagicMock()
        backend = MagicMock()
        backend.generate.return_value = ("ok", _make_metrics())

        model = Model(_make_metadata(), _make_engine(backend))

        with patch("octomil.get_reporter", return_value=global_reporter):
            model.predict(_make_request())

        global_reporter.report_generation_started.assert_called_once()
        global_reporter.report_generation_completed.assert_called_once()

    def test_override_reporter_takes_precedence(self):
        global_reporter = MagicMock()
        override_reporter = MagicMock()
        backend = MagicMock()
        backend.generate.return_value = ("ok", _make_metrics())

        model = Model(
            _make_metadata(), _make_engine(backend), _reporter=override_reporter
        )

        with patch("octomil.get_reporter", return_value=global_reporter):
            model.predict(_make_request())

        override_reporter.report_generation_started.assert_called_once()
        global_reporter.report_generation_started.assert_not_called()
