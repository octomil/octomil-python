"""High-level Model wrapper with integrated telemetry.

Provides ``Model.predict()`` and ``Model.predict_stream()`` which delegate
to an ``EnginePlugin``-created backend while automatically reporting
inference lifecycle events via the ``TelemetryReporter``.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

if TYPE_CHECKING:
    from .telemetry import TelemetryReporter

from .engines.base import EnginePlugin
from .serve import GenerationChunk, GenerationRequest, InferenceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Descriptor for a model in the registry."""

    model_id: str
    name: str
    version: str
    experiment_id: Optional[str] = None


@dataclass
class Prediction:
    """Result of a non-streaming predict call."""

    text: str
    metrics: InferenceMetrics


class Model:
    """High-level model wrapper with telemetry.

    Parameters
    ----------
    metadata:
        Model identity (id, name, version).
    engine:
        Engine plugin used to create the inference backend.
    engine_kwargs:
        Extra keyword arguments forwarded to ``engine.create_backend()``.
    _reporter:
        Optional telemetry reporter.  Internal — callers should rely on
        the global reporter via ``octomil.init()`` instead.
    """

    def __init__(
        self,
        metadata: ModelMetadata,
        engine: EnginePlugin,
        engine_kwargs: dict[str, Any] | None = None,
        _reporter: TelemetryReporter | None = None,
    ) -> None:
        self.metadata = metadata
        self._engine = engine
        self._backend = engine.create_backend(metadata.name, **(engine_kwargs or {}))
        self._reporter_override = _reporter

    # ------------------------------------------------------------------
    # Reporter resolution
    # ------------------------------------------------------------------

    def _get_reporter(self) -> TelemetryReporter | None:
        """Return the reporter to use, falling back to the global one."""
        if self._reporter_override is not None:
            return self._reporter_override
        import octomil

        return octomil.get_reporter()

    # ------------------------------------------------------------------
    # predict (non-streaming)
    # ------------------------------------------------------------------

    def predict(self, request: GenerationRequest) -> Prediction:
        """Run inference and return a ``Prediction``.

        Automatically reports telemetry lifecycle events (started,
        completed, failed) when a reporter is available.
        """
        reporter = self._get_reporter()
        session_id = uuid.uuid4().hex
        model_id = self.metadata.model_id
        version = self.metadata.version

        if reporter:
            try:
                reporter.report_generation_started(
                    model_id=model_id,
                    version=version,
                    session_id=session_id,
                )
            except Exception:
                pass

        gen_start = time.monotonic()
        try:
            text, metrics = self._backend.generate(request)
        except Exception:
            if reporter:
                try:
                    reporter.report_generation_failed(
                        session_id=session_id,
                        model_id=model_id,
                        version=version,
                    )
                except Exception:
                    pass
            raise

        gen_elapsed_ms = (time.monotonic() - gen_start) * 1000

        if reporter:
            try:
                throughput = (
                    metrics.total_tokens / (gen_elapsed_ms / 1000)
                    if gen_elapsed_ms > 0
                    else 0.0
                )
                reporter.report_generation_completed(
                    session_id=session_id,
                    model_id=model_id,
                    version=version,
                    total_chunks=metrics.total_tokens,
                    total_duration_ms=gen_elapsed_ms,
                    ttfc_ms=metrics.ttfc_ms,
                    throughput=throughput,
                )
            except Exception:
                pass

            # Emit experiment metrics when model is part of an experiment
            if self.metadata.experiment_id:
                try:
                    reporter.report_experiment_metric(
                        experiment_id=self.metadata.experiment_id,
                        metric_name="inference.duration_ms",
                        metric_value=gen_elapsed_ms,
                    )
                    reporter.report_experiment_metric(
                        experiment_id=self.metadata.experiment_id,
                        metric_name="inference.ttfc_ms",
                        metric_value=metrics.ttfc_ms,
                    )
                    reporter.report_experiment_metric(
                        experiment_id=self.metadata.experiment_id,
                        metric_name="inference.throughput_tps",
                        metric_value=throughput,
                    )
                except Exception:
                    pass

        return Prediction(text=text, metrics=metrics)

    # ------------------------------------------------------------------
    # predict_stream (async generator)
    # ------------------------------------------------------------------

    async def predict_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationChunk]:
        """Stream inference results as an async generator.

        Reports ``generation_started`` before the first chunk,
        ``chunk_produced`` for each chunk, ``generation_completed``
        after the stream ends, and ``generation_failed`` on error.
        """
        reporter = self._get_reporter()
        session_id = uuid.uuid4().hex
        model_id = self.metadata.model_id
        version = self.metadata.version

        if reporter:
            try:
                reporter.report_generation_started(
                    model_id=model_id,
                    version=version,
                    session_id=session_id,
                )
            except Exception:
                pass

        chunk_index = 0
        stream_start = time.monotonic()
        first_chunk_time: float | None = None

        try:
            async for chunk in self._backend.generate_stream(request):
                now = time.monotonic()
                if first_chunk_time is None and chunk.text:
                    first_chunk_time = now

                if reporter and chunk.text:
                    try:
                        ttfc = (
                            (first_chunk_time - stream_start) * 1000
                            if first_chunk_time is not None and chunk_index == 0
                            else None
                        )
                        reporter.report_chunk_produced(
                            session_id=session_id,
                            model_id=model_id,
                            version=version,
                            chunk_index=chunk_index,
                            ttfc_ms=ttfc,
                        )
                    except Exception:
                        pass
                    chunk_index += 1

                yield chunk
        except Exception:
            if reporter:
                try:
                    reporter.report_generation_failed(
                        session_id=session_id,
                        model_id=model_id,
                        version=version,
                    )
                except Exception:
                    pass
            raise

        if reporter:
            try:
                total_duration_ms = (time.monotonic() - stream_start) * 1000
                ttfc_ms = (
                    (first_chunk_time - stream_start) * 1000
                    if first_chunk_time is not None
                    else 0.0
                )
                throughput = (
                    chunk_index / (total_duration_ms / 1000)
                    if total_duration_ms > 0
                    else 0.0
                )
                reporter.report_generation_completed(
                    session_id=session_id,
                    model_id=model_id,
                    version=version,
                    total_chunks=chunk_index,
                    total_duration_ms=total_duration_ms,
                    ttfc_ms=ttfc_ms,
                    throughput=throughput,
                )
            except Exception:
                pass

            # Emit experiment metrics when model is part of an experiment
            if self.metadata.experiment_id:
                try:
                    reporter.report_experiment_metric(
                        experiment_id=self.metadata.experiment_id,
                        metric_name="inference.duration_ms",
                        metric_value=total_duration_ms,
                    )
                    reporter.report_experiment_metric(
                        experiment_id=self.metadata.experiment_id,
                        metric_name="inference.ttfc_ms",
                        metric_value=ttfc_ms,
                    )
                    reporter.report_experiment_metric(
                        experiment_id=self.metadata.experiment_id,
                        metric_name="inference.throughput_tps",
                        metric_value=throughput,
                    )
                except Exception:
                    pass
