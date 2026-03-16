"""Tests for octomil.text — OctomilText namespace."""

from __future__ import annotations

from typing import AsyncIterator, Optional

import pytest

from octomil.model_ref import ModelRef, ModelRefFactory
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
)
from octomil.text import OctomilText
from octomil.text.predictor import OctomilPredictor


class _MockPredictionRuntime(ModelRuntime):
    """Mock runtime that returns fixed prediction suggestions."""

    def __init__(self, suggestions: str = "fox\njumped\nover") -> None:
        self._suggestions = suggestions

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(text=self._suggestions)

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        yield RuntimeChunk(text=self._suggestions)


class TestOctomilText:
    @pytest.mark.asyncio
    async def test_predict(self) -> None:
        mock_runtime = _MockPredictionRuntime("fox\njumped\nover")

        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return mock_runtime

        text = OctomilText(runtime_resolver=resolver)
        suggestions = await text.predict("The quick brown")

        assert suggestions == ["fox", "jumped", "over"]

    @pytest.mark.asyncio
    async def test_predict_max_suggestions(self) -> None:
        mock_runtime = _MockPredictionRuntime("a\nb\nc\nd\ne")

        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return mock_runtime

        text = OctomilText(runtime_resolver=resolver)
        suggestions = await text.predict("test", max_suggestions=2)

        assert len(suggestions) == 2

    @pytest.mark.asyncio
    async def test_predict_no_runtime_raises(self) -> None:
        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return None

        text = OctomilText(runtime_resolver=resolver)
        with pytest.raises(RuntimeError, match="No runtime"):
            await text.predict("test")

    def test_predictor_returns_instance(self) -> None:
        mock_runtime = _MockPredictionRuntime()

        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return mock_runtime

        text = OctomilText(runtime_resolver=resolver)
        predictor = text.predictor()
        assert isinstance(predictor, OctomilPredictor)

    def test_predictor_no_runtime_returns_none(self) -> None:
        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return None

        text = OctomilText(runtime_resolver=resolver)
        predictor = text.predictor()
        assert predictor is None

    def test_predictor_for_returns_instance(self) -> None:
        mock_runtime = _MockPredictionRuntime()

        def resolver(ref: ModelRef) -> Optional[ModelRuntime]:
            return mock_runtime

        text = OctomilText(runtime_resolver=resolver)
        ref = ModelRefFactory.id("custom-model")
        predictor = text.predictor_for(ref)
        assert isinstance(predictor, OctomilPredictor)


class TestOctomilPredictor:
    @pytest.mark.asyncio
    async def test_predict(self) -> None:
        mock_runtime = _MockPredictionRuntime("suggestion1\nsuggestion2")
        predictor = OctomilPredictor(runtime=mock_runtime, model_id="test")

        suggestions = await predictor.predict("prefix")
        assert suggestions == ["suggestion1", "suggestion2"]

    @pytest.mark.asyncio
    async def test_predict_max_suggestions(self) -> None:
        mock_runtime = _MockPredictionRuntime("a\nb\nc")
        predictor = OctomilPredictor(runtime=mock_runtime, model_id="test")

        suggestions = await predictor.predict("prefix", max_suggestions=1)
        assert len(suggestions) == 1

    def test_close(self) -> None:
        mock_runtime = _MockPredictionRuntime()
        predictor = OctomilPredictor(runtime=mock_runtime, model_id="test")
        predictor.close()

    def test_context_manager(self) -> None:
        mock_runtime = _MockPredictionRuntime()
        with OctomilPredictor(runtime=mock_runtime, model_id="test") as predictor:
            assert predictor is not None
