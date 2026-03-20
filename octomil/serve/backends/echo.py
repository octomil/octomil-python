"""Fallback backend that echoes input -- useful for testing the API layer."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from ..types import GenerationChunk, GenerationRequest, InferenceBackend, InferenceMetrics

logger = logging.getLogger(__name__)


class EchoBackend(InferenceBackend):
    """Fallback backend that echoes input -- useful for testing the API layer."""

    name = "echo"

    def __init__(self) -> None:
        super().__init__()
        self._model_name: str = ""

    def load_model(self, model_name: str) -> None:
        self._model_name = model_name
        logger.warning(
            "No inference backend available. Using echo backend for '%s'. "
            "Install mlx-lm (Apple Silicon) or llama-cpp-python for real inference: "
            "pip install 'octomil[mlx]' or pip install 'octomil[llama]'",
            model_name,
        )

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        last_msg = request.messages[-1]["content"] if request.messages else ""
        text = f"[echo:{self._model_name}] {last_msg}"
        metrics = InferenceMetrics(total_tokens=len(text.split()))
        return text, metrics

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        import asyncio

        last_msg = request.messages[-1]["content"] if request.messages else ""
        words = f"[echo:{self._model_name}] {last_msg}".split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield GenerationChunk(
                text=word + ("" if is_last else " "),
                finish_reason="stop" if is_last else None,
            )
            await asyncio.sleep(0.02)

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []
