"""OctomilPredictor — stateful text predictor."""

from __future__ import annotations

from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import RuntimeRequest


class OctomilPredictor:
    """Stateful text predictor that keeps the model warm between calls.

    Created via ``OctomilText.predictor()``.

    Usage::

        predictor = client.text.predictor()
        suggestions = await predictor.predict("The quick brown")
        predictor.close()

    Or as a context manager::

        with client.text.predictor() as p:
            suggestions = await p.predict("The quick brown")
    """

    def __init__(self, runtime: ModelRuntime, model_id: str) -> None:
        self._runtime = runtime
        self._model_id = model_id

    async def predict(self, prefix: str, max_suggestions: int = 3) -> list[str]:
        """Generate text completions for the given prefix.

        Args:
            prefix: The text typed so far.
            max_suggestions: Maximum number of suggestions (default: 3).

        Returns:
            List of completion suggestions.
        """
        request = RuntimeRequest(
            prompt=prefix,
            max_tokens=32,
            temperature=0.3,
        )
        response = await self._runtime.run(request)
        raw = [line.strip() for line in response.text.split("\n") if line.strip()]
        return raw[:max_suggestions]

    def close(self) -> None:
        """Release the warm model resources."""
        self._runtime.close()

    def __enter__(self) -> OctomilPredictor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
