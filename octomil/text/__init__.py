"""OctomilText — text prediction namespace on OctomilClient."""

from __future__ import annotations

from typing import Callable, Optional

from octomil._generated.message_role import MessageRole
from octomil._generated.model_capability import ModelCapability
from octomil.model_ref import ModelRef, ModelRefFactory, _ModelRefCapability, _ModelRefId
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import GenerationConfig, RuntimeContentPart, RuntimeMessage, RuntimeRequest
from octomil.text.predictor import OctomilPredictor


class OctomilText:
    """Namespace for text prediction APIs on OctomilClient.

    Usage::

        suggestions = await client.text.predict("The quick brown")

        # Or create a stateful predictor
        predictor = client.text.predictor()
    """

    def __init__(
        self,
        runtime_resolver: Callable[[ModelRef], Optional[ModelRuntime]],
    ) -> None:
        self._runtime_resolver = runtime_resolver

    async def predict(
        self,
        prefix: str,
        *,
        model: Optional[ModelRef] = None,
        max_suggestions: int = 3,
    ) -> list[str]:
        """Generate text completion suggestions.

        Args:
            prefix: The text typed so far.
            model: Model reference. Defaults to text_completion capability.
            max_suggestions: Maximum suggestions to return (default: 3).

        Returns:
            List of completion suggestions.
        """
        ref = model or ModelRefFactory.capability(ModelCapability.TEXT_COMPLETION)
        runtime = self._runtime_resolver(ref)
        if runtime is None:
            raise RuntimeError("No runtime available for text prediction model")

        request = RuntimeRequest(
            messages=[RuntimeMessage(role=MessageRole.USER, parts=[RuntimeContentPart.text_part(prefix)])],
            generation_config=GenerationConfig(max_tokens=32, temperature=0.3),
        )
        response = await runtime.run(request)
        raw = [line.strip() for line in response.text.split("\n") if line.strip()]
        return raw[:max_suggestions]

    def predictor(
        self,
        capability: ModelCapability = ModelCapability.TEXT_COMPLETION,
    ) -> Optional[OctomilPredictor]:
        """Create a stateful predictor that keeps the model warm.

        Args:
            capability: The model capability to use (default: TEXT_COMPLETION).

        Returns:
            An OctomilPredictor instance, or None if no runtime is available.
        """
        ref = ModelRefFactory.capability(capability)
        runtime = self._runtime_resolver(ref)
        if runtime is None:
            return None
        return OctomilPredictor(runtime=runtime, model_id=capability.value)

    def predictor_for(self, ref: ModelRef) -> Optional[OctomilPredictor]:
        """Create a stateful predictor for a specific model reference.

        Args:
            ref: The model reference.

        Returns:
            An OctomilPredictor instance, or None if no runtime is available.
        """
        runtime = self._runtime_resolver(ref)
        if runtime is None:
            return None
        if isinstance(ref, _ModelRefId):
            model_id = ref.model_id
        elif isinstance(ref, _ModelRefCapability):
            model_id = ref.capability.value
        else:
            model_id = str(ref)
        return OctomilPredictor(runtime=runtime, model_id=model_id)


__all__ = [
    "OctomilText",
    "OctomilPredictor",
]
