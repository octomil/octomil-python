"""ModelRuntimeRegistry — family-based runtime resolution."""

from __future__ import annotations

from typing import Optional

from octomil.runtime.core.model_runtime import ModelRuntime, RuntimeFactory


class ModelRuntimeRegistry:
    """Global registry for ModelRuntime factories.

    Resolution order: exact family -> prefix match -> default factory.
    """

    _instance: Optional[ModelRuntimeRegistry] = None

    def __init__(self) -> None:
        self._families: dict[str, RuntimeFactory] = {}
        self.default_factory: Optional[RuntimeFactory] = None

    @classmethod
    def shared(cls) -> ModelRuntimeRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, family: str, factory: RuntimeFactory) -> None:
        self._families[family.lower()] = factory

    def resolve(self, model_id: str) -> Optional[ModelRuntime]:
        lowered = model_id.lower()

        # 1. Exact family match
        if lowered in self._families:
            result = self._families[lowered](model_id)
            if result is not None:
                return result

        # 2. Prefix match — longest prefix wins
        prefix = max(
            (k for k in self._families if lowered.startswith(k)),
            key=len,
            default=None,
        )
        if prefix is not None:
            result = self._families[prefix](model_id)
            if result is not None:
                return result

        # 3. Default factory
        if self.default_factory is not None:
            return self.default_factory(model_id)

        return None

    def clear(self) -> None:
        self._families.clear()
        self.default_factory = None
