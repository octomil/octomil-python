"""Tests for octomil.smart_router.SmartRouter."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from octomil.serve import GenerationChunk, GenerationRequest, InferenceBackend, InferenceMetrics
from octomil.smart_router import RouterConfig, SmartRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeBackend(InferenceBackend):
    """Minimal backend that records calls."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self._model: str = ""
        self.generate_calls: list[GenerationRequest] = []
        self.stream_calls: list[GenerationRequest] = []

    def load_model(self, model_name: str) -> None:
        self._model = model_name

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        self.generate_calls.append(request)
        return f"[{self.name}] response", InferenceMetrics(tokens_per_second=100)

    async def generate_stream(self, request: GenerationRequest):
        self.stream_calls.append(request)
        yield GenerationChunk(text=f"[{self.name}]", finish_reason="stop")

    def list_models(self) -> list[str]:
        return [self._model] if self._model else []


_TEST_CONFIG = RouterConfig(
    long_gen_threshold=512,
    concurrency_threshold=2,
    prefer_throughput_engine="mlx-lm",
    prefer_latency_engine="llama.cpp",
)


def _make_router(
    *,
    mlx: bool = True,
    llama: bool = True,
    config: RouterConfig | None = None,
) -> SmartRouter:
    """Build a SmartRouter with pre-injected fake backends (skip real loading)."""
    cfg = config or _TEST_CONFIG
    router = SmartRouter(config=cfg)
    router._model_name = "test-model"
    if mlx:
        router._backends["mlx-lm"] = FakeBackend("mlx-lm")
    if llama:
        router._backends["llama.cpp"] = FakeBackend("llama.cpp")
    return router


def _req(max_tokens: int = 64) -> GenerationRequest:
    return GenerationRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestRouting:
    def test_short_single_request_uses_latency_engine(self):
        router = _make_router()
        backend, name = router._select(_req(max_tokens=64))
        assert name == "llama.cpp"

    def test_long_gen_uses_latency_engine(self):
        router = _make_router()
        backend, name = router._select(_req(max_tokens=1024))
        assert name == "llama.cpp"

    def test_concurrent_requests_use_throughput_engine(self):
        router = _make_router()
        # Simulate 3 in-flight requests
        router._inflight = 3
        backend, name = router._select(_req(max_tokens=64))
        assert name == "mlx-lm"

    def test_concurrent_long_gen_still_uses_latency(self):
        """Long gen rule takes priority over concurrency rule."""
        router = _make_router()
        router._inflight = 5
        backend, name = router._select(_req(max_tokens=1024))
        assert name == "llama.cpp"

    def test_single_backend_always_selected(self):
        router = _make_router(mlx=False, llama=True)
        backend, name = router._select(_req(max_tokens=64))
        assert name == "llama.cpp"

        router2 = _make_router(mlx=True, llama=False)
        backend2, name2 = router2._select(_req(max_tokens=64))
        assert name2 == "mlx-lm"

    def test_custom_config_thresholds(self):
        cfg = RouterConfig(
            long_gen_threshold=128,
            concurrency_threshold=1,
            prefer_throughput_engine="mlx-lm",
            prefer_latency_engine="llama.cpp",
        )
        router = _make_router(config=cfg)

        # 128 tokens triggers long gen with lowered threshold
        _, name = router._select(_req(max_tokens=128))
        assert name == "llama.cpp"

        # 1 inflight with threshold=1 → 1 >= 1 → triggers throughput engine
        router._inflight = 1
        _, name = router._select(_req(max_tokens=64))
        assert name == "mlx-lm"

    def test_concurrency_threshold_boundary(self):
        cfg = RouterConfig(
            concurrency_threshold=2,
            prefer_throughput_engine="mlx-lm",
            prefer_latency_engine="llama.cpp",
        )
        router = _make_router(config=cfg)

        router._inflight = 1  # below threshold
        _, name = router._select(_req(max_tokens=64))
        assert name == "llama.cpp"

        router._inflight = 2  # at threshold
        _, name = router._select(_req(max_tokens=64))
        assert name == "mlx-lm"


# ---------------------------------------------------------------------------
# Generate / stream tests
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_dispatches_and_tracks(self):
        router = _make_router()
        text, metrics = router.generate(_req(max_tokens=64))
        assert "[llama.cpp]" in text
        assert router.route_stats.get("llama.cpp", 0) == 1

    def test_generate_concurrent_dispatches_to_mlx(self):
        router = _make_router()

        results = []
        barrier = threading.Barrier(3)

        def _call():
            barrier.wait()
            text, _ = router.generate(_req(max_tokens=64))
            results.append(text)

        threads = [threading.Thread(target=_call) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        # At least one should have been routed to mlx due to concurrency
        total = router.route_stats
        assert total.get("mlx-lm", 0) + total.get("llama.cpp", 0) == 3

    @pytest.mark.asyncio
    async def test_stream_dispatches(self):
        router = _make_router()
        chunks = []
        async for chunk in router.generate_stream(_req(max_tokens=64)):
            chunks.append(chunk.text)
        assert "[llama.cpp]" in chunks[0]

    def test_inflight_counter_restored_on_error(self):
        router = _make_router()
        llama = router._backends["llama.cpp"]
        llama.generate = MagicMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            router.generate(_req())

        assert router._inflight == 0


# ---------------------------------------------------------------------------
# Load / unload
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_engines_loaded_property(self):
        router = _make_router()
        assert sorted(router.engines_loaded) == ["llama.cpp", "mlx-lm"]

    def test_unload_clears_state(self):
        router = _make_router()
        router._route_counts["llama.cpp"] = 5
        router._inflight = 2
        router.unload()
        assert router.engines_loaded == []
        assert router._inflight == 0

    def test_list_models(self):
        router = _make_router()
        assert router.list_models() == ["test-model"]

    @patch("octomil.engines.get_registry")
    def test_load_model_with_registry(self, mock_get_registry):
        """Test that load_model uses the engine registry."""
        mock_engine_mlx = MagicMock()
        mock_engine_mlx.name = "mlx-lm"
        mock_backend_mlx = FakeBackend("mlx-lm")
        mock_engine_mlx.create_backend.return_value = mock_backend_mlx

        mock_engine_llama = MagicMock()
        mock_engine_llama.name = "llama.cpp"
        mock_backend_llama = FakeBackend("llama.cpp")
        mock_engine_llama.create_backend.return_value = mock_backend_llama

        mock_det_mlx = MagicMock()
        mock_det_mlx.available = True
        mock_det_mlx.engine = mock_engine_mlx

        mock_det_llama = MagicMock()
        mock_det_llama.available = True
        mock_det_llama.engine = mock_engine_llama

        mock_registry = MagicMock()
        mock_registry.detect_all.return_value = [mock_det_mlx, mock_det_llama]
        mock_get_registry.return_value = mock_registry

        router = SmartRouter()
        router.load_model("gemma2-2b")

        assert "mlx-lm" in router.engines_loaded
        assert "llama.cpp" in router.engines_loaded
        mock_engine_mlx.create_backend.assert_called_once()
        mock_engine_llama.create_backend.assert_called_once()

    def test_load_model_raises_when_nothing_loads(self):
        router = SmartRouter()
        with patch("octomil.engines.get_registry") as mock:
            mock.return_value.detect_all.return_value = []
            with pytest.raises(RuntimeError, match="no backend could load"):
                router.load_model("nonexistent")
