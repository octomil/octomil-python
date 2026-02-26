"""Tests for octomil.model — Model class and Client.load_model()."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from octomil.model import Model, ModelMetadata, Prediction
from octomil.serve import GenerationChunk, GenerationRequest, InferenceMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(**overrides: Any) -> ModelMetadata:
    defaults = {
        "model_id": "model-abc",
        "model_name": "test-model",
        "version": "1.0.0",
        "format": "gguf",
        "model_path": "/tmp/test-model.gguf",
        "engine_name": "echo",
    }
    defaults.update(overrides)
    return ModelMetadata(**defaults)


def _make_engine() -> MagicMock:
    engine = MagicMock()
    engine.name = "echo"
    return engine


# ---------------------------------------------------------------------------
# ModelMetadata
# ---------------------------------------------------------------------------


class TestModelMetadata:
    def test_fields(self):
        meta = _make_metadata()
        assert meta.model_id == "model-abc"
        assert meta.model_name == "test-model"
        assert meta.version == "1.0.0"
        assert meta.format == "gguf"
        assert meta.model_path == "/tmp/test-model.gguf"
        assert meta.engine_name == "echo"

    def test_custom_fields(self):
        meta = _make_metadata(model_id="xyz", version="2.0.0", format="onnx")
        assert meta.model_id == "xyz"
        assert meta.version == "2.0.0"
        assert meta.format == "onnx"


# ---------------------------------------------------------------------------
# Model properties
# ---------------------------------------------------------------------------


class TestModelProperties:
    def test_properties_delegate_to_metadata(self):
        meta = _make_metadata()
        model = Model(metadata=meta, engine=_make_engine())

        assert model.model_id == "model-abc"
        assert model.model_name == "test-model"
        assert model.version == "1.0.0"
        assert model.format == "gguf"
        assert model.model_path == "/tmp/test-model.gguf"
        assert model.engine_name == "echo"

    def test_is_loaded_false_initially(self):
        model = Model(metadata=_make_metadata(), engine=_make_engine())
        assert model.is_loaded is False

    def test_is_loaded_true_after_predict(self):
        engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("hello", InferenceMetrics())
        engine.create_backend.return_value = backend

        model = Model(metadata=_make_metadata(), engine=engine)
        model.predict([{"role": "user", "content": "hi"}])
        assert model.is_loaded is True

    def test_is_loaded_false_after_dispose(self):
        engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("hello", InferenceMetrics())
        engine.create_backend.return_value = backend

        model = Model(metadata=_make_metadata(), engine=engine)
        model.predict([{"role": "user", "content": "hi"}])
        assert model.is_loaded is True

        model.dispose()
        assert model.is_loaded is False


# ---------------------------------------------------------------------------
# Lazy backend loading
# ---------------------------------------------------------------------------


class TestLazyLoading:
    def test_backend_not_created_until_predict(self):
        engine = _make_engine()
        model = Model(metadata=_make_metadata(), engine=engine)

        engine.create_backend.assert_not_called()
        assert model._backend is None

    def test_backend_created_on_first_predict(self):
        engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("result", InferenceMetrics())
        engine.create_backend.return_value = backend

        meta = _make_metadata()
        model = Model(metadata=meta, engine=engine)
        model.predict([{"role": "user", "content": "test"}])

        engine.create_backend.assert_called_once_with(
            "test-model",
        )
        backend.load_model.assert_called_once_with("test-model")

    def test_backend_reused_on_subsequent_calls(self):
        engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("result", InferenceMetrics())
        engine.create_backend.return_value = backend

        model = Model(metadata=_make_metadata(), engine=engine)
        model.predict([{"role": "user", "content": "a"}])
        model.predict([{"role": "user", "content": "b"}])

        engine.create_backend.assert_called_once()
        backend.load_model.assert_called_once()
        assert backend.generate.call_count == 2

    def test_engine_kwargs_forwarded(self):
        engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        engine.create_backend.return_value = backend

        model = Model(
            metadata=_make_metadata(),
            engine=engine,
            engine_kwargs={"cache_size_mb": 4096, "cache_enabled": True},
        )
        model.predict([{"role": "user", "content": "hi"}])

        engine.create_backend.assert_called_once_with(
            "test-model", cache_size_mb=4096, cache_enabled=True
        )


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_is_str(self):
        p = Prediction("Hello", InferenceMetrics())
        assert isinstance(p, str)
        assert p == "Hello"

    def test_metrics_attribute(self):
        m = InferenceMetrics(tokens_per_second=42.0, total_tokens=10)
        p = Prediction("text", m)
        assert p.metrics is m
        assert p.metrics.tokens_per_second == 42.0

    def test_repr(self):
        p = Prediction("hi", InferenceMetrics())
        assert "Prediction(" in repr(p)

    def test_str_ops(self):
        p = Prediction("Hello world", InferenceMetrics())
        assert p.upper() == "HELLO WORLD"
        assert p.split() == ["Hello", "world"]
        assert f"said: {p}" == "said: Hello world"


class TestPredict:
    def test_returns_prediction(self):
        engine = _make_engine()
        backend = MagicMock()
        metrics = InferenceMetrics(tokens_per_second=42.0, total_tokens=10)
        backend.generate.return_value = ("Hello world", metrics)
        engine.create_backend.return_value = backend

        model = Model(metadata=_make_metadata(), engine=engine)
        text = model.predict([{"role": "user", "content": "hi"}])

        assert isinstance(text, Prediction)
        assert text == "Hello world"
        assert text.metrics.tokens_per_second == 42.0
        assert text.metrics.total_tokens == 10

    def test_passes_params_to_request(self):
        engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        engine.create_backend.return_value = backend

        model = Model(metadata=_make_metadata(), engine=engine)
        model.predict(
            [{"role": "user", "content": "test"}],
            max_tokens=256,
            temperature=0.5,
            top_p=0.9,
        )

        call_args = backend.generate.call_args[0]
        request = call_args[0]
        assert isinstance(request, GenerationRequest)
        assert request.model == "test-model"
        assert request.max_tokens == 256
        assert request.temperature == 0.5
        assert request.top_p == 0.9
        assert request.stream is False

    def test_default_params(self):
        engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        engine.create_backend.return_value = backend

        model = Model(metadata=_make_metadata(), engine=engine)
        model.predict([{"role": "user", "content": "test"}])

        request = backend.generate.call_args[0][0]
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 1.0


# ---------------------------------------------------------------------------
# predict_stream()
# ---------------------------------------------------------------------------


class TestPredictStream:
    def test_yields_chunks(self):
        engine = _make_engine()
        backend = MagicMock()
        chunks = [
            GenerationChunk(text="Hello"),
            GenerationChunk(text=" world", finish_reason="stop"),
        ]

        async def fake_stream(request):
            for c in chunks:
                yield c

        backend.generate_stream = fake_stream
        engine.create_backend.return_value = backend

        model = Model(metadata=_make_metadata(), engine=engine)

        async def run():
            collected = []
            async for chunk in model.predict_stream(
                [{"role": "user", "content": "hi"}]
            ):
                collected.append(chunk)
            return collected

        result = asyncio.get_event_loop().run_until_complete(run())
        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == " world"
        assert result[1].finish_reason == "stop"

    def test_stream_passes_params(self):
        engine = _make_engine()
        backend = MagicMock()
        captured_request = {}

        async def fake_stream(request):
            captured_request["req"] = request
            return
            yield  # make it an async generator

        backend.generate_stream = fake_stream
        engine.create_backend.return_value = backend

        model = Model(metadata=_make_metadata(), engine=engine)

        async def run():
            async for _ in model.predict_stream(
                [{"role": "user", "content": "test"}],
                max_tokens=128,
                temperature=0.3,
                top_p=0.8,
            ):
                pass

        asyncio.get_event_loop().run_until_complete(run())

        req = captured_request["req"]
        assert req.model == "test-model"
        assert req.max_tokens == 128
        assert req.temperature == 0.3
        assert req.top_p == 0.8
        assert req.stream is True


# ---------------------------------------------------------------------------
# dispose()
# ---------------------------------------------------------------------------


class TestDispose:
    def test_dispose_prevents_predict(self):
        model = Model(metadata=_make_metadata(), engine=_make_engine())
        model.dispose()

        with pytest.raises(RuntimeError, match="Model has been disposed"):
            model.predict([{"role": "user", "content": "hi"}])

    def test_dispose_prevents_predict_stream(self):
        model = Model(metadata=_make_metadata(), engine=_make_engine())
        model.dispose()

        async def run():
            async for _ in model.predict_stream([{"role": "user", "content": "hi"}]):
                pass

        with pytest.raises(RuntimeError, match="Model has been disposed"):
            asyncio.get_event_loop().run_until_complete(run())

    def test_dispose_after_use(self):
        engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        engine.create_backend.return_value = backend

        model = Model(metadata=_make_metadata(), engine=engine)
        model.predict([{"role": "user", "content": "hi"}])
        assert model.is_loaded is True

        model.dispose()
        assert model.is_loaded is False
        assert model._backend is None

        with pytest.raises(RuntimeError, match="Model has been disposed"):
            model.predict([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Client.load_model() integration
# ---------------------------------------------------------------------------


class TestClientLoadModel:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_load_model_basic(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-123"
        mock_registry.get_latest_version.return_value = "2.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/model.gguf"}

        fake_engine = _make_engine()
        fake_engine.name = "mlx-lm"

        with patch("octomil.engines.registry.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            model = c.load_model("my-model")

        assert isinstance(model, Model)
        assert model.model_id == "model-123"
        assert model.model_name == "my-model"
        assert model.version == "2.0.0"
        assert model.format == "auto"
        assert model.model_path == "/tmp/model.gguf"
        assert model.engine_name == "mlx-lm"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_load_model_with_options(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-456"
        mock_registry.download.return_value = {"model_path": "/tmp/model.onnx"}

        fake_engine = _make_engine()
        fake_engine.name = "llama.cpp"

        with patch("octomil.engines.registry.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            model = c.load_model(
                "my-model",
                version="3.0.0",
                format="onnx",
                engine="llama.cpp",
                cache_size_mb=4096,
                cache_enabled=False,
            )

        assert model.version == "3.0.0"
        assert model.format == "onnx"
        assert model.engine_name == "llama.cpp"
        assert model._engine_kwargs == {
            "cache_size_mb": 4096,
            "cache_enabled": False,
        }

        mock_reg_instance.auto_select.assert_called_once_with(
            "my-model", engine_override="llama.cpp"
        )

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_load_model_calls_pull(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-789"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.gguf"}

        fake_engine = _make_engine()

        with patch("octomil.engines.registry.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            c.load_model("my-model", destination="/opt/models")

        mock_registry.download.assert_called_once_with(
            model_id="model-789",
            version="1.0.0",
            destination="/opt/models",
            format=None,
        )


# ---------------------------------------------------------------------------
# Client.predict() — one-call DX
# ---------------------------------------------------------------------------


class TestClientPredict:
    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_predict_one_call(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-1"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.gguf"}

        fake_engine = _make_engine()
        backend = MagicMock()
        metrics = InferenceMetrics(tokens_per_second=99.0)
        backend.generate.return_value = ("Four.", metrics)
        fake_engine.create_backend.return_value = backend

        with patch("octomil.engines.registry.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            text = c.predict(
                "my-model",
                [{"role": "user", "content": "What is 2+2?"}],
            )

        assert isinstance(text, Prediction)
        assert text == "Four."
        assert text.metrics.tokens_per_second == 99.0

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_predict_caches_model(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-1"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.gguf"}

        fake_engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        fake_engine.create_backend.return_value = backend

        with patch("octomil.engines.registry.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            c.predict("my-model", [{"role": "user", "content": "a"}])
            c.predict("my-model", [{"role": "user", "content": "b"}])

        # load_model called once (model cached), but generate called twice
        mock_registry.download.assert_called_once()
        assert backend.generate.call_count == 2

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_predict_passes_params(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-1"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.gguf"}

        fake_engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        fake_engine.create_backend.return_value = backend

        with patch("octomil.engines.registry.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            c.predict(
                "my-model",
                [{"role": "user", "content": "test"}],
                max_tokens=128,
                temperature=0.3,
            )

        request = backend.generate.call_args[0][0]
        assert request.max_tokens == 128
        assert request.temperature == 0.3

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_dispose_clears_models(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-1"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.gguf"}

        fake_engine = _make_engine()
        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        fake_engine.create_backend.return_value = backend

        with patch("octomil.engines.registry.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            c.predict("my-model", [{"role": "user", "content": "a"}])
            assert len(c._models) == 1

            c.dispose()
            assert len(c._models) == 0
