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
        "name": "test-model",
        "version": "1.0.0",
    }
    defaults.update(overrides)
    return ModelMetadata(**defaults)


def _make_engine(backend: MagicMock | None = None) -> MagicMock:
    engine = MagicMock()
    engine.name = "echo"
    engine.manages_own_download = False
    engine.create_backend.return_value = backend or MagicMock()
    return engine


def _make_request(**overrides: Any) -> GenerationRequest:
    defaults = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(overrides)
    return GenerationRequest(**defaults)


# ---------------------------------------------------------------------------
# ModelMetadata
# ---------------------------------------------------------------------------


class TestModelMetadata:
    def test_fields(self):
        meta = _make_metadata()
        assert meta.model_id == "model-abc"
        assert meta.name == "test-model"
        assert meta.version == "1.0.0"

    def test_custom_fields(self):
        meta = _make_metadata(model_id="xyz", version="2.0.0")
        assert meta.model_id == "xyz"
        assert meta.version == "2.0.0"


# ---------------------------------------------------------------------------
# Model properties
# ---------------------------------------------------------------------------


class TestModelProperties:
    def test_metadata_accessible(self):
        meta = _make_metadata()
        model = Model(metadata=meta, engine=_make_engine())

        assert model.metadata.model_id == "model-abc"
        assert model.metadata.name == "test-model"
        assert model.metadata.version == "1.0.0"


# ---------------------------------------------------------------------------
# Backend creation
# ---------------------------------------------------------------------------


class TestBackendCreation:
    def test_backend_created_eagerly(self):
        engine = _make_engine()
        Model(metadata=_make_metadata(), engine=engine)

        engine.create_backend.assert_called_once_with("test-model")

    def test_backend_reused_on_subsequent_calls(self):
        backend = MagicMock()
        backend.generate.return_value = ("result", InferenceMetrics())
        engine = _make_engine(backend)

        model = Model(metadata=_make_metadata(), engine=engine)
        model.predict(_make_request())
        model.predict(_make_request())

        engine.create_backend.assert_called_once()
        assert backend.generate.call_count == 2

    def test_engine_kwargs_forwarded(self):
        engine = _make_engine()

        Model(
            metadata=_make_metadata(),
            engine=engine,
            engine_kwargs={"cache_size_mb": 4096, "cache_enabled": True},
        )

        engine.create_backend.assert_called_once_with(
            "test-model", cache_size_mb=4096, cache_enabled=True
        )


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_is_dataclass(self):
        p = Prediction(text="Hello", metrics=InferenceMetrics())
        assert isinstance(p, Prediction)
        assert p.text == "Hello"

    def test_metrics_attribute(self):
        m = InferenceMetrics(tokens_per_second=42.0, total_tokens=10)
        p = Prediction(text="text", metrics=m)
        assert p.metrics is m
        assert p.metrics.tokens_per_second == 42.0

    def test_repr(self):
        p = Prediction(text="hi", metrics=InferenceMetrics())
        assert "Prediction(" in repr(p)

    def test_text_ops(self):
        p = Prediction(text="Hello world", metrics=InferenceMetrics())
        assert p.text.upper() == "HELLO WORLD"
        assert p.text.split() == ["Hello", "world"]
        assert f"said: {p.text}" == "said: Hello world"


class TestPredict:
    def test_returns_prediction(self):
        backend = MagicMock()
        metrics = InferenceMetrics(tokens_per_second=42.0, total_tokens=10)
        backend.generate.return_value = ("Hello world", metrics)
        engine = _make_engine(backend)

        model = Model(metadata=_make_metadata(), engine=engine)
        result = model.predict(_make_request())

        assert isinstance(result, Prediction)
        assert result.text == "Hello world"
        assert result.metrics.tokens_per_second == 42.0
        assert result.metrics.total_tokens == 10

    def test_passes_request_to_backend(self):
        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        engine = _make_engine(backend)

        model = Model(metadata=_make_metadata(), engine=engine)
        req = _make_request(max_tokens=256, temperature=0.5, top_p=0.9)
        model.predict(req)

        call_args = backend.generate.call_args[0]
        request = call_args[0]
        assert isinstance(request, GenerationRequest)
        assert request.model == "test-model"
        assert request.max_tokens == 256
        assert request.temperature == 0.5
        assert request.top_p == 0.9
        assert request.stream is False

    def test_default_request_params(self):
        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        engine = _make_engine(backend)

        model = Model(metadata=_make_metadata(), engine=engine)
        req = _make_request()
        model.predict(req)

        request = backend.generate.call_args[0][0]
        assert request.max_tokens == 512  # GenerationRequest default
        assert request.temperature == 0.7
        assert request.top_p == 1.0


# ---------------------------------------------------------------------------
# predict_stream()
# ---------------------------------------------------------------------------


class TestPredictStream:
    def test_yields_chunks(self):
        backend = MagicMock()
        chunks = [
            GenerationChunk(text="Hello"),
            GenerationChunk(text=" world", finish_reason="stop"),
        ]

        async def fake_stream(request):
            for c in chunks:
                yield c

        backend.generate_stream = fake_stream
        engine = _make_engine(backend)

        model = Model(metadata=_make_metadata(), engine=engine)

        async def run():
            collected = []
            async for chunk in model.predict_stream(_make_request()):
                collected.append(chunk)
            return collected

        result = asyncio.run(run())
        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == " world"
        assert result[1].finish_reason == "stop"

    def test_stream_passes_request(self):
        backend = MagicMock()
        captured_request = {}

        async def fake_stream(request):
            captured_request["req"] = request
            return
            yield  # make it an async generator

        backend.generate_stream = fake_stream
        engine = _make_engine(backend)

        model = Model(metadata=_make_metadata(), engine=engine)

        async def run():
            req = _make_request(max_tokens=128, temperature=0.3, top_p=0.8)
            async for _ in model.predict_stream(req):
                pass

        asyncio.run(run())

        req = captured_request["req"]
        assert req.model == "test-model"
        assert req.max_tokens == 128
        assert req.temperature == 0.3
        assert req.top_p == 0.8
        assert req.stream is False  # stream flag is on the request as-is


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

        with patch("octomil.engines.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            model = c.load_model("my-model")

        assert isinstance(model, Model)
        assert model.metadata.model_id == "model-123"
        assert model.metadata.name == "my-model"
        assert model.metadata.version == "2.0.0"

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_load_model_with_version(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-456"
        mock_registry.download.return_value = {"model_path": "/tmp/model.onnx"}

        fake_engine = _make_engine()
        fake_engine.name = "llama.cpp"

        with patch("octomil.engines.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            model = c.load_model(
                "my-model",
                version="3.0.0",
                engine="llama.cpp",
            )

        assert model.metadata.version == "3.0.0"

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

        with patch("octomil.engines.get_registry") as mock_get_reg:
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

        backend = MagicMock()
        metrics = InferenceMetrics(tokens_per_second=99.0)
        backend.generate.return_value = ("Four.", metrics)
        fake_engine = _make_engine(backend)

        with patch("octomil.engines.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            result = c.predict(
                "my-model",
                [{"role": "user", "content": "What is 2+2?"}],
            )

        assert isinstance(result, Prediction)
        assert result.text == "Four."
        assert result.metrics.tokens_per_second == 99.0

    @patch("octomil.client.RolloutsAPI")
    @patch("octomil.client.ModelRegistry")
    @patch("octomil.client._ApiClient")
    def test_predict_caches_model(self, mock_api, mock_registry_cls, mock_rollouts):
        from octomil.client import Client

        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve_model_id.return_value = "model-1"
        mock_registry.get_latest_version.return_value = "1.0.0"
        mock_registry.download.return_value = {"model_path": "/tmp/m.gguf"}

        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        fake_engine = _make_engine(backend)

        with patch("octomil.engines.get_registry") as mock_get_reg:
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

        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        fake_engine = _make_engine(backend)

        with patch("octomil.engines.get_registry") as mock_get_reg:
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

        backend = MagicMock()
        backend.generate.return_value = ("ok", InferenceMetrics())
        fake_engine = _make_engine(backend)

        with patch("octomil.engines.get_registry") as mock_get_reg:
            mock_reg_instance = MagicMock()
            mock_reg_instance.auto_select.return_value = (fake_engine, [])
            mock_get_reg.return_value = mock_reg_instance

            c = Client(api_key="key")
            c.predict("my-model", [{"role": "user", "content": "a"}])
            assert len(c._models) == 1

            c.dispose()
            assert len(c._models) == 0
