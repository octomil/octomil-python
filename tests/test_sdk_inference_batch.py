"""Validate that octomil inference and batch work end-to-end from Python.

Tests the full SDK surface for inference (sync, async, streaming) and
batch (queue submit, stream, overflow, stats) using mocked backends so
no real model download or GPU is required.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from octomil.auth import OrgApiKeyAuth
from octomil.batch import (
    QueueStats,
    RequestQueue,
)
from octomil.client import OctomilClient
from octomil.model import Model, ModelMetadata, Prediction
from octomil.serve import (
    EchoBackend,
    GenerationChunk,
    GenerationRequest,
    InferenceMetrics,
    create_app,
)
from octomil.streaming import StreamToken

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_app():
    """Create a FastAPI app backed by EchoBackend."""
    with patch("octomil.serve.app._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo
        app = create_app("test-model")

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())
    return app


@pytest.fixture
def echo_app_with_queue():
    """Create a FastAPI app backed by EchoBackend with request queue."""
    with patch("octomil.serve.app._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo
        app = create_app("test-model", max_queue_depth=8)

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())
    return app


# ---------------------------------------------------------------------------
# 1. Inference via OctomilClient.predict() — mocked backend
# ---------------------------------------------------------------------------


class TestClientPredict:
    """OctomilClient.predict() downloads, loads, and infers in one call."""

    def test_predict_returns_prediction(self):
        """predict() should return a Prediction with text and metrics."""
        mock_backend = MagicMock()
        mock_backend.generate.return_value = (
            "42",
            InferenceMetrics(total_tokens=5, ttfc_ms=10.0, tokens_per_second=50.0),
        )

        mock_engine = MagicMock()
        mock_engine.create_backend.return_value = mock_backend
        mock_engine.manages_own_download = True

        with (
            patch.object(OctomilClient, "_get_model") as mock_get,
        ):
            model = Model(
                metadata=ModelMetadata(model_id="m1", name="phi-4-mini", version="1.0.0"),
                engine=mock_engine,
            )
            mock_get.return_value = model

            client = OctomilClient(auth=OrgApiKeyAuth(api_key="test-key", org_id="test-org"))
            result = client.predict(
                "phi-4-mini",
                [{"role": "user", "content": "What is 6*7?"}],
                max_tokens=10,
            )

        assert isinstance(result, Prediction)
        assert result.text == "42"
        assert result.metrics.total_tokens == 5
        assert result.metrics.ttfc_ms == 10.0

    def test_predict_passes_generation_request(self):
        """predict() should build a GenerationRequest with correct params."""
        captured_req = None

        def fake_generate(req):
            nonlocal captured_req
            captured_req = req
            return ("ok", InferenceMetrics())

        mock_backend = MagicMock()
        mock_backend.generate.side_effect = fake_generate

        mock_engine = MagicMock()
        mock_engine.create_backend.return_value = mock_backend
        mock_engine.manages_own_download = True

        with patch.object(OctomilClient, "_get_model") as mock_get:
            model = Model(
                metadata=ModelMetadata(model_id="m1", name="test", version="1.0.0"),
                engine=mock_engine,
            )
            mock_get.return_value = model

            client = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
            client.predict(
                "test",
                [{"role": "user", "content": "hi"}],
                max_tokens=100,
                temperature=0.5,
                top_p=0.9,
            )

        assert captured_req is not None
        assert captured_req.max_tokens == 100
        assert captured_req.temperature == 0.5
        assert captured_req.top_p == 0.9
        assert captured_req.messages == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# 2. Inference via OctomilClient.predict_stream() — async streaming
# ---------------------------------------------------------------------------


class TestClientPredictStream:
    @pytest.mark.asyncio
    async def test_predict_stream_yields_chunks(self):
        """predict_stream() should yield GenerationChunk objects."""
        chunks = [
            GenerationChunk(text="Hello", token_count=1),
            GenerationChunk(text=" world", token_count=1),
            GenerationChunk(text="", token_count=0, finish_reason="stop"),
        ]

        async def fake_stream(req):
            for c in chunks:
                yield c

        mock_backend = MagicMock()
        mock_backend.generate_stream = fake_stream

        mock_engine = MagicMock()
        mock_engine.create_backend.return_value = mock_backend
        mock_engine.manages_own_download = True

        with patch.object(OctomilClient, "_get_model") as mock_get:
            model = Model(
                metadata=ModelMetadata(model_id="m1", name="test", version="1.0.0"),
                engine=mock_engine,
            )
            mock_get.return_value = model

            client = OctomilClient(auth=OrgApiKeyAuth(api_key="key", org_id="default"))
            result = []
            async for chunk in client.predict_stream(
                "test",
                [{"role": "user", "content": "hi"}],
            ):
                result.append(chunk)

        assert len(result) == 3
        assert result[0].text == "Hello"
        assert result[1].text == " world"
        assert result[2].finish_reason == "stop"


# ---------------------------------------------------------------------------
# 3. Cloud streaming inference (SSE) via OctomilClient.stream_predict()
# ---------------------------------------------------------------------------


class TestClientStreamPredict:
    def test_stream_predict_yields_tokens(self):
        """stream_predict() delegates to stream_inference and yields StreamTokens."""
        expected = [
            StreamToken(token="The", done=False, provider="ollama"),
            StreamToken(token=" answer", done=False),
            StreamToken(token="", done=True, latency_ms=42.0, session_id="s1"),
        ]

        with patch("octomil.streaming.stream_inference", return_value=iter(expected)):
            client = OctomilClient(
                auth=OrgApiKeyAuth(api_key="key", org_id="default", api_base="https://api.test.com/api/v1")
            )
            tokens = list(client.stream_predict("phi-4-mini", "What is 2+2?"))

        assert len(tokens) == 3
        assert tokens[0].token == "The"
        assert tokens[2].done is True
        assert tokens[2].latency_ms == 42.0

    def test_stream_predict_with_chat_messages(self):
        """stream_predict() should accept chat-style message lists."""
        with patch("octomil.streaming.stream_inference", return_value=iter([])) as mock_fn:
            client = OctomilClient(
                auth=OrgApiKeyAuth(api_key="key", org_id="default", api_base="https://api.test.com/api/v1")
            )
            msgs = [{"role": "user", "content": "hello"}]
            list(client.stream_predict("phi-4-mini", msgs))

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["input_data"] == msgs


# ---------------------------------------------------------------------------
# 4. OctomilClient._chat_create() — collects stream into a dict
# ---------------------------------------------------------------------------


class TestClientChat:
    def test_chat_create_returns_dict_with_message(self):
        """_chat_create() should collect tokens and return {message, latency_ms}."""
        tokens = [
            StreamToken(token="Hello", done=False),
            StreamToken(token=" world", done=False),
            StreamToken(token="", done=True),
        ]

        with patch("octomil.streaming.stream_inference", return_value=iter(tokens)):
            client = OctomilClient(
                auth=OrgApiKeyAuth(api_key="key", org_id="default", api_base="https://api.test.com/api/v1")
            )
            result = client._chat_create(
                "phi-4-mini",
                [{"role": "user", "content": "hi"}],
                max_tokens=50,
            )

        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"] == "Hello world"
        assert "latency_ms" in result
        assert result["latency_ms"] > 0


# ---------------------------------------------------------------------------
# 5. HTTP /v1/chat/completions — non-streaming
# ---------------------------------------------------------------------------


class TestHTTPInference:
    @pytest.mark.asyncio
    async def test_completions_non_streaming(self, echo_app):
        """POST /v1/chat/completions should return OpenAI-compatible response."""
        transport = ASGITransport(app=echo_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Validate inference"}],
                    "max_tokens": 50,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "Validate inference" in data["choices"][0]["message"]["content"]
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_completions_streaming(self, echo_app):
        """POST /v1/chat/completions with stream=true should return SSE."""
        transport = ASGITransport(app=echo_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Stream test"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        lines = resp.text.strip().split("\n\n")

        # Parse SSE events
        events = []
        for line in lines:
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))

        assert len(events) > 0
        for evt in events:
            assert evt["object"] == "chat.completion.chunk"
        assert lines[-1] == "data: [DONE]"


# ---------------------------------------------------------------------------
# 6. Batch queue — submit and process requests
# ---------------------------------------------------------------------------


class TestBatchQueue:
    @pytest.mark.asyncio
    async def test_batch_single_generate(self):
        """A single generate through the queue should work."""
        queue = RequestQueue(max_depth=4)
        queue.start()

        def gen(req):
            return (f"echo: {req}", {"tokens": 3})

        result = await queue.submit_generate("hello", gen)
        assert result == ("echo: hello", {"tokens": 3})
        await queue.stop()

    @pytest.mark.asyncio
    async def test_batch_multiple_concurrent(self):
        """Multiple concurrent requests should all complete."""
        queue = RequestQueue(max_depth=8)
        queue.start()

        results = []

        async def submit(i):
            def gen(req):
                return (f"result-{req}", {})

            return await queue.submit_generate(i, gen)

        tasks = [asyncio.create_task(submit(i)) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for i, r in enumerate(results):
            assert r == (f"result-{i}", {})
        await queue.stop()

    @pytest.mark.asyncio
    async def test_batch_stream(self):
        """Streaming requests through the queue should yield all chunks."""
        queue = RequestQueue(max_depth=4)
        queue.start()

        async def stream_gen(req):
            for i in range(3):
                yield GenerationChunk(text=f"word{i}", token_count=1)

        chunks = []
        async for chunk in queue.submit_generate_stream("req", stream_gen):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].text == "word0"
        assert chunks[2].text == "word2"
        await queue.stop()

    @pytest.mark.asyncio
    async def test_batch_stats(self):
        """Queue stats should reflect current state."""
        queue = RequestQueue(max_depth=16)
        queue.start()

        stats = queue.stats()
        assert isinstance(stats, QueueStats)
        assert stats.pending == 0
        assert stats.active == 0
        assert stats.max_depth == 16
        await queue.stop()


# ---------------------------------------------------------------------------
# 7. Batch queue via HTTP — /v1/chat/completions with queue
# ---------------------------------------------------------------------------


class TestBatchHTTP:
    @pytest.mark.asyncio
    async def test_completions_through_queue(self, echo_app_with_queue):
        """Requests should flow through the queue when enabled."""
        transport = ASGITransport(app=echo_app_with_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "batch test"}],
                    "max_tokens": 50,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "batch test" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_streaming_through_queue(self, echo_app_with_queue):
        """Streaming should also work through the queue."""
        transport = ASGITransport(app=echo_app_with_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "stream batch"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        events = []
        for line in resp.text.strip().split("\n\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_queue_stats_endpoint(self, echo_app_with_queue):
        """GET /v1/queue/stats should return queue info."""
        transport = ASGITransport(app=echo_app_with_queue)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/queue/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["max_depth"] == 8


# ---------------------------------------------------------------------------
# 8. Model.predict() with telemetry
# ---------------------------------------------------------------------------


class TestModelPredict:
    def test_predict_reports_telemetry(self):
        """Model.predict() should report inference lifecycle events."""
        mock_backend = MagicMock()
        mock_backend.generate.return_value = (
            "result",
            InferenceMetrics(total_tokens=10, ttfc_ms=5.0),
        )

        mock_engine = MagicMock()
        mock_engine.create_backend.return_value = mock_backend

        mock_reporter = MagicMock()

        model = Model(
            metadata=ModelMetadata(model_id="m1", name="test", version="1.0.0"),
            engine=mock_engine,
            _reporter=mock_reporter,
        )

        req = GenerationRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        result = model.predict(req)

        assert result.text == "result"
        mock_reporter.report_inference_started.assert_called_once()
        mock_reporter.report_inference_completed.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_stream_yields_and_reports(self):
        """Model.predict_stream() should yield chunks and report telemetry."""
        chunks = [
            GenerationChunk(text="A", token_count=1),
            GenerationChunk(text="B", token_count=1),
        ]

        async def fake_stream(req):
            for c in chunks:
                yield c

        mock_backend = MagicMock()
        mock_backend.generate_stream = fake_stream

        mock_engine = MagicMock()
        mock_engine.create_backend.return_value = mock_backend

        mock_reporter = MagicMock()

        model = Model(
            metadata=ModelMetadata(model_id="m1", name="test", version="1.0.0"),
            engine=mock_engine,
            _reporter=mock_reporter,
        )

        req = GenerationRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )

        result = []
        async for chunk in model.predict_stream(req):
            result.append(chunk)

        assert len(result) == 2
        assert result[0].text == "A"
        mock_reporter.report_inference_started.assert_called_once()
        mock_reporter.report_inference_completed.assert_called_once()
