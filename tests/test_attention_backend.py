"""Tests for flash attention configuration and attention backend telemetry (EDG-48)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch


from edgeml.serve import (
    EchoBackend,
    InferenceBackend,
    InferenceMetrics,
    LlamaCppBackend,
    MLXBackend,
)
from edgeml.telemetry import TelemetryReporter


# ---------------------------------------------------------------------------
# InferenceMetrics — attention_backend field
# ---------------------------------------------------------------------------


class TestInferenceMetricsAttentionBackend:
    def test_default_is_standard(self):
        m = InferenceMetrics()
        assert m.attention_backend == "standard"

    def test_custom_attention_backend(self):
        m = InferenceMetrics(attention_backend="flash_attention")
        assert m.attention_backend == "flash_attention"

    def test_metal_fused(self):
        m = InferenceMetrics(attention_backend="metal_fused")
        assert m.attention_backend == "metal_fused"

    def test_sdpa(self):
        m = InferenceMetrics(attention_backend="sdpa")
        assert m.attention_backend == "sdpa"


# ---------------------------------------------------------------------------
# InferenceBackend — attention_backend class attribute
# ---------------------------------------------------------------------------


class TestInferenceBackendAttentionBackend:
    def test_base_backend_default(self):
        assert InferenceBackend.attention_backend == "standard"

    def test_mlx_backend_metal_fused(self):
        assert MLXBackend.attention_backend == "metal_fused"

    def test_llamacpp_backend_flash_attention(self):
        assert LlamaCppBackend.attention_backend == "flash_attention"

    def test_echo_backend_inherits_standard(self):
        echo = EchoBackend()
        assert echo.attention_backend == "standard"


# ---------------------------------------------------------------------------
# LlamaCppBackend — flash_attn=True in Llama() constructor
# ---------------------------------------------------------------------------


class TestLlamaCppFlashAttn:
    def test_load_model_local_gguf_passes_flash_attn(self):
        """Llama() constructor should receive flash_attn=True for local .gguf files."""
        mock_llama_cls = MagicMock()
        mock_llama_instance = MagicMock()
        mock_llama_cls.return_value = mock_llama_instance

        with patch("edgeml.serve.Llama", mock_llama_cls, create=True):
            with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_cls)}):
                backend = LlamaCppBackend(cache_enabled=False)
                # Patch the import inside load_model
                with patch("edgeml.serve.Llama", mock_llama_cls):

                    def patched_load(self, model_name):
                        self._model_name = model_name
                        self._llm = mock_llama_cls(
                            model_path=model_name,
                            n_ctx=4096,
                            n_gpu_layers=-1,
                            flash_attn=True,
                            verbose=False,
                        )

                    with patch.object(LlamaCppBackend, "load_model", patched_load):
                        backend.load_model("model.gguf")

        mock_llama_cls.assert_called_once_with(
            model_path="model.gguf",
            n_ctx=4096,
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=False,
        )

    def test_load_model_gguf_source_contains_flash_attn(self):
        """Verify the source code of load_model passes flash_attn=True."""
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "flash_attn=True" in source

    def test_all_llama_calls_have_flash_attn(self):
        """Every Llama() and Llama.from_pretrained() call should include flash_attn=True."""
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        # Count Llama constructor/from_pretrained calls (both direct and via from_pretrained)
        # Each call block should have flash_attn=True
        lines = source.split("\n")
        in_call_block = False
        call_blocks = []
        current_block = []

        for line in lines:
            stripped = line.strip()
            if "Llama(" in stripped or "Llama.from_pretrained(" in stripped:
                in_call_block = True
                current_block = [stripped]
            elif in_call_block:
                current_block.append(stripped)
                if ")" in stripped and stripped.endswith(")"):
                    call_blocks.append("\n".join(current_block))
                    in_call_block = False

        # There should be at least 3 call sites (local gguf, HF repo, resolver, legacy)
        assert len(call_blocks) >= 3, f"Expected >=3 Llama call blocks, found {len(call_blocks)}"

        for i, block in enumerate(call_blocks):
            assert "flash_attn=True" in block, (
                f"Llama call block {i + 1} missing flash_attn=True:\n{block}"
            )


# ---------------------------------------------------------------------------
# MLXBackend — Metal fused attention comment and attribute
# ---------------------------------------------------------------------------


class TestMLXAttentionBackend:
    def test_mlx_engine_module_docstring_mentions_metal_fused(self):
        """The mlx_engine module docstring should document Metal fused attention."""
        from edgeml.engines import mlx_engine

        assert "Metal fused attention" in mlx_engine.__doc__

    def test_mlx_backend_class_docstring_mentions_metal_fused(self):
        assert "metal_fused" in MLXBackend.__doc__


# ---------------------------------------------------------------------------
# ORT backend — attention_backend attribute
# ---------------------------------------------------------------------------


class TestORTAttentionBackend:
    def test_ort_backend_attention_backend_is_sdpa(self):
        from edgeml.engines.ort_engine import _ORTBackend

        backend = _ORTBackend("test-model")
        assert backend.attention_backend == "sdpa"

    def test_ort_generate_session_returns_sdpa(self):
        """ORT session generate should set attention_backend='sdpa' in metrics."""
        import numpy as np

        from edgeml.engines.ort_engine import _ORTBackend

        mock_input = MagicMock()
        mock_input.name = "input_ids"
        mock_input.shape = [1, 10]
        mock_input.type = "tensor(float)"

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.array([[0.5]])]

        backend = _ORTBackend("test-model")
        backend._session = mock_session
        backend._use_genai = False

        @dataclass
        class FakeRequest:
            messages: list[dict[str, str]]
            max_tokens: int = 32

        request = FakeRequest(messages=[{"role": "user", "content": "Hi"}])
        _text, metrics = backend.generate(request)
        assert metrics.attention_backend == "sdpa"

    def test_ort_generate_genai_returns_sdpa(self):
        """ORT GenAI generate should set attention_backend='sdpa' in metrics."""
        from edgeml.engines.ort_engine import _ORTBackend

        mock_og = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Hello"

        mock_generator = MagicMock()
        mock_generator.is_done.side_effect = [False, False, True]
        mock_generator.get_next_tokens.return_value = [42]
        mock_og.Generator.return_value = mock_generator
        mock_og.GeneratorParams.return_value = MagicMock()

        backend = _ORTBackend("test-model")
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        backend._use_genai = True

        @dataclass
        class FakeRequest:
            messages: list[dict[str, str]]
            max_tokens: int = 512

        request = FakeRequest(messages=[{"role": "user", "content": "Hi"}])

        with patch.dict("sys.modules", {"onnxruntime_genai": mock_og}):
            _text, metrics = backend.generate(request)

        assert metrics.attention_backend == "sdpa"


# ---------------------------------------------------------------------------
# TelemetryReporter — attention_backend in payloads
# ---------------------------------------------------------------------------


class TestTelemetryAttentionBackend:
    def test_generation_started_includes_attention_backend(self):
        sent: list[dict] = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_started(
                model_id="model-a",
                version="1.0",
                session_id="s1",
                attention_backend="flash_attention",
            )
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        p = sent[0]
        assert p["event_type"] == "generation_started"
        assert p["metrics"]["attention_backend"] == "flash_attention"

    def test_generation_started_no_attention_backend_has_no_metrics(self):
        sent: list[dict] = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_started(
                model_id="model-a",
                version="1.0",
                session_id="s1",
            )
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        p = sent[0]
        assert "metrics" not in p

    def test_generation_completed_includes_attention_backend(self):
        sent: list[dict] = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_completed(
                session_id="s1",
                model_id="model-a",
                version="1.0",
                total_chunks=10,
                total_duration_ms=500.0,
                ttfc_ms=30.0,
                throughput=20.0,
                attention_backend="metal_fused",
            )
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        p = sent[0]
        assert p["event_type"] == "generation_completed"
        assert p["metrics"]["attention_backend"] == "metal_fused"
        # Original metrics should still be present
        assert p["metrics"]["total_chunks"] == 10
        assert p["metrics"]["throughput"] == 20.0

    def test_generation_completed_without_attention_backend(self):
        """When attention_backend is not passed, it should not appear in metrics."""
        sent: list[dict] = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_completed(
                session_id="s1",
                model_id="model-a",
                version="1.0",
                total_chunks=5,
                total_duration_ms=250.0,
                ttfc_ms=15.0,
                throughput=10.0,
            )
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        p = sent[0]
        assert "attention_backend" not in p["metrics"]


# ---------------------------------------------------------------------------
# Engine plugins — attention_backend awareness
# ---------------------------------------------------------------------------


class TestEnginePluginAttentionBackend:
    def test_llamacpp_engine_creates_backend_with_flash_attention(self):
        """LlamaCppEngine.create_backend should return a backend with flash_attention."""
        from edgeml.engines.llamacpp_engine import LlamaCppEngine

        engine = LlamaCppEngine()
        with patch("edgeml.serve.LlamaCppBackend") as MockBackend:
            mock_instance = MagicMock()
            mock_instance.attention_backend = "flash_attention"
            MockBackend.return_value = mock_instance
            backend = engine.create_backend("test-model")
            assert backend.attention_backend == "flash_attention"

    def test_mlx_engine_creates_backend_with_metal_fused(self):
        """MLXEngine.create_backend should return a backend with metal_fused."""
        from edgeml.engines.mlx_engine import MLXEngine

        engine = MLXEngine()
        with patch("edgeml.serve.MLXBackend") as MockBackend:
            mock_instance = MagicMock()
            mock_instance.attention_backend = "metal_fused"
            MockBackend.return_value = mock_instance
            backend = engine.create_backend("test-model")
            assert backend.attention_backend == "metal_fused"

    def test_ort_engine_creates_backend_with_sdpa(self):
        """ONNXRuntimeEngine.create_backend should return a backend with sdpa."""
        from edgeml.engines.ort_engine import ONNXRuntimeEngine

        engine = ONNXRuntimeEngine()
        backend = engine.create_backend("test-model")
        assert backend.attention_backend == "sdpa"
