"""Tests for MLC-LLM engine plugin (EDG-59)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from edgeml.engines.base import BenchmarkResult
from edgeml.engines.mlc_engine import (
    MLCBackend,
    MLCEngine,
    _MLC_CATALOG,
    _MLC_QUANT_SUFFIXES,
    _detect_gpu,
    _get_mlc_version,
    _has_mlc_llm,
)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


class TestHasMlcLlm:
    def test_available(self) -> None:
        mock_mlc = MagicMock()
        with patch.dict("sys.modules", {"mlc_llm": mock_mlc}):
            assert _has_mlc_llm() is True

    def test_unavailable(self) -> None:
        import sys

        # Remove mlc_llm from sys.modules and use a None sentinel to block import
        saved = sys.modules.pop("mlc_llm", None)
        # Setting a module to None in sys.modules causes ImportError on import
        sys.modules["mlc_llm"] = None  # type: ignore[assignment]
        try:
            assert _has_mlc_llm() is False
        finally:
            if saved is not None:
                sys.modules["mlc_llm"] = saved
            else:
                sys.modules.pop("mlc_llm", None)


class TestGetMlcVersion:
    def test_returns_version(self) -> None:
        mock_mlc = MagicMock()
        mock_mlc.__version__ = "0.1.0"
        with patch.dict("sys.modules", {"mlc_llm": mock_mlc}):
            assert _get_mlc_version() == "0.1.0"

    def test_returns_unknown_when_no_version(self) -> None:
        mock_mlc = MagicMock(spec=[])
        with patch.dict("sys.modules", {"mlc_llm": mock_mlc}):
            assert _get_mlc_version() == "unknown"

    def test_returns_empty_when_unavailable(self) -> None:
        import sys

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name == "mlc_llm"
                else __import__(name, *a, **kw)
            ),
        ):
            saved = sys.modules.pop("mlc_llm", None)
            try:
                assert _get_mlc_version() == ""
            finally:
                if saved is not None:
                    sys.modules["mlc_llm"] = saved


class TestDetectGpu:
    def test_metal_on_apple_silicon(self) -> None:
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            assert _detect_gpu() == "metal"

    def test_no_metal_on_intel_mac(self) -> None:
        import sys

        saved_tvm = sys.modules.pop("tvm", None)
        sys.modules["tvm"] = None  # type: ignore[assignment]
        try:
            with (
                patch("platform.system", return_value="Darwin"),
                patch("platform.machine", return_value="x86_64"),
                patch("shutil.which", return_value=None),
            ):
                assert _detect_gpu() is None
        finally:
            if saved_tvm is not None:
                sys.modules["tvm"] = saved_tvm
            else:
                sys.modules.pop("tvm", None)

    def test_cuda_via_tvm(self) -> None:
        mock_tvm = MagicMock()
        mock_tvm.cuda.return_value.exist = True
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
            patch.dict("sys.modules", {"tvm": mock_tvm}),
        ):
            assert _detect_gpu() == "cuda"

    def test_cuda_via_nvidia_smi(self) -> None:
        import sys

        saved_tvm = sys.modules.pop("tvm", None)
        sys.modules["tvm"] = None  # type: ignore[assignment]
        try:
            with (
                patch("platform.system", return_value="Linux"),
                patch("platform.machine", return_value="x86_64"),
                patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
                patch("subprocess.run") as mock_run,
            ):
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="NVIDIA GeForce RTX 4090\n"
                )
                assert _detect_gpu() == "cuda"
        finally:
            if saved_tvm is not None:
                sys.modules["tvm"] = saved_tvm
            else:
                sys.modules.pop("tvm", None)

    def test_no_gpu_detected(self) -> None:
        import sys

        saved_tvm = sys.modules.pop("tvm", None)
        sys.modules["tvm"] = None  # type: ignore[assignment]
        try:
            with (
                patch("platform.system", return_value="Linux"),
                patch("platform.machine", return_value="x86_64"),
                patch("shutil.which", return_value=None),
            ):
                assert _detect_gpu() is None
        finally:
            if saved_tvm is not None:
                sys.modules["tvm"] = saved_tvm
            else:
                sys.modules.pop("tvm", None)


# ---------------------------------------------------------------------------
# MLCEngine
# ---------------------------------------------------------------------------


class TestMLCEngine:
    def setup_method(self) -> None:
        self.engine = MLCEngine()

    def test_name(self) -> None:
        assert self.engine.name == "mlc-llm"

    def test_display_name_with_gpu(self) -> None:
        with patch("edgeml.engines.mlc_engine._detect_gpu", return_value="cuda"):
            assert "CUDA" in self.engine.display_name

    def test_display_name_with_metal(self) -> None:
        with patch("edgeml.engines.mlc_engine._detect_gpu", return_value="metal"):
            assert "METAL" in self.engine.display_name

    def test_display_name_without_gpu(self) -> None:
        with patch("edgeml.engines.mlc_engine._detect_gpu", return_value=None):
            assert "GPU" in self.engine.display_name

    def test_priority(self) -> None:
        assert self.engine.priority == 18

    def test_priority_between_mnn_and_llamacpp(self) -> None:
        from edgeml.engines.llamacpp_engine import LlamaCppEngine
        from edgeml.engines.mnn_engine import MNNEngine

        mnn = MNNEngine()
        llama = LlamaCppEngine()
        assert mnn.priority < self.engine.priority < llama.priority

    def test_detect_with_mlc_and_gpu(self) -> None:
        with (
            patch("edgeml.engines.mlc_engine._has_mlc_llm", return_value=True),
            patch("edgeml.engines.mlc_engine._detect_gpu", return_value="cuda"),
        ):
            assert self.engine.detect() is True

    def test_detect_without_mlc(self) -> None:
        with (
            patch("edgeml.engines.mlc_engine._has_mlc_llm", return_value=False),
        ):
            assert self.engine.detect() is False

    def test_detect_without_gpu(self) -> None:
        with (
            patch("edgeml.engines.mlc_engine._has_mlc_llm", return_value=True),
            patch("edgeml.engines.mlc_engine._detect_gpu", return_value=None),
        ):
            assert self.engine.detect() is False

    def test_detect_info_with_mlc(self) -> None:
        with (
            patch(
                "edgeml.engines.mlc_engine._get_mlc_version",
                return_value="0.1.0",
            ),
            patch("edgeml.engines.mlc_engine._detect_gpu", return_value="cuda"),
        ):
            info = self.engine.detect_info()
            assert "mlc_llm 0.1.0" in info
            assert "cuda" in info

    def test_detect_info_empty_when_unavailable(self) -> None:
        with patch("edgeml.engines.mlc_engine._get_mlc_version", return_value=""):
            assert self.engine.detect_info() == ""

    def test_supports_catalog_models(self) -> None:
        for model_name in _MLC_CATALOG:
            assert self.engine.supports_model(model_name) is True

    def test_supports_mlc_repo_id(self) -> None:
        assert self.engine.supports_model("mlc-ai/gemma-2b-it-q4f16_1-MLC") is True

    def test_supports_hf_repo_id(self) -> None:
        assert self.engine.supports_model("some-user/some-model") is True

    def test_supports_mlc_suffix(self) -> None:
        assert self.engine.supports_model("some-model-MLC") is True

    def test_does_not_support_bare_nonsense(self) -> None:
        # Bare names not in catalog are not supported (no / and no -MLC suffix)
        assert self.engine.supports_model("totally-fake-model-xyz") is False

    def test_benchmark_unavailable(self) -> None:
        with patch("edgeml.engines.mlc_engine._has_mlc_llm", return_value=False):
            result = self.engine.benchmark("gemma-1b")
            assert result.ok is False
            assert "not available" in result.error

    def test_benchmark_no_gpu(self) -> None:
        with (
            patch("edgeml.engines.mlc_engine._has_mlc_llm", return_value=True),
            patch("edgeml.engines.mlc_engine._detect_gpu", return_value=None),
        ):
            result = self.engine.benchmark("gemma-1b")
            assert result.ok is False
            assert "No GPU" in result.error

    def test_benchmark_success(self) -> None:
        # Mock the streaming response
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"

        mock_completions = MagicMock()
        mock_completions.create.return_value = [mock_chunk]

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_engine_instance = MagicMock()
        mock_engine_instance.chat = mock_chat

        mock_mlc_engine_cls = MagicMock(return_value=mock_engine_instance)
        mock_mlc = MagicMock()
        mock_mlc.MLCEngine = mock_mlc_engine_cls

        with (
            patch("edgeml.engines.mlc_engine._has_mlc_llm", return_value=True),
            patch("edgeml.engines.mlc_engine._detect_gpu", return_value="cuda"),
            patch.dict("sys.modules", {"mlc_llm": mock_mlc}),
        ):
            result = self.engine.benchmark("mlc-ai/gemma-2b-it-q4f16_1-MLC")
            assert result.ok
            assert result.engine_name == "mlc-llm"
            assert result.tokens_per_second > 0
            assert result.metadata["method"] == "chat.completions"
            assert result.metadata["device"] == "cuda"

    def test_benchmark_exception(self) -> None:
        mock_mlc = MagicMock()
        mock_mlc.MLCEngine.side_effect = RuntimeError("CUDA OOM")

        with (
            patch("edgeml.engines.mlc_engine._has_mlc_llm", return_value=True),
            patch("edgeml.engines.mlc_engine._detect_gpu", return_value="cuda"),
            patch.dict("sys.modules", {"mlc_llm": mock_mlc}),
        ):
            result = self.engine.benchmark("gemma-1b")
            assert result.ok is False
            assert "CUDA OOM" in result.error

    def test_create_backend(self) -> None:
        backend = self.engine.create_backend("gemma-1b")
        assert isinstance(backend, MLCBackend)
        assert backend.name == "mlc-llm"

    def test_create_backend_has_generate(self) -> None:
        backend = self.engine.create_backend("gemma-1b")
        assert hasattr(backend, "generate")
        assert hasattr(backend, "generate_stream")
        assert hasattr(backend, "unload")
        assert hasattr(backend, "load_model")
        assert hasattr(backend, "list_models")

    def test_resolve_repo_id_passthrough(self) -> None:
        """Full repo IDs are passed through unchanged."""
        repo = self.engine._resolve_repo_id("mlc-ai/gemma-2b-it-q4f16_1-MLC")
        assert repo == "mlc-ai/gemma-2b-it-q4f16_1-MLC"

    def test_resolve_repo_id_catalog_lookup(self) -> None:
        """Catalog names resolve to MLC repo IDs."""
        repo = self.engine._resolve_repo_id("gemma-1b")
        assert repo == "mlc-ai/gemma-2b-it-q4f16_1-MLC"

    def test_resolve_repo_id_fallback(self) -> None:
        """Unknown names are returned as-is."""
        repo = self.engine._resolve_repo_id("unknown-model")
        assert repo == "unknown-model"


# ---------------------------------------------------------------------------
# MLCBackend
# ---------------------------------------------------------------------------


class TestMLCBackend:
    def test_name(self) -> None:
        backend = MLCBackend("gemma-1b")
        assert backend.name == "mlc-llm"

    def test_list_models(self) -> None:
        backend = MLCBackend("gemma-1b")
        assert backend.list_models() == ["gemma-1b"]

    def test_list_models_empty(self) -> None:
        backend = MLCBackend("")
        assert backend.list_models() == []

    def test_load_model(self) -> None:
        mock_engine_instance = MagicMock()
        mock_mlc_engine_cls = MagicMock(return_value=mock_engine_instance)
        mock_mlc = MagicMock()
        mock_mlc.MLCEngine = mock_mlc_engine_cls

        with (
            patch.dict("sys.modules", {"mlc_llm": mock_mlc}),
            patch("edgeml.engines.mlc_engine._detect_gpu", return_value="cuda"),
        ):
            backend = MLCBackend("mlc-ai/test-model-MLC")
            backend.load_model("mlc-ai/test-model-MLC")
            mock_mlc_engine_cls.assert_called_once_with(
                model="mlc-ai/test-model-MLC", device="cuda"
            )
            assert backend._engine is mock_engine_instance

    def test_load_model_metal_uses_auto(self) -> None:
        """On Metal, device should be 'auto' (not 'metal')."""
        mock_engine_instance = MagicMock()
        mock_mlc_engine_cls = MagicMock(return_value=mock_engine_instance)
        mock_mlc = MagicMock()
        mock_mlc.MLCEngine = mock_mlc_engine_cls

        with (
            patch.dict("sys.modules", {"mlc_llm": mock_mlc}),
            patch("edgeml.engines.mlc_engine._detect_gpu", return_value="metal"),
        ):
            backend = MLCBackend("mlc-ai/test-model-MLC")
            backend.load_model("mlc-ai/test-model-MLC")
            mock_mlc_engine_cls.assert_called_once_with(
                model="mlc-ai/test-model-MLC", device="auto"
            )

    def test_generate(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world!"

        mock_engine = MagicMock()
        mock_engine.chat.completions.create.return_value = mock_response

        backend = MLCBackend("gemma-1b")
        backend._engine = mock_engine

        result = backend.generate("Say hello")

        assert result == "Hello world!"
        mock_engine.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=512,
            temperature=0.7,
            stop=None,
            stream=False,
        )

    def test_generate_with_params(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Short"

        mock_engine = MagicMock()
        mock_engine.chat.completions.create.return_value = mock_response

        backend = MLCBackend("gemma-1b")
        backend._engine = mock_engine

        result = backend.generate(
            "Say hello",
            max_tokens=32,
            temperature=0.5,
            stop=[".", "\n"],
        )

        assert result == "Short"
        mock_engine.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            temperature=0.5,
            stop=[".", "\n"],
            stream=False,
        )

    def test_generate_empty_content(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_engine = MagicMock()
        mock_engine.chat.completions.create.return_value = mock_response

        backend = MLCBackend("gemma-1b")
        backend._engine = mock_engine

        result = backend.generate("Say hello")
        assert result == ""

    def test_generate_auto_loads_model(self) -> None:
        """generate() calls load_model when no engine is loaded."""
        backend = MLCBackend("gemma-1b")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "loaded!"

        def setup_engine(name: str) -> None:
            mock_engine = MagicMock()
            mock_engine.chat.completions.create.return_value = mock_response
            backend._engine = mock_engine

        with patch.object(backend, "load_model", side_effect=setup_engine) as mock_load:
            result = backend.generate("hello")
            mock_load.assert_called_once_with("gemma-1b")
            assert result == "loaded!"

    def test_generate_stream(self) -> None:
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None  # Empty chunk (e.g. finish)

        mock_engine = MagicMock()
        mock_engine.chat.completions.create.return_value = [chunk1, chunk2, chunk3]

        backend = MLCBackend("gemma-1b")
        backend._engine = mock_engine

        tokens = list(backend.generate_stream("Say hello"))

        assert tokens == ["Hello", " world"]
        mock_engine.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=512,
            temperature=0.7,
            stop=None,
            stream=True,
        )

    def test_generate_stream_with_params(self) -> None:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "ok"

        mock_engine = MagicMock()
        mock_engine.chat.completions.create.return_value = [chunk]

        backend = MLCBackend("gemma-1b")
        backend._engine = mock_engine

        tokens = list(
            backend.generate_stream("test", max_tokens=64, temperature=0.2, stop=["!"])
        )

        assert tokens == ["ok"]
        mock_engine.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=64,
            temperature=0.2,
            stop=["!"],
            stream=True,
        )

    def test_generate_stream_auto_loads_model(self) -> None:
        backend = MLCBackend("gemma-1b")

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "hi"

        def setup_engine(name: str) -> None:
            mock_engine = MagicMock()
            mock_engine.chat.completions.create.return_value = [chunk]
            backend._engine = mock_engine

        with patch.object(backend, "load_model", side_effect=setup_engine) as mock_load:
            tokens = list(backend.generate_stream("hello"))
            mock_load.assert_called_once_with("gemma-1b")
            assert tokens == ["hi"]

    def test_unload(self) -> None:
        mock_engine = MagicMock()
        backend = MLCBackend("gemma-1b")
        backend._engine = mock_engine

        backend.unload()
        assert backend._engine is None

    def test_unload_when_not_loaded(self) -> None:
        backend = MLCBackend("gemma-1b")
        backend.unload()  # Should not raise
        assert backend._engine is None


# ---------------------------------------------------------------------------
# MLC quant suffixes
# ---------------------------------------------------------------------------


class TestMLCQuantSuffixes:
    def test_common_suffixes_present(self) -> None:
        assert "q4f16_1" in _MLC_QUANT_SUFFIXES
        assert "q4f32_1" in _MLC_QUANT_SUFFIXES
        assert "q0f16" in _MLC_QUANT_SUFFIXES
        assert "q0f32" in _MLC_QUANT_SUFFIXES

    def test_suffix_count(self) -> None:
        assert len(_MLC_QUANT_SUFFIXES) >= 6


# ---------------------------------------------------------------------------
# MLC catalog
# ---------------------------------------------------------------------------


class TestMLCCatalog:
    def test_mlc_models_in_catalog(self) -> None:
        from edgeml.models.catalog import CATALOG

        mlc_models = [
            name for name, entry in CATALOG.items() if "mlc-llm" in entry.engines
        ]
        assert len(mlc_models) >= 5

    def test_gemma_1b_has_mlc(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["gemma-1b"]
        assert "mlc-llm" in entry.engines
        assert entry.variants["4bit"].mlc is not None
        assert "mlc-ai/" in entry.variants["4bit"].mlc

    def test_llama_1b_has_mlc(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["llama-1b"]
        assert "mlc-llm" in entry.engines
        assert entry.variants["4bit"].mlc is not None

    def test_llama_3b_has_mlc(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["llama-3b"]
        assert "mlc-llm" in entry.engines
        assert entry.variants["4bit"].mlc is not None

    def test_llama_8b_has_mlc(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["llama-8b"]
        assert "mlc-llm" in entry.engines
        assert entry.variants["4bit"].mlc is not None

    def test_phi_mini_has_mlc(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["phi-mini"]
        assert "mlc-llm" in entry.engines
        assert entry.variants["4bit"].mlc is not None

    def test_gemma_4b_has_mlc(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["gemma-4b"]
        assert "mlc-llm" in entry.engines
        assert entry.variants["4bit"].mlc is not None

    def test_variant_spec_has_mlc_field(self) -> None:
        from edgeml.models.catalog import VariantSpec

        spec = VariantSpec(mlc="mlc-ai/test-model-MLC")
        assert spec.mlc == "mlc-ai/test-model-MLC"

    def test_variant_spec_mlc_default_none(self) -> None:
        from edgeml.models.catalog import VariantSpec

        spec = VariantSpec()
        assert spec.mlc is None

    def test_whisper_does_not_have_mlc(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["whisper-base"]
        assert "mlc-llm" not in entry.engines


# ---------------------------------------------------------------------------
# Resolver integration
# ---------------------------------------------------------------------------


class TestMLCResolver:
    def test_engine_alias_mlc(self) -> None:
        from edgeml.models.resolver import _normalize_engine

        assert _normalize_engine("mlc") == "mlc-llm"
        assert _normalize_engine("mlc-llm") == "mlc-llm"
        assert _normalize_engine("mlcllm") == "mlc-llm"

    def test_engine_in_priority(self) -> None:
        from edgeml.models.resolver import _ENGINE_PRIORITY

        assert "mlc-llm" in _ENGINE_PRIORITY

    def test_mlc_priority_between_mnn_and_llamacpp(self) -> None:
        from edgeml.models.resolver import _ENGINE_PRIORITY

        mnn_idx = _ENGINE_PRIORITY.index("mnn")
        mlc_idx = _ENGINE_PRIORITY.index("mlc-llm")
        llama_idx = _ENGINE_PRIORITY.index("llama.cpp")
        assert mnn_idx < mlc_idx < llama_idx

    def test_resolve_with_mlc_engine(self) -> None:
        from edgeml.models.resolver import resolve

        result = resolve("gemma-1b", engine="mlc-llm")
        assert result.engine == "mlc-llm"
        assert result.hf_repo == "mlc-ai/gemma-2b-it-q4f16_1-MLC"

    def test_resolve_with_mlc_alias(self) -> None:
        from edgeml.models.resolver import resolve

        result = resolve("gemma-1b", engine="mlc")
        assert result.engine == "mlc-llm"

    def test_resolve_llama_with_mlc(self) -> None:
        from edgeml.models.resolver import resolve

        result = resolve("llama-1b", engine="mlc-llm")
        assert result.engine == "mlc-llm"
        assert "mlc-ai/" in result.hf_repo

    def test_resolve_phi_mini_with_mlc(self) -> None:
        from edgeml.models.resolver import resolve

        result = resolve("phi-mini", engine="mlc-llm")
        assert result.engine == "mlc-llm"
        assert "mlc-ai/" in result.hf_repo

    def test_resolve_picks_mlc_when_only_available(self) -> None:
        from edgeml.models.resolver import resolve

        result = resolve("gemma-1b", available_engines=["mlc-llm"])
        assert result.engine == "mlc-llm"

    def test_resolve_passthrough_repo_id(self) -> None:
        from edgeml.models.resolver import resolve

        result = resolve("mlc-ai/custom-model-MLC", engine="mlc-llm")
        assert result.engine == "mlc-llm"
        assert result.hf_repo == "mlc-ai/custom-model-MLC"

    def test_pick_engine_prefers_mlx_over_mlc(self) -> None:
        """mlx-lm has higher priority than mlc-llm."""
        from edgeml.models.resolver import resolve

        result = resolve("gemma-1b", available_engines=["mlx-lm", "mlc-llm"])
        assert result.engine == "mlx-lm"

    def test_pick_engine_prefers_mlc_over_llamacpp(self) -> None:
        """mlc-llm has higher priority than llama.cpp."""
        from edgeml.models.resolver import resolve

        result = resolve("gemma-1b", available_engines=["mlc-llm", "llama.cpp"])
        assert result.engine == "mlc-llm"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestMLCRegistry:
    def test_engine_registered(self) -> None:
        from edgeml.engines.registry import EngineRegistry

        registry = EngineRegistry()
        engine = MLCEngine()
        registry.register(engine)
        assert registry.get_engine("mlc-llm") is engine

    def test_auto_register_includes_mlc(self) -> None:
        from edgeml.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)
        assert registry.get_engine("mlc-llm") is not None

    def test_global_registry_has_mlc(self) -> None:
        from edgeml.engines.registry import get_registry, reset_registry

        reset_registry()
        try:
            reg = get_registry()
            names = [e.name for e in reg.engines]
            assert "mlc-llm" in names
        finally:
            reset_registry()

    def test_priority_ordering(self) -> None:
        """mlc-llm (18) should be after mnn (15) and before llama.cpp (20)."""
        from edgeml.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)

        mlc = registry.get_engine("mlc-llm")
        mnn = registry.get_engine("mnn")
        llama = registry.get_engine("llama.cpp")

        assert mlc is not None
        assert mnn is not None
        assert llama is not None
        assert mnn.priority < mlc.priority < llama.priority

    def test_no_duplicate_registration(self) -> None:
        from edgeml.engines.registry import EngineRegistry

        registry = EngineRegistry()
        registry.register(MLCEngine())
        registry.register(MLCEngine())
        mlc_count = sum(1 for e in registry.engines if e.name == "mlc-llm")
        assert mlc_count == 1

    def test_detect_all_includes_mlc(self) -> None:
        from edgeml.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)
        results = registry.detect_all()
        engine_names = [r.engine.name for r in results]
        assert "mlc-llm" in engine_names


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestMLCEdgeCases:
    def test_benchmark_returns_benchmark_result(self) -> None:
        engine = MLCEngine()
        with (
            patch("edgeml.engines.mlc_engine._has_mlc_llm", return_value=False),
        ):
            result = engine.benchmark("gemma-1b")
            assert isinstance(result, BenchmarkResult)

    def test_benchmark_result_engine_name(self) -> None:
        engine = MLCEngine()
        with (
            patch("edgeml.engines.mlc_engine._has_mlc_llm", return_value=False),
        ):
            result = engine.benchmark("gemma-1b")
            assert result.engine_name == "mlc-llm"

    def test_backend_kwargs_preserved(self) -> None:
        backend = MLCBackend("gemma-1b", some_param="value")
        assert backend._kwargs == {"some_param": "value"}

    def test_engine_is_subclass(self) -> None:
        from edgeml.engines.base import EnginePlugin

        assert issubclass(MLCEngine, EnginePlugin)

    def test_engine_instance_check(self) -> None:
        from edgeml.engines.base import EnginePlugin

        engine = MLCEngine()
        assert isinstance(engine, EnginePlugin)
