"""Tests for CactusEngine plugin."""

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from octomil.engines.cactus_engine import CactusEngine, _try_import_cactus


class TestCactusEngineProperties(unittest.TestCase):
    """Test basic engine properties."""

    def setUp(self):
        self.engine = CactusEngine()

    def test_name(self):
        self.assertEqual(self.engine.name, "cactus")

    def test_display_name_contains_cactus(self):
        self.assertIn("Cactus", self.engine.display_name)

    def test_priority_between_llamacpp_and_executorch(self):
        # llama.cpp is 20, ExecuTorch is 25
        self.assertEqual(self.engine.priority, 22)


class TestCactusEngineDetection(unittest.TestCase):
    """Test detect() with mocked imports."""

    def test_detect_returns_true_when_cactus_available(self):
        engine = CactusEngine()
        fake_cactus = types.ModuleType("cactus")
        fake_cactus.cactus_init = MagicMock()
        fake_cactus.cactus_complete = MagicMock()
        fake_cactus.cactus_destroy = MagicMock()
        fake_cactus.cactus_get_last_error = MagicMock()

        with patch.dict(sys.modules, {"cactus": fake_cactus}):
            result = engine.detect()
        self.assertTrue(result)

    def test_detect_returns_false_when_not_installed(self):
        engine = CactusEngine()
        # Ensure cactus is not importable
        with patch.dict(sys.modules, {"cactus": None, "cactus.cactus": None}):
            # _try_import_cactus should return None when imports fail
            with patch(
                "octomil.engines.cactus_engine._try_import_cactus",
                return_value=None,
            ):
                result = engine.detect()
        self.assertFalse(result)

    def test_detect_info_mentions_cactus(self):
        engine = CactusEngine()
        info = engine.detect_info()
        self.assertIn("Cactus", info)


class TestCactusEngineModelSupport(unittest.TestCase):
    """Test supports_model()."""

    def setUp(self):
        self.engine = CactusEngine()

    def test_supports_catalog_model(self):
        # Any model with llama.cpp engine in the catalog should be supported
        self.assertTrue(self.engine.supports_model("gemma-1b"))

    def test_supports_gguf_file(self):
        self.assertTrue(self.engine.supports_model("some-model.gguf"))

    def test_supports_huggingface_repo(self):
        self.assertTrue(
            self.engine.supports_model("LiquidAI/LFM2-1.2B")
        )

    def test_supports_local_cactus_dir(self):
        with patch("os.path.isdir", return_value=True), patch(
            "os.path.isfile", return_value=True
        ):
            self.assertTrue(self.engine.supports_model("/path/to/model"))

    def test_rejects_unknown_model(self):
        with patch("os.path.isdir", return_value=False):
            self.assertFalse(
                self.engine.supports_model("nonexistent-model-xyz")
            )


class TestCactusEngineBenchmark(unittest.TestCase):
    """Test benchmark() with mocked Cactus FFI."""

    def test_benchmark_success(self):
        engine = CactusEngine()

        mock_cactus = MagicMock()
        mock_handle = MagicMock()
        mock_cactus.cactus_init.return_value = mock_handle
        mock_cactus.cactus_complete.return_value = json.dumps(
            {
                "success": True,
                "response": "Hello! I'm doing well.",
                "decode_tps": 42.5,
                "time_to_first_token_ms": 123.4,
                "ram_usage_mb": 256.0,
                "total_tokens": 10,
                "prefill_tps": 100.0,
                "confidence": 0.85,
                "cloud_handoff": False,
            }
        )

        with patch(
            "octomil.engines.cactus_engine._try_import_cactus",
            return_value=mock_cactus,
        ):
            result = engine.benchmark("gemma-1b", n_tokens=32)

        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.tokens_per_second, 42.5)
        self.assertAlmostEqual(result.ttft_ms, 123.4)
        self.assertAlmostEqual(result.memory_mb, 256.0)
        mock_cactus.cactus_destroy.assert_called_once_with(mock_handle)

    def test_benchmark_init_failure(self):
        engine = CactusEngine()

        mock_cactus = MagicMock()
        mock_cactus.cactus_init.return_value = None
        mock_cactus.cactus_get_last_error.return_value = "model not found"

        with patch(
            "octomil.engines.cactus_engine._try_import_cactus",
            return_value=mock_cactus,
        ):
            result = engine.benchmark("bad-model", n_tokens=32)

        self.assertFalse(result.ok)
        self.assertIn("model not found", result.error)

    def test_benchmark_cactus_unavailable(self):
        engine = CactusEngine()

        with patch(
            "octomil.engines.cactus_engine._try_import_cactus",
            return_value=None,
        ):
            result = engine.benchmark("gemma-1b", n_tokens=32)

        self.assertFalse(result.ok)
        self.assertIn("not available", result.error)


class TestCactusEngineBackend(unittest.TestCase):
    """Test create_backend() returns a valid backend object."""

    def test_create_backend_returns_backend(self):
        engine = CactusEngine()
        backend = engine.create_backend("gemma-1b")
        self.assertEqual(backend.model_name, "gemma-1b")
        self.assertIsNotNone(backend.list_models)


class TestTryImportCactus(unittest.TestCase):
    """Test the _try_import_cactus helper."""

    def test_returns_none_when_not_installed(self):
        with patch.dict(
            sys.modules, {"cactus": None, "cactus.cactus": None}
        ):
            result = _try_import_cactus()
        self.assertIsNone(result)

    def test_returns_module_with_cactus_init(self):
        fake_cactus = types.ModuleType("cactus")
        fake_cactus.cactus_init = MagicMock()
        with patch.dict(sys.modules, {"cactus": fake_cactus}):
            result = _try_import_cactus()
        self.assertIsNotNone(result)


class TestCactusEngineInRegistry(unittest.TestCase):
    """Test that CactusEngine is registered in the global registry."""

    def test_registry_contains_cactus(self):
        from octomil.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)
        names = [e.name for e in registry.engines]
        self.assertIn("cactus", names)

    def test_cactus_after_llamacpp_in_registry(self):
        from octomil.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)
        names = [e.name for e in registry.engines]
        llama_idx = names.index("llama.cpp")
        cactus_idx = names.index("cactus")
        self.assertGreater(cactus_idx, llama_idx)


if __name__ == "__main__":
    unittest.main()
