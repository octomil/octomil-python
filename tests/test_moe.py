"""Tests for Mixture of Experts (MoE) model support.

Covers:
- Catalog: MoE model entries, metadata, architecture field
- Resolver: MoE detection in resolved models
- Telemetry: expert routing metrics, load balance scoring
- Serve: MoE config, model info endpoint
- Engine plugins: MoE model detection
"""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import patch

import pytest

# Ensure the octomil package is importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octomil.models.catalog import (
    CATALOG,
    MODEL_ALIASES,
    MoEMetadata,
    get_model,
    get_moe_metadata,
    is_moe_model,
    list_moe_models,
    list_models,
    supports_engine,
)
from octomil.models.resolver import ResolvedModel, resolve
from octomil.telemetry import TelemetryReporter, _compute_load_balance


# =====================================================================
# Catalog — MoE model entries
# =====================================================================


class TestMoECatalogEntries:
    """Verify MoE models are correctly defined in the catalog."""

    def test_mixtral_8x7b_exists(self) -> None:
        entry = CATALOG.get("mixtral-8x7b")
        assert entry is not None
        assert entry.publisher == "Mistral AI"
        assert entry.architecture == "moe"

    def test_mixtral_8x7b_moe_metadata(self) -> None:
        entry = CATALOG["mixtral-8x7b"]
        assert entry.moe is not None
        assert entry.moe.num_experts == 8
        assert entry.moe.active_experts == 2
        assert entry.moe.expert_size == "7B"
        assert entry.moe.total_params == "46.7B"
        assert entry.moe.active_params == "12.9B"

    def test_mixtral_8x22b_exists(self) -> None:
        entry = CATALOG.get("mixtral-8x22b")
        assert entry is not None
        assert entry.architecture == "moe"
        assert entry.moe is not None
        assert entry.moe.num_experts == 8
        assert entry.moe.active_experts == 2

    def test_dbrx_exists(self) -> None:
        entry = CATALOG.get("dbrx")
        assert entry is not None
        assert entry.publisher == "Databricks"
        assert entry.architecture == "moe"
        assert entry.moe is not None
        assert entry.moe.num_experts == 16
        assert entry.moe.active_experts == 4

    def test_deepseek_v3_exists(self) -> None:
        entry = CATALOG.get("deepseek-v3")
        assert entry is not None
        assert entry.publisher == "DeepSeek"
        assert entry.architecture == "moe"
        assert entry.moe is not None
        assert entry.moe.num_experts == 256
        assert entry.moe.active_experts == 8
        assert entry.moe.total_params == "671B"

    def test_deepseek_v2_lite_exists(self) -> None:
        entry = CATALOG.get("deepseek-v2-lite")
        assert entry is not None
        assert entry.architecture == "moe"
        assert entry.moe is not None
        assert entry.moe.num_experts == 64
        assert entry.moe.active_experts == 6

    def test_qwen_moe_exists(self) -> None:
        entry = CATALOG.get("qwen-moe-14b")
        assert entry is not None
        assert entry.publisher == "Qwen"
        assert entry.architecture == "moe"
        assert entry.moe is not None
        assert entry.moe.num_experts == 60
        assert entry.moe.active_experts == 4

    def test_dense_models_have_no_moe_metadata(self) -> None:
        """Dense models should have architecture='dense' and moe=None."""
        for name in ["gemma-1b", "llama-8b", "phi-mini", "mistral-7b"]:
            entry = CATALOG.get(name)
            assert entry is not None, f"{name} not in catalog"
            assert entry.architecture == "dense", f"{name} should be dense"
            assert entry.moe is None, f"{name} should have no MoE metadata"

    def test_moe_models_have_variants(self) -> None:
        """All MoE models should have at least one variant."""
        for name, entry in CATALOG.items():
            if entry.architecture == "moe":
                assert len(entry.variants) > 0, f"MoE model {name} has no variants"

    def test_moe_models_support_engines(self) -> None:
        """All MoE models should support at least one engine."""
        for name, entry in CATALOG.items():
            if entry.architecture == "moe":
                assert len(entry.engines) > 0, f"MoE model {name} has no engines"


# =====================================================================
# Catalog — MoE helper functions
# =====================================================================


class TestMoECatalogHelpers:
    """Test MoE-specific catalog helper functions."""

    def test_is_moe_model_true(self) -> None:
        assert is_moe_model("mixtral-8x7b") is True
        assert is_moe_model("dbrx") is True
        assert is_moe_model("deepseek-v3") is True

    def test_is_moe_model_false(self) -> None:
        assert is_moe_model("gemma-1b") is False
        assert is_moe_model("llama-8b") is False
        assert is_moe_model("phi-mini") is False

    def test_is_moe_model_unknown(self) -> None:
        assert is_moe_model("nonexistent-model") is False

    def test_is_moe_model_alias(self) -> None:
        """Aliases should resolve to MoE models."""
        assert is_moe_model("mixtral") is True
        assert is_moe_model("mixtral-instruct") is True
        assert is_moe_model("dbrx-instruct") is True
        assert is_moe_model("qwen-moe") is True

    def test_list_moe_models(self) -> None:
        moe_models = list_moe_models()
        assert isinstance(moe_models, list)
        assert (
            len(moe_models) >= 5
        )  # mixtral-8x7b, 8x22b, dbrx, deepseek-v3, v2-lite, qwen-moe
        assert "mixtral-8x7b" in moe_models
        assert "dbrx" in moe_models
        assert "deepseek-v3" in moe_models
        # Dense models should not be in this list
        assert "gemma-1b" not in moe_models

    def test_list_moe_models_sorted(self) -> None:
        moe_models = list_moe_models()
        assert moe_models == sorted(moe_models)

    def test_get_moe_metadata(self) -> None:
        meta = get_moe_metadata("mixtral-8x7b")
        assert meta is not None
        assert isinstance(meta, MoEMetadata)
        assert meta.num_experts == 8
        assert meta.active_experts == 2

    def test_get_moe_metadata_dense_model(self) -> None:
        meta = get_moe_metadata("gemma-1b")
        assert meta is None

    def test_get_moe_metadata_unknown(self) -> None:
        meta = get_moe_metadata("nonexistent")
        assert meta is None

    def test_get_moe_metadata_alias(self) -> None:
        meta = get_moe_metadata("mixtral")
        assert meta is not None
        assert meta.num_experts == 8


# =====================================================================
# Catalog — MoE aliases
# =====================================================================


class TestMoEAliases:
    """Test that MoE model aliases resolve correctly."""

    def test_mixtral_alias(self) -> None:
        assert MODEL_ALIASES["mixtral"] == "mixtral-8x7b"

    def test_mixtral_instruct_alias(self) -> None:
        assert MODEL_ALIASES["mixtral-instruct"] == "mixtral-8x7b"

    def test_dbrx_instruct_alias(self) -> None:
        assert MODEL_ALIASES["dbrx-instruct"] == "dbrx"

    def test_qwen_moe_alias(self) -> None:
        assert MODEL_ALIASES["qwen-moe"] == "qwen-moe-14b"

    def test_alias_resolves_via_get_model(self) -> None:
        entry = get_model("mixtral")
        assert entry is not None
        assert entry.architecture == "moe"
        assert entry.moe is not None
        assert entry.moe.num_experts == 8


# =====================================================================
# Resolver — MoE detection
# =====================================================================


class TestMoEResolver:
    """Test MoE-aware model resolution."""

    def test_resolve_mixtral_has_moe_metadata(self) -> None:
        r = resolve("mixtral-8x7b", engine="llama.cpp")
        assert r.architecture == "moe"
        assert r.moe is not None
        assert r.moe.num_experts == 8
        assert r.moe.active_experts == 2
        assert r.is_moe is True

    def test_resolve_mixtral_with_variant(self) -> None:
        r = resolve("mixtral-8x7b:4bit", engine="llama.cpp")
        assert r.is_moe is True
        assert r.quant == "4bit"
        assert r.is_gguf is True

    def test_resolve_dbrx(self) -> None:
        r = resolve("dbrx", engine="llama.cpp")
        assert r.is_moe is True
        assert r.moe is not None
        assert r.moe.num_experts == 16

    def test_resolve_deepseek_v3(self) -> None:
        r = resolve("deepseek-v3", engine="llama.cpp")
        assert r.is_moe is True
        assert r.moe is not None
        assert r.moe.num_experts == 256

    def test_resolve_dense_model_not_moe(self) -> None:
        r = resolve("gemma-1b", engine="llama.cpp")
        assert r.architecture == "dense"
        assert r.moe is None
        assert r.is_moe is False

    def test_resolve_via_alias(self) -> None:
        r = resolve("mixtral", engine="llama.cpp")
        assert r.is_moe is True
        assert r.family == "mixtral-8x7b"

    def test_resolve_passthrough_not_moe(self) -> None:
        """Passthrough models (local files, full repos) are not detected as MoE."""
        r = resolve("./model.gguf")
        assert r.is_moe is False
        assert r.architecture == "dense"

    def test_resolve_mixtral_mlx(self) -> None:
        r = resolve("mixtral-8x7b", engine="mlx-lm")
        assert r.is_moe is True
        assert r.hf_repo == "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit"

    def test_resolve_mixtral_8bit(self) -> None:
        r = resolve("mixtral-8x7b:8bit", engine="llama.cpp")
        assert r.is_moe is True
        assert r.quant == "8bit"


# =====================================================================
# MoEMetadata — frozen dataclass
# =====================================================================


class TestMoEMetadata:
    """Test the MoEMetadata dataclass."""

    def test_creation(self) -> None:
        meta = MoEMetadata(
            num_experts=8,
            active_experts=2,
            expert_size="7B",
            total_params="46.7B",
            active_params="12.9B",
        )
        assert meta.num_experts == 8
        assert meta.active_experts == 2
        assert meta.expert_size == "7B"

    def test_frozen(self) -> None:
        meta = MoEMetadata(
            num_experts=8,
            active_experts=2,
            expert_size="7B",
            total_params="46.7B",
            active_params="12.9B",
        )
        with pytest.raises(AttributeError):
            meta.num_experts = 16  # type: ignore[misc]

    def test_equality(self) -> None:
        m1 = MoEMetadata(8, 2, "7B", "46.7B", "12.9B")
        m2 = MoEMetadata(8, 2, "7B", "46.7B", "12.9B")
        assert m1 == m2

    def test_inequality(self) -> None:
        m1 = MoEMetadata(8, 2, "7B", "46.7B", "12.9B")
        m2 = MoEMetadata(16, 4, "8B", "132B", "36B")
        assert m1 != m2


# =====================================================================
# Telemetry — load balance scoring
# =====================================================================


class TestLoadBalanceScoring:
    """Test the _compute_load_balance helper function."""

    def test_perfect_balance(self) -> None:
        """All experts get equal activations -> score 1.0."""
        counts = {0: 100, 1: 100, 2: 100, 3: 100}
        score = _compute_load_balance(counts, num_experts=4)
        assert score == pytest.approx(1.0)

    def test_single_expert_imbalance(self) -> None:
        """All tokens to one expert -> score near 0.0."""
        counts = {0: 1000, 1: 0, 2: 0, 3: 0}
        score = _compute_load_balance(counts, num_experts=4)
        assert score < 0.1  # Should be very low

    def test_moderate_imbalance(self) -> None:
        """Moderate skew -> score between 0 and 1."""
        counts = {0: 200, 1: 100, 2: 50, 3: 50}
        score = _compute_load_balance(counts, num_experts=4)
        assert 0.0 < score < 1.0

    def test_empty_counts(self) -> None:
        score = _compute_load_balance({}, num_experts=4)
        assert score == 1.0

    def test_single_expert(self) -> None:
        """One expert total -> always balanced."""
        score = _compute_load_balance({0: 100}, num_experts=1)
        assert score == 1.0

    def test_zero_total(self) -> None:
        counts = {0: 0, 1: 0, 2: 0}
        score = _compute_load_balance(counts, num_experts=3)
        assert score == 1.0

    def test_two_expert_balance(self) -> None:
        counts = {0: 50, 1: 50}
        score = _compute_load_balance(counts, num_experts=2)
        assert score == pytest.approx(1.0)

    def test_two_expert_full_imbalance(self) -> None:
        counts = {0: 100, 1: 0}
        score = _compute_load_balance(counts, num_experts=2)
        assert score == pytest.approx(0.0)

    def test_many_experts_mostly_balanced(self) -> None:
        """8 experts with slight variance should score high."""
        counts = {i: 100 + (i % 3) * 5 for i in range(8)}
        score = _compute_load_balance(counts, num_experts=8)
        assert score > 0.9

    def test_score_always_in_range(self) -> None:
        """Score should always be between 0.0 and 1.0."""
        import random

        rng = random.Random(42)
        for _ in range(50):
            n = rng.randint(2, 16)
            counts = {i: rng.randint(0, 1000) for i in range(n)}
            score = _compute_load_balance(counts, num_experts=n)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {counts}"


# =====================================================================
# Telemetry — MoE routing event
# =====================================================================


class TestTelemetryMoERouting:
    """Test the report_moe_routing telemetry method."""

    def setup_method(self) -> None:
        self.sent: list[dict] = []

        def mock_send(client, url, headers, payload):
            self.sent.append(payload)

        self._patch = patch.object(TelemetryReporter, "_send", side_effect=mock_send)
        self._patch.start()
        self.reporter = TelemetryReporter(
            api_key="test-key",
            api_base="https://api.test.com/api/v1",
            org_id="test-org",
            device_id="dev-moe",
        )

    def teardown_method(self) -> None:
        self.reporter.close()
        self._patch.stop()

    def _wait_for_events(self, expected_count: int = 1, timeout: float = 2.0) -> None:
        deadline = time.time() + timeout
        while len(self.sent) < expected_count and time.time() < deadline:
            time.sleep(0.05)

    def test_moe_routing_event_structure(self) -> None:
        self.reporter.report_moe_routing(
            session_id="sess-1",
            model_id="mixtral-8x7b",
            version="1.0",
            num_experts=8,
            active_experts=2,
            total_tokens_routed=100,
        )
        self._wait_for_events()
        assert len(self.sent) >= 1
        payload = self.sent[0]
        assert payload["event_type"] == "moe_routing"
        assert payload["model_id"] == "mixtral-8x7b"
        assert payload["session_id"] == "sess-1"
        assert payload["device_id"] == "dev-moe"
        assert payload["org_id"] == "test-org"
        metrics = payload["metrics"]
        assert metrics["num_experts"] == 8
        assert metrics["active_experts"] == 2
        assert metrics["total_tokens_routed"] == 100

    def test_moe_routing_with_activation_counts(self) -> None:
        counts = {0: 30, 1: 25, 2: 20, 3: 15, 4: 5, 5: 3, 6: 1, 7: 1}
        self.reporter.report_moe_routing(
            session_id="sess-2",
            model_id="mixtral-8x7b",
            version="1.0",
            num_experts=8,
            active_experts=2,
            expert_activation_counts=counts,
            total_tokens_routed=100,
        )
        self._wait_for_events()
        assert len(self.sent) >= 1
        metrics = self.sent[0]["metrics"]
        assert metrics["expert_activation_counts"] == counts
        # Load balance score should be auto-computed
        assert "load_balance_score" in metrics
        assert 0.0 <= metrics["load_balance_score"] <= 1.0

    def test_moe_routing_explicit_load_balance(self) -> None:
        self.reporter.report_moe_routing(
            session_id="sess-3",
            model_id="dbrx",
            version="1.0",
            num_experts=16,
            active_experts=4,
            load_balance_score=0.85,
            total_tokens_routed=500,
        )
        self._wait_for_events()
        metrics = self.sent[0]["metrics"]
        assert metrics["load_balance_score"] == 0.85

    def test_moe_routing_with_memory_info(self) -> None:
        self.reporter.report_moe_routing(
            session_id="sess-4",
            model_id="deepseek-v3",
            version="1.0",
            num_experts=256,
            active_experts=8,
            expert_memory_mb=4096.5,
            total_tokens_routed=1000,
        )
        self._wait_for_events()
        metrics = self.sent[0]["metrics"]
        assert metrics["expert_memory_mb"] == 4096.5

    def test_moe_routing_load_balance_overrides_computed(self) -> None:
        """Explicit load_balance_score should override auto-computed value."""
        counts = {0: 100, 1: 0}
        self.reporter.report_moe_routing(
            session_id="sess-5",
            model_id="test-moe",
            version="1.0",
            num_experts=2,
            active_experts=1,
            expert_activation_counts=counts,
            load_balance_score=0.99,  # Override
        )
        self._wait_for_events()
        metrics = self.sent[0]["metrics"]
        # Explicit value wins
        assert metrics["load_balance_score"] == 0.99


# =====================================================================
# Serve — MoE config
# =====================================================================


class TestMoEConfig:
    """Test MoE configuration dataclass."""

    def test_default_config(self) -> None:
        from octomil.serve import MoEConfig

        cfg = MoEConfig()
        assert cfg.enabled is True
        assert cfg.expert_memory_limit_mb == 0
        assert cfg.log_expert_routing is False
        assert cfg.offload_inactive is False

    def test_custom_config(self) -> None:
        from octomil.serve import MoEConfig

        cfg = MoEConfig(
            enabled=True,
            expert_memory_limit_mb=8192,
            log_expert_routing=True,
            offload_inactive=True,
        )
        assert cfg.expert_memory_limit_mb == 8192
        assert cfg.log_expert_routing is True
        assert cfg.offload_inactive is True

    def test_disabled_config(self) -> None:
        from octomil.serve import MoEConfig

        cfg = MoEConfig(enabled=False)
        assert cfg.enabled is False


class TestServerStateMoE:
    """Test MoE fields on ServerState."""

    def test_server_state_defaults(self) -> None:
        from octomil.serve import MoEConfig, ServerState

        state = ServerState()
        assert state.is_moe_model is False
        assert state.moe_metadata is None
        assert isinstance(state.moe_config, MoEConfig)
        assert state.moe_config.enabled is True

    def test_server_state_with_moe(self) -> None:
        from octomil.serve import MoEConfig, ServerState

        meta = MoEMetadata(8, 2, "7B", "46.7B", "12.9B")
        state = ServerState(
            is_moe_model=True,
            moe_metadata=meta,
            moe_config=MoEConfig(log_expert_routing=True),
        )
        assert state.is_moe_model is True
        assert state.moe_metadata.num_experts == 8
        assert state.moe_config.log_expert_routing is True


# =====================================================================
# Engine plugins — MoE detection
# =====================================================================


class TestLlamaCppMoEDetection:
    """Test MoE model detection in LlamaCppEngine."""

    def test_is_moe_model_mixtral(self) -> None:
        from octomil.engines.llamacpp_engine import LlamaCppEngine

        engine = LlamaCppEngine()
        assert engine.is_moe_model("mixtral-8x7b") is True

    def test_is_moe_model_dbrx(self) -> None:
        from octomil.engines.llamacpp_engine import LlamaCppEngine

        engine = LlamaCppEngine()
        assert engine.is_moe_model("dbrx") is True

    def test_is_moe_model_dense(self) -> None:
        from octomil.engines.llamacpp_engine import LlamaCppEngine

        engine = LlamaCppEngine()
        assert engine.is_moe_model("gemma-1b") is False

    def test_is_moe_model_unknown(self) -> None:
        from octomil.engines.llamacpp_engine import LlamaCppEngine

        engine = LlamaCppEngine()
        assert engine.is_moe_model("not-a-model") is False


class TestMLXMoEDetection:
    """Test MoE model detection in MLXEngine."""

    def test_is_moe_model_mixtral(self) -> None:
        from octomil.engines.mlx_engine import MLXEngine

        engine = MLXEngine()
        assert engine.is_moe_model("mixtral-8x7b") is True

    def test_is_moe_model_dbrx(self) -> None:
        from octomil.engines.mlx_engine import MLXEngine

        engine = MLXEngine()
        assert engine.is_moe_model("dbrx") is True

    def test_is_moe_model_dense(self) -> None:
        from octomil.engines.mlx_engine import MLXEngine

        engine = MLXEngine()
        assert engine.is_moe_model("llama-8b") is False


# =====================================================================
# ResolvedModel — is_moe property
# =====================================================================


class TestResolvedModelIsMoE:
    """Test the is_moe property on ResolvedModel."""

    def test_moe_resolved_model(self) -> None:
        meta = MoEMetadata(8, 2, "7B", "46.7B", "12.9B")
        r = ResolvedModel(
            family="mixtral-8x7b",
            quant="4bit",
            engine="llama.cpp",
            hf_repo="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
            architecture="moe",
            moe=meta,
        )
        assert r.is_moe is True

    def test_dense_resolved_model(self) -> None:
        r = ResolvedModel(
            family="gemma-1b",
            quant="4bit",
            engine="mlx-lm",
            hf_repo="mlx-community/gemma-3-1b-it-4bit",
        )
        assert r.is_moe is False

    def test_moe_without_metadata(self) -> None:
        """architecture='moe' but moe=None should be False."""
        r = ResolvedModel(
            family="test",
            quant="4bit",
            engine="llama.cpp",
            hf_repo="test/repo",
            architecture="moe",
            moe=None,
        )
        assert r.is_moe is False


# =====================================================================
# Integration — MoE models support correct engines
# =====================================================================


class TestMoEEngineSupport:
    """Verify MoE models have the expected engine support."""

    def test_mixtral_supports_llamacpp(self) -> None:
        assert supports_engine("mixtral-8x7b", "llama.cpp") is True

    def test_mixtral_supports_mlx(self) -> None:
        assert supports_engine("mixtral-8x7b", "mlx-lm") is True

    def test_dbrx_supports_llamacpp(self) -> None:
        assert supports_engine("dbrx", "llama.cpp") is True

    def test_deepseek_v3_supports_llamacpp(self) -> None:
        assert supports_engine("deepseek-v3", "llama.cpp") is True

    def test_moe_models_in_list_models(self) -> None:
        """MoE models should appear in the full model listing."""
        all_models = list_models()
        assert "mixtral-8x7b" in all_models
        assert "dbrx" in all_models
        assert "deepseek-v3" in all_models


# =====================================================================
# Top-level exports
# =====================================================================


class TestTopLevelExports:
    """Verify MoE utilities are exported from octomil package."""

    def test_import_moe_metadata(self) -> None:
        from octomil import MoEMetadata

        assert MoEMetadata is not None

    def test_import_is_moe_model(self) -> None:
        from octomil import is_moe_model

        assert callable(is_moe_model)

    def test_import_list_moe_models(self) -> None:
        from octomil import list_moe_models

        assert callable(list_moe_models)

    def test_import_get_moe_metadata(self) -> None:
        from octomil import get_moe_metadata

        assert callable(get_moe_metadata)

    def test_import_moe_config(self) -> None:
        from octomil.serve import MoEConfig

        assert MoEConfig is not None
