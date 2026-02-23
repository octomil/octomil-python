"""Tests for octomil.model_optimizer — pure computation, no mocking needed."""

from __future__ import annotations

import pytest

from octomil.hardware._types import (
    CPUInfo,
    GPUDetectionResult,
    GPUInfo,
    GPUMemory,
    HardwareProfile,
)
from octomil.model_optimizer import (
    MemoryStrategy,
    ModelOptimizer,
    QuantOffloadResult,
    SpeedEstimate,
    _kv_cache_gb,
    _model_memory_gb,
    _resolve_model_size,
    _total_memory_gb,
)


# ---------------------------------------------------------------------------
# Helper to build test profiles
# ---------------------------------------------------------------------------


def _make_profile(
    vram_gb: float = 0.0,
    ram_gb: float = 16.0,
    available_ram_gb: float = 12.0,
    backend: str = "cuda",
    speed_coeff: int = 100,
    threads: int = 16,
    platform: str = "linux",
) -> HardwareProfile:
    gpu = None
    if vram_gb > 0:
        gpu = GPUDetectionResult(
            gpus=[
                GPUInfo(
                    index=0,
                    name="Test GPU",
                    memory=GPUMemory(total_gb=vram_gb),
                    speed_coefficient=speed_coeff,
                )
            ],
            backend=backend,
            total_vram_gb=vram_gb,
            speed_coefficient=speed_coeff,
        )
    cpu = CPUInfo(
        brand="Test CPU",
        cores=8,
        threads=threads,
        base_speed_ghz=3.0,
        architecture="x86_64",
    )
    return HardwareProfile(
        gpu=gpu,
        cpu=cpu,
        total_ram_gb=ram_gb,
        available_ram_gb=available_ram_gb,
        platform=platform,
        best_backend=backend if vram_gb > 0 else "cpu",
    )


# ---------------------------------------------------------------------------
# Quantization selection (pick_quant_and_offload)
# ---------------------------------------------------------------------------


class TestQuantSelection:
    def test_7b_24gb_vram_q8_full_gpu(self):
        """7B model + 24 GB VRAM → Q8_0 full_gpu."""
        profile = _make_profile(vram_gb=24.0)
        opt = ModelOptimizer(profile)
        result = opt.pick_quant_and_offload(7.0)

        assert result.quantization == "Q8_0"
        assert result.strategy == MemoryStrategy.FULL_GPU
        assert result.gpu_layers == -1

    def test_7b_8gb_vram_q8_partial_offload(self):
        """7B model + 8 GB VRAM → Q8_0 partial_offload (higher quality preferred)."""
        profile = _make_profile(vram_gb=8.0)
        opt = ModelOptimizer(profile)
        result = opt.pick_quant_and_offload(7.0)

        # usable_vram = 8.0 * 0.9 = 7.2 GB
        # Q8_0: 7.0 * 1.0 + kv(7, 4096) ≈ 7.61 GB → doesn't fit full GPU
        # But fits in vram + ram → partial offload preferred over lower quant
        assert result.quantization == "Q8_0"
        assert result.strategy == MemoryStrategy.PARTIAL_OFFLOAD

    def test_70b_24gb_vram_partial_or_cpu(self):
        """70B model + 24 GB VRAM → partial offload or CPU strategy."""
        profile = _make_profile(vram_gb=24.0, ram_gb=64.0, available_ram_gb=48.0)
        opt = ModelOptimizer(profile)
        result = opt.pick_quant_and_offload(70.0)

        # usable_vram = 24 * 0.9 = 21.6 GB
        # Q4_K_M model alone = 70 * 0.625 = 43.75 GB + kv
        # Definitely won't fit full GPU
        assert result.strategy in (
            MemoryStrategy.PARTIAL_OFFLOAD,
            MemoryStrategy.CPU_ONLY,
        )

    def test_7b_no_gpu_16gb_ram_cpu_only(self):
        """7B model + 0 VRAM + 16 GB RAM → CPU_ONLY."""
        profile = _make_profile(vram_gb=0.0, ram_gb=16.0, available_ram_gb=12.0)
        opt = ModelOptimizer(profile)
        result = opt.pick_quant_and_offload(7.0)

        assert result.strategy == MemoryStrategy.CPU_ONLY
        assert result.gpu_layers == 0

    def test_405b_8gb_vram_16gb_ram_aggressive_quant(self):
        """405B model + 8 GB VRAM + 16 GB RAM → aggressive_quant with warning."""
        profile = _make_profile(vram_gb=8.0, ram_gb=16.0, available_ram_gb=12.0)
        opt = ModelOptimizer(profile)
        result = opt.pick_quant_and_offload(405.0)

        assert result.strategy == MemoryStrategy.AGGRESSIVE_QUANT
        assert result.warning is not None
        assert len(result.warning) > 0

    @pytest.mark.parametrize(
        "model_size_b,vram_gb,ram_gb,expected_strategy",
        [
            (0.5, 4.0, 8.0, MemoryStrategy.FULL_GPU),
            (3.0, 8.0, 16.0, MemoryStrategy.FULL_GPU),
            (13.0, 0.0, 32.0, MemoryStrategy.CPU_ONLY),
        ],
    )
    def test_parametrized_strategies(
        self,
        model_size_b,
        vram_gb,
        ram_gb,
        expected_strategy,
    ):
        profile = _make_profile(
            vram_gb=vram_gb, ram_gb=ram_gb, available_ram_gb=ram_gb * 0.75
        )
        opt = ModelOptimizer(profile)
        result = opt.pick_quant_and_offload(model_size_b)
        assert result.strategy == expected_strategy


# ---------------------------------------------------------------------------
# Speed prediction
# ---------------------------------------------------------------------------


class TestSpeedPrediction:
    def test_full_gpu_high_confidence(self):
        """FULL_GPU strategy → high confidence."""
        profile = _make_profile(vram_gb=24.0, speed_coeff=100)
        opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=5.0,
            ram_gb=0.0,
            total_gb=5.0,
        )
        speed = opt.predict_speed(7.0, config)
        assert speed.confidence == "high"
        assert speed.tokens_per_second > 0

    def test_partial_offload_medium_confidence_penalty(self):
        """PARTIAL_OFFLOAD → medium confidence and 0.6x penalty."""
        profile = _make_profile(vram_gb=24.0, speed_coeff=100)
        opt = ModelOptimizer(profile)

        full_config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=5.0,
            ram_gb=0.0,
            total_gb=5.0,
        )
        partial_config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=20,
            strategy=MemoryStrategy.PARTIAL_OFFLOAD,
            vram_gb=3.0,
            ram_gb=2.0,
            total_gb=5.0,
        )

        full_speed = opt.predict_speed(7.0, full_config)
        partial_speed = opt.predict_speed(7.0, partial_config)

        assert partial_speed.confidence == "medium"
        # partial should be ~0.6x of full
        ratio = partial_speed.tokens_per_second / full_speed.tokens_per_second
        assert 0.55 <= ratio <= 0.65

    def test_cpu_only_low_confidence_penalty(self):
        """CPU_ONLY → low confidence and 0.15x penalty."""
        profile = _make_profile(vram_gb=24.0, speed_coeff=100)
        opt = ModelOptimizer(profile)

        full_config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=5.0,
            ram_gb=0.0,
            total_gb=5.0,
        )
        cpu_config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=0,
            strategy=MemoryStrategy.CPU_ONLY,
            vram_gb=0.0,
            ram_gb=5.0,
            total_gb=5.0,
        )

        full_speed = opt.predict_speed(7.0, full_config)
        cpu_speed = opt.predict_speed(7.0, cpu_config)

        assert cpu_speed.confidence == "low"
        ratio = cpu_speed.tokens_per_second / full_speed.tokens_per_second
        assert 0.12 <= ratio <= 0.18

    def test_zero_model_size_returns_zero_tps(self):
        """model_size_b=0 → 0 tok/s."""
        profile = _make_profile(vram_gb=24.0)
        opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=0.0,
            ram_gb=0.0,
            total_gb=0.0,
        )
        speed = opt.predict_speed(0, config)
        assert speed.tokens_per_second == 0.0
        assert speed.confidence == "low"

    def test_aggressive_quant_penalty(self):
        """AGGRESSIVE_QUANT gets 0.12x penalty."""
        profile = _make_profile(vram_gb=24.0, speed_coeff=100)
        opt = ModelOptimizer(profile)

        full_config = QuantOffloadResult(
            quantization="Q2_K",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=5.0,
            ram_gb=0.0,
            total_gb=5.0,
        )
        agg_config = QuantOffloadResult(
            quantization="Q2_K",
            gpu_layers=5,
            strategy=MemoryStrategy.AGGRESSIVE_QUANT,
            vram_gb=2.0,
            ram_gb=3.0,
            total_gb=5.0,
        )

        full_speed = opt.predict_speed(7.0, full_config)
        agg_speed = opt.predict_speed(7.0, agg_config)

        ratio = agg_speed.tokens_per_second / full_speed.tokens_per_second
        assert 0.10 <= ratio <= 0.15


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_priority_speed_sorted_by_tps_descending(self):
        """priority='speed' → sorted by tok/s descending."""
        profile = _make_profile(
            vram_gb=24.0, ram_gb=32.0, available_ram_gb=24.0, speed_coeff=100
        )
        opt = ModelOptimizer(profile)
        recs = opt.recommend(priority="speed")

        assert len(recs) > 0
        tps_values = [r.speed.tokens_per_second for r in recs]
        assert tps_values == sorted(tps_values, reverse=True)

    def test_priority_quality_sorted_by_model_size_descending(self):
        """priority='quality' → larger models first."""
        profile = _make_profile(
            vram_gb=24.0, ram_gb=64.0, available_ram_gb=48.0, speed_coeff=100
        )
        opt = ModelOptimizer(profile)
        recs = opt.recommend(priority="quality")

        assert len(recs) > 0
        # First recommendation should be the largest model that can run
        sizes = [float(r.model_size.replace("B", "")) for r in recs]
        # Quality-sorted means largest model size first
        assert sizes[0] >= sizes[-1]

    def test_priority_balanced_fast_enough_first(self):
        """priority='balanced' → models >= 10 tok/s come first."""
        profile = _make_profile(
            vram_gb=24.0, ram_gb=32.0, available_ram_gb=24.0, speed_coeff=100
        )
        opt = ModelOptimizer(profile)
        recs = opt.recommend(priority="balanced")

        assert len(recs) > 0
        # Find the boundary between fast-enough and too-slow
        fast_enough_done = False
        for rec in recs:
            if rec.speed.tokens_per_second < 10.0:
                fast_enough_done = True
            elif fast_enough_done:
                # A fast model appearing after a slow one means wrong ordering
                pytest.fail(
                    f"Fast model ({rec.model_size} at {rec.speed.tokens_per_second} tok/s) "
                    "appeared after slow models in balanced sorting"
                )

    def test_recommend_skips_oom_models(self):
        """Models that would OOM are excluded."""
        profile = _make_profile(vram_gb=0.0, ram_gb=4.0, available_ram_gb=2.0)
        opt = ModelOptimizer(profile)
        recs = opt.recommend()

        # Very small RAM — large models should be filtered out
        for rec in recs:
            if rec.config.warning and "OOM" in rec.config.warning:
                pytest.fail(f"OOM model {rec.model_size} should have been skipped")

    def test_recommendations_have_serve_command(self):
        """Every recommendation includes a valid serve_command."""
        from unittest.mock import patch

        with patch(
            "octomil.model_optimizer.shutil.which", return_value="/usr/bin/octomil"
        ):
            profile = _make_profile(vram_gb=24.0, ram_gb=32.0, available_ram_gb=24.0)
            opt = ModelOptimizer(profile)
        recs = opt.recommend()

        for rec in recs:
            assert "octomil serve" in rec.serve_command
            assert rec.model_size.lower() in rec.serve_command.lower()


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------


class TestEnvVars:
    def test_cuda_backend_includes_flash_attention(self):
        """CUDA backend → OLLAMA_FLASH_ATTENTION=1."""
        profile = _make_profile(vram_gb=24.0, backend="cuda")
        opt = ModelOptimizer(profile)
        env = opt.env_vars()

        assert env["OLLAMA_FLASH_ATTENTION"] == "1"

    def test_metal_backend_includes_flash_attention(self):
        """Metal backend → OLLAMA_FLASH_ATTENTION=1."""
        profile = _make_profile(vram_gb=16.0, backend="metal")
        opt = ModelOptimizer(profile)
        env = opt.env_vars()

        assert env["OLLAMA_FLASH_ATTENTION"] == "1"

    def test_cpu_backend_no_flash_attention(self):
        """CPU backend → no OLLAMA_FLASH_ATTENTION."""
        profile = _make_profile(vram_gb=0.0)
        opt = ModelOptimizer(profile)
        env = opt.env_vars()

        assert "OLLAMA_FLASH_ATTENTION" not in env

    def test_rocm_backend_no_flash_attention(self):
        """ROCm backend → no OLLAMA_FLASH_ATTENTION."""
        profile = _make_profile(vram_gb=16.0, backend="rocm")
        opt = ModelOptimizer(profile)
        env = opt.env_vars()

        assert "OLLAMA_FLASH_ATTENTION" not in env

    def test_32_threads_num_parallel_4(self):
        """32 threads → OLLAMA_NUM_PARALLEL=4 (min(4, 32//4)=4)."""
        profile = _make_profile(threads=32)
        opt = ModelOptimizer(profile)
        env = opt.env_vars()

        assert env["OLLAMA_NUM_PARALLEL"] == "4"

    def test_8_threads_num_parallel_2(self):
        """8 threads → OLLAMA_NUM_PARALLEL=2."""
        profile = _make_profile(threads=8)
        opt = ModelOptimizer(profile)
        env = opt.env_vars()

        assert env["OLLAMA_NUM_PARALLEL"] == "2"

    def test_64gb_ram_max_loaded_models_8(self):
        """64 GB RAM → OLLAMA_MAX_LOADED_MODELS=8."""
        profile = _make_profile(ram_gb=64.0, available_ram_gb=48.0)
        opt = ModelOptimizer(profile)
        env = opt.env_vars()

        assert env["OLLAMA_MAX_LOADED_MODELS"] == "8"

    def test_16gb_ram_max_loaded_models_2(self):
        """16 GB RAM → OLLAMA_MAX_LOADED_MODELS=2."""
        profile = _make_profile(ram_gb=16.0, available_ram_gb=12.0)
        opt = ModelOptimizer(profile)
        env = opt.env_vars()

        assert env["OLLAMA_MAX_LOADED_MODELS"] == "2"

    def test_gpu_overhead_always_present(self):
        """OLLAMA_GPU_OVERHEAD=256 is always set."""
        profile = _make_profile()
        opt = ModelOptimizer(profile)
        env = opt.env_vars()

        assert env["OLLAMA_GPU_OVERHEAD"] == "256"


# ---------------------------------------------------------------------------
# serve_command (multi-engine)
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_octomil_explicit_full_gpu(self):
        """Explicit octomil engine → 'octomil serve tag'."""
        profile = _make_profile(vram_gb=24.0)
        opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=5.0,
            ram_gb=0.0,
            total_gb=5.0,
        )
        cmd = opt.serve_command("phi-mini", config, engine="octomil")
        assert cmd == "octomil serve phi-mini"

    def test_octomil_cpu_only(self):
        """CPU-only → 'octomil serve tag --engine cpu'."""
        profile = _make_profile(vram_gb=0.0)
        opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=0,
            strategy=MemoryStrategy.CPU_ONLY,
            vram_gb=0.0,
            ram_gb=5.0,
            total_gb=5.0,
        )
        cmd = opt.serve_command("phi-mini", config, engine="octomil")
        assert cmd == "octomil serve phi-mini --engine cpu"

    def test_auto_detects_octomil(self):
        """Auto-detect picks octomil when on PATH."""
        from unittest.mock import patch

        with patch(
            "octomil.model_optimizer.shutil.which",
            side_effect=lambda x: "/usr/bin/octomil" if x == "octomil" else None,
        ):
            profile = _make_profile(vram_gb=24.0)
            opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=5.0,
            ram_gb=0.0,
            total_gb=5.0,
        )
        cmd = opt.serve_command("phi-mini", config)
        assert cmd == "octomil serve phi-mini"

    def test_auto_detects_ollama(self):
        """Auto-detect picks ollama when octomil not on PATH."""
        from unittest.mock import patch

        def _which(name: str) -> str | None:
            if name == "ollama":
                return "/usr/bin/ollama"
            return None

        with patch("octomil.model_optimizer.shutil.which", side_effect=_which):
            profile = _make_profile(vram_gb=24.0)
            opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=5.0,
            ram_gb=0.0,
            total_gb=5.0,
        )
        cmd = opt.serve_command("phi-mini", config)
        assert cmd == "ollama run phi-mini"

    def test_auto_detects_llamacpp(self):
        """Auto-detect picks llama.cpp when only llama-server on PATH."""
        from unittest.mock import patch

        def _which(name: str) -> str | None:
            if name == "llama-server":
                return "/usr/bin/llama-server"
            return None

        with patch("octomil.model_optimizer.shutil.which", side_effect=_which):
            profile = _make_profile(vram_gb=24.0)
            opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=5.0,
            ram_gb=0.0,
            total_gb=5.0,
        )
        cmd = opt.serve_command("phi-mini", config)
        assert "llama-server" in cmd

    def test_ollama_engine_full_gpu(self):
        """Ollama engine → 'ollama run tag'."""
        profile = _make_profile(vram_gb=24.0)
        opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=-1,
            strategy=MemoryStrategy.FULL_GPU,
            vram_gb=5.0,
            ram_gb=0.0,
            total_gb=5.0,
        )
        cmd = opt.serve_command("llama2:7b-q4_k_m", config, engine="ollama")
        assert cmd == "ollama run llama2:7b-q4_k_m"

    def test_ollama_engine_gpu_layers_0(self):
        """Ollama engine gpu_layers=0 → 'OLLAMA_NUM_GPU=0 ollama run tag'."""
        profile = _make_profile(vram_gb=0.0)
        opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=0,
            strategy=MemoryStrategy.CPU_ONLY,
            vram_gb=0.0,
            ram_gb=5.0,
            total_gb=5.0,
        )
        cmd = opt.serve_command("llama2:7b-q4_k_m", config, engine="ollama")
        assert cmd == "OLLAMA_NUM_GPU=0 ollama run llama2:7b-q4_k_m"

    def test_ollama_engine_partial_offload(self):
        """Ollama engine gpu_layers=28 → 'OLLAMA_NUM_GPU=28 ollama run tag'."""
        profile = _make_profile(vram_gb=12.0)
        opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=28,
            strategy=MemoryStrategy.PARTIAL_OFFLOAD,
            vram_gb=8.0,
            ram_gb=4.0,
            total_gb=12.0,
        )
        cmd = opt.serve_command("llama2:13b-q4_k_m", config, engine="ollama")
        assert cmd == "OLLAMA_NUM_GPU=28 ollama run llama2:13b-q4_k_m"

    def test_llamacpp_engine(self):
        """llama.cpp engine → 'llama-server -m tag -ngl N'."""
        profile = _make_profile(vram_gb=12.0)
        opt = ModelOptimizer(profile)
        config = QuantOffloadResult(
            quantization="Q4_K_M",
            gpu_layers=28,
            strategy=MemoryStrategy.PARTIAL_OFFLOAD,
            vram_gb=8.0,
            ram_gb=4.0,
            total_gb=12.0,
        )
        cmd = opt.serve_command("model.gguf", config, engine="llama.cpp")
        assert cmd == "llama-server -m model.gguf -ngl 28"


# ---------------------------------------------------------------------------
# KV cache calculation
# ---------------------------------------------------------------------------


class TestKVCache:
    def test_7b_4096_context(self):
        """7B model + 4096 context → ~0.6 GB."""
        kv = _kv_cache_gb(7.0, 4096)
        assert 0.55 <= kv <= 0.70

    def test_7b_32768_context(self):
        """7B model + 32768 context → ~4.9 GB."""
        kv = _kv_cache_gb(7.0, 32768)
        assert 4.5 <= kv <= 5.2

    def test_tiny_model_uses_default(self):
        """Very small model uses the smallest KV cache coefficient."""
        kv = _kv_cache_gb(0.5, 1000)
        # 0.5B >= threshold 1 → per_1k = 0.05 → 0.05 * 1.0 = 0.05
        assert kv == pytest.approx(0.05, abs=0.01)

    def test_70b_model_high_kv(self):
        """70B model has higher KV per 1K tokens."""
        kv = _kv_cache_gb(70.0, 4096)
        # 70B threshold → per_1k = 1.0 → 1.0 * 4.096 = 4.096
        assert kv == pytest.approx(4.096, abs=0.01)

    def test_zero_context(self):
        """0 context tokens → 0 KV cache."""
        kv = _kv_cache_gb(7.0, 0)
        assert kv == 0.0


# ---------------------------------------------------------------------------
# Model memory helpers
# ---------------------------------------------------------------------------


class TestModelMemory:
    def test_model_memory_gb(self):
        """7B at Q4_K_M → 7 * 0.625 = 4.375 GB."""
        mem = _model_memory_gb(7.0, "Q4_K_M")
        assert mem == pytest.approx(4.375)

    def test_total_memory_includes_kv_cache(self):
        """Total memory = model weights + KV cache."""
        total = _total_memory_gb(7.0, "Q4_K_M", 4096)
        model = _model_memory_gb(7.0, "Q4_K_M")
        kv = _kv_cache_gb(7.0, 4096)
        assert total == pytest.approx(model + kv)


# ---------------------------------------------------------------------------
# Apple Silicon / Metal unified memory
# ---------------------------------------------------------------------------


class TestMetalUnifiedMemory:
    def test_metal_uses_total_ram_as_vram(self):
        """Metal backend treats total_ram - 4 GB as usable VRAM."""
        profile = _make_profile(
            vram_gb=16.0, backend="metal", ram_gb=32.0, available_ram_gb=24.0
        )
        opt = ModelOptimizer(profile)
        # usable_vram = total_ram_gb - 4 = 32 - 4 = 28 GB
        assert opt.usable_vram == 28.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_gpu_cpu_speed_from_gflops(self):
        """CPU-only profile uses estimated_gflops for speed coefficient."""
        cpu = CPUInfo(
            brand="Test CPU",
            cores=8,
            threads=16,
            base_speed_ghz=3.0,
            architecture="x86_64",
            estimated_gflops=500.0,
        )
        profile = HardwareProfile(
            gpu=None,
            cpu=cpu,
            total_ram_gb=32.0,
            available_ram_gb=24.0,
            platform="linux",
            best_backend="cpu",
        )
        opt = ModelOptimizer(profile)
        # speed_coeff = max(500.0 / 50.0, 1.0) = 10.0
        assert opt._speed_coeff == 10.0

    def test_no_gpu_zero_gflops_floor_at_1(self):
        """Zero estimated_gflops → speed_coeff floor is 1.0."""
        cpu = CPUInfo(
            brand="Test CPU",
            cores=1,
            threads=1,
            base_speed_ghz=0.5,
            architecture="x86_64",
            estimated_gflops=0.0,
        )
        profile = HardwareProfile(
            gpu=None,
            cpu=cpu,
            total_ram_gb=4.0,
            available_ram_gb=2.0,
            platform="linux",
            best_backend="cpu",
        )
        opt = ModelOptimizer(profile)
        assert opt._speed_coeff == 1.0

    def test_recommendation_reason_includes_warning(self):
        """_recommendation_reason includes warning when present."""
        config = QuantOffloadResult(
            quantization="Q2_K",
            gpu_layers=0,
            strategy=MemoryStrategy.AGGRESSIVE_QUANT,
            vram_gb=0.0,
            ram_gb=5.0,
            total_gb=5.0,
            warning="aggressive quant",
        )
        speed = SpeedEstimate(
            tokens_per_second=1.5,
            backend="cpu",
            strategy=MemoryStrategy.AGGRESSIVE_QUANT,
            confidence="low",
        )
        reason = ModelOptimizer._recommendation_reason("7B", config, speed)
        assert "aggressive quant" in reason
        assert "7B" in reason


# ---------------------------------------------------------------------------
# _resolve_model_size
# ---------------------------------------------------------------------------


class TestResolveModelSize:
    @pytest.mark.parametrize(
        "tag,expected",
        [
            ("llama3.1:8b", 8.0),
            ("mistral:7b", 7.0),
            ("llama3.1:70b", 70.0),
            ("qwen2:0.5b", 0.5),
            ("phi-mini", 3.8),
            ("smollm", 0.36),
            ("mixtral", 46.7),
            ("whisper-tiny", 0.039),
        ],
    )
    def test_known_models(self, tag: str, expected: float) -> None:
        assert _resolve_model_size(tag) == pytest.approx(expected, rel=0.01)

    def test_unknown_model_returns_none(self) -> None:
        assert _resolve_model_size("totally-unknown-model") is None

    def test_size_in_name(self) -> None:
        """Model name with embedded size like 'llama3.1:8b-instruct'."""
        assert _resolve_model_size("llama3.1:8b-instruct") == 8.0

    def test_decimal_size(self) -> None:
        assert _resolve_model_size("model:1.5b") == 1.5


# ---------------------------------------------------------------------------
# _has_explicit_quant (CLI helper)
# ---------------------------------------------------------------------------


class TestHasExplicitQuant:
    def test_no_colon_no_quant(self) -> None:
        from octomil.cli import _has_explicit_quant

        assert _has_explicit_quant("llama3.1") is False

    def test_size_variant_not_quant(self) -> None:
        from octomil.cli import _has_explicit_quant

        assert _has_explicit_quant("llama3.1:8b") is False

    @pytest.mark.parametrize(
        "tag",
        [
            "llama3.1:q4_k_m",
            "gemma-1b:8bit",
            "model:fp16",
            "model:q8_0",
            "model:Q6_K",
        ],
    )
    def test_explicit_quant_detected(self, tag: str) -> None:
        from octomil.cli import _has_explicit_quant

        assert _has_explicit_quant(tag) is True
