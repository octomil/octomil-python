"""Cross-model perf knobs in ``octomil.runtime.engines.sherpa.engine``.

Pins:
  - voice catalog cache (process-lifetime; cleared via the public
    release hook).
  - default ONNX-Runtime thread count for sherpa-onnx TTS (cores
    capped at 4, env-overridable).

Both apply uniformly to Kokoro / Piper / Pocket via the shared
sherpa-onnx code path, hence the cross-model framing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from octomil.runtime.engines.sherpa.engine import (
    _VOICE_CATALOG_CACHE,
    _default_sherpa_num_threads,
    release_voice_catalog_cache,
    resolve_voice_catalog,
)

# ---------------------------------------------------------------------------
# Voice catalog cache
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_voice_cache_between_tests():
    release_voice_catalog_cache()
    yield
    release_voice_catalog_cache()


def test_voice_catalog_cache_hits_second_call_without_disk_read(tmp_path):
    """First call reads ``voices.txt`` + ``VERSION`` from disk;
    second call returns the cached object verbatim. The disk-read
    counter is the regression: pre-cache, every TTS dispatch
    re-stat'd and re-read these sidecars."""
    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    (artifact_dir / "voices.txt").write_text("af_alpha\naf_beta\naf_gamma\n")
    (artifact_dir / "VERSION").write_text("kokoro-en-v0_19")

    read_count = 0
    real_open = open

    def _counting_open(path, *args, **kwargs):
        nonlocal read_count
        if "voices.txt" in str(path) or path.endswith("VERSION"):
            read_count += 1
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=_counting_open):
        first = resolve_voice_catalog("kokoro-en-v0_19", prepared_model_dir=str(artifact_dir))
        post_first = read_count
        second = resolve_voice_catalog("kokoro-en-v0_19", prepared_model_dir=str(artifact_dir))
        post_second = read_count

    assert first.voices == ("af_alpha", "af_beta", "af_gamma")
    assert second is first, "cache should return the same object identity"
    assert post_first > 0, "first call must hit disk"
    assert post_second == post_first, (
        f"second call must NOT hit disk (delta {post_second - post_first} reads); " "voice catalog cache regressed"
    )


def test_voice_catalog_cache_distinguishes_artifact_dirs(tmp_path):
    """Two prepared dirs at different paths must be cached
    independently, even for the same model name. Otherwise a v1
    cache would shadow a v2 dir."""
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / "voices.txt").write_text("af_v1\n")

    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "voices.txt").write_text("af_v2\n")

    cat_v1 = resolve_voice_catalog("kokoro-en-v0_19", prepared_model_dir=str(v1))
    cat_v2 = resolve_voice_catalog("kokoro-en-v0_19", prepared_model_dir=str(v2))

    assert cat_v1.voices == ("af_v1",)
    assert cat_v2.voices == ("af_v2",)
    assert cat_v1 is not cat_v2


def test_release_voice_catalog_cache_clears_entries(tmp_path):
    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    (artifact_dir / "voices.txt").write_text("af_alpha\n")

    resolve_voice_catalog("kokoro-en-v0_19", prepared_model_dir=str(artifact_dir))
    assert len(_VOICE_CATALOG_CACHE) == 1

    release_voice_catalog_cache()

    assert _VOICE_CATALOG_CACHE == {}


def test_release_warmed_backends_clears_voice_catalog_cache(tmp_path):
    """The kernel's ``release_warmed_backends`` is the public 'drop
    my caches' surface; it must clear the voice catalog cache too."""
    from octomil.execution.kernel import ExecutionKernel

    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    (artifact_dir / "voices.txt").write_text("af_alpha\n")
    resolve_voice_catalog("kokoro-en-v0_19", prepared_model_dir=str(artifact_dir))
    assert len(_VOICE_CATALOG_CACHE) == 1

    kernel = ExecutionKernel()
    kernel.release_warmed_backends()

    assert _VOICE_CATALOG_CACHE == {}


# ---------------------------------------------------------------------------
# Default ONNX Runtime thread count
# ---------------------------------------------------------------------------


def test_default_num_threads_caps_at_four_on_high_core_machines(monkeypatch):
    """16-core box must NOT default to 16 threads — diminishing
    returns above 4 + thrash on shared CI runners."""
    monkeypatch.delenv("OCTOMIL_SHERPA_NUM_THREADS", raising=False)
    with patch("octomil.runtime.engines.sherpa.engine.os.cpu_count", return_value=16):
        assert _default_sherpa_num_threads() == 4


def test_default_num_threads_uses_actual_core_count_below_cap(monkeypatch):
    """3-core machine: default = 3, not 4 (don't oversubscribe)."""
    monkeypatch.delenv("OCTOMIL_SHERPA_NUM_THREADS", raising=False)
    with patch("octomil.runtime.engines.sherpa.engine.os.cpu_count", return_value=3):
        assert _default_sherpa_num_threads() == 3


def test_default_num_threads_floors_at_one(monkeypatch):
    monkeypatch.delenv("OCTOMIL_SHERPA_NUM_THREADS", raising=False)
    with patch("octomil.runtime.engines.sherpa.engine.os.cpu_count", return_value=None):
        # cpu_count -> None => fall through to 2 (legacy default).
        result = _default_sherpa_num_threads()
        assert result >= 1


def test_octomil_sherpa_num_threads_env_overrides_default(monkeypatch):
    monkeypatch.setenv("OCTOMIL_SHERPA_NUM_THREADS", "8")
    with patch("octomil.runtime.engines.sherpa.engine.os.cpu_count", return_value=2):
        # env override wins even when cores < env value.
        assert _default_sherpa_num_threads() == 8


def test_octomil_sherpa_num_threads_env_invalid_falls_back_to_default(monkeypatch):
    """Garbage env value must NOT crash dispatch — fall back to
    the cores-capped-at-4 default."""
    monkeypatch.setenv("OCTOMIL_SHERPA_NUM_THREADS", "not-a-number")
    with patch("octomil.runtime.engines.sherpa.engine.os.cpu_count", return_value=8):
        assert _default_sherpa_num_threads() == 4


def test_octomil_sherpa_num_threads_env_zero_floors_to_one(monkeypatch):
    """``num_threads=0`` is a configuration footgun (means 'all
    cores' in some libraries, undefined in ONNX Runtime). Floor to
    1 so we never feed 0 into sherpa-onnx."""
    monkeypatch.setenv("OCTOMIL_SHERPA_NUM_THREADS", "0")
    assert _default_sherpa_num_threads() == 1
