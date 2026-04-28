"""Issue E: ``octomil.runtime.engines.sherpa.engine`` must be
importable on stripped embedded Pythons without ``audioop``.

CPython's stdlib ``wave`` transitively imports ``audioop``, which
is excluded from Ren'Py / PyInstaller / Bazel-stripped CPython 3.9
builds. Importing the Sherpa engine at module load time would then
fail before the SDK could surface PR #467's disambiguated TTS
error messages — ``_sherpa_tts_runtime_loadable`` would just
return False and the dispatch path would emit the generic
``local_tts_runtime_unavailable``.

These regressions:

  1. Importing the engine module must succeed even when the
     ``audioop`` import fails (we simulate the exclusion via
     ``sys.modules`` manipulation).
  2. ``_samples_to_wav`` must produce a valid RIFF/WAVE/fmt /data
     header without ``wave``.
  3. ``_sherpa_tts_runtime_loadable`` must log the engine-import
     failure at WARNING (rather than silently swallowing it) so
     embedded users can see *why* the runtime is unavailable.
"""

from __future__ import annotations

import logging
import struct
import sys

# ---------------------------------------------------------------------------
# (1) Engine module imports without ``audioop`` / ``wave``
# ---------------------------------------------------------------------------


def test_sherpa_engine_top_level_does_not_import_wave():
    """Pure-source check: ``import wave`` must not appear in the
    engine's top-level imports. Even on a Python that has
    ``audioop`` available, depending on it pulls a stdlib edge that
    Ren'Py / stripped PyInstaller bundles ship without."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent / "octomil" / "runtime" / "engines" / "sherpa" / "engine.py"
    ).read_text()
    # Read only the import block at the top of the file.
    header = src.split("def ", 1)[0]
    assert "import wave" not in header, (
        "octomil/runtime/engines/sherpa/engine.py must not ``import wave`` at module load — "
        "it transitively imports ``audioop``, which is missing on Ren'Py / stripped CPython 3.9 builds."
    )


def test_engine_imports_under_simulated_missing_audioop(monkeypatch):
    """Force-import the engine module after blocking ``audioop`` /
    ``wave`` so a fresh import on Ren'Py-style stripped CPython is
    exercised. Both the module load AND the WAV-encoding helper
    must keep working."""
    # Pre-clear so the next ``import`` re-runs the module body.
    for mod in list(sys.modules):
        if mod.startswith("octomil.runtime.engines.sherpa"):
            sys.modules.pop(mod, None)
    sys.modules.pop("wave", None)
    sys.modules.pop("audioop", None)

    # Block the next ``import audioop`` so anything that transitively
    # reaches it (like ``wave``) fails. The engine must NOT touch
    # either at module load.
    blocked = {"audioop"}

    class _BlockMissingAudioop:
        def find_spec(self, name, path=None, target=None):  # noqa: ARG002
            if name in blocked:
                raise ImportError(f"simulated missing module: {name!r}")
            return None

    finder = _BlockMissingAudioop()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    # The engine module must import cleanly.
    from octomil.runtime.engines.sherpa import engine as sherpa_engine

    # And the WAV-encoder must work without ``wave``/``audioop``.
    samples = [0.0, 0.5, -0.5, 1.0, -1.0]
    wav = sherpa_engine._samples_to_wav(samples, sample_rate=24000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    # Verify the engine didn't pull ``wave`` in despite our hook
    # being active. (Some indirect helper might still trigger it
    # later; that's outside this test's scope.)
    assert "audioop" not in sys.modules, "engine import path must not transitively load ``audioop``"


# ---------------------------------------------------------------------------
# (2) _samples_to_wav builds a valid RIFF/WAVE/fmt /data header
# ---------------------------------------------------------------------------


def test_samples_to_wav_emits_valid_pcm16_header():
    """Hand-built RIFF/WAVE encoder must round-trip through the
    ``wave`` module's reader (when ``wave`` IS available, as in
    normal CPython on this CI runner). That confirms the header is
    byte-identical to what ``wave.open(..., "wb")`` used to produce."""
    import wave

    from octomil.runtime.engines.sherpa.engine import _samples_to_wav

    samples = [0.0, 0.25, 0.5, 0.75, 1.0, -0.5, -1.0]
    sample_rate = 22050
    blob = _samples_to_wav(samples, sample_rate)

    # Spot-check magic bytes + chunk sizes via struct directly so
    # the test doesn't depend on ``wave``'s availability.
    assert blob[:4] == b"RIFF"
    riff_size = struct.unpack_from("<I", blob, 4)[0]
    assert blob[8:12] == b"WAVE"
    assert blob[12:16] == b"fmt "
    fmt_size = struct.unpack_from("<I", blob, 16)[0]
    assert fmt_size == 16
    audio_format, n_channels, sr, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", blob, 20)
    assert audio_format == 1  # PCM
    assert n_channels == 1
    assert sr == sample_rate
    assert bits == 16
    assert block_align == n_channels * (bits // 8)
    assert byte_rate == sample_rate * block_align
    # data chunk follows the 36-byte header preamble.
    assert blob[36:40] == b"data"
    pcm_size = struct.unpack_from("<I", blob, 40)[0]
    assert pcm_size == len(samples) * 2  # PCM16 mono → 2 bytes/sample
    assert riff_size == 4 + 8 + fmt_size + 8 + pcm_size
    assert len(blob) == 44 + pcm_size

    # Cross-check with the stdlib reader so the header is fully
    # parseable. (This part requires ``wave`` to be importable in
    # the test runner; on stripped Pythons it would be skipped, but
    # our regular CI has it.)
    import io

    with wave.open(io.BytesIO(blob), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == sample_rate
        assert wf.getnframes() == len(samples)


def test_samples_to_wav_clips_outside_unit_range():
    """Values outside [-1, 1] must clip to PCM16 min/max rather than
    overflow the ``struct.pack('<h', ...)`` integer range."""
    from octomil.runtime.engines.sherpa.engine import _samples_to_wav

    blob = _samples_to_wav([2.5, -3.7], sample_rate=8000)
    pcm = blob[44:]  # past the 44-byte standard header
    s0, s1 = struct.unpack("<hh", pcm)
    assert s0 == 32767
    assert s1 == -32767


# ---------------------------------------------------------------------------
# (3) Kernel gate logs engine-import failures
# ---------------------------------------------------------------------------


def test_runtime_loadable_logs_engine_import_failure(monkeypatch, caplog):
    """When ``octomil.runtime.engines.sherpa`` raises on import,
    ``_sherpa_tts_runtime_loadable`` must log a WARNING explaining
    why — not just silently return False. Embedded-Python users
    need this to diagnose ``local_tts_runtime_unavailable``."""
    from octomil.execution.kernel import ExecutionKernel

    # Force the next ``from octomil.runtime.engines.sherpa import …``
    # inside ``_sherpa_tts_runtime_loadable`` to raise. Easiest way:
    # remove the module from sys.modules so it re-imports, then
    # install a meta-path finder that blocks it.
    sys.modules.pop("octomil.runtime.engines.sherpa", None)

    class _BlockSherpa:
        def find_spec(self, name, path=None, target=None):  # noqa: ARG002
            if name == "octomil.runtime.engines.sherpa":
                raise ImportError("simulated stripped-Python missing audioop")
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockSherpa(), *sys.meta_path])

    with caplog.at_level(logging.WARNING, logger="octomil.execution.kernel"):
        result = ExecutionKernel._sherpa_tts_runtime_loadable("kokoro-82m")

    assert result is False
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "Sherpa TTS engine module failed to import" in m for m in msgs
    ), f"expected a WARNING explaining the engine-import failure; got: {msgs!r}"
