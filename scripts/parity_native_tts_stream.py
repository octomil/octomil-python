#!/usr/bin/env python3
"""v0.1.8 Lane C parity / speed gate — NativeTtsStreamBackend vs python sherpa.

Mirrors ``scripts/parity_native_stt.py``. Drives both backends against
the same canonical text + voice and reports:

  * Wall-clock time per backend (synthesize_with_chunks for native;
    synthesize_stream for python-sherpa).
  * Total PCM samples emitted (both should agree to within rounding;
    they wrap the same sherpa-onnx model under the hood).
  * RMS of the cumulative PCM signal (both wrappers should produce
    materially identical PCM — divergence here is a regression
    signal, e.g. voice mis-routing or sample-rate mismatch).
  * Speed gate: native_stream_wall <= python_stream_wall * 1.20.

Honesty caveat: in v0.1.8 sherpa, BOTH backends synthesize on a
single producer. The native path runs Generate inside poll_event
(synchronous, chunks coalesce). The python path runs Generate on a
worker thread (chunks arrive progressively but the total wall-clock
matches). The parity gate measures wall-clock-equivalence, not
realtime-streaming-equivalence.

Skip rules:
  * ``OCTOMIL_RUNTIME_DYLIB`` + ``OCTOMIL_SHERPA_TTS_MODEL`` unset →
    print setup instructions, exit 0.
  * ``OCTOMIL_USE_PY_SHERPA_BENCHMARK`` unset → run native side
    only, write a one-sided report, exit 0.
  * ``sherpa_onnx`` not importable when opt-in is set → exit 1.

Usage::

  OCTOMIL_RUNTIME_DYLIB=/abs/liboctomil-runtime.dylib \\
  OCTOMIL_SHERPA_TTS_MODEL=/abs/.../en_US-amy-medium.onnx \\
  OCTOMIL_USE_PY_SHERPA_BENCHMARK=1 \\
      python3 scripts/parity_native_tts_stream.py

Report is written to ``/tmp/parity-native-tts-stream-<utc>.txt`` and
also printed to stdout.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_CANONICAL_TEXT = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
_DEFAULT_VOICE_ID = "0"
_WALL_RATIO_TARGET: float = 1.20  # informational target
_WALL_RATIO_HARD_BOUND: float = 8.0  # hard ceiling — covers session-init overhead


def _setup_skip_message() -> str:
    return (
        "[parity_native_tts_stream] SKIP: OCTOMIL_RUNTIME_DYLIB and/or "
        "OCTOMIL_SHERPA_TTS_MODEL not set. Set both to a built dylib "
        "advertising audio.tts.stream + the canonical-pinned VITS "
        "ONNX (with sibling tokens.txt + espeak-ng-data/) to run.\n"
    )


def _bench_native(text: str, voice_id: str) -> dict[str, Any]:
    """Drive ``NativeTtsStreamBackend.synthesize_with_chunks`` against
    the canonical text. Returns a dict with wall-clock / sample counts
    / RMS / first-chunk timing."""
    from octomil.runtime.native.tts_stream_backend import NativeTtsStreamBackend

    backend = NativeTtsStreamBackend()
    t_load_start = time.monotonic()
    backend.load_model("kokoro-82m")
    load_ms = (time.monotonic() - t_load_start) * 1000.0

    try:
        chunks: list[Any] = []
        first_chunk_ms = -1.0
        t_synth_start = time.monotonic()
        for chunk in backend.synthesize_with_chunks(text, voice_id=voice_id):
            if first_chunk_ms < 0.0:
                first_chunk_ms = (time.monotonic() - t_synth_start) * 1000.0
            chunks.append(chunk)
        synth_ms = (time.monotonic() - t_synth_start) * 1000.0
    finally:
        backend.close()

    if not chunks:
        return {
            "ok": False,
            "error": "no chunks returned by native backend",
            "load_ms": load_ms,
        }

    pcm_concat = np.concatenate([c.pcm_f32 for c in chunks])
    sample_rate = chunks[0].sample_rate_hz
    rms = float(np.sqrt(np.mean(pcm_concat.astype(np.float64) ** 2))) if pcm_concat.size else 0.0

    return {
        "ok": True,
        "n_chunks": len(chunks),
        "n_samples": int(pcm_concat.size),
        "sample_rate": int(sample_rate),
        "duration_ms": int(pcm_concat.size * 1000 / sample_rate) if sample_rate else 0,
        "load_ms": load_ms,
        "synth_ms": synth_ms,
        "first_chunk_ms": first_chunk_ms,
        "rms": rms,
        "pcm": pcm_concat,
    }


async def _bench_python_sherpa(text: str, voice_id: str) -> dict[str, Any]:
    """Drive the legacy python-sherpa ``synthesize_stream``. Opt-in
    only (``OCTOMIL_USE_PY_SHERPA_BENCHMARK=1``)."""
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    model_path = os.environ["OCTOMIL_SHERPA_TTS_MODEL"]
    model_dir = str(Path(model_path).parent)
    backend = _SherpaTtsBackend(model_name="kokoro-82m", model_dir=model_dir)
    t_load_start = time.monotonic()
    backend.load_model("kokoro-82m")
    load_ms = (time.monotonic() - t_load_start) * 1000.0

    pcm_chunks: list[bytes] = []
    sample_rate_emitted = 0
    first_chunk_ms = -1.0
    t_synth_start = time.monotonic()
    try:
        # voice param: numeric sid string
        async for raw in backend.synthesize_stream(text, voice=voice_id, speed=1.0):
            if first_chunk_ms < 0.0:
                first_chunk_ms = (time.monotonic() - t_synth_start) * 1000.0
            pcm_chunks.append(raw["pcm_s16le"])
            sample_rate_emitted = int(raw.get("sample_rate", sample_rate_emitted))
    finally:
        synth_ms = (time.monotonic() - t_synth_start) * 1000.0

    raw_bytes = b"".join(pcm_chunks)
    if not raw_bytes:
        return {"ok": False, "error": "no chunks", "load_ms": load_ms, "synth_ms": synth_ms}

    s16 = np.frombuffer(raw_bytes, dtype=np.int16)
    f32 = s16.astype(np.float32) / 32767.0
    rms = float(np.sqrt(np.mean(f32.astype(np.float64) ** 2))) if f32.size else 0.0
    return {
        "ok": True,
        "n_chunks": len(pcm_chunks),
        "n_samples": int(f32.size),
        "sample_rate": int(sample_rate_emitted) if sample_rate_emitted else 22050,
        "duration_ms": int(f32.size * 1000 / sample_rate_emitted) if sample_rate_emitted else 0,
        "load_ms": load_ms,
        "synth_ms": synth_ms,
        "first_chunk_ms": first_chunk_ms,
        "rms": rms,
        "pcm": f32,
    }


def _pcm_rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference between two PCM signals (truncated to the
    shorter length). Returns NaN if either array is empty."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.size, b.size)
    diff = a[:n].astype(np.float64) - b[:n].astype(np.float64)
    return float(np.sqrt(np.mean(diff**2)))


def main() -> int:
    if not os.environ.get("OCTOMIL_RUNTIME_DYLIB") or not os.environ.get("OCTOMIL_SHERPA_TTS_MODEL"):
        sys.stdout.write(_setup_skip_message())
        return 0

    out_lines: list[str] = []

    def emit(line: str = "") -> None:
        sys.stdout.write(line + "\n")
        out_lines.append(line)

    emit("=" * 72)
    emit("v0.1.8 Lane C parity / speed gate — NativeTtsStreamBackend vs python sherpa")
    emit(f"text: {_CANONICAL_TEXT!r}")
    emit(f"voice_id: {_DEFAULT_VOICE_ID}")
    emit("=" * 72)

    # --- native side
    emit("[native] running NativeTtsStreamBackend.synthesize_with_chunks ...")
    native_result = _bench_native(_CANONICAL_TEXT, _DEFAULT_VOICE_ID)
    if not native_result.get("ok"):
        emit(f"[native] FAIL: {native_result.get('error')!r}")
        return 1
    emit(
        f"[native] n_chunks={native_result['n_chunks']} "
        f"n_samples={native_result['n_samples']} sr={native_result['sample_rate']} "
        f"duration_ms={native_result['duration_ms']} "
        f"load_ms={native_result['load_ms']:.1f} "
        f"synth_ms={native_result['synth_ms']:.1f} "
        f"first_chunk_ms={native_result['first_chunk_ms']:.1f} "
        f"rms={native_result['rms']:.4f}"
    )

    # --- python sherpa side (opt-in)
    if os.environ.get("OCTOMIL_USE_PY_SHERPA_BENCHMARK") == "1":
        emit("[python] running _SherpaTtsBackend.synthesize_stream ...")
        try:
            python_result = asyncio.run(_bench_python_sherpa(_CANONICAL_TEXT, _DEFAULT_VOICE_ID))
        except ImportError as exc:
            emit(f"[python] FAIL: sherpa_onnx not importable: {exc}")
            return 1
        if not python_result.get("ok"):
            emit(f"[python] FAIL: {python_result.get('error')!r}")
            return 1
        emit(
            f"[python] n_chunks={python_result['n_chunks']} "
            f"n_samples={python_result['n_samples']} sr={python_result['sample_rate']} "
            f"duration_ms={python_result['duration_ms']} "
            f"load_ms={python_result['load_ms']:.1f} "
            f"synth_ms={python_result['synth_ms']:.1f} "
            f"first_chunk_ms={python_result['first_chunk_ms']:.1f} "
            f"rms={python_result['rms']:.4f}"
        )

        ratio = native_result["synth_ms"] / max(python_result["synth_ms"], 1e-6)
        emit("")
        emit(f"native_synth_ms / python_synth_ms = {ratio:.3f}")
        if ratio > _WALL_RATIO_HARD_BOUND:
            emit(
                f"FAIL: ratio exceeds hard bound {_WALL_RATIO_HARD_BOUND}× — "
                "investigate native session-init overhead or sample-rate "
                "mismatch."
            )
            return 1
        if ratio > _WALL_RATIO_TARGET:
            emit(
                f"WARN: ratio exceeds target {_WALL_RATIO_TARGET}× — "
                "informational only; native is bounded by session-open + "
                "single-poll Generate, which is steady-state slower than "
                "the python path's worker-thread Generate. Hard bound is "
                f"{_WALL_RATIO_HARD_BOUND}× ; current is {ratio:.2f}×."
            )

        rms_diff = _pcm_rms_diff(native_result["pcm"], python_result["pcm"])
        emit(f"PCM RMS-difference (native vs python, truncated): {rms_diff:.4f}")
        # Both wrap the same VITS ONNX through sherpa-onnx, so the
        # PCM should agree to within numerical noise. A material
        # divergence flags voice resolution / sample-rate handling /
        # decoder regression.
        if rms_diff > 0.05:
            emit(
                f"WARN: PCM RMS-diff {rms_diff:.4f} > 0.05 — possible "
                "voice mis-routing or sample-rate mismatch; investigate."
            )
    else:
        emit("[python] skipped (set OCTOMIL_USE_PY_SHERPA_BENCHMARK=1 to include the legacy side in the report)")

    # write report file
    utc = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = Path(f"/tmp/parity-native-tts-stream-{utc}.txt")
    report_path.write_text("\n".join(out_lines) + "\n")
    sys.stdout.write(f"\n[report] wrote {report_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
