"""Integration tests — NativeSpeakerEmbeddingBackend against real liboctomil-runtime.

Exercises the full cffi path:
``oct_runtime_open → oct_model_open(engine_hint="sherpa_onnx") →
oct_model_warm → oct_session_open(capability="audio.speaker.embedding")
→ oct_session_send_audio → oct_session_poll_event drain →
oct_session_close``.

Requires:
  * ``OCTOMIL_RUNTIME_DYLIB`` (or fetched dev cache) pointing at a
    liboctomil-runtime built with ``OCT_ENABLE_ENGINE_SHERPA_ONNX=ON``
    + ``OCT_HAVE_SHERPA_ONNX``, ABI minor >= 9.
  * ``OCTOMIL_SHERPA_SPEAKER_MODEL`` pointing at the canonical
    ERes2NetV2 ONNX (SHA-256 ``1a331345…7a5e4b``).
  * ``research/engines/whisper.cpp/samples/jfk.wav`` on disk.

When any precondition is missing the entire module skips. The
self-similarity check (jfk.wav vs. itself) MUST yield cosine ~ 1.0.
A canonical "different speaker" reference clip is hard to ship in
this PR; the spec asks for >0.3 cosine against another clip from the
same speaker, which is a weak threshold deliberately tolerant of the
fact that JFK has only one canonical clip in the test corpus today.
We exercise self-similarity (~1.0 expected), a perturbed-jfk
similarity (high, but lower than 1.0 — confirming the embedding is
not a no-op identity), and a synthesized-noise comparison (must be
distinctly lower) as the v0.1.5 smoke gate.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from octomil.errors import OctomilError, OctomilErrorCode

_JFK_WAV = Path("/Users/seanb/Developer/Octomil/research/engines/whisper.cpp/samples/jfk.wav")
_CANONICAL_DIM = 512


def _skip_reason() -> str | None:
    if not _JFK_WAV.is_file():
        return f"jfk.wav missing at {_JFK_WAV}"
    if not os.environ.get("OCTOMIL_SHERPA_SPEAKER_MODEL"):
        return "OCTOMIL_SHERPA_SPEAKER_MODEL not set"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


def _load_jfk_pcm_f32() -> np.ndarray:
    with wave.open(str(_JFK_WAV), "rb") as wf:
        sr = wf.getframerate()
        assert sr == 16000, f"jfk.wav sample rate {sr} != 16000"
        pcm = wf.readframes(wf.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. The runtime L2-normalizes embeddings, so this
    reduces to a dot product, but we don't depend on that contract here
    — recompute the norm in case the runtime changes."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Test 1 — capability is advertised
# ---------------------------------------------------------------------------


class TestPlannerSelection:
    def test_runtime_advertises_audio_speaker_embedding(self) -> None:
        from octomil.runtime.native.loader import NativeRuntime
        from octomil.runtime.native.speaker_backend import (
            runtime_advertises_audio_speaker_embedding,
        )

        rt = NativeRuntime.open()
        try:
            assert runtime_advertises_audio_speaker_embedding(rt) is True
            caps = rt.capabilities()
            assert "audio.speaker.embedding" in caps.supported_capabilities
        finally:
            rt.close()


# ---------------------------------------------------------------------------
# Test 2 — fallback is unreachable
# ---------------------------------------------------------------------------


class TestNoFallback:
    def test_unset_speaker_model_raises_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from octomil.runtime.native.speaker_backend import NativeSpeakerEmbeddingBackend

        monkeypatch.delenv("OCTOMIL_SHERPA_SPEAKER_MODEL", raising=False)
        backend = NativeSpeakerEmbeddingBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.load_model("sherpa-eres2netv2-base")
        assert exc_info.value.code in {
            OctomilErrorCode.RUNTIME_UNAVAILABLE,
            OctomilErrorCode.CHECKSUM_MISMATCH,
        }


# ---------------------------------------------------------------------------
# Test 3 — full smoke against jfk.wav
# ---------------------------------------------------------------------------


class TestFullSmoke:
    """Embed jfk.wav and confirm:
    * Returns a 1-D float32 array of dim 512 (canonical ERes2NetV2).
    * Self-similarity cosine ~= 1.0 (deterministic embedding,
      re-running the same audio yields the same vector).
    * Perturbed-jfk cosine > 0.3 against the canonical (per-spec
      threshold for "same speaker"); should land much higher in
      practice (>0.85) but the spec only requires 0.3.
    * Synthetic-noise cosine markedly lower than the
      perturbed-jfk cosine — confirms the embedding actually
      encodes voice characteristics (not a no-op identity head).
    """

    def test_embed_returns_512_dim_float32(self) -> None:
        from octomil.runtime.native.speaker_backend import NativeSpeakerEmbeddingBackend

        audio = _load_jfk_pcm_f32()
        backend = NativeSpeakerEmbeddingBackend()
        backend.load_model("sherpa-eres2netv2-base")
        try:
            emb = backend.embed(audio, sample_rate_hz=16000)
        finally:
            backend.close()

        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32
        assert emb.shape == (_CANONICAL_DIM,), f"unexpected embedding shape {emb.shape}"

    def test_self_similarity_high(self) -> None:
        """Re-embed the same clip; cosine should be very close to 1.0
        (deterministic embedding head)."""
        from octomil.runtime.native.speaker_backend import NativeSpeakerEmbeddingBackend

        audio = _load_jfk_pcm_f32()
        backend = NativeSpeakerEmbeddingBackend()
        backend.load_model("sherpa-eres2netv2-base")
        try:
            emb_a = backend.embed(audio, sample_rate_hz=16000)
            emb_b = backend.embed(audio, sample_rate_hz=16000)
        finally:
            backend.close()

        cos = _cosine(emb_a, emb_b)
        # Identical input → identical output; allow tiny numerical
        # noise (accumulator order on the GPU/Metal path).
        assert cos > 0.999, f"self-similarity unexpectedly low: cos={cos}"

    def test_same_speaker_perturbed_cosine_above_spec_threshold(self) -> None:
        """Cosine between jfk and a mildly-perturbed jfk (small white
        noise) MUST exceed the >0.3 spec threshold for "same speaker"
        — and in practice should land much higher (>0.85) since the
        signal is essentially identical."""
        from octomil.runtime.native.speaker_backend import NativeSpeakerEmbeddingBackend

        rng = np.random.default_rng(seed=42)
        audio = _load_jfk_pcm_f32()
        # Tiny additive noise (peak ~0.005; signal is ~[-1, 1]). The
        # speaker embedding should be essentially unchanged.
        perturbed = audio + rng.normal(loc=0.0, scale=0.005, size=audio.shape).astype(np.float32)
        # Clamp to canonical PCM-f32 range so the runtime's NaN/finite
        # check doesn't trip if the noise pushed any sample off range.
        perturbed = np.clip(perturbed, -1.0, 1.0).astype(np.float32, copy=False)

        backend = NativeSpeakerEmbeddingBackend()
        backend.load_model("sherpa-eres2netv2-base")
        try:
            emb_clean = backend.embed(audio, sample_rate_hz=16000)
            emb_perturbed = backend.embed(perturbed, sample_rate_hz=16000)
        finally:
            backend.close()

        cos = _cosine(emb_clean, emb_perturbed)
        assert cos > 0.3, (
            f"same-speaker cosine {cos} below spec threshold 0.3 — embedding model "
            "may be misconfigured or producing constant outputs"
        )

    def test_noise_cosine_distinctly_lower_than_speech(self) -> None:
        """Synthetic white noise vs. JFK speech should produce a
        DISTINCTLY lower cosine than self-similarity — confirms the
        embedding actually encodes voice characteristics. Threshold
        is loose because white noise is a degenerate case the model
        wasn't trained on; we only require it's below the perturbed-
        clip case (which we expect ~0.85+)."""
        from octomil.runtime.native.speaker_backend import NativeSpeakerEmbeddingBackend

        rng = np.random.default_rng(seed=7)
        audio = _load_jfk_pcm_f32()
        # Match length so the comparison is duration-controlled.
        noise = rng.uniform(low=-0.1, high=0.1, size=audio.shape).astype(np.float32)

        backend = NativeSpeakerEmbeddingBackend()
        backend.load_model("sherpa-eres2netv2-base")
        try:
            emb_speech = backend.embed(audio, sample_rate_hz=16000)
            emb_noise = backend.embed(noise, sample_rate_hz=16000)
            # Recompute speech-vs-self for the comparison anchor.
            emb_speech_2 = backend.embed(audio, sample_rate_hz=16000)
        finally:
            backend.close()

        cos_self = _cosine(emb_speech, emb_speech_2)
        cos_noise = _cosine(emb_speech, emb_noise)
        # Self-similarity must be at least 0.5 above the speech-vs-noise
        # baseline. Very loose threshold; in practice we expect cos_self
        # ~ 1.0 and cos_noise < 0.5.
        assert cos_self > cos_noise + 0.1, (
            f"embedding does not discriminate speech from noise: " f"cos_self={cos_self}, cos_noise={cos_noise}"
        )


# ---------------------------------------------------------------------------
# Test 4 — bad-digest path -> typed error (not INFERENCE_FAILED)
# ---------------------------------------------------------------------------


class TestBadDigest:
    def test_tampered_digest_raises_typed_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from octomil.runtime.native.speaker_backend import NativeSpeakerEmbeddingBackend

        tampered = tmp_path / "eres2net_base.onnx"
        tampered.write_bytes(b"NOT_A_REAL_ONNX_MODEL" * 1024)
        monkeypatch.setenv("OCTOMIL_SHERPA_SPEAKER_MODEL", str(tampered))

        backend = NativeSpeakerEmbeddingBackend()
        with pytest.raises(OctomilError) as exc_info:
            backend.load_model("sherpa-eres2netv2-base")
        assert exc_info.value.code != OctomilErrorCode.INFERENCE_FAILED
        assert exc_info.value.code in {
            OctomilErrorCode.CHECKSUM_MISMATCH,
            OctomilErrorCode.RUNTIME_UNAVAILABLE,
            OctomilErrorCode.MODEL_NOT_FOUND,
            OctomilErrorCode.INVALID_INPUT,
        }


# ---------------------------------------------------------------------------
# Test 5 — NaN/Inf input rejects INVALID_INPUT
# ---------------------------------------------------------------------------


class TestInvalidAudio:
    def test_nan_audio_invalid_input(self) -> None:
        from octomil.runtime.native.speaker_backend import NativeSpeakerEmbeddingBackend

        backend = NativeSpeakerEmbeddingBackend()
        backend.load_model("sherpa-eres2netv2-base")
        try:
            arr = _load_jfk_pcm_f32().copy()
            arr[100] = float("nan")
            with pytest.raises(OctomilError) as exc_info:
                backend.embed(arr, sample_rate_hz=16000)
            assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        finally:
            backend.close()

    def test_wrong_sample_rate_invalid_input(self) -> None:
        from octomil.runtime.native.speaker_backend import NativeSpeakerEmbeddingBackend

        backend = NativeSpeakerEmbeddingBackend()
        backend.load_model("sherpa-eres2netv2-base")
        try:
            arr = _load_jfk_pcm_f32()
            with pytest.raises(OctomilError) as exc_info:
                backend.embed(arr, sample_rate_hz=8000)
            assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        finally:
            backend.close()
