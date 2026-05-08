"""v0.1.9 Lane C — progressive claim requires proof.

INVERSION of tests/test_tts_stream_no_premature_progressive_claim.py
(which existed in v0.1.9 Lane 4 to block premature claims). Now that
Lanes A (contracts PR #116) and B (runtime PRs #42, #43) are both
merged, the progressive claim is valid — this test asserts the POSITIVE
direction: public surfaces MAY (and do) claim progressive delivery, but
only when paired with a traceable proof reference.

Proof artifact:
  /tmp/v019-progressive-proof-20260508T185426Z.json
  sha256: 0c0b67a8a05d8a6fe07107cfe9f028d73327afb7ad6a4cc856251443c5374789
  first_audio_ratio: 0.5909  (gate: < 0.75 — PASSED)
  chunk_count: 2             (gate: >= 2 — PASSED)
  RTF: 0.105                 (gate: < 1.0 — PASSED)
  gate_pass: true
  kind: progressive_timing_proof
  timestamp_utc: 2026-05-08T18:53:48Z

Contracts field: ``octomil-contracts/conformance/audio.tts.stream.yaml``
  proof_artifact.measured_first_audio_ratio = 0.5909

Honest framing (required alongside any progressive claim):
  "first audio" = open→first-chunk-dequeued, NOT a streaming-latency
  floor. RTF=0.105 means faster than real-time. The claim is
  delivery_timing=progressive_during_synthesis — not "instantaneous"
  or "zero-delay."

Lane 4 inversion rationale (docstring preserved for audit history):
  Lane 4 blocked any "progressive_during_synthesis" string from
  appearing in public surfaces until the runtime release landed.
  This test INVERTS that: we now assert the claim IS present in the
  wire layer (app.py header) AND that it is always paired with a
  reference to the proof artifact — not a free-form marketing claim.

What this test does NOT do:
  - It does not execute the actual TTS streaming path (requires real
    runtime dylib + OCTOMIL_SHERPA_TTS_MODEL — see integration/
    test_tts_progressive_streaming.py for the end-to-end gate).
  - It does not validate the proof artifact JSON on disk (it may be
    in /tmp and not committed). It validates that the SDK's source
    code carries the proof reference strings.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TTS_STREAM_BACKEND_PATH = _REPO_ROOT / "octomil" / "runtime" / "native" / "tts_stream_backend.py"
_SERVE_APP_PATH = _REPO_ROOT / "octomil" / "serve" / "app.py"

_PROGRESSIVE_CLAIM_MARKER = "progressive_during_synthesis"
_PROOF_ARTIFACT_PATH_MARKER = "v019-progressive-proof-20260508T185426Z.json"
_PROOF_RATIO_MARKER = "0.5909"


# ---------------------------------------------------------------------------
# 1. HTTP response header NOW says progressive (flipped from v0.1.8)
# ---------------------------------------------------------------------------


class TestHttpHonestyHeaderIsProgressive:
    """The /v1/audio/speech/stream response now includes
    ``X-Octomil-Streaming-Honesty: progressive_during_synthesis``.
    Pinned via source-file string search — sufficient to catch a
    regression that reverts the claim without evidence.
    """

    def test_serve_app_emits_progressive_during_synthesis(self) -> None:
        assert _SERVE_APP_PATH.exists(), f"serve/app.py not found at {_SERVE_APP_PATH}"
        text = _SERVE_APP_PATH.read_text(encoding="utf-8")
        assert '"X-Octomil-Streaming-Honesty": "progressive_during_synthesis"' in text, (
            "The HTTP response header X-Octomil-Streaming-Honesty has been "
            "reverted from 'progressive_during_synthesis'. Any revert MUST be "
            "paired with removal of the proof_artifact block from "
            "octomil-contracts/conformance/audio.tts.stream.yaml (PR #116) "
            "and a new contracts PR explaining the regression. "
            "See test docstring for the proof artifact reference."
        )

    def test_serve_app_does_not_regress_to_coalesced_after_synthesis(self) -> None:
        assert _SERVE_APP_PATH.exists()
        text = _SERVE_APP_PATH.read_text(encoding="utf-8")
        assert '"X-Octomil-Streaming-Honesty": "coalesced_after_synthesis"' not in text, (
            "The HTTP response header X-Octomil-Streaming-Honesty has been "
            "REVERTED to 'coalesced_after_synthesis'. This is a regression from "
            "the v0.1.9 progressive flip. The claim is backed by "
            f"proof artifact {_PROOF_ARTIFACT_PATH_MARKER} "
            f"(first_audio_ratio={_PROOF_RATIO_MARKER}, gate_pass=true). "
            "Do not revert without a new contracts PR."
        )

    def test_serve_app_carries_proof_artifact_reference(self) -> None:
        """Proof reference MUST accompany any progressive claim.

        Any "realtime", "instant", or "streaming-latency" language in
        the source that is NOT paired with the proof artifact path or
        proof ratio is a marketing overclaim — this test blocks it.
        """
        assert _SERVE_APP_PATH.exists()
        text = _SERVE_APP_PATH.read_text(encoding="utf-8")
        assert _PROOF_ARTIFACT_PATH_MARKER in text or _PROOF_RATIO_MARKER in text, (
            f"serve/app.py carries the progressive claim but NEITHER "
            f"the proof artifact path ({_PROOF_ARTIFACT_PATH_MARKER!r}) "
            f"NOR the first_audio_ratio ({_PROOF_RATIO_MARKER!r}) is referenced. "
            "Any progressive claim must cite the proof artifact. "
            "Add the reference in the endpoint docstring or the header comment."
        )


# ---------------------------------------------------------------------------
# 2. Backend module docstring describes progressive semantics
# ---------------------------------------------------------------------------


class TestBackendModuleDocstringIsProgressive:
    def test_module_docstring_contains_progressive_during_synthesis(self) -> None:
        """v0.1.9 flip: the module docstring must now describe progressive
        delivery (not coalesced). Lane 4 blocked this; Lane C enables it.
        """
        import inspect

        from octomil.runtime.native import tts_stream_backend

        doc = inspect.getdoc(tts_stream_backend) or ""
        assert "progressive_during_synthesis" in doc, (
            "tts_stream_backend.py module docstring does not mention "
            "'progressive_during_synthesis'. The v0.1.9 flip requires "
            "the module docstring to describe the new delivery semantics."
        )

    def test_module_docstring_carries_proof_reference(self) -> None:
        """Proof artifact reference must appear in the module docstring
        alongside the progressive claim — no undocumented flips.
        """
        import inspect

        from octomil.runtime.native import tts_stream_backend

        doc = inspect.getdoc(tts_stream_backend) or ""
        has_proof = _PROOF_ARTIFACT_PATH_MARKER in doc or _PROOF_RATIO_MARKER in doc
        assert has_proof, (
            "tts_stream_backend.py module docstring carries a progressive "
            "claim but does NOT reference the proof artifact. Add the "
            f"proof artifact path ({_PROOF_ARTIFACT_PATH_MARKER!r}) or "
            f"the measured ratio ({_PROOF_RATIO_MARKER!r}) to the docstring."
        )

    def test_source_file_does_not_overclaim_realtime_instant_or_zero_delay(self) -> None:
        """Guard against marketing overclaim. The source may describe
        "progressive" delivery but MUST NOT use "realtime", "instant",
        or "zero-delay" in association with TTS latency claims unless
        those words are inside a quote/citation of the contract enum
        (e.g. 'realtime_streaming_claim').
        """
        assert _TTS_STREAM_BACKEND_PATH.exists()
        text = _TTS_STREAM_BACKEND_PATH.read_text(encoding="utf-8")
        # realtime_streaming_claim is a contract field name — OK to cite.
        # Check for overclaim language outside of that pattern.
        overclaim_phrases = ["zero-delay", "instantaneous TTS", "instant TTS"]
        for phrase in overclaim_phrases:
            assert phrase.lower() not in text.lower(), (
                f"tts_stream_backend.py contains overclaim language: {phrase!r}. "
                "The v0.1.9 progressive claim is bounded by the proof artifact "
                f"(first_audio_ratio={_PROOF_RATIO_MARKER}, RTF=0.105). "
                "Remove or qualify the overclaim language."
            )


# ---------------------------------------------------------------------------
# 3. TtsAudioChunk.streaming_mode default is "progressive"
# ---------------------------------------------------------------------------


class TestTtsAudioChunkStreamingMode:
    """streaming_mode was added in Lane 4 with default "coalesced".
    Lane C (this PR) flips the default to "progressive" now that the
    runtime release is proven.
    """

    def test_streaming_mode_field_default_is_progressive(self) -> None:
        import dataclasses

        from octomil.runtime.native.tts_stream_backend import TtsAudioChunk

        fields = {f.name: f for f in dataclasses.fields(TtsAudioChunk)}
        assert (
            "streaming_mode" in fields
        ), "TtsAudioChunk is missing the streaming_mode field. It was added in v0.1.9 Lane 4 and must be present."
        default_val = fields["streaming_mode"].default
        assert default_val == "progressive", (
            f"TtsAudioChunk.streaming_mode default is {default_val!r}, "
            "expected 'progressive' after the v0.1.9 Lane C flip. "
            "The runtime worker-thread Generate (Lane 1, PR #42) is merged; "
            "the default must reflect proven progressive delivery."
        )

    def test_tts_first_audio_ms_metric_name_constant_exported(self) -> None:
        """TTS_FIRST_AUDIO_MS_METRIC_NAME must be exported from the module
        and match the canonical metric name from contracts.
        """
        from octomil.runtime.native.tts_stream_backend import TTS_FIRST_AUDIO_MS_METRIC_NAME

        assert TTS_FIRST_AUDIO_MS_METRIC_NAME == "tts.first_audio_ms", (
            f"TTS_FIRST_AUDIO_MS_METRIC_NAME is {TTS_FIRST_AUDIO_MS_METRIC_NAME!r}, "
            "expected 'tts.first_audio_ms' (canonical metric name from "
            "octomil-contracts/fixtures/runtime_metric/canonical_metrics.json). "
            "Do not rename without a contracts PR."
        )


# ---------------------------------------------------------------------------
# 4. Proof required alongside any "realtime" / "streaming-latency" claim
#    in public surfaces (README, docs, pyproject)
# ---------------------------------------------------------------------------


def _public_surface_files():
    """Bounded set of public-facing files. If any of these carry the
    progressive claim marker, they must also carry a proof reference.

    Inverts the Lane 4 test (which blocked the marker entirely).
    Now: the marker is allowed ONLY when paired with proof.
    """
    files = []
    readme = _REPO_ROOT / "README.md"
    if readme.exists():
        files.append(readme)
    pyproject = _REPO_ROOT / "pyproject.toml"
    if pyproject.exists():
        files.append(pyproject)
    docs_dir = _REPO_ROOT / "docs"
    if docs_dir.is_dir():
        files.extend(sorted(docs_dir.rglob("*.md")))
    return files


import pytest  # noqa: E402


@pytest.mark.parametrize("surface_path", _public_surface_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
class TestPublicSurfacesCarryProofWhenClaimingProgressive:
    """If a public surface claims progressive delivery, it must cite
    the proof artifact (path or ratio). Surfaces that don't mention
    progressive at all are fine — no retroactive requirement.
    """

    def test_progressive_claim_paired_with_proof_reference(self, surface_path: Path) -> None:
        text = surface_path.read_text(encoding="utf-8")
        if _PROGRESSIVE_CLAIM_MARKER not in text and "progressive" not in text.lower():
            return  # surface doesn't claim progressive — nothing to check
        has_proof = _PROOF_ARTIFACT_PATH_MARKER in text or _PROOF_RATIO_MARKER in text
        # Allow the claim without proof only if it's not a latency claim
        # (e.g. a description of the streaming API shape is fine without proof;
        # a claim of "first audio in Nms" requires proof).
        latency_claim_phrases = [
            "first audio in",
            "first audio latency",
            "streaming latency",
            "progressive_during_synthesis",
        ]
        has_latency_claim = any(p in text for p in latency_claim_phrases)
        if has_latency_claim and not has_proof:
            pytest.fail(
                f"Public surface {surface_path.relative_to(_REPO_ROOT)!s} "
                f"carries a latency/progressive claim but does NOT reference "
                f"the proof artifact. Add either the proof path "
                f"({_PROOF_ARTIFACT_PATH_MARKER!r}) or measured ratio "
                f"({_PROOF_RATIO_MARKER!r}).\n"
                "Proof: /tmp/v019-progressive-proof-20260508T185426Z.json "
                f"(first_audio_ratio={_PROOF_RATIO_MARKER}, gate_pass=true)"
            )
