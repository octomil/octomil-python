"""v0.1.9 Lane 4 — guard test: block premature progressive claim.

This test fails the moment someone PR's a public-claim flip from
``coalesced_after_synthesis`` to ``progressive_during_synthesis``
without the runtime release landing first. The fix is: gate the
claim flip on a runtime version check that proves Lane 1 + Lane 2
have shipped.

Specifically, this test pins:

1. The HTTP response header
   ``X-Octomil-Streaming-Honesty: coalesced_after_synthesis``
   on ``/v1/audio/speech/stream`` (in
   ``octomil/serve/app.py``) is still ``coalesced_after_synthesis``.

2. The module docstring on
   ``octomil/runtime/native/tts_stream_backend.py`` still mentions
   "coalesced" semantics and DOES NOT contain the marker string
   ``progressive_during_synthesis`` (outside the v0.1.9 forward-
   compatibility comment that explains why the field exists today).

3. The public method ``synthesize_with_chunks`` still describes the
   iterator shape (NOT a realtime ``stream`` method).

Why this test exists:
    The SDK MUST continue to honestly say "coalesced" until the
    runtime release proves progressive. We've made the SDK forward-
    compatible (TtsAudioChunk.streaming_mode added; iterator pattern
    unchanged) but the public wire claim has NOT changed. A drift
    test catches anyone who rips the honesty header out without
    flipping the runtime version gate at the same time.

Loosening this test:
    The legitimate path to flip the public claim is:
      a. Lane 1 (runtime worker-thread Generate) ships in
         octomil-runtime and the dylib is rebuilt.
      b. Lane 2 (runtime release tag bumped) lands and the SDK
         binding consumes the new release (capability hint exposed,
         OR per-chunk streaming_mode='progressive' set by the
         backend drain).
      c. The SDK PR that flips the claim ALSO updates this test
         to match — replacing the 'coalesced_after_synthesis'
         assertion with a runtime-version-gated branch.

    Updating this test in isolation (without the runtime release)
    is the bug class this test exists to prevent. Reviewers MUST
    require evidence in the same PR that Lane 1 + Lane 2 are
    shipped (linked PR / runtime tag / capability hint).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import List

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TTS_STREAM_BACKEND_PATH = _REPO_ROOT / "octomil" / "runtime" / "native" / "tts_stream_backend.py"
_SERVE_APP_PATH = _REPO_ROOT / "octomil" / "serve" / "app.py"

_PROGRESSIVE_CLAIM_MARKER = "progressive_during_synthesis"


# ---------------------------------------------------------------------------
# 1. HTTP response header still says coalesced
# ---------------------------------------------------------------------------


class TestHttpHonestyHeaderStillCoalesced:
    """The /v1/audio/speech/stream response includes
    ``X-Octomil-Streaming-Honesty: coalesced_after_synthesis``. We
    pin via source-file string search because exercising the
    StreamingResponse path requires a real or stubbed runtime;
    a string check is sufficient to catch a public-claim flip.
    """

    def test_serve_app_still_emits_coalesced_after_synthesis(self) -> None:
        assert _SERVE_APP_PATH.exists(), f"serve/app.py not found at {_SERVE_APP_PATH}"
        text = _SERVE_APP_PATH.read_text(encoding="utf-8")
        assert '"X-Octomil-Streaming-Honesty": "coalesced_after_synthesis"' in text, (
            "The HTTP response header X-Octomil-Streaming-Honesty has been "
            "changed away from 'coalesced_after_synthesis'. This is a public-"
            "claim flip and is BLOCKED until v0.1.9 Lane 1 (runtime worker-"
            "thread Generate) + Lane 2 (runtime release) ship. See "
            "tests/test_tts_stream_no_premature_progressive_claim.py docstring "
            "for the legitimate path to flip this claim."
        )

    def test_serve_app_does_not_emit_progressive_during_synthesis(self) -> None:
        assert _SERVE_APP_PATH.exists()
        text = _SERVE_APP_PATH.read_text(encoding="utf-8")
        assert "progressive_during_synthesis" not in text, (
            "The string 'progressive_during_synthesis' appeared in serve/app.py. "
            "This is the marker for a public progressive claim — BLOCKED until "
            "the runtime release lands. If you're legitimately flipping the claim, "
            "update this test in the SAME PR, with linked evidence that Lane 1 + "
            "Lane 2 are merged + released."
        )


# ---------------------------------------------------------------------------
# 2. Module docstring still describes coalesced semantics
# ---------------------------------------------------------------------------


class TestBackendModuleDocstringStillCoalesced:
    def test_module_docstring_contains_coalesced(self) -> None:
        from octomil.runtime.native import tts_stream_backend

        doc = inspect.getdoc(tts_stream_backend) or ""
        assert "coalesced" in doc.lower(), (
            "tts_stream_backend.py module docstring no longer mentions "
            "'coalesced'. This is a public-claim flip — BLOCKED until "
            "the runtime release lands. See test docstring for the "
            "legitimate flip path."
        )

    def test_module_docstring_does_not_describe_progressive_during_synthesis(self) -> None:
        """The string 'progressive_during_synthesis' is the canonical
        public marker for a real progressive claim. The forward-
        compatibility comment on the streaming_mode dataclass field
        does NOT use that exact marker — it talks about a future flip.
        So this test stays clean today.

        If someone adds 'progressive_during_synthesis' to the module
        docstring, that's a public-claim flip and triggers the gate.
        """
        from octomil.runtime.native import tts_stream_backend

        doc = inspect.getdoc(tts_stream_backend) or ""
        assert "progressive_during_synthesis" not in doc, (
            "tts_stream_backend.py module docstring contains the "
            "'progressive_during_synthesis' public-claim marker. BLOCKED "
            "until the runtime release lands. See test docstring."
        )

    def test_source_file_only_uses_progressive_marker_in_forward_compat_comments(self) -> None:
        """Stricter: scan the WHOLE source file (not just the
        docstring). The forward-compatibility comment on the
        streaming_mode field references 'progressive' as a Literal
        value, but does NOT use the 'progressive_during_synthesis'
        marker string. So 'progressive_during_synthesis' must not
        appear anywhere in the source — that string is reserved for
        the legitimate post-runtime-release flip.
        """
        assert _TTS_STREAM_BACKEND_PATH.exists()
        text = _TTS_STREAM_BACKEND_PATH.read_text(encoding="utf-8")
        assert "progressive_during_synthesis" not in text, (
            "tts_stream_backend.py source contains the "
            "'progressive_during_synthesis' marker. This is the public "
            "progressive-claim string — BLOCKED until the runtime release "
            "lands. The forward-compatibility comments on streaming_mode "
            "do not need this exact marker; if you've added it, you're "
            "flipping the claim."
        )


# ---------------------------------------------------------------------------
# 3. Method name still synthesize_with_chunks (not 'stream')
# ---------------------------------------------------------------------------


class TestPublicMethodNameStillSynthesizeWithChunks:
    """The method name MUST stay ``synthesize_with_chunks``; the
    iterator-of-chunks shape is what the v0.1.8 contract says.
    Renaming to ``stream`` (or anything that suggests realtime) is
    a public-claim flip."""

    def test_synthesize_with_chunks_present(self) -> None:
        from octomil.runtime.native.tts_stream_backend import NativeTtsStreamBackend

        assert hasattr(NativeTtsStreamBackend, "synthesize_with_chunks"), (
            "synthesize_with_chunks method missing on NativeTtsStreamBackend. "
            "BLOCKED — keeping the v0.1.8 method name is part of the honesty "
            "discipline; renaming is a public-claim flip."
        )

    def test_no_realtime_or_progressive_named_method(self) -> None:
        """No alias method that suggests realtime / progressive
        delivery (``stream``, ``stream_progressive``,
        ``synthesize_progressive``, etc.). The dataclass FIELD can
        carry 'progressive' once the runtime release supports it,
        but the METHOD name stays iterator-shape."""
        from octomil.runtime.native.tts_stream_backend import NativeTtsStreamBackend

        forbidden_method_names = (
            "stream",
            "stream_progressive",
            "synthesize_progressive",
            "synthesize_realtime",
            "stream_realtime",
        )
        for name in forbidden_method_names:
            assert not hasattr(NativeTtsStreamBackend, name), (
                f"NativeTtsStreamBackend grew a '{name}' method. This signals a "
                "public progressive/realtime claim — BLOCKED until the runtime "
                "release lands."
            )


# ---------------------------------------------------------------------------
# 4. Other public surfaces don't carry the progressive claim either
# ---------------------------------------------------------------------------


def _public_surface_files() -> List[Path]:
    """Bounded set of public-facing files that could plausibly carry
    a TTS-streaming public claim. We scan each for the
    ``progressive_during_synthesis`` marker; any hit means a public-
    claim flip happened on a surface that the serve/app.py + backend
    pins do not cover.

    Surfaces included:
      * ``README.md`` — top-level project README, ships to PyPI.
      * ``docs/**/*.md`` — public docs site sources.
      * ``pyproject.toml`` — project metadata (``description`` /
        ``readme`` content, both render on PyPI).

    Surfaces deliberately excluded (already pinned in earlier
    classes): ``octomil/serve/app.py`` and
    ``octomil/runtime/native/tts_stream_backend.py``.

    Surfaces deliberately NOT included: generated OpenAPI specs.
    Today the SDK doesn't ship a generated spec containing
    streaming-mode honesty text; if that changes, extend this list.
    """
    files: List[Path] = []
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


@pytest.mark.parametrize("surface_path", _public_surface_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
class TestPublicSurfacesDoNotCarryProgressiveClaim:
    """Bounded scan over README + docs + pyproject. Each must NOT
    contain the ``progressive_during_synthesis`` marker until the
    runtime release lands.

    If a future surface (e.g. a generated OpenAPI spec, a CHANGELOG
    entry that quotes the marker) starts legitimately needing the
    string, extend ``_public_surface_files`` with an explicit
    forward-compat exclusion + a comment explaining why.
    """

    def test_surface_does_not_carry_progressive_during_synthesis_marker(self, surface_path: Path) -> None:
        text = surface_path.read_text(encoding="utf-8")
        assert _PROGRESSIVE_CLAIM_MARKER not in text, (
            f"Public surface {surface_path.relative_to(_REPO_ROOT)!s} contains "
            f"the marker {_PROGRESSIVE_CLAIM_MARKER!r}. This is a public-claim "
            "flip and is BLOCKED until v0.1.9 Lane 1 + Lane 2 ship. If the "
            "marker is being introduced legitimately as part of the runtime "
            "release, update this test in the SAME PR with linked evidence."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
