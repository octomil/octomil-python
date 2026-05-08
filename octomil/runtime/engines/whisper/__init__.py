"""Whisper engine package — model-name detection only.

v0.1.5 PR-2B retired the legacy ``pywhispercpp`` engine from the
production registry; the product STT path runs through
:class:`octomil.runtime.native.stt_backend.NativeSttBackend`. This
package keeps :func:`is_whisper_model` importable so the serve layer
can still detect whisper-prefixed model names. v0.1.6 PR2 moved
:func:`is_whisper_model` into the non-legacy
:mod:`octomil.runtime.engines.whisper.model_names` module so the
product path no longer pulls the legacy pywhispercpp shim into
``sys.modules`` at import time. The legacy class ``WhisperCppEngine``
remains reachable via
``octomil.runtime.engines.whisper._legacy_pywhisper`` for benchmark /
parity use ONLY (gated by ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1``
in :mod:`scripts.parity_native_stt`).
"""

from octomil.runtime.engines.whisper.model_names import is_whisper_model

TIER = "supported"

__all__ = ["TIER", "is_whisper_model"]
