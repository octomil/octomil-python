"""Whisper engine package — model-name detection only.

v0.1.5 PR-2B retired the legacy ``pywhispercpp`` engine from the
production registry; the product STT path runs through
:class:`octomil.runtime.native.stt_backend.NativeSttBackend`. This
package keeps :func:`is_whisper_model` importable so the serve
layer can still detect whisper-prefixed model names without
pulling in the legacy pywhispercpp shim. The legacy class
``WhisperCppEngine`` is reachable via
``octomil.runtime.engines.whisper._legacy_pywhisper`` for benchmark /
parity use ONLY (gated by ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1``
in :mod:`scripts.parity_native_stt`).
"""

from octomil.runtime.engines.whisper._legacy_pywhisper import is_whisper_model

TIER = "supported"

__all__ = ["TIER", "is_whisper_model"]
