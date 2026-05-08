"""v0.1.9 Lane 4 — telemetry sink forward-path: constant exposure pin.

Honesty pin for the runtime->SDK telemetry forward path.

SCOPE LIMITATION (read first):
    This test verifies that the SDK *exposes* the constants needed
    for the eventual forward path — the OCT_EVENT_METRIC enum value
    and the ``tts.first_audio_ms`` metric-name string. It does NOT
    prove that the cffi binding actually forwards a real metric event
    end-to-end. That is a runtime-side test, pending Lane 1 + Lane 2
    merge — at which point a follow-up PR can register a real
    ``oct_telemetry_sink_fn``, fire a real metric event through cffi,
    and assert the sink callback receives ``name="tts.first_audio_ms"``
    byte-for-byte.

    Lane 4 is "prep, not proof". An earlier draft of this test used
    a closure side-channel that pretended to exercise the forward
    path; that was misleading and has been removed.

Background (RM-RT-001):
    The runtime delivers ``OCT_EVENT_METRIC`` events through the
    runtime-scope telemetry sink (``oct_telemetry_sink_fn`` per
    ``runtime.h``) AND, depending on adapter, through ``poll_event``
    on a session. ``OCT_EVENT_METRIC`` payloads are
    ``data.metric { const char* name; double value; }`` — a string
    name + a double. The runtime treats the metric vocabulary as
    open-ended; new metric names appear without ABI bumps.

    Per RM-RT-001, the SDK is NOT supposed to introspect or filter
    on the ``name`` string at the sink boundary. It forwards every
    metric event through to the registered Python sink unchanged.
    Filtering / aggregation is the consumer's job, not the binding's.

What this file pins:

1. ``OCT_EVENT_METRIC`` is importable from
   ``octomil.runtime.native.loader`` and re-exported on the
   ``octomil.runtime.native`` package. Hiding the constant would
   force consumers to rely on numeric literals — a compatibility
   hazard.

2. The string ``tts.first_audio_ms`` is referenced in the SDK's
   native-runtime layer. A future runtime release (Lane 1 + Lane 2)
   will start emitting ``OCT_EVENT_METRIC`` with this name to mark
   first-audio-chunk arrival; the SDK side needs it documented now
   so consumers + reviewers know where it will surface.

3. The native-runtime layer does NOT carry an allowlist that filters
   metric events on name. ``tts.first_audio_ms`` is not — and must
   never become — a magic-known string in the binding's metric path.
   We pin this by source-scanning the loader for filtering patterns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from octomil.runtime.native.loader import (
    OCT_EVENT_METRIC,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOADER_PATH = _REPO_ROOT / "octomil" / "runtime" / "native" / "loader.py"
_TTS_STREAM_BACKEND_PATH = _REPO_ROOT / "octomil" / "runtime" / "native" / "tts_stream_backend.py"


# ---------------------------------------------------------------------------
# 1. The OCT_EVENT_METRIC enum value is in the binding
# ---------------------------------------------------------------------------


class TestMetricEventConstantExposed:
    """``OCT_EVENT_METRIC`` MUST be importable from
    ``octomil.runtime.native.loader`` and re-exported on the package.
    A regression that hides the constant would force consumers to
    rely on numeric literals — a compatibility hazard."""

    def test_loader_exports_metric_constant(self) -> None:
        # Imported at top of file; if the import fails at all, this
        # test file fails to collect — pytest will surface it.
        assert isinstance(OCT_EVENT_METRIC, int)
        assert OCT_EVENT_METRIC > 0

    def test_package_re_exports_metric_constant(self) -> None:
        from octomil.runtime import native as native_pkg

        assert hasattr(native_pkg, "OCT_EVENT_METRIC")
        assert native_pkg.OCT_EVENT_METRIC == OCT_EVENT_METRIC


# ---------------------------------------------------------------------------
# 2. The tts.first_audio_ms metric-name string is referenced at the SDK
#    layer (binding + backend module). Lane 4 ships this as documentation
#    + readiness; Lane 1 + Lane 2 wire the actual emit + forward.
# ---------------------------------------------------------------------------


class TestTtsFirstAudioMsConstantReferenced:
    """The metric name string ``tts.first_audio_ms`` MUST appear at
    the SDK layer so reviewers + consumers can grep for it.

    NOT pinned here: that the binding ACTUALLY forwards an event with
    this name — that's a runtime-integration test, pending Lane 1 +
    Lane 2 merge.
    """

    METRIC_NAME = "tts.first_audio_ms"

    def test_metric_name_referenced_in_native_runtime_module(self) -> None:
        """The string appears somewhere in the native-runtime SDK
        layer (loader.py OR tts_stream_backend.py). Reviewers need
        a discoverable anchor before Lane 1 + Lane 2 wire the live
        forward path."""
        assert _LOADER_PATH.exists(), f"loader.py not found at {_LOADER_PATH}"
        assert _TTS_STREAM_BACKEND_PATH.exists()
        loader_text = _LOADER_PATH.read_text(encoding="utf-8")
        backend_text = _TTS_STREAM_BACKEND_PATH.read_text(encoding="utf-8")
        assert self.METRIC_NAME in loader_text or self.METRIC_NAME in backend_text, (
            f"Metric name {self.METRIC_NAME!r} is not referenced in the SDK's "
            "native-runtime layer (loader.py or tts_stream_backend.py). "
            "Lane 1 + Lane 2 will start emitting OCT_EVENT_METRIC events "
            "with this name; the SDK needs at least a documented anchor "
            "now so reviewers can grep for it."
        )


# ---------------------------------------------------------------------------
# 3. Source-scan: no name-allowlist on the metric forward path
# ---------------------------------------------------------------------------


class TestNoMetricNameAllowlistInLoader:
    """Per RM-RT-001 the binding does NOT introspect or filter on the
    metric ``name`` string. We can't run the live forward path until
    Lane 1 + Lane 2 land, but we CAN catch the regression class —
    someone adding a hard-coded allowlist or name-equality switch on
    the metric path — by source-scanning the loader.

    This is a smoke pin, not a guarantee. The real forward-path test
    arrives with the runtime release.
    """

    def test_loader_does_not_introduce_metric_name_allowlist(self) -> None:
        assert _LOADER_PATH.exists()
        text = _LOADER_PATH.read_text(encoding="utf-8")
        forbidden_patterns = (
            "_METRIC_NAME_ALLOWLIST",
            "METRIC_NAME_ALLOWLIST",
            "metric_name_allowlist",
            "ALLOWED_METRIC_NAMES",
            "allowed_metric_names",
        )
        for pat in forbidden_patterns:
            assert pat not in text, (
                f"loader.py contains the pattern {pat!r}, which looks like "
                "a metric-name allowlist on the binding's forward path. "
                "Per RM-RT-001 the binding MUST forward all metric events "
                "regardless of name; filtering is a consumer concern."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
