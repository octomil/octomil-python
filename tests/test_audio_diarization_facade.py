from __future__ import annotations


def test_diarization_facade_exports_native_surface() -> None:
    from octomil.audio.diarization import (
        DiarizationSegment,
        NativeDiarizationBackend,
        open_diarization_backend,
        runtime_advertises_audio_diarization,
    )

    assert DiarizationSegment is not None
    assert NativeDiarizationBackend is not None
    assert callable(open_diarization_backend)
    assert callable(runtime_advertises_audio_diarization)


def test_open_diarization_backend_closes_on_exit(monkeypatch) -> None:
    from octomil.audio import diarization as mod

    calls: list[str] = []

    class FakeBackend:
        def open(self) -> None:
            calls.append("open")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(mod, "NativeDiarizationBackend", FakeBackend)
    with mod.open_diarization_backend() as backend:
        assert isinstance(backend, FakeBackend)
        calls.append("body")

    assert calls == ["open", "body", "close"]
