"""Native-only product routes for low-level audio capabilities."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from octomil.audio import (
    FacadeAudio,
    FacadeDiarization,
    FacadeSpeakerEmbedding,
    FacadeVad,
)
from octomil.errors import OctomilError, OctomilErrorCode
from octomil.serve import create_app


def _create_server_app():
    pytest.importorskip("fastapi")
    return create_app("test-model", max_queue_depth=0)


class _FakeVadSession:
    def __init__(self, owner: "_FakeVadBackend") -> None:
        self.owner = owner

    def __enter__(self) -> "_FakeVadSession":
        return self

    def __exit__(self, *args: object) -> None:
        self.owner.session_closed = True

    def feed_chunk(self, audio: object, *, sample_rate_hz: int | None = None) -> None:
        self.owner.feed_args = (audio, sample_rate_hz)

    def poll_transitions(
        self,
        *,
        deadline_ms: int | None = None,
        drain_until_completed: bool = False,
    ):
        self.owner.poll_args = (deadline_ms, drain_until_completed)
        return iter(
            [
                SimpleNamespace(kind="speech_start", timestamp_ms=120, confidence=0.91),
                SimpleNamespace(kind="speech_end", timestamp_ms=980, confidence=0.87),
            ]
        )


class _FakeVadBackend:
    instances: list["_FakeVadBackend"] = []

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.session_closed = False
        self.open_session_sample_rate: int | None = None
        self.feed_args: tuple[object, int | None] | None = None
        self.poll_args: tuple[int | None, bool] | None = None
        self.instances.append(self)

    def open(self) -> None:
        self.opened = True

    def open_session(self, *, sample_rate_hz: int = 16000) -> _FakeVadSession:
        self.open_session_sample_rate = sample_rate_hz
        return _FakeVadSession(self)

    def close(self) -> None:
        self.closed = True


class _FakeSpeakerEmbeddingBackend:
    instances: list["_FakeSpeakerEmbeddingBackend"] = []

    def __init__(self) -> None:
        self.loaded_model = ""
        self.embed_args: tuple[object, int, int | None] | None = None
        self.closed = False
        self.instances.append(self)

    def load_model(self, model_name: str = "sherpa-eres2netv2-base") -> None:
        self.loaded_model = model_name

    def embed(
        self,
        audio: object,
        *,
        sample_rate_hz: int = 16000,
        deadline_ms: int | None = None,
    ) -> list[float]:
        self.embed_args = (audio, sample_rate_hz, deadline_ms)
        return [0.25, -0.25, 0.5]

    def close(self) -> None:
        self.closed = True


class _FakeDiarizationBackend:
    instances: list["_FakeDiarizationBackend"] = []

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.diarize_args: tuple[object, int, int] | None = None
        self.instances.append(self)

    def open(self) -> None:
        self.opened = True

    def diarize(
        self,
        audio: object,
        *,
        sample_rate_hz: int = 16000,
        deadline_ms: int = 300_000,
    ) -> list[SimpleNamespace]:
        self.diarize_args = (audio, sample_rate_hz, deadline_ms)
        return [
            SimpleNamespace(start_ms=0, end_ms=500, speaker_id=1, speaker_label="speaker_1"),
            SimpleNamespace(start_ms=500, end_ms=900, speaker_id=2, speaker_label="speaker_2"),
        ]

    def close(self) -> None:
        self.closed = True


def _payload() -> dict[str, object]:
    return {"audio_base64": base64.b64encode(b"\x00\x00\x00\x00").decode("ascii")}


def test_facade_audio_exposes_native_low_level_namespaces() -> None:
    audio = FacadeAudio(kernel=SimpleNamespace())

    assert isinstance(audio.vad, FacadeVad)
    assert callable(audio.vad.detect)
    assert isinstance(audio.speaker_embedding, FacadeSpeakerEmbedding)
    assert callable(audio.speaker_embedding.create)
    assert isinstance(audio.diarization, FacadeDiarization)
    assert callable(audio.diarization.create)


def test_facade_vad_detect_uses_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from octomil.audio import vad as vad_module

    _FakeVadBackend.instances = []
    monkeypatch.setattr(vad_module, "NativeVadBackend", _FakeVadBackend)

    result = asyncio.run(
        FacadeVad().detect(
            audio=b"\x00\x00\x00\x00",
            sample_rate_hz=16000,
            deadline_ms=1234,
        )
    )

    backend = _FakeVadBackend.instances[0]
    assert backend.opened is True
    assert backend.closed is True
    assert backend.session_closed is True
    assert backend.poll_args == (1234, True)
    assert [transition.kind for transition in result.transitions] == ["speech_start", "speech_end"]


def test_facade_speaker_embedding_uses_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from octomil.audio import speaker_embedding as speaker_module

    _FakeSpeakerEmbeddingBackend.instances = []
    monkeypatch.setattr(
        speaker_module,
        "NativeSpeakerEmbeddingBackend",
        _FakeSpeakerEmbeddingBackend,
    )

    result = asyncio.run(
        FacadeSpeakerEmbedding().create(
            audio=b"\x00\x00\x00\x00",
            model="sherpa-eres2netv2-base",
            deadline_ms=99,
        )
    )

    backend = _FakeSpeakerEmbeddingBackend.instances[0]
    assert backend.loaded_model == "sherpa-eres2netv2-base"
    assert backend.closed is True
    assert backend.embed_args == (b"\x00\x00\x00\x00", 16000, 99)
    assert result.embedding == [0.25, -0.25, 0.5]
    assert result.dimensions == 3


def test_facade_diarization_uses_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from octomil.audio import diarization as diarization_module

    _FakeDiarizationBackend.instances = []
    monkeypatch.setattr(
        diarization_module,
        "NativeDiarizationBackend",
        _FakeDiarizationBackend,
    )

    result = asyncio.run(
        FacadeDiarization().create(
            audio=b"\x00\x00\x00\x00",
            sample_rate_hz=16000,
            deadline_ms=456,
        )
    )

    backend = _FakeDiarizationBackend.instances[0]
    assert backend.opened is True
    assert backend.closed is True
    assert backend.diarize_args == (b"\x00\x00\x00\x00", 16000, 456)
    assert [segment.speaker_label for segment in result.segments] == ["speaker_1", "speaker_2"]


def test_vad_server_route_uses_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from octomil.audio import vad as vad_module

    _FakeVadBackend.instances = []
    monkeypatch.setattr(vad_module, "NativeVadBackend", _FakeVadBackend)
    app = _create_server_app()

    async def _request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/audio/vad", json={**_payload(), "deadline_ms": 123})

    resp = asyncio.run(_request())

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "audio.vad"
    assert body["transitions"][0] == {
        "kind": "speech_start",
        "timestamp_ms": 120,
        "confidence": 0.91,
    }
    assert _FakeVadBackend.instances[0].opened is True


def test_speaker_embedding_server_route_uses_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from octomil.audio import speaker_embedding as speaker_module

    _FakeSpeakerEmbeddingBackend.instances = []
    monkeypatch.setattr(
        speaker_module,
        "NativeSpeakerEmbeddingBackend",
        _FakeSpeakerEmbeddingBackend,
    )
    app = _create_server_app()

    async def _request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(
                "/v1/audio/speaker_embeddings",
                json={**_payload(), "model": "sherpa-eres2netv2-base", "deadline_ms": 777},
            )

    resp = asyncio.run(_request())

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "audio.speaker.embedding"
    assert body["embedding"] == [0.25, -0.25, 0.5]
    assert body["dimensions"] == 3
    assert _FakeSpeakerEmbeddingBackend.instances[0].embed_args == (
        b"\x00\x00\x00\x00",
        16000,
        777,
    )


def test_diarization_server_route_uses_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from octomil.audio import diarization as diarization_module

    _FakeDiarizationBackend.instances = []
    monkeypatch.setattr(
        diarization_module,
        "NativeDiarizationBackend",
        _FakeDiarizationBackend,
    )
    app = _create_server_app()

    async def _request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/audio/diarizations", json={**_payload(), "deadline_ms": 321})

    resp = asyncio.run(_request())

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "audio.diarization"
    assert body["segments"] == [
        {"start_ms": 0, "end_ms": 500, "speaker_id": 1, "speaker_label": "speaker_1"},
        {"start_ms": 500, "end_ms": 900, "speaker_id": 2, "speaker_label": "speaker_2"},
    ]
    assert _FakeDiarizationBackend.instances[0].diarize_args == (
        b"\x00\x00\x00\x00",
        16000,
        321,
    )


def test_native_audio_route_error_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from octomil.audio import vad as vad_module

    class FailingVadBackend(_FakeVadBackend):
        def open(self) -> None:
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message="native audio.vad unavailable",
            )

    monkeypatch.setattr(vad_module, "NativeVadBackend", FailingVadBackend)
    app = _create_server_app()

    async def _request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/audio/vad", json=_payload())

    resp = asyncio.run(_request())

    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "runtime_unavailable"
    assert "native audio.vad unavailable" in body["message"]
