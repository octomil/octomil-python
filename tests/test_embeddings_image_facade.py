"""Unit tests for the public ``client.embeddings.image(...)`` facade.

Five guarantees pinned here, all without requiring a real
``liboctomil-runtime`` dylib:

  1. Capability NOT advertised by the runtime → ``OctomilUnsupportedError``
     naming ``embeddings.image``. This is the canonical refusal path on
     Linux/Android (runtime adapter refuses) and on darwin-arm64 builds
     missing sherpa-onnx or the SigLIP artifact env var.
  2. Capability ADVERTISED and the underlying ``NativeSession`` returns a
     vector → :class:`ImageEmbedding` with the right ``vector`` /
     ``n_dim`` / ``model``.
  3. RGB8 buffer of the wrong size → ``ValueError`` naming
     ``OCT_IMAGE_RGB8_FIXED_BYTES`` and the expected vs got count.
  4. ``OCT_IMAGE_MIME_UNKNOWN`` → ``ValueError``.
  5. ``OCT_IMAGE_MIME_WEBP`` → ``OctomilUnsupportedError`` with
     "WebP" in the message (vendored stb_image does not decode WebP).

The tests stub :class:`NativeRuntime` + :class:`NativeSession` at the
import site of ``FacadeEmbeddings.image`` so no real dylib is needed.
This is consistent with ``test_runtime_native_image_bindings.py`` —
the runtime adapter is the source of truth, Python just forwards.
"""

from __future__ import annotations

from typing import Any

import pytest

from octomil.facade import FacadeEmbeddings, ImageEmbedding
from octomil.runtime.native.loader import (
    OCT_IMAGE_MIME_JPEG,
    OCT_IMAGE_MIME_PNG,
    OCT_IMAGE_MIME_RGB8,
    OCT_IMAGE_MIME_UNKNOWN,
    OCT_IMAGE_MIME_WEBP,
    OCT_IMAGE_RGB8_FIXED_BYTES,
    OctomilUnsupportedError,
)


class _StubCapabilities:
    """Stand-in for the return shape of ``NativeRuntime.capabilities()``."""

    def __init__(self, supported: tuple[str, ...]) -> None:
        self.supported_capabilities = supported


class _StubSession:
    """Stand-in for ``NativeSession`` — captures the request shape and
    returns a canned vector. ``embeddings_image`` is what the facade
    calls; ``close`` is called from the facade's finally-block."""

    def __init__(self, *, vector: list[float] | None = None) -> None:
        self._vector = vector if vector is not None else [0.1] * 768
        self.closed = False
        self.calls: list[dict[str, Any]] = []

    def embeddings_image(self, image_bytes: Any, *, mime: int, deadline_ms: int = 0) -> list[float]:
        self.calls.append({"image_bytes": bytes(image_bytes), "mime": mime, "deadline_ms": deadline_ms})
        return self._vector

    def close(self) -> None:
        self.closed = True


class _StubRuntime:
    """Stand-in for ``NativeRuntime`` — exposes ``capabilities()`` +
    ``open_session(...)`` and tracks ``close()``."""

    def __init__(self, *, advertised: tuple[str, ...], session: _StubSession | None = None) -> None:
        self._caps = _StubCapabilities(advertised)
        self._session = session
        self.closed = False
        self.open_session_kwargs: dict[str, Any] = {}

    def capabilities(self) -> _StubCapabilities:
        return self._caps

    def open_session(self, **kwargs: Any) -> _StubSession:
        self.open_session_kwargs = kwargs
        # Return a fresh session if one wasn't pre-injected, so tests
        # that don't care about the vector still get a valid object.
        return self._session or _StubSession()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def patch_runtime(monkeypatch: pytest.MonkeyPatch):
    """Factory: return a closure that patches NativeRuntime.open at the
    facade's import site with a stub runtime that has the given
    advertised capabilities + optional pre-injected session."""

    def _patch(*, advertised: tuple[str, ...], session: _StubSession | None = None) -> _StubRuntime:
        rt = _StubRuntime(advertised=advertised, session=session)
        # The facade imports NativeRuntime lazily inside .image(), so
        # we patch on the loader module — that's the actual symbol the
        # facade dereferences.
        from octomil.runtime.native import loader as _loader

        monkeypatch.setattr(_loader.NativeRuntime, "open", classmethod(lambda cls: rt))
        return rt

    return _patch


def _facade() -> FacadeEmbeddings:
    """The facade only touches ``self._client`` on the cloud path and
    ``self._kernel`` on the app/policy path; the image() method
    bypasses both."""

    class _Dummy:
        pass

    return FacadeEmbeddings(_Dummy(), kernel=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Capability NOT advertised → OctomilUnsupportedError
# ---------------------------------------------------------------------------


def test_embeddings_image_capability_not_advertised_raises(patch_runtime):
    """Refusal path: darwin-arm64 dylib that doesn't advertise
    embeddings.image (no sherpa-onnx) OR any Linux/Android dylib.
    Must raise OctomilUnsupportedError naming the capability."""
    rt = patch_runtime(advertised=("chat.completion", "embeddings.text"))

    with pytest.raises(OctomilUnsupportedError) as exc_info:
        _facade().image(b"\x89PNG\r\n\x1a\n\x00", mime=OCT_IMAGE_MIME_PNG)

    err = exc_info.value
    assert err.capability == "embeddings.image"
    assert "embeddings.image" in str(err)
    assert "not advertised" in str(err).lower()
    # Even on the refusal path the runtime is closed cleanly.
    assert rt.closed is True


# ---------------------------------------------------------------------------
# 2. Capability ADVERTISED → returns a vector
# ---------------------------------------------------------------------------


def test_embeddings_image_capability_advertised_returns_vector(patch_runtime, monkeypatch):
    """Live path: runtime advertises the capability and the underlying
    NativeSession returns a 768-dim vector. The facade must wrap it
    into an ImageEmbedding and close runtime+session cleanly."""
    monkeypatch.setenv("OCTOMIL_EMBEDDINGS_IMAGE_MODEL_PATH", "/tmp/siglip.onnx")
    expected = [float(i) / 768.0 for i in range(768)]
    session = _StubSession(vector=expected)
    rt = patch_runtime(advertised=("embeddings.image",), session=session)

    result = _facade().image(b"\x89PNG\r\n\x1a\n\x00", mime=OCT_IMAGE_MIME_PNG)

    assert isinstance(result, ImageEmbedding)
    assert result.vector == expected
    assert result.n_dim == 768
    assert result.model == "/tmp/siglip.onnx"
    # open_session was called with the right capability + model_uri
    assert rt.open_session_kwargs.get("capability") == "embeddings.image"
    assert rt.open_session_kwargs.get("model_uri") == "/tmp/siglip.onnx"
    assert rt.open_session_kwargs.get("locality") == "on_device"
    # Session call shape: PNG, the right deadline_ms, the same bytes.
    assert session.calls[0]["mime"] == OCT_IMAGE_MIME_PNG
    assert session.calls[0]["image_bytes"] == b"\x89PNG\r\n\x1a\n\x00"
    # Cleanup ran in the finally-block.
    assert session.closed is True
    assert rt.closed is True


# ---------------------------------------------------------------------------
# 3. RGB8 wrong size → ValueError naming OCT_IMAGE_RGB8_FIXED_BYTES
# ---------------------------------------------------------------------------


def test_embeddings_image_wrong_rgb8_size_raises_value_error(patch_runtime):
    """200-byte buffer with mime=RGB8 is way under the canonical
    150528 byte SigLIP-base input shape. Must raise ValueError with a
    message mentioning OCT_IMAGE_RGB8_FIXED_BYTES + the expected /
    got counts. We patch the runtime so a real dylib isn't needed,
    even though the validator triggers before we touch it."""
    patch_runtime(advertised=("embeddings.image",))

    with pytest.raises(ValueError) as exc_info:
        _facade().image(b"\x00" * 200, mime=OCT_IMAGE_MIME_RGB8)

    msg = str(exc_info.value)
    assert "OCT_IMAGE_RGB8_FIXED_BYTES" in msg
    assert str(OCT_IMAGE_RGB8_FIXED_BYTES) in msg  # 150528
    assert "200" in msg


def test_embeddings_image_rgb8_correct_size_does_not_raise(patch_runtime, monkeypatch):
    """Sanity: a 150528-byte RGB8 buffer passes validation and reaches
    the session. Pins that the validator uses the exact constant
    rather than an off-by-one."""
    monkeypatch.setenv("OCTOMIL_EMBEDDINGS_IMAGE_MODEL_PATH", "/tmp/siglip.onnx")
    session = _StubSession(vector=[1.0] * 768)
    patch_runtime(advertised=("embeddings.image",), session=session)

    payload = b"\x80" * OCT_IMAGE_RGB8_FIXED_BYTES  # exactly 150528 bytes
    result = _facade().image(payload, mime=OCT_IMAGE_MIME_RGB8)
    assert isinstance(result, ImageEmbedding)
    assert result.n_dim == 768
    assert session.calls[0]["mime"] == OCT_IMAGE_MIME_RGB8


# ---------------------------------------------------------------------------
# 4. UNKNOWN mime → ValueError
# ---------------------------------------------------------------------------


def test_embeddings_image_unknown_mime_raises(patch_runtime):
    """OCT_IMAGE_MIME_UNKNOWN is the forward-compat sentinel; callers
    should never pass it. Must raise ValueError naming the sentinel."""
    patch_runtime(advertised=("embeddings.image",))

    with pytest.raises(ValueError) as exc_info:
        _facade().image(b"anything", mime=OCT_IMAGE_MIME_UNKNOWN)

    msg = str(exc_info.value)
    assert "OCT_IMAGE_MIME_UNKNOWN" in msg


# ---------------------------------------------------------------------------
# 5. WEBP mime → OctomilUnsupportedError naming "WebP"
# ---------------------------------------------------------------------------


def test_embeddings_image_webp_raises_unsupported(patch_runtime):
    """The vendored stb_image in the v0.1.14 runtime does not include
    WebP. Surface this at the Python facade with a clearer diagnostic
    so callers don't get a confusing UNSUPPORTED from below the ABI."""
    patch_runtime(advertised=("embeddings.image",))

    with pytest.raises(OctomilUnsupportedError) as exc_info:
        _facade().image(b"RIFF\x00\x00\x00\x00WEBP", mime=OCT_IMAGE_MIME_WEBP)

    err = exc_info.value
    assert err.capability == "embeddings.image"
    assert "WebP" in str(err)


def test_embeddings_image_jpeg_advertised_path(patch_runtime, monkeypatch):
    """Sanity: JPEG flows through the same advertised path as PNG.
    Pins that the facade doesn't accidentally allow-list a single
    encoded MIME."""
    monkeypatch.setenv("OCTOMIL_EMBEDDINGS_IMAGE_MODEL_PATH", "/tmp/siglip.onnx")
    session = _StubSession(vector=[0.5] * 768)
    patch_runtime(advertised=("embeddings.image",), session=session)

    result = _facade().image(b"\xff\xd8\xff\xe0\x00\x10JFIF", mime=OCT_IMAGE_MIME_JPEG)
    assert isinstance(result, ImageEmbedding)
    assert session.calls[0]["mime"] == OCT_IMAGE_MIME_JPEG
