"""Unit tests for the optional ABI minor 11 image-input bindings.

Three guarantees pinned here, all without requiring a real
``liboctomil-runtime`` dylib:

  1. The cffi cdef compiles cleanly (the new ``oct_image_view_t``
     struct + ``oct_image_view_size`` + ``oct_session_send_image``
     parse alongside the rest of the cdef).
  2. With a mock runtime advertising ABI minor=10, the loader does
     NOT bind ``oct_session_send_image`` and does NOT raise — image
     input is OPT-IN, the required ABI floor stays at 10.
  3. With a mock runtime advertising ABI minor=11 but NOT the
     ``embeddings.image`` capability, the public-facing image path
     raises :class:`OctomilUnsupportedError` naming the capability.

These tests deliberately avoid the ``requires_runtime`` marker —
they're pure-Python contract checks on the cdef parse and the
gate functions, so they run on CI without a fetched dylib.

NOTE: there is intentionally NO end-to-end integration test
exercising the function call against a real runtime.
``embeddings.image`` is BLOCKED_WITH_PROOF; no live capability
exists yet. The :meth:`NativeSession.embeddings_image` public
surface is wired to ``raise NotImplementedError`` and we assert
that contract here.
"""

from __future__ import annotations

import pytest

cffi = pytest.importorskip("cffi", reason="cffi extra not installed")  # noqa: F841


# ---------------------------------------------------------------------------
# 1. cdef compiles cleanly
# ---------------------------------------------------------------------------


def test_cdef_parses_with_image_input_symbols():
    """The cdef in loader._CDEF includes the v0.1.12 / ABI minor 11
    image-input additions (oct_image_view_t, oct_image_view_size,
    oct_session_send_image). Parse must succeed independently of
    whether a dylib is available."""
    from cffi import FFI

    from octomil.runtime.native import loader

    ffi = FFI()
    ffi.cdef(loader._CDEF)
    # struct must be cdef'd
    assert ffi.sizeof("oct_image_view_t") > 0
    # function decls must be visible to the parser
    assert "oct_image_view_size" in loader._CDEF
    assert "oct_session_send_image" in loader._CDEF
    assert "oct_image_view_t" in loader._CDEF


def test_image_mime_constants_match_runtime_header():
    """Closed-enum values pinned against runtime.h (PR #86 / 1d92e35).
    A drift would mean a binding constant disagrees with the dylib's
    interpretation of ``oct_image_view_t.mime``."""
    from octomil.runtime.native import loader

    assert loader.OCT_IMAGE_MIME_UNKNOWN == 0
    assert loader.OCT_IMAGE_MIME_PNG == 1
    assert loader.OCT_IMAGE_MIME_JPEG == 2
    assert loader.OCT_IMAGE_MIME_WEBP == 3
    assert loader.OCT_IMAGE_MIME_RGB8 == 4


def test_image_clip_pooling_constant_appended():
    """v0.1.12 appends OCT_EMBED_POOLING_IMAGE_CLIP=5 to the existing
    pooling enum without changing values 0..4."""
    from octomil.runtime.native import loader

    assert loader.OCT_EMBED_POOLING_IMAGE_CLIP == 5


def test_required_abi_minor_is_unchanged_at_10():
    """HARD INVARIANT for this PR: the binding's required ABI floor
    MUST remain at 10. Bumping to 11 would force the entire SDK
    to refuse older minor=10 dylibs even though image-input is
    optional and capability gates are still in place."""
    from octomil.runtime.native import loader

    assert loader._REQUIRED_ABI_MINOR == 10


# ---------------------------------------------------------------------------
# 2. Conditional symbol bind on a minor=10 dylib does NOT raise
# ---------------------------------------------------------------------------


class _FakeLibMinor10:
    """Stand-in for a real cffi ``lib`` whose dylib advertises
    ABI minor=10 and does NOT export the v0.1.12 image-input
    symbols. Accessing the new symbols raises AttributeError,
    matching cffi's behavior on missing dylib exports."""

    def __getattr__(self, name: str):
        if name in ("oct_image_view_size", "oct_session_send_image"):
            raise AttributeError(name)
        raise AttributeError(name)


class _FakeLibMinor11:
    """Stand-in for a real cffi ``lib`` whose dylib advertises
    ABI minor=11 and exports the optional image-input symbols.
    The function values themselves are just sentinel callables —
    nothing in these tests invokes them.
    """

    def __init__(self) -> None:
        self.oct_image_view_size = lambda: 24  # sentinel
        self.oct_session_send_image = lambda *_args, **_kwargs: 2  # OCT_STATUS_UNSUPPORTED


def test_resolve_image_symbols_minor10_skips_without_raising():
    """On a minor=10 dylib, the optional image-input slots stay None
    AND the resolver does not raise. This is the hard rule: image
    input is OPT-IN."""
    from octomil.runtime.native import loader

    fake_lib = _FakeLibMinor10()
    # Should NOT raise on a minor=10 dylib.
    loader._resolve_optional_image_symbols(fake_lib, dylib_minor=10)
    assert loader._OPTIONAL_IMAGE_SYMBOLS["oct_session_send_image"] is None
    assert loader._OPTIONAL_IMAGE_SYMBOLS["oct_image_view_size"] is None


def test_resolve_image_symbols_minor11_binds_when_present():
    """On a minor=11 dylib that exports both symbols, the resolver
    populates both slots. This is the path that lets future
    capability-flipped runtimes get exercised."""
    from octomil.runtime.native import loader

    fake_lib = _FakeLibMinor11()
    loader._resolve_optional_image_symbols(fake_lib, dylib_minor=11)
    try:
        assert loader._OPTIONAL_IMAGE_SYMBOLS["oct_image_view_size"] is fake_lib.oct_image_view_size
        assert loader._OPTIONAL_IMAGE_SYMBOLS["oct_session_send_image"] is fake_lib.oct_session_send_image
    finally:
        # Reset so other tests / modules see the canonical None state.
        loader._OPTIONAL_IMAGE_SYMBOLS["oct_image_view_size"] = None
        loader._OPTIONAL_IMAGE_SYMBOLS["oct_session_send_image"] = None


def test_resolve_image_symbols_minor11_missing_symbol_stays_none():
    """Defensive: if the dylib advertises minor=11 but the symbol is
    unexpectedly missing (corrupt build, partial relink), the slot
    stays None rather than raising. The capability gate at call
    time surfaces the same UNSUPPORTED error path."""
    from octomil.runtime.native import loader

    fake_lib = _FakeLibMinor10()  # raises AttributeError on the new symbols
    loader._resolve_optional_image_symbols(fake_lib, dylib_minor=11)
    assert loader._OPTIONAL_IMAGE_SYMBOLS["oct_session_send_image"] is None
    assert loader._OPTIONAL_IMAGE_SYMBOLS["oct_image_view_size"] is None


# ---------------------------------------------------------------------------
# 3. Capability gate: minor=11 + capability NOT advertised raises clearly
# ---------------------------------------------------------------------------


class _StubCapabilities:
    """Mimic the return shape of ``NativeRuntime.capabilities()`` —
    only ``supported_capabilities`` is read by the image-input gate."""

    def __init__(self, supported: tuple[str, ...]) -> None:
        self.supported_capabilities = supported


class _StubOwner:
    """Stand-in for the NativeRuntime owning a session. Returns a
    canned capability set and an empty last_error."""

    def __init__(self, advertised: tuple[str, ...]) -> None:
        self._advertised = advertised

    def capabilities(self) -> _StubCapabilities:
        return _StubCapabilities(self._advertised)

    def last_error(self) -> str:
        return ""


def _make_stub_session(monkeypatch: pytest.MonkeyPatch, owner: _StubOwner):
    """Construct a NativeSession-like instance without going through
    NativeRuntime / oct_session_open. The send_image gate only reads
    ``_check_open``, ``_owner``, ``_handle``, ``_ffi``, ``_lib`` —
    we stub what's needed."""
    from octomil.runtime.native import loader

    # The gate reads from _OPTIONAL_IMAGE_SYMBOLS via the module
    # global, so anything we patch here lives at the module level.
    sess = loader.NativeSession.__new__(loader.NativeSession)
    sess._ffi = None  # not exercised in the gate-rejection paths
    sess._lib = None
    sess._handle = object()
    sess._owner = owner  # type: ignore[assignment]  # stub stands in for NativeRuntime
    sess._closed = False
    sess._handle_invalid = False
    sess._event_buf = None
    sess._borrowed_model = None
    return sess


def test_send_image_raises_unsupported_when_symbol_unresolved(monkeypatch):
    """Gate 1: even if the runtime were to advertise
    ``embeddings.image`` (it does not yet), an unresolved symbol on
    a minor=10 dylib MUST cause OctomilUnsupportedError. We also
    assert the message names the capability."""
    from octomil.runtime.native import loader

    # Pin both gates: symbol unresolved + capability not advertised.
    monkeypatch.setitem(loader._OPTIONAL_IMAGE_SYMBOLS, "oct_session_send_image", None)

    # Stub abi_version to avoid touching a real dylib.
    monkeypatch.setattr(loader, "abi_version", lambda: (0, 10, 0))

    owner = _StubOwner(advertised=("chat.completion",))  # capability absent
    sess = _make_stub_session(monkeypatch, owner)

    with pytest.raises(loader.OctomilUnsupportedError) as exc_info:
        sess.send_image(b"\x89PNG\r\n\x1a\n\x00", mime=loader.OCT_IMAGE_MIME_PNG)
    err = exc_info.value
    assert err.capability == "embeddings.image"
    assert err.status == loader.OCT_STATUS_UNSUPPORTED
    assert "embeddings.image" in str(err)
    assert "minor" in str(err).lower()


def test_send_image_raises_unsupported_when_capability_not_advertised(monkeypatch):
    """Gate 2: symbol IS resolved (minor=11 dylib loaded) but the
    runtime does NOT advertise ``embeddings.image``. MUST raise
    OctomilUnsupportedError with a message naming the capability —
    this is the canonical 'BLOCKED_WITH_PROOF' refusal path."""
    from octomil.runtime.native import loader

    # Pretend the symbol resolved (we never call it on this path).
    monkeypatch.setitem(loader._OPTIONAL_IMAGE_SYMBOLS, "oct_session_send_image", lambda *a, **k: 0)

    monkeypatch.setattr(loader, "abi_version", lambda: (0, 11, 0))

    # Advertise only OTHER capabilities, NOT embeddings.image.
    owner = _StubOwner(advertised=("chat.completion", "embeddings.text"))
    sess = _make_stub_session(monkeypatch, owner)

    with pytest.raises(loader.OctomilUnsupportedError) as exc_info:
        sess.send_image(b"\x89PNG\r\n\x1a\n\x00", mime=loader.OCT_IMAGE_MIME_PNG)
    err = exc_info.value
    assert err.capability == "embeddings.image"
    assert err.status == loader.OCT_STATUS_UNSUPPORTED
    assert "embeddings.image" in str(err)
    assert "not advertised" in str(err).lower()


def test_embeddings_image_public_surface_forwards_to_send_image_and_polls(monkeypatch):
    """v0.1.14: ``NativeSession.embeddings_image`` is no longer a
    NotImplementedError placeholder. It MUST forward to
    :meth:`send_image` and drain ``OCT_EVENT_EMBEDDING_VECTOR`` +
    ``OCT_EVENT_SESSION_COMPLETED``, returning a list[float]. This
    pins the post-flip contract paired with runtime PR #91."""
    from octomil.runtime.native import loader

    ffi = _build_ffi_with_cdef()

    # Pretend the runtime advertises the capability AND the symbol is
    # resolved.
    sent = {}

    def fake_send_image(handle, view_ptr):
        sent["n_bytes"] = int(view_ptr.n_bytes)
        sent["mime"] = int(view_ptr.mime)
        return loader.OCT_STATUS_OK

    monkeypatch.setitem(loader._OPTIONAL_IMAGE_SYMBOLS, "oct_session_send_image", fake_send_image)
    monkeypatch.setattr(loader, "abi_version", lambda: (0, 11, 0))

    owner = _StubOwner(advertised=("embeddings.image",))
    sess = _make_stub_session(monkeypatch, owner)
    sess._ffi = ffi

    # Stub poll_event to emit one EMBEDDING_VECTOR followed by
    # SESSION_COMPLETED(OK). We construct NativeEvent objects directly
    # via the public constructor.
    expected_vector = [0.1, 0.2, 0.3, 0.4]

    from typing import Any as _Any

    def _ev(ev_type: int, **extra: _Any) -> loader.NativeEvent:
        return loader.NativeEvent(
            type=ev_type,
            version=loader.OCT_EVENT_VERSION,
            monotonic_ns=0,
            user_data_ptr=0,
            **extra,
        )

    polls = iter(
        [
            _ev(loader.OCT_EVENT_SESSION_STARTED),
            _ev(
                loader.OCT_EVENT_EMBEDDING_VECTOR,
                values=expected_vector,
                n_dim=len(expected_vector),
                index=0,
                pooling_type=loader.OCT_EMBED_POOLING_IMAGE_CLIP,
                is_normalized=True,
            ),
            _ev(
                loader.OCT_EVENT_SESSION_COMPLETED,
                terminal_status=loader.OCT_STATUS_OK,
            ),
        ]
    )
    monkeypatch.setattr(sess, "poll_event", lambda timeout_ms=0: next(polls))

    result = sess.embeddings_image(b"\x89PNG\r\n\x1a\n\x00", mime=loader.OCT_IMAGE_MIME_PNG)
    assert result == expected_vector
    assert sent["mime"] == loader.OCT_IMAGE_MIME_PNG


def _build_ffi_with_cdef():
    """Build a fresh cffi FFI parsing the loader's full cdef. Lets us
    materialize a real ``oct_image_view_t*`` without loading the dylib."""
    from cffi import FFI

    from octomil.runtime.native import loader

    ffi = FFI()
    ffi.cdef(loader._CDEF)
    return ffi


def test_send_image_uses_byte_count_for_non_byte_memoryview(monkeypatch):
    """Regression for the memoryview byte-count bug: when the caller
    passes a ``memoryview`` whose underlying buffer has itemsize > 1
    (e.g., ``memoryview(array.array('I', [...]))``), ``len(mv)`` is the
    ELEMENT count, NOT the BYTE count. The C struct
    ``oct_image_view_t.n_bytes`` MUST be the true byte length. This
    test pins the fix by stubbing the C-call and asserting on the
    populated view fields."""
    import array

    from octomil.runtime.native import loader

    ffi = _build_ffi_with_cdef()

    captured: dict[str, object] = {}

    def fake_send_image(handle, view_ptr):
        # cffi exposes the struct fields by attribute access on the
        # dereferenced pointer; capture n_bytes for assertion below.
        captured["n_bytes"] = int(view_ptr.n_bytes)
        captured["mime"] = int(view_ptr.mime)
        captured["handle"] = handle
        return loader.OCT_STATUS_OK

    monkeypatch.setitem(loader._OPTIONAL_IMAGE_SYMBOLS, "oct_session_send_image", fake_send_image)
    monkeypatch.setattr(loader, "abi_version", lambda: (0, 11, 0))

    owner = _StubOwner(advertised=("embeddings.image",))
    sess = _make_stub_session(monkeypatch, owner)
    sess._ffi = ffi  # real ffi so view = ffi.new(...) works

    # array.array('I') -> itemsize=4 on every platform we ship to;
    # len(mv) returns N (element count), but n_bytes MUST be N * 4.
    arr = array.array("I", [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0xFEEDFACE])
    mv = memoryview(arr)
    assert mv.itemsize == 4
    assert len(mv) == 4  # element count
    assert mv.nbytes == 16  # byte count

    sess.send_image(mv, mime=loader.OCT_IMAGE_MIME_RGB8)

    # The fix: n_bytes is the TRUE byte length (16), not the element
    # count (4). If this asserts 4 the bug regressed.
    assert (
        captured["n_bytes"] == 16
    ), f"n_bytes should be byte count (16), got {captured['n_bytes']}; memoryview byte-count bug regressed"
    assert captured["mime"] == loader.OCT_IMAGE_MIME_RGB8


def test_send_image_byte_memoryview_round_trips_correctly(monkeypatch):
    """Sanity: a plain ``memoryview(bytes)`` (itemsize=1) still
    populates ``n_bytes`` correctly after the cast-to-byte
    normalization."""
    from octomil.runtime.native import loader

    ffi = _build_ffi_with_cdef()
    captured: dict[str, int] = {}

    def fake_send_image(handle, view_ptr):
        captured["n_bytes"] = int(view_ptr.n_bytes)
        return loader.OCT_STATUS_OK

    monkeypatch.setitem(loader._OPTIONAL_IMAGE_SYMBOLS, "oct_session_send_image", fake_send_image)
    monkeypatch.setattr(loader, "abi_version", lambda: (0, 11, 0))

    owner = _StubOwner(advertised=("embeddings.image",))
    sess = _make_stub_session(monkeypatch, owner)
    sess._ffi = ffi

    payload = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0d"
    sess.send_image(memoryview(payload), mime=loader.OCT_IMAGE_MIME_PNG)
    assert captured["n_bytes"] == len(payload)


def test_send_image_bytearray_uses_byte_count(monkeypatch):
    """Sanity: bytearray path still populates ``n_bytes`` correctly
    after the memoryview-cast normalization."""
    from octomil.runtime.native import loader

    ffi = _build_ffi_with_cdef()
    captured: dict[str, int] = {}

    def fake_send_image(handle, view_ptr):
        captured["n_bytes"] = int(view_ptr.n_bytes)
        return loader.OCT_STATUS_OK

    monkeypatch.setitem(loader._OPTIONAL_IMAGE_SYMBOLS, "oct_session_send_image", fake_send_image)
    monkeypatch.setattr(loader, "abi_version", lambda: (0, 11, 0))

    owner = _StubOwner(advertised=("embeddings.image",))
    sess = _make_stub_session(monkeypatch, owner)
    sess._ffi = ffi

    payload = bytearray(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    sess.send_image(payload, mime=loader.OCT_IMAGE_MIME_JPEG)
    assert captured["n_bytes"] == len(payload)


def test_unsupported_error_is_subclass_of_native_runtime_error():
    """Callers that already catch :class:`NativeRuntimeError` for the
    UNSUPPORTED status code MUST still catch
    :class:`OctomilUnsupportedError` — it is a strict subclass."""
    from octomil.runtime.native import loader

    err = loader.OctomilUnsupportedError("embeddings.image", "test")
    assert isinstance(err, loader.NativeRuntimeError)
    assert err.status == loader.OCT_STATUS_UNSUPPORTED
    assert err.capability == "embeddings.image"
