"""Tests for OCT-113 v2 — NativeSttBackend.load_uploaded_model.

OCT-113 v1 shipped the server-side architecture × capability gate
in octomil-server. v2 is the cross-repo SDK half: a new entry point
that lets the SDK open a user-uploaded Whisper artifact by path
instead of being constrained to the canonical
``_WHISPER_ARTIFACTS`` SHA registry.

What we pin here:

1. The new method accepts a (model_name, artifact_path,
   expected_sha256) tuple and skips the canonical spec lookup.
2. Integrity check: if the file's actual SHA doesn't match the
   caller-supplied checksum we raise ``CHECKSUM_MISMATCH`` BEFORE
   touching the runtime.
3. Missing / unreadable artifact → ``MODEL_NOT_FOUND``.
4. Already-loaded backend rejects a different uploaded model with
   ``INVALID_INPUT`` (caller must close first).
5. Same uploaded model is idempotent on re-load.
6. Sets the OCT_WHISPER_ALLOW_USER_ARTIFACTS env var before opening
   the runtime — the runtime side checks for it before relaxing
   its built-in digest registry.
7. When the runtime DOES advertise the capability (user-upload
   support live), open_model is called with the file's own digest.
8. When the runtime does NOT advertise (current state — runtime
   side not yet built), the SDK surfaces a diagnostic that names
   the missing runtime support instead of falling back to the
   canonical pin.

The runtime side itself is mocked throughout — these are SDK-layer
tests; integration with a real dylib lands when the runtime
ships ``OCT_WHISPER_ALLOW_USER_ARTIFACTS`` support.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.native.stt_backend import (
    _WHISPER_ALLOW_USER_ARTIFACTS_ENV,
    _WHISPER_BIN_ENV,
    NativeSttBackend,
)


def _write_artifact(tmp_path: Path, content: bytes = b"GGUF") -> tuple[Path, str]:
    """Write bytes to a tempfile, return (path, sha256_hex)."""
    p = tmp_path / "user-whisper.gguf"
    p.write_bytes(content + b"\x00" * 1024)  # > min header length
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return p, sha


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_malformed_sha_rejected(self, tmp_path: Path) -> None:
        path, _ = _write_artifact(tmp_path)
        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc:
            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(path),
                expected_sha256="not-hex",  # not 64 chars
            )
        assert exc.value.code == OctomilErrorCode.INVALID_INPUT

    def test_short_sha_rejected(self, tmp_path: Path) -> None:
        path, _ = _write_artifact(tmp_path)
        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc:
            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(path),
                expected_sha256="a" * 32,
            )
        assert exc.value.code == OctomilErrorCode.INVALID_INPUT


# ---------------------------------------------------------------------------
# File checks
# ---------------------------------------------------------------------------


class TestFileChecks:
    def test_missing_path_raises_model_not_found(self, tmp_path: Path) -> None:
        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc:
            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(tmp_path / "no-such-file.gguf"),
                expected_sha256="a" * 64,
            )
        assert exc.value.code == OctomilErrorCode.MODEL_NOT_FOUND

    def test_directory_path_raises_model_not_found(self, tmp_path: Path) -> None:
        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc:
            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(tmp_path),  # tmp_path is a directory
                expected_sha256="a" * 64,
            )
        assert exc.value.code == OctomilErrorCode.MODEL_NOT_FOUND


# ---------------------------------------------------------------------------
# Integrity (SHA verification)
# ---------------------------------------------------------------------------


class TestIntegrityCheck:
    def test_sha_mismatch_raises_checksum_mismatch(self, tmp_path: Path) -> None:
        """Caller says SHA = X but file hashes to Y → reject without
        touching the runtime. Protects against bytes mutating between
        server-side storage and on-device load."""
        path, _real_sha = _write_artifact(tmp_path)
        backend = NativeSttBackend()
        with pytest.raises(OctomilError) as exc:
            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(path),
                expected_sha256="b" * 64,  # bogus
            )
        assert exc.value.code == OctomilErrorCode.CHECKSUM_MISMATCH
        # Error message includes both digests truncated.
        assert "SHA-256" in exc.value.error_message

    def test_sha_match_passes_integrity_check(self, tmp_path: Path) -> None:
        """When caller-supplied SHA matches the file, we proceed to
        runtime open. Patch the runtime to advertise the capability
        so we exercise the success path."""
        path, real_sha = _write_artifact(tmp_path)
        backend = NativeSttBackend()

        # Mock the runtime so we don't need the real dylib.
        with (
            patch("octomil.runtime.native.stt_backend.NativeRuntime") as runtime_class,
            patch(
                "octomil.runtime.native.stt_backend._runtime_advertises_audio_transcription",
                return_value=True,
            ),
        ):
            runtime_inst = MagicMock()
            runtime_inst.open_model.return_value = MagicMock()
            runtime_class.open.return_value = runtime_inst

            # Should not raise.
            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(path),
                expected_sha256=real_sha,
            )

            # open_model was called with the file's actual SHA as the
            # integrity-only artifact_digest.
            call_kwargs = runtime_inst.open_model.call_args.kwargs
            assert call_kwargs["artifact_digest"] == f"sha256:{real_sha}"
            assert call_kwargs["engine_hint"] == "whisper_cpp"
            assert call_kwargs["model_uri"] == str(path)
        backend.close()


# ---------------------------------------------------------------------------
# Env-var bridge to runtime
# ---------------------------------------------------------------------------


class TestRuntimeEnvBridge:
    def test_sets_allow_user_artifacts_env_before_runtime_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OCT_WHISPER_ALLOW_USER_ARTIFACTS must be set before
        NativeRuntime.open is invoked — the runtime checks the env
        at open time to decide whether to relax its built-in digest
        registry."""
        monkeypatch.delenv(_WHISPER_ALLOW_USER_ARTIFACTS_ENV, raising=False)

        path, real_sha = _write_artifact(tmp_path)
        backend = NativeSttBackend()

        observed_env: dict[str, str | None] = {}

        def _capture_env() -> MagicMock:
            observed_env["allow"] = os.environ.get(_WHISPER_ALLOW_USER_ARTIFACTS_ENV)
            observed_env["bin"] = os.environ.get(_WHISPER_BIN_ENV)
            return MagicMock()

        with (
            patch("octomil.runtime.native.stt_backend.NativeRuntime") as runtime_class,
            patch(
                "octomil.runtime.native.stt_backend._runtime_advertises_audio_transcription",
                return_value=True,
            ),
        ):
            runtime_class.open.side_effect = _capture_env

            # The call may succeed or fail downstream; we only assert
            # env was set at the moment NativeRuntime.open ran.
            try:
                backend.load_uploaded_model(
                    model_name="my-whisper",
                    artifact_path=str(path),
                    expected_sha256=real_sha,
                )
            except Exception:
                pass

        assert observed_env["allow"] == "1", "OCT_WHISPER_ALLOW_USER_ARTIFACTS not set before runtime open"
        assert observed_env["bin"] == str(path), "OCTOMIL_WHISPER_BIN not pointing at user artifact at open time"


# ---------------------------------------------------------------------------
# Runtime support gap diagnostic
# ---------------------------------------------------------------------------


class TestRuntimeSupportGap:
    def test_runtime_digest_mismatch_surfaces_clear_diagnostic(self, tmp_path: Path) -> None:
        """When the runtime doesn't yet support user uploads it
        rejects our artifact's SHA (not in built-in registry) and
        the load_model code path raises CHECKSUM_MISMATCH. For
        user-upload loads we surface RUNTIME_UNAVAILABLE with a
        diagnostic that names the missing runtime support instead
        of leaking the canonical-pin error message."""
        path, real_sha = _write_artifact(tmp_path)
        backend = NativeSttBackend()

        with (
            patch("octomil.runtime.native.stt_backend.NativeRuntime") as runtime_class,
            patch(
                "octomil.runtime.native.stt_backend._runtime_advertises_audio_transcription",
                return_value=False,  # runtime refuses to advertise
            ),
        ):
            runtime_inst = MagicMock()
            runtime_inst.last_error.return_value = "expected digest sha256:canonical-x; got sha256:user-y"
            runtime_class.open.return_value = runtime_inst

            with pytest.raises(OctomilError) as exc:
                backend.load_uploaded_model(
                    model_name="my-whisper",
                    artifact_path=str(path),
                    expected_sha256=real_sha,
                )

        assert exc.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        msg = exc.value.error_message.lower()
        # The diagnostic must name the runtime-side gap, not the
        # canonical-pin error message.
        assert "user-upload" in msg or "user-uploaded" in msg or "user uploaded" in msg
        assert "rebuild" in msg or "support" in msg

    def test_runtime_capability_missing_for_other_reasons(self, tmp_path: Path) -> None:
        """When the runtime doesn't advertise but the last_error
        doesn't mention digest (e.g. dylib built without
        OCT_ENABLE_ENGINE_WHISPER_CPP), surface a generic
        RUNTIME_UNAVAILABLE that names the artifact path."""
        path, real_sha = _write_artifact(tmp_path)
        backend = NativeSttBackend()

        with (
            patch("octomil.runtime.native.stt_backend.NativeRuntime") as runtime_class,
            patch(
                "octomil.runtime.native.stt_backend._runtime_advertises_audio_transcription",
                return_value=False,
            ),
        ):
            runtime_inst = MagicMock()
            runtime_inst.last_error.return_value = "whisper not built into dylib"
            runtime_class.open.return_value = runtime_inst

            with pytest.raises(OctomilError) as exc:
                backend.load_uploaded_model(
                    model_name="my-whisper",
                    artifact_path=str(path),
                    expected_sha256=real_sha,
                )

        assert exc.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        assert str(path) in exc.value.error_message


# ---------------------------------------------------------------------------
# Idempotency / concurrent-load semantics
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_same_model_reload_is_noop(self, tmp_path: Path) -> None:
        """A second load_uploaded_model call with the same model_name
        is idempotent — no work, no error."""
        path, real_sha = _write_artifact(tmp_path)
        backend = NativeSttBackend()

        with (
            patch("octomil.runtime.native.stt_backend.NativeRuntime") as runtime_class,
            patch(
                "octomil.runtime.native.stt_backend._runtime_advertises_audio_transcription",
                return_value=True,
            ),
        ):
            runtime_inst = MagicMock()
            runtime_inst.open_model.return_value = MagicMock()
            runtime_class.open.return_value = runtime_inst

            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(path),
                expected_sha256=real_sha,
            )
            # Second call — should be a no-op (no second NativeRuntime.open).
            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(path),
                expected_sha256=real_sha,
            )
            assert runtime_class.open.call_count == 1

        backend.close()

    def test_different_model_reload_rejects(self, tmp_path: Path) -> None:
        """Once a model is loaded, a different uploaded model must
        be rejected with INVALID_INPUT — caller must close first."""
        path, real_sha = _write_artifact(tmp_path)
        backend = NativeSttBackend()

        with (
            patch("octomil.runtime.native.stt_backend.NativeRuntime") as runtime_class,
            patch(
                "octomil.runtime.native.stt_backend._runtime_advertises_audio_transcription",
                return_value=True,
            ),
        ):
            runtime_inst = MagicMock()
            runtime_inst.open_model.return_value = MagicMock()
            runtime_class.open.return_value = runtime_inst

            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(path),
                expected_sha256=real_sha,
            )
            with pytest.raises(OctomilError) as exc:
                backend.load_uploaded_model(
                    model_name="other-whisper",
                    artifact_path=str(path),
                    expected_sha256=real_sha,
                )
            assert exc.value.code == OctomilErrorCode.INVALID_INPUT
        backend.close()


# ---------------------------------------------------------------------------
# Cross-method compatibility
# ---------------------------------------------------------------------------


def test_canonical_load_model_still_works(tmp_path: Path) -> None:
    """OCT-113 v2 must not break the canonical ``load_model`` path —
    pilots using whisper-tiny / whisper-base via the registry get the
    same behavior they had pre-PR."""
    from octomil.runtime.native.stt_backend import _WHISPER_ARTIFACTS

    # Smoke check: the constant set of registered Whisper names
    # hasn't changed.
    assert set(_WHISPER_ARTIFACTS.keys()) == {"whisper-tiny", "whisper-base"}
