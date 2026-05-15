"""Tests for OCT-113 v2 review2 findings (octomil-python).

Two P1 fixes:

P1 #2 — idempotency keyed on model_name only. A second
``load_uploaded_model("phi-mini", new_path, new_sha)`` returned
immediately even though the artifact changed, silently keeping the
SDK on old bytes.
Fix: idempotency keys on the full ``(model_name, artifact_path,
expected_sha256)`` triple. Same triple → no-op. Same name +
different path/sha → INVALID_INPUT.

P1 #3 — env mutation without restore. ``load_uploaded_model`` set
``OCT_WHISPER_ALLOW_USER_ARTIFACTS=1`` and
``OCTOMIL_WHISPER_BIN=<user_path>`` globally and ``close()`` only
released runtime handles. Subsequent canonical loads in the same
process inherited the user-upload env.
Fix: snapshot prior env on first set, restore in ``close()`` and
on every failure path.
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


def _write_artifact(tmp_path: Path, name: str = "user-whisper.gguf", content: bytes = b"GGUF") -> tuple[Path, str]:
    p = tmp_path / name
    p.write_bytes(content + b"\x00" * 1024)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return p, sha


# ---------------------------------------------------------------------------
# P1 #2 — idempotency keyed on the full triple
# ---------------------------------------------------------------------------


class TestIdempotencyTriple:
    def test_same_triple_is_idempotent(self, tmp_path: Path) -> None:
        """Reloading the same (name, path, sha) is a no-op — the
        backend already has those bytes warmed."""
        path, sha = _write_artifact(tmp_path)
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
                expected_sha256=sha,
            )
            # Same triple — no second NativeRuntime.open call.
            backend.load_uploaded_model(
                model_name="my-whisper",
                artifact_path=str(path),
                expected_sha256=sha,
            )
            assert runtime_class.open.call_count == 1
        backend.close()

    def test_same_name_new_path_rejects(self, tmp_path: Path) -> None:
        """The bug OCT-113 v2 review2 P1 #2 caught: same slug but a
        new path was silently a no-op. Now we refuse — caller must
        close() before re-loading with a new artifact."""
        path1, sha1 = _write_artifact(tmp_path, "v1.gguf")
        path2, sha2 = _write_artifact(tmp_path, "v2.gguf", content=b"GGUF2")
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
                artifact_path=str(path1),
                expected_sha256=sha1,
            )
            # Same slug, different path → reject.
            with pytest.raises(OctomilError) as exc:
                backend.load_uploaded_model(
                    model_name="my-whisper",
                    artifact_path=str(path2),
                    expected_sha256=sha2,
                )
            assert exc.value.code == OctomilErrorCode.INVALID_INPUT
            assert (
                "different artifact_path" in exc.value.error_message or "different" in exc.value.error_message.lower()
            )
        backend.close()

    def test_same_name_new_sha_rejects(self, tmp_path: Path) -> None:
        """Same slug + same path but the caller passes a different
        expected_sha — also a NEW load, not a no-op. The catalog may
        have updated the canonical checksum and the SDK must not
        silently keep serving old bytes."""
        path, sha = _write_artifact(tmp_path)
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
                expected_sha256=sha,
            )

            # Same slug, same path — but caller now says expected =
            # a DIFFERENT sha (e.g. catalog updated). We must refuse
            # to no-op. The new SHA won't match the on-disk file, so
            # this naturally raises CHECKSUM_MISMATCH (the integrity
            # check fires before idempotency on a fresh load), but
            # the IMPORTANT thing is we do NOT silently return. Pin
            # either INVALID_INPUT (idempotency check fires first)
            # or CHECKSUM_MISMATCH (integrity check fires) — but
            # never "no-op + serve old bytes".
            with pytest.raises(OctomilError) as exc:
                backend.load_uploaded_model(
                    model_name="my-whisper",
                    artifact_path=str(path),
                    expected_sha256="f" * 64,  # different
                )
            assert exc.value.code in (
                OctomilErrorCode.INVALID_INPUT,
                OctomilErrorCode.CHECKSUM_MISMATCH,
            )
        backend.close()


# ---------------------------------------------------------------------------
# P1 #3 — env restore on close + failure paths
# ---------------------------------------------------------------------------


class TestEnvRestore:
    def test_close_restores_unset_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If env var was UNSET pre-load, close() must DELETE the var
        rather than leave it at the user's path."""
        monkeypatch.delenv(_WHISPER_ALLOW_USER_ARTIFACTS_ENV, raising=False)
        monkeypatch.delenv(_WHISPER_BIN_ENV, raising=False)

        path, sha = _write_artifact(tmp_path)
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
                expected_sha256=sha,
            )
            # During the load, env is set.
            assert os.environ.get(_WHISPER_ALLOW_USER_ARTIFACTS_ENV) == "1"
            assert os.environ.get(_WHISPER_BIN_ENV) == str(path)

            backend.close()

            # After close, both env vars are back to unset.
            assert _WHISPER_ALLOW_USER_ARTIFACTS_ENV not in os.environ
            assert _WHISPER_BIN_ENV not in os.environ

    def test_close_restores_prior_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If env vars were set to operator values pre-load, close()
        must restore those exact values, not "" or our user path."""
        monkeypatch.setenv(_WHISPER_BIN_ENV, "/operator/canonical.bin")
        monkeypatch.setenv(_WHISPER_ALLOW_USER_ARTIFACTS_ENV, "0")

        path, sha = _write_artifact(tmp_path)
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
                expected_sha256=sha,
            )
            backend.close()

        # Operator's pre-load values restored verbatim.
        assert os.environ.get(_WHISPER_BIN_ENV) == "/operator/canonical.bin"
        assert os.environ.get(_WHISPER_ALLOW_USER_ARTIFACTS_ENV) == "0"

    def test_runtime_open_failure_restores_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If NativeRuntime.open raises, the env we just set must be
        restored before we raise — caller doesn't know to call close()
        when load_uploaded_model itself blew up partway."""
        monkeypatch.setenv(_WHISPER_BIN_ENV, "/operator/canonical.bin")

        path, sha = _write_artifact(tmp_path)
        backend = NativeSttBackend()

        with patch("octomil.runtime.native.stt_backend.NativeRuntime") as runtime_class:
            runtime_class.open.side_effect = ImportError("dylib missing")

            with pytest.raises(OctomilError) as exc:
                backend.load_uploaded_model(
                    model_name="my-whisper",
                    artifact_path=str(path),
                    expected_sha256=sha,
                )
            assert exc.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

        # Even though load failed, env is restored to pre-load.
        assert os.environ.get(_WHISPER_BIN_ENV) == "/operator/canonical.bin"

    def test_capability_advertise_failure_restores_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the runtime doesn't advertise the capability (the
        common runtime-gap path today), we call self.close() which
        must restore env before the diagnostic raises."""
        monkeypatch.setenv(_WHISPER_BIN_ENV, "/operator/canonical.bin")

        path, sha = _write_artifact(tmp_path)
        backend = NativeSttBackend()

        with (
            patch("octomil.runtime.native.stt_backend.NativeRuntime") as runtime_class,
            patch(
                "octomil.runtime.native.stt_backend._runtime_advertises_audio_transcription",
                return_value=False,
            ),
        ):
            runtime_inst = MagicMock()
            runtime_inst.last_error.return_value = "digest mismatch"
            runtime_class.open.return_value = runtime_inst

            with pytest.raises(OctomilError) as exc:
                backend.load_uploaded_model(
                    model_name="my-whisper",
                    artifact_path=str(path),
                    expected_sha256=sha,
                )
            assert exc.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

        assert os.environ.get(_WHISPER_BIN_ENV) == "/operator/canonical.bin"

    def test_close_is_idempotent_on_unloaded_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A backend that never loaded anything can still be closed.
        The env-restore path must no-op rather than crash."""
        monkeypatch.setenv(_WHISPER_BIN_ENV, "/operator/canonical.bin")

        backend = NativeSttBackend()
        backend.close()  # should not raise

        # Env untouched (we never set it).
        assert os.environ.get(_WHISPER_BIN_ENV) == "/operator/canonical.bin"


# ---------------------------------------------------------------------------
# Cross-instance isolation
# ---------------------------------------------------------------------------


def test_second_instance_inherits_no_user_upload_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One backend loads + closes a user upload. A second backend
    in the same process must see clean env — no leftover
    OCTOMIL_WHISPER_BIN pointing at the first instance's user
    artifact."""
    monkeypatch.delenv(_WHISPER_BIN_ENV, raising=False)

    path, sha = _write_artifact(tmp_path)
    backend1 = NativeSttBackend()

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

        backend1.load_uploaded_model(
            model_name="m1",
            artifact_path=str(path),
            expected_sha256=sha,
        )
        backend1.close()

    # Second instance constructed AFTER close — should see clean env.
    backend2 = NativeSttBackend()  # noqa: F841
    assert _WHISPER_BIN_ENV not in os.environ
