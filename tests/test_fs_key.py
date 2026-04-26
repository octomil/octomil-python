"""Tests for the shared filesystem-key helper."""

from __future__ import annotations

import pytest

from octomil.runtime.lifecycle._fs_key import (
    DEFAULT_MAX_VISIBLE_CHARS,
    safe_filesystem_key,
)
from octomil.runtime.lifecycle.file_lock import FileLock
from octomil.runtime.lifecycle.prepare_manager import PrepareManager

# --- Direct helper coverage ---------------------------------------------


def test_helper_caps_long_ascii_at_byte_count():
    key = safe_filesystem_key("a" * 5000)
    assert len(key.encode("utf-8")) <= DEFAULT_MAX_VISIBLE_CHARS + 13
    assert key.endswith("-" + key.rsplit("-", 1)[-1])


def test_helper_caps_long_non_ascii_emoji_at_byte_count():
    # Each emoji is 4 bytes UTF-8. A naive char-count cap would let
    # 96 emoji = 384 bytes through.
    key = safe_filesystem_key("😀" * 5000)
    assert len(key.encode("utf-8")) <= DEFAULT_MAX_VISIBLE_CHARS + 13
    # All emoji should be replaced; only ASCII allowlist + hash remain.
    assert all(ch.isascii() for ch in key)


@pytest.mark.parametrize(
    "name",
    [
        "model<x>.bin",
        "weights:colon",
        'name"with"quotes',
        "pipe|name",
        "ask?bin",
        "star*bin",
        "back\\slash",
    ],
)
def test_helper_strips_windows_reserved_characters(name):
    key = safe_filesystem_key(name)
    assert all(ch not in '<>:"/\\|?*' for ch in key)
    # Original input still distinguishes via the hash suffix.
    assert key != safe_filesystem_key(name + "-other")


def test_helper_is_deterministic_and_disambiguates():
    a = safe_filesystem_key("kokoro/v1")
    b = safe_filesystem_key("kokoro v1")
    assert a == safe_filesystem_key("kokoro/v1")
    assert a != b  # different hash suffixes


def test_helper_collapses_dot_only_inputs():
    assert safe_filesystem_key(".").startswith("id-")
    assert safe_filesystem_key("..").startswith("id-")
    assert safe_filesystem_key("").startswith("id-")


def test_helper_rejects_nul_byte():
    with pytest.raises(ValueError, match="NUL"):
        safe_filesystem_key("with\x00null")


# --- FileLock regression -------------------------------------------------


def test_filelock_long_emoji_id_stays_under_name_max(tmp_path):
    # Reviewer's reproducer: FileLock('😀' * 5000) used to produce a
    # 402-byte filename. With the shared helper it must fit NAME_MAX.
    lock = FileLock("😀" * 5000, lock_dir=tmp_path)
    name = lock.lock_path.name
    assert len(name.encode("utf-8")) <= 255, f"got {len(name.encode('utf-8'))} bytes"


def test_filelock_strips_windows_reserved_chars(tmp_path):
    lock = FileLock("name<with>:bad?chars", lock_dir=tmp_path)
    name = lock.lock_path.name
    for ch in '<>:"|?*':
        assert ch not in name


def test_filelock_distinct_inputs_produce_distinct_lock_files(tmp_path):
    a = FileLock("a/b", lock_dir=tmp_path).lock_path.name
    b = FileLock("a:b", lock_dir=tmp_path).lock_path.name
    assert a != b


# --- PrepareManager parity ----------------------------------------------


def test_prepare_manager_and_filelock_share_key_shape(tmp_path):
    mgr = PrepareManager(cache_dir=tmp_path / "cache")
    artifact_dir = mgr.artifact_dir_for("kokoro 😀 v1")
    lock = FileLock("kokoro 😀 v1", lock_dir=tmp_path / "locks")
    # The artifact dir name and the lock file's basename (minus .lock)
    # come from the same helper, so they match.
    assert lock.lock_path.stem == artifact_dir.name


def test_prepare_manager_emoji_id_lands_under_cache(tmp_path):
    mgr = PrepareManager(cache_dir=tmp_path)
    d = mgr.artifact_dir_for("😀" * 5000)
    artifacts_root = (tmp_path / "artifacts").resolve()
    assert d.is_relative_to(artifacts_root)
    assert len(d.name.encode("utf-8")) <= 255
