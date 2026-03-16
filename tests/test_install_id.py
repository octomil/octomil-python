"""Tests for octomil.install_id — persistent install identifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from octomil.install_id import get_install_id, reset_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the module-level cache before and after each test."""
    reset_cache()
    yield
    reset_cache()


class TestGetInstallId:
    def test_generates_uuid_hex(self, tmp_path: Path) -> None:
        install_id = get_install_id(install_dir=tmp_path)
        assert len(install_id) == 32
        # Should be valid hex
        int(install_id, 16)

    def test_persists_to_file(self, tmp_path: Path) -> None:
        install_id = get_install_id(install_dir=tmp_path)
        file_path = tmp_path / "install_id"
        assert file_path.exists()
        assert file_path.read_text().strip() == install_id

    def test_reads_existing_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "install_id"
        file_path.write_text("existing_id_12345\n")
        install_id = get_install_id(install_dir=tmp_path)
        assert install_id == "existing_id_12345"

    def test_stable_across_calls(self, tmp_path: Path) -> None:
        first = get_install_id(install_dir=tmp_path)
        # Clear cache to force re-read from file
        reset_cache()
        second = get_install_id(install_dir=tmp_path)
        assert first == second

    def test_cache_avoids_repeated_reads(self, tmp_path: Path) -> None:
        first = get_install_id(install_dir=tmp_path)
        # Overwrite file — cached value should still be returned
        (tmp_path / "install_id").write_text("overwritten\n")
        second = get_install_id(install_dir=tmp_path)
        assert first == second

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "nested" / "octomil"
        install_id = get_install_id(install_dir=nested_dir)
        assert (nested_dir / "install_id").exists()
        assert len(install_id) == 32

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "install_id"
        file_path.write_text("")
        install_id = get_install_id(install_dir=tmp_path)
        # Should generate a new one since file was empty
        assert len(install_id) == 32
        int(install_id, 16)

    def test_handles_unreadable_directory(self, tmp_path: Path) -> None:
        # Use a path that can't be created (nested under a file)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        install_id = get_install_id(install_dir=blocker)
        # Should still return a valid ID even if persistence fails
        assert len(install_id) == 32


class TestResetCache:
    def test_reset_forces_reread(self, tmp_path: Path) -> None:
        first = get_install_id(install_dir=tmp_path)
        reset_cache()
        # Overwrite file
        (tmp_path / "install_id").write_text("new_value\n")
        second = get_install_id(install_dir=tmp_path)
        assert second == "new_value"
        assert first != second
