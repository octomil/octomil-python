"""Tests for octomil.models_namespace — SDK Facade Contract models API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from octomil.auth import OrgApiKeyAuth
from octomil.models_namespace import ModelStatus, OctomilModels

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client():
    """Return a minimal mock OctomilClient with the attributes OctomilModels needs."""
    client = MagicMock()
    client._models = {}
    return client


@pytest.fixture()
def ns(mock_client):
    """Return an OctomilModels namespace backed by the mock client."""
    return OctomilModels(mock_client)


# ---------------------------------------------------------------------------
# ModelStatus enum
# ---------------------------------------------------------------------------


class TestModelStatusEnum:
    def test_values(self):
        assert ModelStatus.NOT_CACHED == "not_cached"
        assert ModelStatus.DOWNLOADING == "downloading"
        assert ModelStatus.READY == "ready"
        assert ModelStatus.ERROR == "error"

    def test_is_str_subclass(self):
        assert isinstance(ModelStatus.READY, str)


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


class TestStatus:
    def test_unknown_model_returns_not_cached(self, ns):
        assert ns.status("nonexistent-model") == ModelStatus.NOT_CACHED

    def test_downloading_state(self, ns):
        ns._downloading.add("my-model")
        assert ns.status("my-model") == ModelStatus.DOWNLOADING

    def test_error_state(self, ns):
        ns._errors["my-model"] = "connection failed"
        assert ns.status("my-model") == ModelStatus.ERROR

    def test_downloading_takes_priority_over_error(self, ns):
        ns._downloading.add("my-model")
        ns._errors["my-model"] = "old error"
        assert ns.status("my-model") == ModelStatus.DOWNLOADING

    def test_loaded_in_memory_returns_ready(self, ns, mock_client):
        mock_client._models["phi-4-mini"] = MagicMock()
        assert ns.status("phi-4-mini") == ModelStatus.READY

    def test_octomil_cache_dir_returns_ready(self, ns, tmp_path):
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            assert ns.status("test-model") == ModelStatus.READY

    def test_octomil_cache_file_returns_ready(self, ns, tmp_path):
        model_file = tmp_path / "test-model.gguf"
        model_file.write_bytes(b"fake-model")
        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            assert ns.status("test-model") == ModelStatus.READY

    def test_hf_repo_id_slash_name_cache_check(self, ns, tmp_path):
        # "org/my-model" should also check "my-model" in ~/.octomil/models/
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            assert ns.status("org/my-model") == ModelStatus.READY

    def test_hf_cache_returns_ready(self, ns):
        mock_repo = MagicMock()
        mock_repo.repo_id = "mlx-community/phi-4-mini-4bit"
        mock_cache = MagicMock()
        mock_cache.repos = [mock_repo]

        with patch("octomil.models_namespace.OctomilModels._check_hf_cache") as mock_check:
            mock_check.return_value = True
            assert ns.status("mlx-community/phi-4-mini-4bit") == ModelStatus.READY

    def test_ollama_cache_returns_ready(self, ns, tmp_path):
        # Create fake Ollama manifest
        manifest_dir = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "gemma3" / "4b"
        manifest_dir.parent.mkdir(parents=True)
        manifest_dir.write_text('{"layers": []}')

        with patch("octomil.models_namespace.OctomilModels._check_ollama_cache") as mock_check:
            mock_check.return_value = True
            assert ns.status("gemma3:4b") == ModelStatus.READY


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestLoad:
    def test_delegates_to_client_load_model(self, ns, mock_client):
        mock_model = MagicMock()
        mock_client.load_model.return_value = mock_model

        result = ns.load("phi-4-mini", version="1.0.0")

        mock_client.load_model.assert_called_once_with("phi-4-mini", version="1.0.0")
        assert result is mock_model

    def test_tracks_downloading_state(self, ns, mock_client):
        # load_model blocks, so we check that _downloading is populated
        # before and cleared after
        states_during_load = []

        def side_effect(name, **kwargs):
            states_during_load.append(name in ns._downloading)
            return MagicMock()

        mock_client.load_model.side_effect = side_effect
        ns.load("my-model")
        assert states_during_load == [True]
        assert "my-model" not in ns._downloading

    def test_records_error_on_failure(self, ns, mock_client):
        mock_client.load_model.side_effect = RuntimeError("download failed")

        with pytest.raises(RuntimeError, match="download failed"):
            ns.load("bad-model")

        assert "bad-model" in ns._errors
        assert ns._errors["bad-model"] == "download failed"
        assert "bad-model" not in ns._downloading

    def test_clears_previous_error_on_retry(self, ns, mock_client):
        ns._errors["retry-model"] = "old failure"
        mock_client.load_model.return_value = MagicMock()

        ns.load("retry-model")

        assert "retry-model" not in ns._errors


# ---------------------------------------------------------------------------
# unload()
# ---------------------------------------------------------------------------


class TestUnload:
    def test_removes_from_client_models(self, ns, mock_client):
        mock_client._models["my-model"] = MagicMock()
        ns.unload("my-model")
        assert "my-model" not in mock_client._models

    def test_noop_for_unknown_model(self, ns, mock_client):
        # Should not raise
        ns.unload("not-loaded")


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


class TestList:
    def test_returns_loaded_models(self, ns, mock_client):
        model = MagicMock()
        model.metadata.version = "2.0.0"
        mock_client._models = {"phi-4-mini": model}

        result = ns.list()
        assert len(result) == 1
        assert result[0]["model_id"] == "phi-4-mini"
        assert result[0]["status"] == "loaded"
        assert result[0]["version"] == "2.0.0"

    def test_returns_cached_models_on_disk(self, ns, mock_client, tmp_path):
        model_dir = tmp_path / "gemma-3b"
        model_dir.mkdir()

        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            result = ns.list()

        assert any(m["model_id"] == "gemma-3b" for m in result)
        cached = [m for m in result if m["model_id"] == "gemma-3b"][0]
        assert cached["status"] == "cached"

    def test_no_duplicates_between_loaded_and_cached(self, ns, mock_client, tmp_path):
        model = MagicMock()
        model.metadata.version = "1.0.0"
        mock_client._models = {"gemma-3b": model}
        # Also exists on disk
        model_dir = tmp_path / "gemma-3b"
        model_dir.mkdir()

        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            result = ns.list()

        names = [m["model_id"] for m in result]
        assert names.count("gemma-3b") == 1
        # The loaded version takes priority
        assert result[0]["status"] == "loaded"

    def test_empty_when_nothing_cached(self, ns, mock_client, tmp_path):
        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            result = ns.list()
        assert result == []


# ---------------------------------------------------------------------------
# clear_cache()
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_clears_in_memory_models(self, ns, mock_client):
        mock_client._models["loaded"] = MagicMock()
        ns._errors["broken"] = "fail"
        ns.clear_cache()
        assert len(mock_client._models) == 0
        assert len(ns._errors) == 0

    def test_removes_files_from_cache_dir(self, ns, mock_client, tmp_path):
        model_dir = tmp_path / "some-model"
        model_dir.mkdir()
        model_file = tmp_path / "other.gguf"
        model_file.write_bytes(b"fake")

        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            ns.clear_cache()

        assert not model_dir.exists()
        assert not model_file.exists()


# ---------------------------------------------------------------------------
# Internal cache probes (unit tests for edge cases)
# ---------------------------------------------------------------------------


class TestCacheProbes:
    def test_check_octomil_cache_dir(self, tmp_path):
        (tmp_path / "my-model").mkdir()
        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            assert OctomilModels._check_octomil_cache("my-model") is True
            assert OctomilModels._check_octomil_cache("other") is False

    def test_check_octomil_cache_file_with_extension(self, tmp_path):
        (tmp_path / "test-model.onnx").write_bytes(b"data")
        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            assert OctomilModels._check_octomil_cache("test-model") is True

    def test_check_octomil_cache_hf_repo_slash(self, tmp_path):
        (tmp_path / "my-model").mkdir()
        with patch("octomil.models_namespace._OCTOMIL_MODELS_DIR", tmp_path):
            assert OctomilModels._check_octomil_cache("org/my-model") is True

    def test_check_ollama_cache(self, tmp_path):
        manifest_path = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "gemma3" / "latest"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text("{}")

        with patch("os.path.expanduser", return_value=str(tmp_path)):
            assert OctomilModels._check_ollama_cache("gemma3") is True
            assert OctomilModels._check_ollama_cache("nonexistent") is False

    def test_check_ollama_cache_with_tag(self, tmp_path):
        manifest_path = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "qwen2" / "7b"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text("{}")

        with patch("os.path.expanduser", return_value=str(tmp_path)):
            assert OctomilModels._check_ollama_cache("qwen2:7b") is True
            assert OctomilModels._check_ollama_cache("qwen2:3b") is False


# ---------------------------------------------------------------------------
# OctomilClient.models property integration
# ---------------------------------------------------------------------------


class TestClientModelsProperty:
    @patch("octomil.client.RolloutsAPI", create=True)
    @patch("octomil.client.ModelRegistry", create=True)
    @patch("octomil.client._ApiClient", create=True)
    def test_models_property_returns_namespace(self, mock_api, mock_reg, mock_roll):
        from octomil.client import OctomilClient

        client = OctomilClient(auth=OrgApiKeyAuth(api_key="test", org_id="default"))
        ns = client.models
        assert isinstance(ns, OctomilModels)

    @patch("octomil.client.RolloutsAPI", create=True)
    @patch("octomil.client.ModelRegistry", create=True)
    @patch("octomil.client._ApiClient", create=True)
    def test_models_property_is_cached(self, mock_api, mock_reg, mock_roll):
        from octomil.client import OctomilClient

        client = OctomilClient(auth=OrgApiKeyAuth(api_key="test", org_id="default"))
        ns1 = client.models
        ns2 = client.models
        assert ns1 is ns2
