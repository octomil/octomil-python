"""Tests for edgeml.ollama — ollama bridge and CLI integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from edgeml.cli import main
from edgeml.ollama import (
    OllamaModel,
    get_ollama_model,
    is_ollama_running,
    list_ollama_models,
    map_quantization,
    resolve_gguf_path,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_TAGS_RESPONSE = {
    "models": [
        {
            "name": "gemma:2b",
            "size": 1_678_000_000,
            "digest": "sha256:abc123def456",
            "modified_at": "2026-01-15T10:30:00Z",
            "details": {
                "family": "gemma",
                "quantization_level": "Q4_K_M",
                "parameter_size": "2B",
            },
        },
        {
            "name": "llama3.2:3b",
            "size": 2_048_000_000,
            "digest": "sha256:789abc012def",
            "modified_at": "2026-01-10T08:00:00Z",
            "details": {
                "family": "llama",
                "quantization_level": "Q4_K_M",
                "parameter_size": "3B",
            },
        },
        {
            "name": "phi-3:mini",
            "size": 2_350_000_000,
            "digest": "sha256:deadbeef1234",
            "modified_at": "2026-01-05T14:00:00Z",
            "details": {
                "family": "phi",
                "quantization_level": "Q4_0",
                "parameter_size": "3.8B",
            },
        },
    ]
}


def _make_model(**overrides) -> OllamaModel:
    defaults = {
        "name": "gemma:2b",
        "size": 1_678_000_000,
        "family": "gemma",
        "quantization": "Q4_K_M",
        "parameter_size": "2B",
        "modified_at": "2026-01-15T10:30:00Z",
        "digest": "sha256:abc123def456",
    }
    defaults.update(overrides)
    return OllamaModel(**defaults)


# ---------------------------------------------------------------------------
# is_ollama_running
# ---------------------------------------------------------------------------


class TestIsOllamaRunning:
    @patch("edgeml.ollama.httpx.get")
    def test_returns_true_when_reachable(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        assert is_ollama_running() is True
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=3.0)

    @patch("edgeml.ollama.httpx.get")
    def test_returns_false_on_connect_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        assert is_ollama_running() is False

    @patch("edgeml.ollama.httpx.get")
    def test_returns_false_on_timeout(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        assert is_ollama_running() is False

    @patch("edgeml.ollama.httpx.get")
    def test_returns_false_on_non_200(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp
        assert is_ollama_running() is False

    @patch("edgeml.ollama.httpx.get")
    def test_custom_base_url(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        assert is_ollama_running(base_url="http://remote:11434") is True
        mock_get.assert_called_once_with("http://remote:11434/api/tags", timeout=3.0)


# ---------------------------------------------------------------------------
# list_ollama_models
# ---------------------------------------------------------------------------


class TestListOllamaModels:
    @patch("edgeml.ollama.resolve_gguf_path", return_value=None)
    @patch("edgeml.ollama.httpx.get")
    def test_returns_models(self, mock_get, mock_resolve):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_TAGS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        models = list_ollama_models()
        assert len(models) == 3
        assert models[0].name == "gemma:2b"
        assert models[0].family == "gemma"
        assert models[0].quantization == "Q4_K_M"
        assert models[1].name == "llama3.2:3b"
        assert models[2].name == "phi-3:mini"

    @patch("edgeml.ollama.httpx.get")
    def test_returns_empty_on_connect_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        assert list_ollama_models() == []

    @patch("edgeml.ollama.httpx.get")
    def test_returns_empty_on_timeout(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        assert list_ollama_models() == []

    @patch("edgeml.ollama.httpx.get")
    def test_returns_empty_on_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
        mock_get.return_value = mock_resp
        assert list_ollama_models() == []


# ---------------------------------------------------------------------------
# get_ollama_model
# ---------------------------------------------------------------------------


class TestGetOllamaModel:
    @patch("edgeml.ollama.resolve_gguf_path", return_value=None)
    @patch("edgeml.ollama.httpx.get")
    def test_found_by_full_name(self, mock_get, mock_resolve):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_TAGS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        model = get_ollama_model("gemma:2b")
        assert model is not None
        assert model.name == "gemma:2b"

    @patch("edgeml.ollama.resolve_gguf_path", return_value=None)
    @patch("edgeml.ollama.httpx.get")
    def test_found_by_base_name(self, mock_get, mock_resolve):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_TAGS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        model = get_ollama_model("gemma")
        assert model is not None
        assert model.name == "gemma:2b"

    @patch("edgeml.ollama.resolve_gguf_path", return_value=None)
    @patch("edgeml.ollama.httpx.get")
    def test_not_found(self, mock_get, mock_resolve):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_TAGS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        model = get_ollama_model("nonexistent:7b")
        assert model is None


# ---------------------------------------------------------------------------
# resolve_gguf_path
# ---------------------------------------------------------------------------


class TestResolveGgufPath:
    def test_returns_path_when_blob_exists(self, tmp_path):
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()
        blob_file = blob_dir / "sha256-abc123def456"
        blob_file.write_bytes(b"fake gguf data")

        model = _make_model(digest="sha256:abc123def456")

        with patch("edgeml.ollama._ollama_models_dir", return_value=str(tmp_path)):
            result = resolve_gguf_path(model)

        assert result == str(blob_file)

    def test_returns_none_when_blob_missing(self, tmp_path):
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()

        model = _make_model(digest="sha256:doesnotexist")

        with patch("edgeml.ollama._ollama_models_dir", return_value=str(tmp_path)):
            result = resolve_gguf_path(model)

        assert result is None

    def test_returns_none_when_no_digest(self):
        model = _make_model(digest="")
        assert resolve_gguf_path(model) is None


# ---------------------------------------------------------------------------
# Quantization mapping
# ---------------------------------------------------------------------------


class TestQuantizationMapping:
    @pytest.mark.parametrize(
        "ollama_quant,expected",
        [
            ("Q4_K_M", "INT4"),
            ("Q4_0", "INT4"),
            ("Q5_K_S", "INT5"),
            ("Q5_K_M", "INT5"),
            ("Q8_0", "INT8"),
            ("F16", "FP16"),
            ("F32", "FP32"),
            ("Q2_K", "INT2"),
            ("Q3_K_M", "INT3"),
            ("Q6_K", "INT6"),
        ],
    )
    def test_known_mappings(self, ollama_quant, expected):
        assert map_quantization(ollama_quant) == expected

    def test_unknown_passes_through(self):
        assert map_quantization("WEIRD_QUANT") == "WEIRD_QUANT"

    def test_model_edgeml_quantization_property(self):
        model = _make_model(quantization="Q4_K_M")
        assert model.edgeml_quantization == "INT4"


# ---------------------------------------------------------------------------
# OllamaModel properties
# ---------------------------------------------------------------------------


class TestOllamaModelProperties:
    def test_size_display_gb(self):
        model = _make_model(size=1_678_000_000)
        assert model.size_display == "1.6 GB"

    def test_size_display_mb(self):
        model = _make_model(size=500_000_000)
        assert model.size_display == "477 MB"

    def test_size_display_small(self):
        model = _make_model(size=50_000_000)
        assert "MB" in model.size_display


# ---------------------------------------------------------------------------
# CLI: edgeml models
# ---------------------------------------------------------------------------


class TestModelsCommand:
    @patch("edgeml.ollama.is_ollama_running", return_value=True)
    @patch("edgeml.ollama.list_ollama_models")
    def test_models_ollama_only(self, mock_list, mock_running):
        mock_list.return_value = [
            _make_model(name="gemma:2b", size=1_678_000_000),
        ]
        runner = CliRunner()
        result = runner.invoke(main, ["models", "--source", "ollama"])
        assert result.exit_code == 0
        assert "Local (ollama):" in result.output
        assert "gemma:2b" in result.output
        assert "Q4_K_M" in result.output

    @patch("edgeml.ollama.is_ollama_running", return_value=False)
    def test_models_ollama_not_running(self, mock_running):
        runner = CliRunner()
        result = runner.invoke(main, ["models", "--source", "ollama"])
        assert result.exit_code == 0
        assert "not running" in result.output

    @patch("edgeml.ollama.is_ollama_running", return_value=True)
    @patch("edgeml.ollama.list_ollama_models", return_value=[])
    def test_models_ollama_no_models(self, mock_list, mock_running):
        runner = CliRunner()
        result = runner.invoke(main, ["models", "--source", "ollama"])
        assert result.exit_code == 0
        assert "no models found" in result.output

    def test_models_registry_no_key(self, monkeypatch):
        monkeypatch.delenv("EDGEML_API_KEY", raising=False)
        runner = CliRunner()
        result = runner.invoke(main, ["models", "--source", "registry"])
        assert result.exit_code == 0
        assert "no API key" in result.output


# ---------------------------------------------------------------------------
# CLI: edgeml deploy --phone with ollama detection
# ---------------------------------------------------------------------------


class TestDeployWithOllama:
    @patch("edgeml.cli.webbrowser.open")
    @patch("edgeml.ollama.get_ollama_model")
    def test_deploy_phone_detects_ollama(self, mock_get_model, mock_open, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")

        mock_get_model.return_value = _make_model(
            name="gemma:2b",
            size=1_678_000_000,
            quantization="Q4_K_M",
        )

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "XYZ789",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "gemma:2b", "--phone"])

        assert result.exit_code == 0
        assert "Detected ollama model: gemma:2b" in result.output
        assert "1.6 GB" in result.output
        assert "Q4_K_M" in result.output

    @patch("edgeml.cli.webbrowser.open")
    @patch("edgeml.ollama.get_ollama_model", return_value=None)
    def test_deploy_phone_no_ollama_match(self, mock_get_model, mock_open, monkeypatch):
        monkeypatch.setenv("EDGEML_API_KEY", "test-key")

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "ABC123",
            "expires_at": "2026-02-18T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "my-custom-model", "--phone"])

        assert result.exit_code == 0
        assert "Detected ollama model" not in result.output
