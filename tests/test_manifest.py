"""Tests for octomil.manifest and octomil.manifest_validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from octomil.manifest import (
    AppManifest,
    AppModelEntry,
    AppRoutingPolicy,
    DeliveryMode,
)
from octomil.manifest_validator import validate_manifest, validate_manifest_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(**overrides: Any) -> AppModelEntry:
    defaults: dict[str, Any] = {
        "id": "test-model",
        "capability": "chat",
        "delivery": DeliveryMode.MANAGED,
    }
    defaults.update(overrides)
    return AppModelEntry(**defaults)


def _manifest(entries: list[AppModelEntry] | None = None) -> AppManifest:
    return AppManifest(models=entries or [_entry()])


# ---------------------------------------------------------------------------
# DeliveryMode / AppRoutingPolicy enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_delivery_mode_values(self):
        assert DeliveryMode.BUNDLED.value == "bundled"
        assert DeliveryMode.MANAGED.value == "managed"
        assert DeliveryMode.CLOUD.value == "cloud"

    def test_routing_policy_values(self):
        assert AppRoutingPolicy.LOCAL_ONLY.value == "local_only"
        assert AppRoutingPolicy.LOCAL_FIRST.value == "local_first"
        assert AppRoutingPolicy.CLOUD_ONLY.value == "cloud_only"


# ---------------------------------------------------------------------------
# AppModelEntry
# ---------------------------------------------------------------------------


class TestAppModelEntry:
    def test_to_dict_minimal(self):
        entry = _entry()
        d = entry.to_dict()
        assert d == {"id": "test-model", "capability": "chat", "delivery": "managed"}
        assert "routing_policy" not in d
        assert "bundled_path" not in d
        assert "required" not in d

    def test_to_dict_with_optional_fields(self):
        entry = _entry(
            routing_policy=AppRoutingPolicy.LOCAL_ONLY,
            bundled_path="model.gguf",
            required=True,
        )
        d = entry.to_dict()
        assert d["routing_policy"] == "local_only"
        assert d["bundled_path"] == "model.gguf"
        assert d["required"] is True

    def test_to_dict_required_false_omitted(self):
        entry = _entry(required=False)
        assert "required" not in entry.to_dict()

    def test_from_dict_minimal(self):
        d = {"id": "m1", "capability": "chat", "delivery": "managed"}
        entry = AppModelEntry.from_dict(d)
        assert entry.id == "m1"
        assert entry.capability == "chat"
        assert entry.delivery == DeliveryMode.MANAGED
        assert entry.routing_policy is None
        assert entry.bundled_path is None
        assert entry.required is False

    def test_from_dict_full(self):
        d = {
            "id": "m2",
            "capability": "transcription",
            "delivery": "bundled",
            "routing_policy": "local_only",
            "bundled_path": "whisper.gguf",
            "required": True,
        }
        entry = AppModelEntry.from_dict(d)
        assert entry.delivery == DeliveryMode.BUNDLED
        assert entry.routing_policy == AppRoutingPolicy.LOCAL_ONLY
        assert entry.bundled_path == "whisper.gguf"
        assert entry.required is True

    def test_effective_routing_policy_explicit(self):
        entry = _entry(routing_policy=AppRoutingPolicy.CLOUD_ONLY)
        assert entry.effective_routing_policy == AppRoutingPolicy.CLOUD_ONLY

    def test_effective_routing_policy_bundled_default(self):
        entry = _entry(delivery=DeliveryMode.BUNDLED)
        assert entry.effective_routing_policy == AppRoutingPolicy.LOCAL_FIRST

    def test_effective_routing_policy_managed_default(self):
        entry = _entry(delivery=DeliveryMode.MANAGED)
        assert entry.effective_routing_policy == AppRoutingPolicy.LOCAL_FIRST

    def test_effective_routing_policy_cloud_default(self):
        entry = _entry(delivery=DeliveryMode.CLOUD)
        assert entry.effective_routing_policy == AppRoutingPolicy.CLOUD_ONLY


# ---------------------------------------------------------------------------
# AppManifest — serialization
# ---------------------------------------------------------------------------


class TestAppManifestSerialization:
    def test_to_dict(self):
        m = _manifest()
        d = m.to_dict()
        assert d["version"] == 1
        assert len(d["models"]) == 1
        assert d["models"][0]["id"] == "test-model"

    def test_to_json(self):
        m = _manifest()
        raw = m.to_json()
        parsed = json.loads(raw)
        assert parsed["version"] == 1
        assert parsed["models"][0]["capability"] == "chat"

    def test_to_json_indent(self):
        m = _manifest()
        raw = m.to_json(indent=4)
        # Should have 4-space indentation
        assert "    " in raw

    def test_to_yaml_returns_string(self):
        m = _manifest()
        yaml_str = m.to_yaml()
        assert isinstance(yaml_str, str)
        assert "version" in yaml_str
        assert "chat" in yaml_str

    def test_from_dict(self):
        data = {
            "version": 1,
            "models": [
                {"id": "a", "capability": "chat", "delivery": "managed"},
                {"id": "b", "capability": "transcription", "delivery": "cloud"},
            ],
        }
        m = AppManifest.from_dict(data)
        assert m.version == 1
        assert len(m.models) == 2
        assert m.models[1].delivery == DeliveryMode.CLOUD

    def test_from_dict_defaults(self):
        m = AppManifest.from_dict({})
        assert m.version == 1
        assert m.models == []


# ---------------------------------------------------------------------------
# AppManifest — YAML round-trip
# ---------------------------------------------------------------------------


class TestAppManifestYamlRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        original = AppManifest(
            models=[
                AppModelEntry(
                    id="smolvlm2-500m",
                    capability="chat",
                    delivery=DeliveryMode.MANAGED,
                ),
                AppModelEntry(
                    id="whisper-tiny",
                    capability="transcription",
                    delivery=DeliveryMode.BUNDLED,
                    bundled_path="whisper-tiny.gguf",
                    required=True,
                ),
            ],
        )

        yaml_path = tmp_path / "octomil.yaml"
        yaml_path.write_text(original.to_yaml(), encoding="utf-8")
        restored = AppManifest.from_yaml(yaml_path)

        assert restored.version == original.version
        assert len(restored.models) == len(original.models)
        for orig, rest in zip(original.models, restored.models):
            assert orig.id == rest.id
            assert orig.capability == rest.capability
            assert orig.delivery == rest.delivery
            assert orig.bundled_path == rest.bundled_path
            assert orig.required == rest.required

    def test_from_yaml_bad_file(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("[not a mapping]", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            AppManifest.from_yaml(bad)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_manifest(self):
        m = _manifest()
        assert m.validate() == []

    def test_bundled_missing_path(self):
        entry = _entry(delivery=DeliveryMode.BUNDLED, bundled_path=None)
        m = _manifest([entry])
        errors = m.validate()
        assert len(errors) == 1
        assert "bundled_path" in errors[0]

    def test_bundled_with_path_ok(self):
        entry = _entry(delivery=DeliveryMode.BUNDLED, bundled_path="model.gguf")
        m = _manifest([entry])
        assert m.validate() == []

    def test_unknown_capability(self):
        entry = _entry(capability="nonexistent_cap")
        m = _manifest([entry])
        errors = m.validate()
        assert len(errors) == 1
        assert "unknown capability" in errors[0]

    def test_duplicate_capability(self):
        e1 = _entry(id="a", capability="chat")
        e2 = _entry(id="b", capability="chat")
        m = _manifest([e1, e2])
        errors = m.validate()
        assert any("duplicate capability" in e for e in errors)

    def test_multiple_errors(self):
        e1 = _entry(id="a", capability="fake", delivery=DeliveryMode.BUNDLED)
        e2 = _entry(id="b", capability="fake")
        m = _manifest([e1, e2])
        errors = m.validate()
        # At least: unknown cap for e1, missing bundled_path for e1, unknown cap for e2, dup for e2
        assert len(errors) >= 3


# ---------------------------------------------------------------------------
# validate_manifest / validate_manifest_file
# ---------------------------------------------------------------------------


class TestValidatorModule:
    def test_validate_manifest_delegates(self):
        m = _manifest()
        assert validate_manifest(m) == []

    def test_validate_manifest_file_valid(self, tmp_path: Path):
        m = _manifest()
        p = tmp_path / "ok.yaml"
        p.write_text(m.to_yaml(), encoding="utf-8")
        assert validate_manifest_file(p) == []

    def test_validate_manifest_file_not_found(self, tmp_path: Path):
        p = tmp_path / "missing.yaml"
        errors = validate_manifest_file(p)
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_validate_manifest_file_invalid_yaml(self, tmp_path: Path):
        p = tmp_path / "bad.yaml"
        p.write_text("{{invalid yaml", encoding="utf-8")
        errors = validate_manifest_file(p)
        assert len(errors) == 1
        assert "parse" in errors[0].lower() or "failed" in errors[0].lower()

    def test_validate_manifest_file_catches_errors(self, tmp_path: Path):
        data = "version: 1\nmodels:\n  - id: x\n    capability: bogus\n    delivery: managed\n"
        p = tmp_path / "warn.yaml"
        p.write_text(data, encoding="utf-8")
        errors = validate_manifest_file(p)
        assert len(errors) >= 1
        assert "unknown capability" in errors[0]


# ---------------------------------------------------------------------------
# CLI commands (smoke tests via Click testing)
# ---------------------------------------------------------------------------


class TestManifestCLI:
    def test_manifest_init(self, tmp_path: Path):
        from click.testing import CliRunner

        from octomil.commands.manifest_cmd import manifest

        runner = CliRunner()
        out = tmp_path / "octomil.yaml"
        result = runner.invoke(manifest, ["init", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        text = out.read_text()
        assert "version: 1" in text
        assert "capability: chat" in text

    def test_manifest_init_no_overwrite(self, tmp_path: Path):
        from click.testing import CliRunner

        from octomil.commands.manifest_cmd import manifest

        runner = CliRunner()
        out = tmp_path / "octomil.yaml"
        out.write_text("existing", encoding="utf-8")
        result = runner.invoke(manifest, ["init", "-o", str(out)])
        assert result.exit_code != 0
        assert out.read_text() == "existing"

    def test_manifest_init_force(self, tmp_path: Path):
        from click.testing import CliRunner

        from octomil.commands.manifest_cmd import manifest

        runner = CliRunner()
        out = tmp_path / "octomil.yaml"
        out.write_text("existing", encoding="utf-8")
        result = runner.invoke(manifest, ["init", "-o", str(out), "--force"])
        assert result.exit_code == 0
        assert "version: 1" in out.read_text()

    def test_manifest_validate_valid(self, tmp_path: Path):
        from click.testing import CliRunner

        from octomil.commands.manifest_cmd import manifest

        runner = CliRunner()
        m = _manifest()
        p = tmp_path / "ok.yaml"
        p.write_text(m.to_yaml(), encoding="utf-8")
        result = runner.invoke(manifest, ["validate", str(p)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_manifest_validate_invalid(self, tmp_path: Path):
        from click.testing import CliRunner

        from octomil.commands.manifest_cmd import manifest

        runner = CliRunner()
        data = "version: 1\nmodels:\n  - id: x\n    capability: bogus\n    delivery: managed\n"
        p = tmp_path / "bad.yaml"
        p.write_text(data, encoding="utf-8")
        result = runner.invoke(manifest, ["validate", str(p)])
        assert result.exit_code != 0
        assert "error" in result.output.lower()
