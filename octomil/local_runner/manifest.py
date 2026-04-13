"""Runner manifest -- tracks the state of the invisible local runner process."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_DEFAULT_MANIFEST_PATH = Path.home() / ".cache" / "octomil" / "local-runner.json"
_DEFAULT_TOKEN_PATH = Path.home() / ".cache" / "octomil" / "local-runner.token"


@dataclass
class RunnerManifest:
    """Persisted manifest describing a running local runner process."""

    pid: int
    port: int
    base_url: str
    token_file: str
    model: str
    engine: str
    artifact_digest: str = ""
    capability: str = "responses"
    started_at: float = 0.0
    last_used_at: float = 0.0
    idle_timeout_seconds: int = 1800
    octomil_version: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> RunnerManifest | None:
        """Load a manifest from disk. Returns None if missing or corrupt."""
        p = path or _DEFAULT_MANIFEST_PATH
        try:
            data = json.loads(p.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None

    def save(self, path: Path | None = None) -> None:
        """Persist the manifest to disk."""
        p = path or _DEFAULT_MANIFEST_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def remove(path: Path | None = None) -> None:
        """Remove the manifest file."""
        p = path or _DEFAULT_MANIFEST_PATH
        p.unlink(missing_ok=True)
