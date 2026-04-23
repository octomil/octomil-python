"""Runtime product policy tests.

Ollama remains an import/deploy source, but it is not an Octomil execution
engine. First-party routing should choose managed SDK runtimes such as MLX or
llama.cpp, never an Ollama server.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_default_engine_registry_does_not_register_ollama() -> None:
    from octomil.runtime.engines.registry import get_registry, reset_registry

    reset_registry()
    try:
        names = [engine.name for engine in get_registry().engines]
    finally:
        reset_registry()

    assert "ollama" not in names
    assert "llama.cpp" in names


def test_device_profile_does_not_advertise_ollama_as_installed_runtime() -> None:
    from octomil.runtime.planner.device_profile import _detect_installed_runtimes

    fake_registry = SimpleNamespace(
        detect_all=lambda: [
            SimpleNamespace(engine=SimpleNamespace(name="ollama"), available=True, info="on PATH"),
            SimpleNamespace(engine=SimpleNamespace(name="llama.cpp"), available=True, info="python binding"),
        ]
    )

    with patch("octomil.runtime.engines.get_registry", return_value=fake_registry):
        runtimes = _detect_installed_runtimes()

    assert [runtime.engine for runtime in runtimes] == ["llama.cpp"]


def test_lifecycle_detection_does_not_report_ollama(monkeypatch) -> None:
    from octomil.runtime.lifecycle.detection import detect_installed_runtimes

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/ollama" if name == "ollama" else None)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    runtimes = detect_installed_runtimes()

    assert all(runtime.engine_id != "ollama" for runtime in runtimes)
