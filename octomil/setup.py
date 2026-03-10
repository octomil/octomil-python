"""Managed venv + engine setup for the PyInstaller binary.

After ``curl -fsSL https://get.octomil.com | sh``, a background process
calls ``octomil setup --foreground`` which:

1. Finds a system Python with venv support.
2. Creates ``~/.octomil/engines/venv/``.
3. Installs ``octomil-sdk[mlx,serve]`` (Apple Silicon) or
   ``octomil-sdk[llama,serve]`` (elsewhere).
4. Downloads the recommended model via huggingface_hub.

When ``octomil serve`` runs from the frozen binary and finds no native
engines, it calls ``get_venv_python()`` + ``is_engine_ready()`` and
re-execs into the venv's Python via ``os.execv()``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OCTOMIL_DIR = Path.home() / ".octomil"
ENGINES_DIR = OCTOMIL_DIR / "engines"
VENV_DIR = ENGINES_DIR / "venv"
STATE_FILE = OCTOMIL_DIR / "setup_state.json"
SETUP_LOG = OCTOMIL_DIR / "setup.log"

# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

PHASE_PENDING = "pending"
PHASE_CREATING_VENV = "creating_venv"
PHASE_INSTALLING_ENGINE = "installing_engine"
PHASE_DOWNLOADING_MODEL = "downloading_model"
PHASE_COMPLETE = "complete"
PHASE_FAILED = "failed"


@dataclass
class SetupState:
    """Persistent state for the setup pipeline."""

    phase: str = PHASE_PENDING
    engine: Optional[str] = None
    package: Optional[str] = None
    engine_installed: bool = False
    model_key: Optional[str] = None
    model_downloaded: bool = False
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


def load_state() -> SetupState:
    """Load setup state from disk."""
    try:
        data = json.loads(STATE_FILE.read_text())
        return SetupState(**{k: v for k, v in data.items() if k in SetupState.__dataclass_fields__})
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return SetupState()


def _save_state(state: SetupState) -> None:
    """Persist setup state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(state), indent=2) + "\n")


# ---------------------------------------------------------------------------
# Public queries
# ---------------------------------------------------------------------------


def get_venv_python() -> Optional[str]:
    """Return path to the managed venv's Python, or None."""
    p = VENV_DIR / "bin" / "python"
    if p.is_file():
        return str(p)
    return None


def is_engine_ready() -> bool:
    """True if the managed venv has an engine installed."""
    state = load_state()
    return state.engine_installed and state.phase == PHASE_COMPLETE


def is_setup_in_progress() -> bool:
    """True if setup is currently running (not finished or failed)."""
    state = load_state()
    return state.phase not in (PHASE_PENDING, PHASE_COMPLETE, PHASE_FAILED)


def is_setup_complete() -> bool:
    """True if setup finished successfully."""
    return load_state().phase == PHASE_COMPLETE


# ---------------------------------------------------------------------------
# System Python detection
# ---------------------------------------------------------------------------

_PYTHON_CANDIDATES = [
    "python3.13",
    "python3.12",
    "python3.11",
    "python3.10",
    "python3.9",
    "python3",
]


def _is_correct_arch(python_path: str) -> bool:
    """Verify the Python binary matches the current architecture.

    On Apple Silicon Macs running Rosetta, a system python3 might be
    x86_64 which would produce an x86_64 venv — no good for mlx-lm.
    """
    machine = platform.machine()
    if machine != "arm64":
        return True  # Only matters on Apple Silicon

    try:
        # Check the binary itself
        arch_result = subprocess.run(
            ["file", python_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "x86_64" in arch_result.stdout and "arm64" not in arch_result.stdout:
            return False
    except Exception:
        pass
    return True


def _has_venv_module(python_path: str) -> bool:
    """Check if the Python binary has the venv module available."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import venv"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def find_system_python() -> Optional[str]:
    """Find a suitable system Python for creating the managed venv.

    Searches python3.13 down to python3.9 on PATH, verifies:
    - venv module is available
    - architecture matches (arm64 on Apple Silicon)

    Returns the full path or None.
    """
    for name in _PYTHON_CANDIDATES:
        path = shutil.which(name)
        if path is None:
            continue
        if not _has_venv_module(path):
            logger.debug("Skipping %s: no venv module", path)
            continue
        if not _is_correct_arch(path):
            logger.debug("Skipping %s: wrong architecture", path)
            continue
        logger.debug("Found system Python: %s", path)
        return path

    return None


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------


def detect_best_engine() -> tuple[str, str]:
    """Detect the best engine for this platform.

    Returns:
        (engine_name, pip_package) — e.g. ("mlx-lm", "octomil-sdk[mlx,serve]")
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        return ("mlx-lm", "octomil-sdk[mlx,serve]")
    else:
        return ("llama.cpp", "octomil-sdk[llama,serve]")


# ---------------------------------------------------------------------------
# Recommended model
# ---------------------------------------------------------------------------


def _get_recommended_model() -> str:
    """Pick the recommended model key for first-time download.

    Uses the same logic as the launcher's auto-select but simplified:
    pick the best model that fits the device memory budget.
    """
    try:
        from octomil.agents.launcher import _build_recommendations

        recommendations = _build_recommendations()
        if recommendations:
            return recommendations[0].key
    except Exception:
        pass
    # Fallback: safe default that runs everywhere
    return "qwen-coder-7b"


# ---------------------------------------------------------------------------
# Venv creation
# ---------------------------------------------------------------------------


def _has_uv() -> bool:
    """Check if uv is on PATH."""
    return shutil.which("uv") is not None


def create_managed_venv(python_path: str) -> str:
    """Create the managed venv at ~/.octomil/engines/venv/.

    Prefers ``uv venv`` for speed, falls back to ``python3 -m venv``.

    Returns the path to the venv's python binary.
    """
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing venv if present
    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)

    if _has_uv():
        logger.info("Creating venv with uv...")
        subprocess.run(
            ["uv", "venv", str(VENV_DIR), "--python", python_path],
            check=True,
            capture_output=True,
        )
    else:
        logger.info("Creating venv with python -m venv...")
        subprocess.run(
            [python_path, "-m", "venv", str(VENV_DIR)],
            check=True,
            capture_output=True,
        )

    venv_python = str(VENV_DIR / "bin" / "python")
    if not os.path.isfile(venv_python):
        raise RuntimeError(f"Venv created but python not found at {venv_python}")

    return venv_python


# ---------------------------------------------------------------------------
# Engine installation
# ---------------------------------------------------------------------------


def install_engine(package: str) -> None:
    """Install the engine package into the managed venv.

    Prefers ``uv pip install`` for speed, falls back to venv pip.
    """
    venv_python = str(VENV_DIR / "bin" / "python")

    if _has_uv():
        logger.info("Installing %s with uv...", package)
        subprocess.run(
            ["uv", "pip", "install", package, "--python", venv_python],
            check=True,
        )
    else:
        logger.info("Installing %s with pip...", package)
        subprocess.run(
            [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [venv_python, "-m", "pip", "install", package],
            check=True,
        )


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


def download_model(model_key: str) -> None:
    """Download a model using the venv's huggingface_hub.

    Runs ``python -c 'from huggingface_hub import snapshot_download; ...'``
    inside the managed venv so we don't need huggingface_hub in the
    frozen binary.
    """
    venv_python = get_venv_python()
    if venv_python is None:
        raise RuntimeError("Managed venv not found — run setup first")

    # Resolve the model key to an HF repo via the venv's octomil
    # This runs the catalog resolution in the venv where all deps exist
    script = f"""\
import sys
try:
    from octomil.models.resolver import resolve
    result = resolve("{model_key}")
    repo = result.hf_repo
    filename = result.filename
except Exception as e:
    # Fallback: try catalog directly
    try:
        from octomil.models.catalog import get_model
        entry = get_model("{model_key}")
        if entry is None:
            print("ERROR:Model not found in catalog", file=sys.stderr)
            sys.exit(1)
        variant = entry.variants.get(entry.default_quant)
        if variant is None:
            print("ERROR:No default variant", file=sys.stderr)
            sys.exit(1)
        # Prefer MLX on Apple Silicon
        import platform
        if platform.machine() == "arm64" and variant.mlx:
            repo = variant.mlx
            filename = None
        elif variant.gguf:
            repo = variant.gguf.repo
            filename = variant.gguf.filename
        else:
            print(f"ERROR:No downloadable artifact for {{entry.default_quant}}", file=sys.stderr)
            sys.exit(1)
    except Exception as e2:
        print(f"ERROR:{{e2}}", file=sys.stderr)
        sys.exit(1)

if filename:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, filename)
    print(f"Downloaded {{filename}} from {{repo}} -> {{path}}")
else:
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo)
    print(f"Downloaded {{repo}} -> {{path}}")
"""
    result = subprocess.run(
        [venv_python, "-c", script],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Model download failed: {stderr}")

    logger.info("Model download output: %s", result.stdout.strip())


# ---------------------------------------------------------------------------
# Full setup pipeline
# ---------------------------------------------------------------------------


def _log_setup_summary(state: SetupState, _log) -> None:  # noqa: ANN001
    """Print a summary of the current setup state."""
    _log("")
    _log("  Setup complete")
    if state.engine:
        _log(f"  Engine:  {state.engine}")
    if state.model_key:
        dl = "downloaded" if state.model_downloaded else "downloads on first use"
        _log(f"  Model:   {state.model_key} ({dl})")
    _log("")


def _register_mcp(model_key: str, _log) -> None:  # noqa: ANN001
    """Register MCP server across AI tools (non-fatal)."""
    _log("  MCP servers")
    try:
        from octomil.mcp.registration import register_mcp_server

        results = register_mcp_server(model=model_key)
        for r in results:
            if r.success:
                _log(f"    \u2713 {r.display}")
            else:
                _log(f"    - {r.display} (skipped)")
    except Exception as e:
        logger.debug("MCP registration failed (non-fatal): %s", e)
        _log(f"    MCP registration skipped: {e}")
    _log("")


def run_setup(*, force: bool = False, foreground: bool = False) -> SetupState:
    """Run the full setup pipeline.

    Args:
        force: Re-run even if already complete.
        foreground: If True, suppresses interactive output (used by install.sh).

    Returns:
        Final SetupState.
    """
    state = load_state()

    def _log(msg: str) -> None:
        logger.info(msg)
        if not foreground:
            import click

            click.echo(msg)

    # If already complete, show status and re-run MCP registration only
    if state.phase == PHASE_COMPLETE and not force:
        _log_setup_summary(state, _log)
        _register_mcp(state.model_key or "qwen-coder-7b", _log)
        return state

    state = SetupState(started_at=time.time())
    _save_state(state)

    # Step 1: Find system Python
    _log("Finding system Python...")
    python_path = find_system_python()
    if python_path is None:
        state.phase = PHASE_FAILED
        state.error = (
            "No suitable Python found. Install python3 with venv support:\n"
            "  macOS: brew install python3\n"
            "  Ubuntu/Debian: sudo apt install python3-venv\n"
            "  Fedora: sudo dnf install python3"
        )
        state.finished_at = time.time()
        _save_state(state)
        _log(f"Setup failed: {state.error}")
        return state

    _log(f"  Using {python_path}")

    # Step 2: Detect best engine
    engine_name, package = detect_best_engine()
    state.engine = engine_name
    state.package = package
    _save_state(state)
    _log(f"  Engine: {engine_name} ({package})")

    # Step 3: Create venv
    state.phase = PHASE_CREATING_VENV
    _save_state(state)
    _log("Creating managed venv...")
    try:
        create_managed_venv(python_path)
        _log(f"  Venv: {VENV_DIR}")
    except Exception as e:
        state.phase = PHASE_FAILED
        state.error = f"Failed to create venv: {e}"
        state.finished_at = time.time()
        _save_state(state)
        _log(f"Setup failed: {state.error}")
        return state

    # Step 4: Install engine
    state.phase = PHASE_INSTALLING_ENGINE
    _save_state(state)
    _log(f"Installing {package} (this may take a few minutes)...")
    try:
        install_engine(package)
        state.engine_installed = True
        _save_state(state)
        _log(f"  {engine_name} installed")
    except Exception as e:
        state.phase = PHASE_FAILED
        state.error = f"Failed to install {package}: {e}"
        state.finished_at = time.time()
        _save_state(state)
        _log(f"Setup failed: {state.error}")
        return state

    # Step 5: Download recommended model
    state.phase = PHASE_DOWNLOADING_MODEL
    model_key = _get_recommended_model()
    state.model_key = model_key
    _save_state(state)
    _log(f"Downloading model: {model_key}...")
    try:
        download_model(model_key)
        state.model_downloaded = True
        _save_state(state)
        _log(f"  Model {model_key} ready")
    except Exception as e:
        # Model download failure is non-fatal — engine still works,
        # user just gets a download on first serve.
        logger.warning("Model download failed (non-fatal): %s", e)
        _log(f"  Model download failed (will download on first use): {e}")

    # Step 6: Register MCP server across AI coding tools
    _register_mcp(model_key, _log)

    # Done
    state.phase = PHASE_COMPLETE
    state.finished_at = time.time()
    _save_state(state)

    elapsed = state.finished_at - (state.started_at or state.finished_at)
    _log(f"Setup complete in {elapsed:.0f}s")
    return state
