"""Launch coding agents powered by local Octomil models."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import click

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate models for the interactive picker (largest → smallest).
# The picker filters this list based on available device memory.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """A model that may appear in the interactive picker."""

    key: str  # catalog key passed to ``octomil serve``
    params_b: float  # total parameter count in billions
    description: str  # one-line description
    size: str  # approximate Q4_K_M download size


_ALL_CANDIDATES: list[_Candidate] = [
    _Candidate("deepseek-v3.2", 685, "DeepSeek V3.2, top open model", "~405 GB"),
    _Candidate("minimax-m2.1", 229, "MiniMax M2.1, strong multi-lang coding", "~138 GB"),
    _Candidate("devstral-123b", 123, "Devstral 2, Mistral's agentic coder", "~75 GB"),
    _Candidate("glm-flash", 30, "GLM-4.7 Flash, fast reasoning & code", "~18.5 GB"),
    _Candidate("qwen-coder-7b", 7, "Best small coding model, purpose-built", "~4.5 GB"),
    _Candidate("llama-8b", 8, "Meta Llama 3.1, solid all-rounder", "~4.5 GB"),
    _Candidate("qwen-coder-3b", 3, "Fast coding model, runs anywhere", "~2 GB"),
    _Candidate("qwen-coder-1.5b", 1.5, "Ultra-light coder, instant responses", "~1 GB"),
]


# ---------------------------------------------------------------------------
# Device-aware model filtering
# ---------------------------------------------------------------------------


def _get_memory_budget_gb() -> float:
    """Return usable memory in GB for model loading (total RAM - OS reserve)."""
    try:
        from ..hardware._unified import UnifiedDetector  # type: ignore[attr-defined]

        hw = UnifiedDetector().detect()
        is_metal = hw.gpu is not None and hw.gpu.backend == "metal"
        if is_metal:
            # Apple Silicon: unified memory, reserve 4 GB for OS
            return float(max(hw.total_ram_gb - 4.0, 0.0))
        if hw.gpu is not None and hw.gpu.total_vram_gb > 0:
            return float(hw.gpu.total_vram_gb * 0.9)
        return float(hw.available_ram_gb * 0.85)
    except Exception:
        logger.debug("Hardware detection failed, assuming 8 GB budget")
        return 8.0


def _model_fits(params_b: float, budget_gb: float) -> bool:
    """Check if a model at Q4_K_M fits within the memory budget."""
    try:
        from ..model_optimizer import _total_memory_gb

        needed = _total_memory_gb(params_b, "Q4_K_M", 4096)
        return needed <= budget_gb
    except Exception:
        # Fallback: rough estimate at 0.625 bytes/param
        return (params_b * 0.625) <= budget_gb


def _is_model_downloaded(key: str) -> bool:
    """Check if a model is already cached locally."""
    try:
        from ..models.catalog import get_model
        from ..sources.huggingface import HuggingFaceSource

        entry = get_model(key)
        if entry is None:
            return False

        variant = entry.variants.get(entry.default_quant)
        if variant is None:
            return False

        hf = HuggingFaceSource()
        if variant.mlx and hf.check_cache(variant.mlx):
            return True
        if variant.gguf and hf.check_cache(variant.gguf.repo, variant.gguf.filename):
            return True
    except Exception:
        pass
    return False


@dataclass(frozen=True)
class RecommendedModel:
    """A model recommendation shown in the interactive picker."""

    key: str
    label: str
    description: str
    size: str
    recommended: bool = False
    downloaded: bool = False


def _build_recommendations() -> list[RecommendedModel]:
    """Build a device-filtered list of recommended models."""
    budget = _get_memory_budget_gb()
    models: list[RecommendedModel] = []

    for c in _ALL_CANDIDATES:
        if _model_fits(c.params_b, budget):
            models.append(
                RecommendedModel(
                    key=c.key,
                    label=c.key,
                    description=c.description,
                    size=c.size,
                    downloaded=_is_model_downloaded(c.key),
                )
            )

    if not models:
        # Always offer the smallest model as a fallback
        smallest = _ALL_CANDIDATES[-1]
        models.append(
            RecommendedModel(
                key=smallest.key,
                label=smallest.key,
                description=smallest.description,
                size=smallest.size,
                downloaded=_is_model_downloaded(smallest.key),
            )
        )

    # Mark the first (largest fitting) model as recommended
    models[0] = RecommendedModel(
        key=models[0].key,
        label=models[0].label,
        description=models[0].description,
        size=models[0].size,
        recommended=True,
        downloaded=models[0].downloaded,
    )
    return models


def _auto_select_model() -> str:
    """Auto-select the best model for the device. No user prompt.

    Prefers an already-downloaded model to avoid cold-start download wait.
    When multiple models are downloaded, picks the largest (it's already
    cached so there's no penalty).  Falls back to the largest fitting model
    when nothing is cached.
    """
    recommendations = _build_recommendations()
    budget = _get_memory_budget_gb()

    downloaded = [m for m in recommendations if m.downloaded]

    if downloaded:
        # Pick the largest downloaded model (first in the list — sorted
        # largest-to-smallest from _ALL_CANDIDATES order).
        best = downloaded[0]
        click.echo(f"Using {best.key} (already downloaded, best for {budget:.0f} GB). Use --select to choose.")
    else:
        best = recommendations[0]
        click.echo(f"Using {best.key} (best for {budget:.0f} GB, will download {best.size}). Use --select to choose.")
    return best.key


def _select_model_tui() -> str:
    """Show a TUI model picker with arrow keys. Falls back to plain list."""
    recommendations = _build_recommendations()
    budget = _get_memory_budget_gb()

    # Check for prompt_toolkit + TTY
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    try:
        if not is_tty:
            raise ImportError("not a TTY")

        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        return _run_model_tui(
            Application,
            KeyBindings,
            Layout,
            Window,
            FormattedTextControl,
            recommendations,
            budget,
        )
    except ImportError:
        return _select_model_fallback(recommendations, budget)


def _run_model_tui(
    Application,  # noqa: N803
    KeyBindings,  # noqa: N803
    Layout,  # noqa: N803
    Window,  # noqa: N803
    FormattedTextControl,  # noqa: N803
    recommendations: list[RecommendedModel],
    budget: float,
) -> str:
    """Full prompt_toolkit TUI for model selection."""
    statuses = {m.key: _is_model_downloaded(m.key) for m in recommendations}

    selected = [0]
    search_mode = [False]
    search_query = [""]
    filtered: list[list[RecommendedModel]] = [list(recommendations)]
    result: list[str | None] = [None]

    def _fuzzy_filter(query: str) -> list[RecommendedModel]:
        if not query:
            return list(recommendations)
        q = query.lower()
        out: list[RecommendedModel] = []
        for m in recommendations:
            target = f"{m.key} {m.description}".lower()
            qi = 0
            for ch in target:
                if qi < len(q) and ch == q[qi]:
                    qi += 1
            if qi == len(q):
                out.append(m)
        return out

    def get_display_text():  # type: ignore[no-untyped-def]
        lines: list[tuple[str, str]] = []
        lines.append(("bold", f"  Model Selection ({budget:.0f} GB available)\n"))
        lines.append(
            (
                "",
                "  Use \u2191\u2193 to navigate, Enter to select, / to search, Esc to cancel\n\n",
            )
        )

        for i, m in enumerate(filtered[0]):
            dl = "downloaded" if statuses.get(m.key) else "not downloaded"
            tag = " \u2605 recommended" if m.recommended else ""
            prefix = " \u25b8 " if i == selected[0] else "   "
            style = "reverse" if i == selected[0] else ""
            lines.append((style, f"{prefix}{m.key}{tag}\n"))
            lines.append(("", f"     {m.description}  |  {m.size}  |  {dl}\n"))

        if search_mode[0]:
            lines.append(("bold", f"\n  Search: {search_query[0]}\u2588\n"))

        return lines

    control = FormattedTextControl(get_display_text)
    bindings = KeyBindings()

    @bindings.add("up")
    def _up(event):  # type: ignore[no-untyped-def]
        selected[0] = max(0, selected[0] - 1)

    @bindings.add("down")
    def _down(event):  # type: ignore[no-untyped-def]
        selected[0] = min(len(filtered[0]) - 1, selected[0] + 1)

    @bindings.add("enter")
    def _enter(event):  # type: ignore[no-untyped-def]
        if filtered[0]:
            result[0] = filtered[0][selected[0]].key
        event.app.exit()

    @bindings.add("/")
    def _search(event):  # type: ignore[no-untyped-def]
        if not search_mode[0]:
            search_mode[0] = True
            search_query[0] = ""

    @bindings.add("escape")
    def _escape(event):  # type: ignore[no-untyped-def]
        if search_mode[0]:
            search_mode[0] = False
            search_query[0] = ""
            filtered[0] = list(recommendations)
            selected[0] = 0
        else:
            event.app.exit()

    @bindings.add("c-c")
    def _ctrl_c(event):  # type: ignore[no-untyped-def]
        event.app.exit()

    @bindings.add("backspace")
    def _backspace(event):  # type: ignore[no-untyped-def]
        if search_mode[0] and search_query[0]:
            search_query[0] = search_query[0][:-1]
            filtered[0] = _fuzzy_filter(search_query[0])
            selected[0] = 0

    @bindings.add("<any>")
    def _any_key(event):  # type: ignore[no-untyped-def]
        if search_mode[0]:
            search_query[0] += event.data
            filtered[0] = _fuzzy_filter(search_query[0])
            selected[0] = 0

    layout = Layout(Window(content=control))
    app: Application[None] = Application(
        layout=layout,
        key_bindings=bindings,
        full_screen=True,
    )
    app.run()

    if result[0] is None:
        raise SystemExit(0)
    return result[0]


def _select_model_fallback(
    recommendations: list[RecommendedModel],
    budget: float,
) -> str:
    """Plain numbered list fallback when TUI is unavailable."""
    click.echo(f"\nModel Selection ({budget:.0f} GB available)\n")

    for i, m in enumerate(recommendations):
        downloaded = _is_model_downloaded(m.key)
        status = "downloaded" if downloaded else "not downloaded"
        marker = " (Recommended)" if m.recommended else ""
        prefix = "  > " if i == 0 else "    "
        click.echo(f"{prefix}{i + 1}. {m.label}{marker}")
        click.echo(f"      {m.description}, {m.size}, ({status})")

    click.echo()

    choices = {str(i + 1): m.key for i, m in enumerate(recommendations)}
    labels = {str(i + 1): m.label for i, m in enumerate(recommendations)}

    hint_parts = [f"{k}={labels[k]}" for k in sorted(choices)]
    hint = ", ".join(hint_parts)

    selection: str = click.prompt(
        f"Select model [{hint}] or enter a model name",
        default="1",
    )

    if selection in choices:
        return choices[selection]
    return selection


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def is_serve_running(host: str = "localhost", port: int = 8080) -> bool:
    """Check if ``octomil serve`` is already running."""
    try:
        urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def _build_serve_cmd(model: str, port: int) -> list[str]:
    """Build the command to start ``octomil serve``.

    When running from a PyInstaller frozen binary, ``sys.executable`` is
    the binary itself — not a Python interpreter — so ``-m octomil`` is
    invalid.  In that case we invoke the binary directly as a subcommand.

    If a managed venv with a native engine exists (set up by
    ``octomil setup``), we use the venv's Python instead — this gives
    direct access to mlx-lm / llama.cpp without Ollama.
    """
    if getattr(sys, "frozen", False):
        from octomil.setup import get_venv_python, is_engine_ready

        venv_py = get_venv_python()
        if venv_py and is_engine_ready():
            return [venv_py, "-m", "octomil", "serve", model, "--port", str(port)]
        # Fallback: frozen binary (Ollama engine)
        return [sys.executable, "serve", model, "--port", str(port)]
    return [sys.executable, "-m", "octomil", "serve", model, "--port", str(port)]


def start_serve_background(model: str, port: int = 8080, timeout: int = 600) -> subprocess.Popen:
    """Start ``octomil serve`` in the background and wait until ready.

    The default 600s timeout accommodates first-run model downloads (~4.5 GB).
    Server stderr is streamed so users can see download progress and errors.
    """
    log_path = os.path.join(os.path.expanduser("~"), ".cache", "octomil", "serve.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")  # noqa: SIM115

    cmd = _build_serve_cmd(model, port)
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    click.echo(f"Waiting for model to load (log: {log_path})...")
    for i in range(timeout):
        if proc.poll() is not None:
            log_file.close()
            try:
                with open(log_path) as f:
                    tail = f.readlines()[-20:]
                click.echo("Server exited unexpectedly. Last log lines:", err=True)
                for line in tail:
                    click.echo(f"  {line.rstrip()}", err=True)
            except Exception:
                pass
            raise RuntimeError(f"octomil serve exited with code {proc.returncode}")
        if is_serve_running(port=port):
            log_file.close()
            return proc
        if i > 0 and i % 15 == 0:
            click.echo(f"  Still loading... ({i}s elapsed)")
        time.sleep(1)
    proc.terminate()
    log_file.close()
    raise RuntimeError(f"octomil serve failed to start within {timeout}s")


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Agent picker
# ---------------------------------------------------------------------------

# Agents shown in the interactive picker (order matters).
_PICKER_AGENTS = ["claude", "codex", "droid", "opencode"]


def _select_agent_tui() -> str:
    """Show a TUI agent picker. Falls back to plain numbered list."""
    from .registry import AgentDef, is_agent_installed, list_agents

    agents = list_agents()
    # Put picker agents first in order, then any extras
    ordered: list[AgentDef] = []
    for name in _PICKER_AGENTS:
        for a in agents:
            if a.name == name:
                ordered.append(a)
                break
    for a in agents:
        if a.name not in _PICKER_AGENTS:
            ordered.append(a)

    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    try:
        if not is_tty:
            raise ImportError("not a TTY")

        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        return _run_agent_tui(
            Application,
            KeyBindings,
            Layout,
            Window,
            FormattedTextControl,
            ordered,
            is_agent_installed,
        )
    except ImportError:
        return _select_agent_fallback(ordered, is_agent_installed)


def _run_agent_tui(
    Application,  # noqa: N803
    KeyBindings,  # noqa: N803
    Layout,  # noqa: N803
    Window,  # noqa: N803
    FormattedTextControl,  # noqa: N803
    agents: list,
    check_installed,  # noqa: ANN001
) -> str:
    """Full prompt_toolkit TUI for agent selection."""
    import shutil as _shutil

    statuses = {a.name: _shutil.which(a.install_check) is not None for a in agents}

    selected = [0]
    result: list[str | None] = [None]

    def get_display_text():  # type: ignore[no-untyped-def]
        lines: list[tuple[str, str]] = []
        lines.append(("bold", "  Select a coding agent\n"))
        lines.append(("", "  Use \u2191\u2193 to navigate, Enter to select, Esc to cancel\n\n"))

        for i, a in enumerate(agents):
            installed = "\u2713 installed" if statuses.get(a.name) else "not installed"
            prefix = " \u25b8 " if i == selected[0] else "   "
            style = "reverse" if i == selected[0] else ""
            lines.append((style, f"{prefix}{a.name}\n"))
            lines.append(("", f"     {a.description}  |  {installed}\n"))

        return lines

    control = FormattedTextControl(get_display_text)
    bindings = KeyBindings()

    @bindings.add("up")
    def _up(event):  # type: ignore[no-untyped-def]
        selected[0] = max(0, selected[0] - 1)

    @bindings.add("down")
    def _down(event):  # type: ignore[no-untyped-def]
        selected[0] = min(len(agents) - 1, selected[0] + 1)

    @bindings.add("enter")
    def _enter(event):  # type: ignore[no-untyped-def]
        if agents:
            result[0] = agents[selected[0]].name
        event.app.exit()

    @bindings.add("escape")
    def _escape(event):  # type: ignore[no-untyped-def]
        event.app.exit()

    @bindings.add("c-c")
    def _ctrl_c(event):  # type: ignore[no-untyped-def]
        event.app.exit()

    layout = Layout(Window(content=control))
    app: Application[None] = Application(
        layout=layout,
        key_bindings=bindings,
        full_screen=True,
    )
    app.run()

    if result[0] is None:
        raise SystemExit(0)
    return result[0]


def _select_agent_fallback(agents: list, check_installed) -> str:  # noqa: ANN001
    """Plain numbered list fallback when TUI is unavailable."""
    import shutil as _shutil

    click.echo("\nSelect a coding agent\n")

    for i, a in enumerate(agents):
        installed = _shutil.which(a.install_check) is not None
        status = "installed" if installed else "not installed"
        prefix = "  > " if i == 0 else "    "
        click.echo(f"{prefix}{i + 1}. {a.name}")
        click.echo(f"      {a.description}  ({status})")

    click.echo()

    choices = {str(i + 1): a.name for i, a in enumerate(agents)}
    labels = {str(i + 1): a.name for i, a in enumerate(agents)}
    hint = ", ".join(f"{k}={labels[k]}" for k in sorted(choices))

    selection = click.prompt(f"Select agent [{hint}]", default="1")
    if selection in choices:
        return choices[selection]
    return selection


def launch_agent(
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    port: int = 8080,
    select: bool = False,
) -> None:
    """Launch a coding agent with a local model backend.

    1. If no agent specified, shows an interactive picker.
    2. Ensures the agent binary is installed (offers to install if not).
    3. If ``--select``, shows a TUI picker. Otherwise auto-selects best model.
    4. Starts ``octomil serve`` in the background if no server is running.
    5. Sets the appropriate env var so the agent talks to the local server.
    6. Execs the agent and tears down the server on exit.
    """
    from .registry import get_agent, is_agent_installed

    if agent_name is None:
        agent_name = _select_agent_tui()

    agent = get_agent(agent_name)
    if agent is None:
        from .registry import list_agents

        available = ", ".join(a.name for a in list_agents())
        raise click.ClickException(f"Unknown agent '{agent_name}'. Available: {available}")

    # Check if agent is installed
    if not is_agent_installed(agent):
        click.echo(f"{agent.display_name} is not installed.")
        click.echo(f"  Install: {agent.install_cmd}")
        if click.confirm("Install now?"):
            subprocess.run(shlex.split(agent.install_cmd), check=True)
        else:
            raise SystemExit(1)

    # Agents that don't need a local model (e.g. Claude Code) are exec'd
    # directly unless the user explicitly passed --model to proxy through
    # the local server.
    use_local_server = agent.needs_local_model or model is not None

    serve_proc: Optional[subprocess.Popen] = None
    env = os.environ.copy()

    if use_local_server:
        base_url = f"http://localhost:{port}/v1"
        if not is_serve_running(port=port):
            if model is None:
                model = _select_model_tui() if select else _auto_select_model()
            click.echo(f"Starting octomil serve {model}...")
            serve_proc = start_serve_background(model, port=port)
            click.echo(f"Model ready at {base_url}")
        else:
            click.echo(f"Using existing server at {base_url}")

        env[agent.env_key] = base_url
        if agent.env_key.startswith("OPENAI"):
            env["OPENAI_API_KEY"] = "octomil-local"

    try:
        click.echo(f"Launching {agent.display_name}...\n")
        result = subprocess.run(shlex.split(agent.exec_cmd), env=env)
    finally:
        if serve_proc is not None:
            serve_proc.terminate()

    sys.exit(result.returncode)
