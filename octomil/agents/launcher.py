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


_CURATED_CANDIDATES: list[_Candidate] = [
    _Candidate("qwen-7b", 7, "Qwen 2.5 Coder, best small coding model", "4.5 GB"),
    _Candidate("llama-8b", 8, "Meta Llama 3.1, solid all-rounder", "4.5 GB"),
    _Candidate("phi-4", 14, "Microsoft Phi-4, strong reasoning", "8.5 GB"),
    _Candidate("gemma-12b", 12, "Google Gemma 3, multilingual", "7.5 GB"),
]

# Recommended keys shown in the top section of the picker
_RECOMMENDED_KEYS = {"qwen-7b", "llama-8b", "phi-4"}


def _params_to_float(params: str) -> float:
    """Convert catalog params string like '7B' or '360M' to float billions."""
    p = params.upper().strip()
    if p.endswith("B"):
        return float(p[:-1])
    if p.endswith("M"):
        return float(p[:-1]) / 1000
    return 0.0


def _estimate_size(params_b: float) -> str:
    """Rough Q4 download size estimate."""
    gb = params_b * 0.625
    if gb < 1:
        return f"{gb * 1000:.0f} MB"
    return f"{gb:.1f} GB"


def _params_to_float_from_key(key: str) -> float:
    """Get param count in billions for a catalog key."""
    try:
        from ..models.catalog import get_model

        entry = get_model(key)
        if entry:
            return _params_to_float(entry.params)
    except Exception:
        pass
    return 0.0


def _build_all_candidates() -> list[_Candidate]:
    """Build candidate list from the live catalog, falling back to curated."""
    try:
        from ..models.catalog import CATALOG

        candidates: list[_Candidate] = []
        for key, entry in CATALOG.items():
            # Skip speech-to-text models
            if "whisper" in key:
                continue
            params_b = _params_to_float(entry.params)
            desc = f"{entry.publisher} {key}, {entry.params} params"
            size = _estimate_size(params_b)
            candidates.append(_Candidate(key, params_b, desc, size))

        # Sort largest → smallest
        candidates.sort(key=lambda c: c.params_b, reverse=True)
        if candidates:
            return candidates
    except Exception:
        pass
    return list(_CURATED_CANDIDATES)


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
    """Build the full model list from catalog, sorted largest→smallest."""
    budget = _get_memory_budget_gb()
    all_candidates = _build_all_candidates()
    models: list[RecommendedModel] = []

    for c in all_candidates:
        fits = _model_fits(c.params_b, budget)
        is_rec = c.key in _RECOMMENDED_KEYS and fits
        models.append(
            RecommendedModel(
                key=c.key,
                label=c.key,
                description=c.description,
                size=c.size,
                recommended=is_rec,
                downloaded=_is_model_downloaded(c.key),
            )
        )

    if not models:
        models.append(
            RecommendedModel(
                key="smollm-360m",
                label="smollm-360m",
                description="Ultra-light, runs anywhere",
                size="225 MB",
            )
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

    # Prefer downloaded models that fit in memory
    fitting = [m for m in recommendations if _model_fits(_params_to_float_from_key(m.key), budget)]
    downloaded = [m for m in fitting if m.downloaded]

    if downloaded:
        best = downloaded[0]
        click.echo(f"Using {best.key} (already downloaded). Use --select to choose.")
    elif fitting:
        best = fitting[0]
        click.echo(f"Using {best.key} (will download ~{best.size}). Use --select to choose.")
    else:
        best = recommendations[-1]  # smallest
        click.echo(f"Using {best.key} (will download ~{best.size}). Use --select to choose.")
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

    # Split into recommended and more
    recommended = [m for m in recommendations if m.recommended]
    more = [m for m in recommendations if not m.recommended]
    all_items = recommended + more

    selected = [0]
    search_mode = [False]
    search_query = [""]
    filtered: list[list[RecommendedModel]] = [list(all_items)]
    result: list[str | None] = [None]

    def _fuzzy_filter(query: str) -> list[RecommendedModel]:
        if not query:
            return list(all_items)
        q = query.lower()
        out: list[RecommendedModel] = []
        for m in all_items:
            target = f"{m.key} {m.description}".lower()
            qi = 0
            for ch in target:
                if qi < len(q) and ch == q[qi]:
                    qi += 1
            if qi == len(q):
                out.append(m)
        return out

    def _status_tag(m: RecommendedModel) -> str:
        if statuses.get(m.key):
            return ""
        return ", (not downloaded)"

    def get_display_text():  # type: ignore[no-untyped-def]
        lines: list[tuple[str, str]] = []

        if search_mode[0]:
            lines.append(("bold", f"  Select model: {search_query[0]}\u2588\n"))
        else:
            lines.append(("bold", "  Select model: "))
            lines.append(("italic", "Type to filter...\n"))

        # Find which section the cursor is in
        flat = filtered[0]

        # Split filtered items back into sections
        f_rec = [m for m in flat if m in recommended]
        f_more = [m for m in flat if m in more]

        idx = 0  # global flat index

        if f_rec:
            lines.append(("", "\n"))
            lines.append(("bold", "  Recommended\n"))
            for m in f_rec:
                is_sel = idx == selected[0]
                prefix = "  \u25b8 " if is_sel else "    "
                style = "bold" if is_sel else ""
                dl = _status_tag(m)
                lines.append((style, f"{prefix}{m.key}\n"))
                lines.append(("", f"      {m.description}, ~{m.size}{dl}\n"))
                idx += 1

        if f_more:
            lines.append(("", "\n"))
            lines.append(("bold", "  More\n"))
            for m in f_more:
                is_sel = idx == selected[0]
                prefix = "  \u25b8 " if is_sel else "    "
                style = "bold" if is_sel else ""
                dl = _status_tag(m)
                lines.append((style, f"{prefix}{m.key}\n"))
                lines.append(("", f"      {m.description}, ~{m.size}{dl}\n"))
                idx += 1

        lines.append(("", "\n"))
        lines.append(("", "  \u2191/\u2193 navigate \u2022 enter select \u2022 / search \u2022 esc cancel\n"))

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
            filtered[0] = list(all_items)
            selected[0] = 0
        else:
            event.app.exit()

    @bindings.add("c-c")
    def _ctrl_c(event):  # type: ignore[no-untyped-def]
        event.app.exit()

    @bindings.add("backspace")
    def _backspace(event):  # type: ignore[no-untyped-def]
        if search_query[0]:
            search_query[0] = search_query[0][:-1]
            filtered[0] = _fuzzy_filter(search_query[0])
            selected[0] = 0
            if not search_query[0]:
                search_mode[0] = False

    @bindings.add("<any>")
    def _any_key(event):  # type: ignore[no-untyped-def]
        ch = event.data
        if ch.isprintable() and len(ch) == 1:
            search_mode[0] = True
            search_query[0] += ch
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
    click.echo("\n  Select model:\n")

    recommended = [m for m in recommendations if m.recommended]
    more = [m for m in recommendations if not m.recommended]

    if recommended:
        click.echo("  Recommended")
        for i, m in enumerate(recommended):
            downloaded = _is_model_downloaded(m.key)
            dl = "" if downloaded else ", (not downloaded)"
            click.echo(f"    {i + 1}. {m.key}")
            click.echo(f"       {m.description}, ~{m.size}{dl}")

    if more:
        click.echo("\n  More")
        for j, m in enumerate(more):
            num = len(recommended) + j + 1
            downloaded = _is_model_downloaded(m.key)
            dl = "" if downloaded else ", (not downloaded)"
            click.echo(f"    {num}. {m.key}")
            click.echo(f"       {m.description}, ~{m.size}{dl}")

    click.echo()

    choices = {str(i + 1): m.key for i, m in enumerate(recommendations)}
    selection: str = click.prompt(
        "  Select model (number or name)",
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


def _ensure_engine_ready() -> None:
    """Ensure the managed venv and native engine are set up.

    Called before starting ``octomil serve`` as a subprocess so that
    ``octomil launch <agent>`` is a single-command experience.  If setup
    hasn't run yet, runs it inline with progress output.
    """
    if not getattr(sys, "frozen", False):
        return  # dev installs have engines available directly

    from octomil.setup import (
        get_venv_python,
        is_engine_ready,
        is_setup_in_progress,
        load_state,
        run_setup,
    )

    if is_setup_in_progress():
        click.echo("\n  Engine setup is in progress, waiting...")
        import time

        waited = 0
        while is_setup_in_progress() and waited < 600:
            time.sleep(2)
            waited += 2
        return

    venv_py = get_venv_python()
    if venv_py and is_engine_ready():
        return  # already set up

    state = load_state()
    if state.phase == "failed":
        return  # previous failure, _build_serve_cmd will use fallback

    click.echo(
        click.style(
            "\n  First run: setting up native inference engine...\n",
            fg="cyan",
        )
    )
    run_setup()


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


def _extract_serve_error(log_path: str) -> str:
    """Extract a human-readable error message from the serve log file."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except Exception:
        return "Server exited unexpectedly."

    for line in reversed(lines):
        stripped = line.strip()
        for prefix in ("ValueError: ", "ModelResolutionError: "):
            if stripped.startswith(prefix):
                return stripped[len(prefix) :]
        if "ERROR" in stripped and "Failed to" in stripped:
            parts = stripped.split("Failed to", 1)
            if len(parts) == 2:
                return "Failed to" + parts[1]

    return f"Server failed to start. See full log: {log_path}"


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
            error_msg = _extract_serve_error(log_path)
            raise click.ClickException(f"{error_msg}\n\n  Full log: {log_path}")
        if is_serve_running(port=port):
            log_file.close()
            return proc
        if i > 0 and i % 15 == 0:
            click.echo(f"  Still loading... ({i}s elapsed)")
        time.sleep(1)
    proc.terminate()
    log_file.close()
    raise click.ClickException(f"Model server failed to start within {timeout}s.\n\n  Full log: {log_path}")


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

    use_local_server = agent.needs_local_model or model is not None

    serve_proc: Optional[subprocess.Popen] = None
    env = os.environ.copy()

    if use_local_server:
        base_url = f"http://localhost:{port}/v1"
        if not is_serve_running(port=port):
            _ensure_engine_ready()
            if model is None:
                model = _select_model_tui() if (select or agent.needs_local_model) else _auto_select_model()
            click.echo(f"Starting octomil serve {model}...")
            serve_proc = start_serve_background(model, port=port)
            click.echo(f"Model ready at {base_url}")
        else:
            click.echo(f"Using existing server at {base_url}")

        # Configure the agent to use the local server
        if agent.configure_local is not None and model is not None:
            click.echo(f"Configuring {agent.display_name} for local model...")
            extra_env = agent.configure_local(base_url, model)
            env.update(extra_env)
        else:
            env[agent.env_key] = base_url
            if agent.env_key.startswith("OPENAI"):
                env["OPENAI_API_KEY"] = "octomil-local"
            if agent.env_key == "ANTHROPIC_BASE_URL":
                env["ANTHROPIC_API_KEY"] = "octomil-local"

    try:
        cmd = shlex.split(agent.exec_cmd)
        if model is not None and agent.model_flag:
            cmd += shlex.split(agent.model_flag.format(model=model))
        click.echo(f"Launching {agent.display_name}...\n")
        result = subprocess.run(cmd, env=env)
    except click.ClickException:
        raise
    except Exception as exc:
        if serve_proc is not None:
            serve_proc.terminate()
        raise click.ClickException(str(exc))
    else:
        if serve_proc is not None:
            serve_proc.terminate()

    sys.exit(result.returncode)
