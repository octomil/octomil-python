"""Direct local invocation commands: run, embed, transcribe.

Also registers advanced API-exact subcommand groups:
  octomil responses create|stream
  octomil embeddings create
  octomil audio transcriptions create

All commands use the shared execution kernel.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import click

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOCAL_RUNTIME_HINT = (
    "Install a local runtime for on-device execution:\n"
    "  pip install 'octomil[mlx]'      # Apple Silicon\n"
    "  pip install 'octomil[llama]'    # Cross-platform\n"
    "Or rerun with --install-runtime to let Octomil install the recommended runtime into this Python environment.\n"
    "Set OCTOMIL_SERVER_KEY to allow hosted cloud fallback."
)


@dataclass(frozen=True)
class RuntimeInstallCandidate:
    engine_name: str
    requirement: str
    extra_name: str
    description: str


def _run_async(coro):
    """Run an async coroutine synchronously for Click commands."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)
    except Exception as exc:
        _raise_click_exception(exc)


def _raise_click_exception(exc: Exception) -> None:
    """Render domain errors as Click failures instead of tracebacks."""
    from octomil.errors import OctomilError
    from octomil.models.resolver import ModelResolutionError

    if isinstance(exc, (OctomilError, ModelResolutionError)):
        raise click.ClickException(str(exc)) from exc
    if isinstance(exc, RuntimeError) and _is_runtime_unavailable_error(str(exc)):
        raise click.ClickException(f"No inference backend available.\n\n{_LOCAL_RUNTIME_HINT}") from exc
    raise exc


def _is_runtime_unavailable_error(message: str) -> bool:
    return (
        message
        in {
            "No runtime available",
            "No local runtime available",
            "No local or cloud backend available for chat.",
        }
        or "no local runtime is available" in message.lower()
    )


def _is_runtime_unavailable_click_exception(exc: click.ClickException) -> bool:
    return "No inference backend available" in exc.message


def _recommended_runtime_install() -> RuntimeInstallCandidate:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return RuntimeInstallCandidate(
            engine_name="mlx-lm",
            requirement="mlx-lm>=0.10.0",
            extra_name="mlx",
            description="Apple Silicon GPU inference",
        )
    return RuntimeInstallCandidate(
        engine_name="llama.cpp",
        requirement="llama-cpp-python>=0.2.0",
        extra_name="llama",
        description="cross-platform local GGUF inference",
    )


def _truthy_env(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _falsey_env(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "n", "off"}


def _can_prompt_runtime_install() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def _should_install_runtime(
    auto_install_runtime: Optional[bool], assume_yes: bool, candidate: RuntimeInstallCandidate
) -> bool:
    if auto_install_runtime is False:
        return False
    if auto_install_runtime is True or assume_yes or _truthy_env(os.environ.get("OCTOMIL_AUTO_INSTALL_RUNTIME")):
        return True
    if _falsey_env(os.environ.get("OCTOMIL_AUTO_INSTALL_RUNTIME")):
        return False
    if not _can_prompt_runtime_install():
        return False

    click.echo(
        f"No local inference runtime is installed. Octomil can install {candidate.engine_name} "
        f"for {candidate.description}.",
        err=True,
    )
    return click.confirm(
        f"Install {candidate.engine_name} now into this Python environment?",
        default=True,
        err=True,
    )


def _install_runtime(candidate: RuntimeInstallCandidate) -> None:
    command = _runtime_install_command(candidate.requirement)
    click.echo(f"Installing {candidate.engine_name} runtime with: {' '.join(command)}", err=True)
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(
            "Failed to install a local inference runtime.\n\n"
            f"Try manually: pip install 'octomil[{candidate.extra_name}]'\n"
            "Then rerun your octomil command."
        ) from exc

    # Make packages installed into the current interpreter importable on retry.
    importlib.invalidate_caches()
    try:
        from octomil.runtime.core import engine_bridge
        from octomil.runtime.engines.registry import reset_registry

        engine_bridge._runtime_cache.clear()
        reset_registry()
    except Exception:
        # Cache refresh is best-effort; a fresh process will still see the installed runtime.
        pass


def _runtime_install_command(requirement: str) -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, requirement]
    return [sys.executable, "-m", "pip", "install", requirement]


def _retry_after_runtime_install(
    exc: click.ClickException,
    auto_install_runtime: Optional[bool],
    assume_yes: bool,
    already_attempted: bool = False,
) -> bool:
    if not _is_runtime_unavailable_click_exception(exc):
        return False
    if already_attempted:
        return False

    candidate = _recommended_runtime_install()
    if not _should_install_runtime(auto_install_runtime, assume_yes, candidate):
        return False

    _install_runtime(candidate)
    return True


def _run_with_runtime_install_retry(
    make_coro,
    auto_install_runtime: Optional[bool],
    assume_yes: bool,
    install_already_attempted: bool = False,
):
    try:
        return _run_async(make_coro())
    except click.ClickException as exc:
        if not _retry_after_runtime_install(exc, auto_install_runtime, assume_yes, install_already_attempted):
            raise
        return _run_async(make_coro())


def _has_real_local_runtime(model: str) -> bool:
    try:
        from octomil.runtime.engines import get_registry

        registry = get_registry()
        return any(detection.available and detection.engine.name != "echo" for detection in registry.detect_all(model))
    except Exception:
        return False


def _chat_policy_prefers_local(
    kernel, model: Optional[str], policy: Optional[str], app: Optional[str]
) -> tuple[bool, str]:
    try:
        from octomil._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy
        from octomil.execution.kernel import _resolve_routing_policy

        defaults = kernel.resolve_chat_defaults(model=model, policy=policy, app=app)
        if inspect.isawaitable(defaults):
            close = getattr(defaults, "close", None)
            if callable(close):
                close()
            return False, ""
        effective_model = defaults.model
        if not isinstance(effective_model, str) or not effective_model:
            return False, ""

        routing_policy = _resolve_routing_policy(defaults)
        prefers_local = (
            routing_policy.mode == ContractRoutingPolicy.LOCAL_ONLY
            or routing_policy.mode == ContractRoutingPolicy.LOCAL_FIRST
            or routing_policy.prefer_local
        )
        return prefers_local and routing_policy.mode != ContractRoutingPolicy.CLOUD_ONLY, effective_model
    except Exception:
        return False, ""


def _capability_runner_target(
    kernel,
    capability: str,
    model: Optional[str],
    policy: Optional[str],
    app: Optional[str],
) -> tuple[bool, str, Optional[str]]:
    """Return whether a capability should use the runner, plus model/policy."""
    try:
        from octomil._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy
        from octomil.config.local import CAPABILITY_CHAT
        from octomil.execution.kernel import _resolve_routing_policy

        if capability == CAPABILITY_CHAT:
            defaults = kernel.resolve_chat_defaults(model=model, policy=policy, app=app)
        else:
            defaults = kernel._resolve(capability, model=model, policy=policy, app=app)
        if inspect.isawaitable(defaults):
            close = getattr(defaults, "close", None)
            if callable(close):
                close()
            return False, "", None

        effective_model = defaults.model
        if not isinstance(effective_model, str) or not effective_model:
            return False, "", None

        routing_policy = _resolve_routing_policy(defaults)
        prefers_local = (
            routing_policy.mode == ContractRoutingPolicy.LOCAL_ONLY
            or routing_policy.mode == ContractRoutingPolicy.LOCAL_FIRST
            or routing_policy.prefer_local
        )
        if routing_policy.mode == ContractRoutingPolicy.CLOUD_ONLY or not prefers_local:
            return False, effective_model, None

        return True, effective_model, defaults.policy_preset or policy or "local_first"
    except Exception:
        return False, "", None


def _planner_engine_for_runner(model: str, capability: str, policy: str) -> Optional[str]:
    """Use the runtime planner to choose the runner engine when possible."""
    try:
        from octomil.config.local import CAPABILITY_CHAT, CAPABILITY_EMBEDDING, CAPABILITY_TRANSCRIPTION
        from octomil.runtime.planner.planner import RuntimePlanner

        planner_capability = {
            CAPABILITY_CHAT: "responses",
            CAPABILITY_EMBEDDING: "embeddings",
            CAPABILITY_TRANSCRIPTION: "transcription",
        }.get(capability, capability)
        selection = RuntimePlanner().resolve(
            model=model,
            capability=planner_capability,
            routing_policy=policy,
        )
        if selection.locality == "local":
            return selection.engine
    except Exception:
        logger.debug("Runtime planner could not choose runner engine", exc_info=True)
    return None


def _prepare_runtime_for_local_run(
    kernel,
    model: Optional[str],
    policy: Optional[str],
    app: Optional[str],
    auto_install_runtime: Optional[bool],
    assume_yes: bool,
) -> bool:
    prefers_local, effective_model = _chat_policy_prefers_local(kernel, model, policy, app)
    if not prefers_local or not effective_model or _has_real_local_runtime(effective_model):
        return False

    candidate = _recommended_runtime_install()
    if not _should_install_runtime(auto_install_runtime, assume_yes, candidate):
        return False

    _install_runtime(candidate)
    return True


def _warn_if_cloud_execution(result) -> None:
    if result.locality != "cloud":
        return
    if result.fallback_used:
        click.echo(
            "Using hosted cloud fallback because no local inference backend was available. "
            "Install a local runtime with `pip install 'octomil[mlx]'` or `pip install 'octomil[llama]'` for on-device execution.",
            err=True,
        )
    else:
        click.echo("Running on hosted cloud per routing policy.", err=True)


def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _kernel():
    from octomil.execution.kernel import ExecutionKernel

    return ExecutionKernel()


def _read_stdin_text() -> Optional[str]:
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


# ---------------------------------------------------------------------------
# Local runner integration
# ---------------------------------------------------------------------------


def _runner_enabled(no_runner: bool) -> bool:
    """Check if the invisible local runner should be used."""
    if no_runner:
        return False
    env = os.environ.get("OCTOMIL_LOCAL_RUNNER", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def _try_runner_response(
    model: Optional[str],
    prompt: str,
    *,
    engine: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    restart_runner: bool = False,
    output_json: bool = False,
) -> Optional[str]:
    """Try to run inference via the local runner. Returns output text or None on failure."""
    if not model:
        return None
    try:
        from ..local_runner.client import LocalRunnerClient
        from ..local_runner.manager import LocalRunnerManager

        mgr = LocalRunnerManager()
        handle = mgr.ensure(model=model, engine=engine, restart=restart_runner)
        client = LocalRunnerClient(handle.base_url, handle.token)

        result = _run_async(
            client.create_response(
                model=handle.model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        )
        if result and "choices" in result:
            text = result["choices"][0]["message"]["content"]
            if output_json:
                return json.dumps(result, indent=2)
            return text
    except Exception as exc:
        logger.debug("Local runner failed, falling back to kernel: %s", exc)
    return None


def _try_runner_embed(
    model: Optional[str],
    text_inputs: list[str],
    *,
    engine: Optional[str] = None,
    restart_runner: bool = False,
) -> Optional[dict]:
    """Try to run embeddings via the local runner. Returns result dict or None."""
    if not model:
        return None
    try:
        from ..local_runner.client import LocalRunnerClient
        from ..local_runner.manager import LocalRunnerManager

        mgr = LocalRunnerManager()
        handle = mgr.ensure(model=model, engine=engine, restart=restart_runner)
        client = LocalRunnerClient(handle.base_url, handle.token)
        result = _run_async(client.create_embedding(model=handle.model, input=text_inputs))
        return result
    except Exception as exc:
        logger.debug("Local runner embedding failed, falling back: %s", exc)
    return None


def _try_runner_transcribe(
    model: Optional[str],
    audio_file: str,
    *,
    engine: Optional[str] = None,
    restart_runner: bool = False,
    language: Optional[str] = None,
) -> Optional[dict]:
    """Try to run transcription via the local runner. Returns result dict or None."""
    if not model:
        return None
    try:
        from ..local_runner.client import LocalRunnerClient
        from ..local_runner.manager import LocalRunnerManager

        mgr = LocalRunnerManager()
        handle = mgr.ensure(model=model, engine=engine, restart=restart_runner)
        client = LocalRunnerClient(handle.base_url, handle.token)
        kw: dict[str, str] = {}
        if language:
            kw["language"] = language
        result = _run_async(client.create_transcription(model=handle.model, file_path=audio_file, **kw))
        return result
    except Exception as exc:
        logger.debug("Local runner transcription failed, falling back: %s", exc)
    return None


# ---------------------------------------------------------------------------
# octomil run
# ---------------------------------------------------------------------------


@click.command("run")
@click.argument("prompt", required=False, default=None)
@click.option("--model", "-m", default=None, help="Model to use.")
@click.option("--app", default=None, help="App context (slug).")
@click.option("--policy", default=None, help="Serving policy preset.")
@click.option("--json", "output_json", is_flag=True, help="Output structured JSON.")
@click.option("--stream/--no-stream", "stream", default=None, help="Enable/disable streaming.")
@click.option("--temperature", "-t", type=float, default=None, help="Sampling temperature.")
@click.option("--max-output-tokens", type=int, default=None, help="Max output tokens.")
@click.option(
    "--install-runtime/--no-install-runtime",
    default=None,
    help="Install a recommended local runtime if no real local backend is available.",
)
@click.option("-y", "--yes", is_flag=True, help="Accept runtime installation prompts.")
@click.option("--no-runner", is_flag=True, help="Disable the invisible local runner.")
@click.option("--restart-runner", is_flag=True, help="Force restart the local runner.")
def run_cmd(
    prompt: Optional[str],
    model: Optional[str],
    app: Optional[str],
    policy: Optional[str],
    output_json: bool,
    stream: Optional[bool],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    install_runtime: Optional[bool],
    yes: bool,
    no_runner: bool,
    restart_runner: bool,
) -> None:
    """Run a one-shot inference request.

    The primary command for local AI inference. Resolves the model,
    downloads if needed, and executes locally or via cloud based on policy.

    \b
    Examples:
        octomil run "What can you help me with?"
        octomil run --model gemma3-1b "Explain transformers"
        cat prompt.txt | octomil run
        octomil run --json "Return a haiku about SQLite"
    """
    # Resolve input
    if prompt is None:
        prompt = _read_stdin_text()
    if not prompt:
        raise click.UsageError("Missing prompt. Pass a prompt or pipe stdin.")

    kernel = _kernel()
    runtime_install_attempted = _prepare_runtime_for_local_run(kernel, model, policy, app, install_runtime, yes)
    should_try_runner, runner_model, runner_policy = _capability_runner_target(kernel, "chat", model, policy, app)

    # Try invisible local runner only when the resolved policy prefers local execution.
    if _runner_enabled(no_runner) and should_try_runner and runner_model and _has_real_local_runtime(runner_model):
        runner_engine = _planner_engine_for_runner(runner_model, "chat", runner_policy or "local_first")
        runner_output = _try_runner_response(
            runner_model,
            prompt,
            engine=runner_engine,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            restart_runner=restart_runner,
            output_json=output_json,
        )
        if runner_output is not None:
            click.echo(runner_output)
            return

    # Default: stream if TTY and not JSON
    should_stream = stream if stream is not None else (_is_tty() and not output_json)

    if should_stream and not output_json:
        _run_with_runtime_install_retry(
            lambda: _stream_run(kernel, prompt, model, policy, app, temperature, max_output_tokens),
            install_runtime,
            yes,
            runtime_install_attempted,
        )
    else:
        result = _run_with_runtime_install_retry(
            lambda: kernel.create_response(
                prompt,
                model=model,
                policy=policy,
                app=app,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
            install_runtime,
            yes,
            runtime_install_attempted,
        )
        if output_json:
            click.echo(json.dumps(_result_to_dict(result), indent=2))
        else:
            _warn_if_cloud_execution(result)
            click.echo(result.output_text)


async def _stream_run(kernel, prompt, model, policy, app, temperature, max_output_tokens):
    final_result = None
    async for chunk in kernel.stream_response(
        prompt,
        model=model,
        policy=policy,
        app=app,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    ):
        if chunk.delta:
            click.echo(chunk.delta, nl=False)
        if chunk.done and chunk.result is not None:
            final_result = chunk.result
    click.echo()  # trailing newline
    if final_result is not None:
        _warn_if_cloud_execution(final_result)


# ---------------------------------------------------------------------------
# octomil embed
# ---------------------------------------------------------------------------


@click.command("embed")
@click.argument("inputs", nargs=-1)
@click.option("--model", "-m", default=None, help="Embedding model to use.")
@click.option("--app", default=None, help="App context (slug).")
@click.option("--policy", default=None, help="Serving policy preset.")
@click.option("--json", "output_json", is_flag=True, help="Output full JSON with vectors.")
@click.option("--jsonl", "output_jsonl", is_flag=True, help="Output NDJSON (one object per input).")
@click.option("--no-runner", is_flag=True, help="Disable the invisible local runner.")
@click.option("--restart-runner", is_flag=True, help="Force restart the local runner.")
def embed_cmd(
    inputs: tuple[str, ...],
    model: Optional[str],
    app: Optional[str],
    policy: Optional[str],
    output_json: bool,
    output_jsonl: bool,
    no_runner: bool,
    restart_runner: bool,
) -> None:
    """Generate embeddings for text or files.

    \b
    Examples:
        octomil embed "On-device AI inference at scale" --json
        octomil embed docs/*.md --json
        cat passages.txt | octomil embed --jsonl
    """
    from pathlib import Path

    # Resolve inputs
    text_inputs: list[str] = []
    if inputs:
        for inp in inputs:
            p = Path(inp)
            if p.is_file():
                text_inputs.append(p.read_text(encoding="utf-8", errors="replace"))
            else:
                text_inputs.append(inp)
    else:
        stdin_text = _read_stdin_text()
        if stdin_text:
            # Split lines for batch
            text_inputs = [line for line in stdin_text.strip().split("\n") if line.strip()]

    if not text_inputs:
        raise click.UsageError("Missing input. Pass text, files, or pipe stdin.")

    kernel = _kernel()
    should_try_runner, runner_model, runner_policy = _capability_runner_target(
        kernel,
        "embedding",
        model,
        policy,
        app,
    )

    # Try invisible local runner only when the resolved policy prefers local execution.
    if _runner_enabled(no_runner) and should_try_runner and runner_model and _has_real_local_runtime(runner_model):
        runner_engine = _planner_engine_for_runner(runner_model, "embedding", runner_policy or "local_first")
        runner_result = _try_runner_embed(
            runner_model,
            text_inputs,
            engine=runner_engine,
            restart_runner=restart_runner,
        )
        if runner_result is not None:
            if output_jsonl and "data" in runner_result:
                for item in runner_result["data"]:
                    click.echo(json.dumps(item))
            elif output_json:
                click.echo(json.dumps(runner_result, indent=2))
            else:
                count = len(runner_result.get("data", []))
                dims = len(runner_result["data"][0].get("embedding", [])) if count > 0 else 0
                click.echo(f"Generated {count} embedding(s) with {dims} dimensions. Use --json to print vectors.")
            return

    result = _run_async(
        kernel.create_embeddings(
            text_inputs,
            model=model,
            policy=policy,
            app=app,
        )
    )

    if output_jsonl and result.embeddings:
        for i, vec in enumerate(result.embeddings):
            click.echo(json.dumps({"index": i, "embedding": vec}))
    elif output_json:
        click.echo(json.dumps(_embed_result_to_dict(result), indent=2))
    else:
        count = len(result.embeddings) if result.embeddings else 0
        dims = result.dimensions or 0
        click.echo(f"Generated {count} embedding(s) with {dims} dimensions. Use --json to print vectors.")


# ---------------------------------------------------------------------------
# octomil transcribe
# ---------------------------------------------------------------------------


@click.command("transcribe")
@click.argument("audio_file", required=False, default=None, type=click.Path(exists=False))
@click.option("--model", "-m", default=None, help="Transcription model.")
@click.option("--app", default=None, help="App context (slug).")
@click.option("--policy", default=None, help="Serving policy preset.")
@click.option("--json", "output_json", is_flag=True, help="Output structured JSON.")
@click.option("--language", default=None, help="Language hint (BCP 47 code).")
@click.option("--no-runner", is_flag=True, help="Disable the invisible local runner.")
@click.option("--restart-runner", is_flag=True, help="Force restart the local runner.")
def transcribe_cmd(
    audio_file: Optional[str],
    model: Optional[str],
    app: Optional[str],
    policy: Optional[str],
    output_json: bool,
    language: Optional[str],
    no_runner: bool,
    restart_runner: bool,
) -> None:
    """Transcribe audio to text.

    \b
    Examples:
        octomil transcribe meeting.wav
        octomil transcribe interview.mp3 --json
        octomil transcribe --model whisper-small voice-note.m4a
    """
    from pathlib import Path

    # Resolve audio data
    audio_data: bytes
    if audio_file:
        p = Path(audio_file)
        if not p.is_file():
            raise click.UsageError(f"Audio file not found: {audio_file}")
        audio_data = p.read_bytes()
    elif not sys.stdin.isatty() and hasattr(sys.stdin, "buffer"):
        audio_data = sys.stdin.buffer.read()
        if not audio_data:
            raise click.UsageError("Missing audio input. Pass a file or pipe audio bytes.")
    else:
        raise click.UsageError("Missing audio input. Pass a file or pipe audio bytes.")

    kernel = _kernel()
    should_try_runner, runner_model, runner_policy = _capability_runner_target(
        kernel,
        "transcription",
        model,
        policy,
        app,
    )

    # Try invisible local runner only when the resolved policy prefers local execution.
    if (
        _runner_enabled(no_runner)
        and should_try_runner
        and runner_model
        and audio_file
        and _has_real_local_runtime(runner_model)
    ):
        runner_engine = _planner_engine_for_runner(runner_model, "transcription", runner_policy or "local_first")
        runner_result = _try_runner_transcribe(
            runner_model,
            audio_file,
            engine=runner_engine,
            restart_runner=restart_runner,
            language=language,
        )
        if runner_result is not None:
            if output_json:
                click.echo(json.dumps(runner_result, indent=2))
            else:
                click.echo(runner_result.get("text", ""))
            return

    result = _run_async(
        kernel.transcribe_audio(
            audio_data,
            model=model,
            policy=policy,
            app=app,
            language=language,
        )
    )

    if output_json:
        click.echo(json.dumps(_transcribe_result_to_dict(result), indent=2))
    else:
        click.echo(result.output_text)


# ---------------------------------------------------------------------------
# API-exact subcommand groups
# ---------------------------------------------------------------------------

# --- octomil responses ---


@click.group("responses")
def responses_group() -> None:
    """API-exact response commands (advanced)."""


@responses_group.command("create")
@click.option("--model", "-m", required=True, help="Model to use.")
@click.option("--input", "input_text", required=True, help="Input text.")
@click.option("--app", default=None, help="App context.")
@click.option("--policy", default=None, help="Serving policy preset.")
@click.option("--temperature", "-t", type=float, default=None)
@click.option("--max-output-tokens", type=int, default=None)
def responses_create(
    model: str,
    input_text: str,
    app: Optional[str],
    policy: Optional[str],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
) -> None:
    """Create a response (API-exact)."""
    kernel = _kernel()
    result = _run_async(
        kernel.create_response(
            input_text,
            model=model,
            policy=policy,
            app=app,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    )
    click.echo(json.dumps(_result_to_dict(result), indent=2))


@responses_group.command("stream")
@click.option("--model", "-m", required=True, help="Model to use.")
@click.option("--input", "input_text", required=True, help="Input text.")
@click.option("--app", default=None, help="App context.")
@click.option("--policy", default=None, help="Serving policy preset.")
@click.option("--temperature", "-t", type=float, default=None)
@click.option("--max-output-tokens", type=int, default=None)
def responses_stream(
    model: str,
    input_text: str,
    app: Optional[str],
    policy: Optional[str],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
) -> None:
    """Stream a response (API-exact)."""
    kernel = _kernel()
    _run_async(_stream_run(kernel, input_text, model, policy, app, temperature, max_output_tokens))


# --- octomil embeddings ---


@click.group("embeddings")
def embeddings_group() -> None:
    """API-exact embedding commands (advanced)."""


@embeddings_group.command("create")
@click.option("--model", "-m", required=True, help="Embedding model.")
@click.option("--input", "input_text", required=True, help="Text to embed.")
@click.option("--app", default=None, help="App context.")
@click.option("--policy", default=None, help="Serving policy preset.")
def embeddings_create(
    model: str,
    input_text: str,
    app: Optional[str],
    policy: Optional[str],
) -> None:
    """Create embeddings (API-exact)."""
    kernel = _kernel()
    result = _run_async(
        kernel.create_embeddings(
            [input_text],
            model=model,
            policy=policy,
            app=app,
        )
    )
    click.echo(json.dumps(_embed_result_to_dict(result), indent=2))


# --- octomil audio ---


@click.group("audio")
def audio_group() -> None:
    """Audio API commands (advanced)."""


@click.group("transcriptions")
def transcriptions_group() -> None:
    """Audio transcription commands."""


@transcriptions_group.command("create")
@click.option("--model", "-m", default=None, help="Transcription model.")
@click.option("--file", "audio_file", required=True, type=click.Path(exists=True), help="Audio file path.")
@click.option("--language", default=None, help="Language hint.")
@click.option("--app", default=None, help="App context.")
@click.option("--policy", default=None, help="Serving policy preset.")
def audio_transcriptions_create(
    model: Optional[str],
    audio_file: str,
    language: Optional[str],
    app: Optional[str],
    policy: Optional[str],
) -> None:
    """Create audio transcription (API-exact)."""
    from pathlib import Path

    audio_data = Path(audio_file).read_bytes()
    kernel = _kernel()
    result = _run_async(
        kernel.transcribe_audio(
            audio_data,
            model=model,
            policy=policy,
            app=app,
            language=language,
        )
    )
    click.echo(json.dumps(_transcribe_result_to_dict(result), indent=2))


audio_group.add_command(transcriptions_group)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _route_metadata_to_dict(route) -> dict:
    """Serialize RouteMetadata to a JSON-safe dict matching the contract wire format."""
    if route is None:
        return {}
    d: dict = {"status": route.status}

    if route.execution is not None:
        d["execution"] = {
            "locality": route.execution.locality,
            "mode": route.execution.mode,
            "engine": route.execution.engine,
        }
    else:
        d["execution"] = None

    d["model"] = {
        "requested": {
            "ref": route.model.requested.ref,
            "kind": route.model.requested.kind,
            "capability": route.model.requested.capability,
        },
        "resolved": None,
    }
    if route.model.resolved is not None:
        d["model"]["resolved"] = {
            "id": route.model.resolved.id,
            "slug": route.model.resolved.slug,
            "version_id": route.model.resolved.version_id,
            "variant_id": route.model.resolved.variant_id,
        }

    if route.artifact is not None:
        d["artifact"] = {
            "id": route.artifact.id,
            "version": route.artifact.version,
            "format": route.artifact.format,
            "digest": route.artifact.digest,
            "cache": {
                "status": route.artifact.cache.status,
                "managed_by": route.artifact.cache.managed_by,
            },
        }
    else:
        d["artifact"] = None

    d["planner"] = {"source": route.planner.source}
    d["fallback"] = {"used": route.fallback.used}
    d["reason"] = {"code": route.reason.code, "message": route.reason.message}

    return d


def _result_to_dict(result) -> dict:
    d = {
        "id": result.id,
        "model": result.model,
        "capability": result.capability,
        "locality": result.locality,
        "fallback_used": result.fallback_used,
        "output_text": result.output_text,
    }
    if result.usage:
        d["usage"] = result.usage
    if result.route is not None:
        d["route"] = _route_metadata_to_dict(result.route)
    return d


def _embed_result_to_dict(result) -> dict:
    d = {
        "model": result.model,
        "capability": result.capability,
        "locality": result.locality,
        "fallback_used": result.fallback_used,
    }
    if result.embeddings:
        d["data"] = [{"index": i, "embedding": v} for i, v in enumerate(result.embeddings)]
    if result.usage:
        d["usage"] = result.usage
    if result.route is not None:
        d["route"] = _route_metadata_to_dict(result.route)
    return d


def _transcribe_result_to_dict(result) -> dict:
    d = {
        "model": result.model,
        "capability": result.capability,
        "locality": result.locality,
        "fallback_used": result.fallback_used,
        "text": result.output_text,
    }
    if result.segments:
        d["segments"] = result.segments
    if result.route is not None:
        d["route"] = _route_metadata_to_dict(result.route)
    return d


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    """Register all inference commands onto the main CLI group."""
    # Primary task verbs
    cli.add_command(run_cmd)
    cli.add_command(embed_cmd)
    cli.add_command(transcribe_cmd)

    # API-exact advanced groups
    cli.add_command(responses_group)
    cli.add_command(embeddings_group)
    cli.add_command(audio_group)
