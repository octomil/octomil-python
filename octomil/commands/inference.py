"""Direct local invocation commands: run, embed, transcribe.

Also registers advanced API-exact subcommand groups:
  octomil responses create|stream
  octomil embeddings create
  octomil audio transcriptions create

All commands use the shared execution kernel.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOCAL_RUNTIME_HINT = (
    "Install a local runtime for on-device execution:\n"
    "  pip install 'octomil[mlx]'      # Apple Silicon\n"
    "  pip install 'octomil[llama]'    # Cross-platform\n"
    "Or set OCTOMIL_SERVER_KEY to allow hosted cloud fallback."
)


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
def run_cmd(
    prompt: Optional[str],
    model: Optional[str],
    app: Optional[str],
    policy: Optional[str],
    output_json: bool,
    stream: Optional[bool],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
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

    # Default: stream if TTY and not JSON
    should_stream = stream if stream is not None else (_is_tty() and not output_json)

    if should_stream and not output_json:
        _run_async(_stream_run(kernel, prompt, model, policy, app, temperature, max_output_tokens))
    else:
        result = _run_async(
            kernel.create_response(
                prompt,
                model=model,
                policy=policy,
                app=app,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
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
def embed_cmd(
    inputs: tuple[str, ...],
    model: Optional[str],
    app: Optional[str],
    policy: Optional[str],
    output_json: bool,
    output_jsonl: bool,
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
def transcribe_cmd(
    audio_file: Optional[str],
    model: Optional[str],
    app: Optional[str],
    policy: Optional[str],
    output_json: bool,
    language: Optional[str],
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
