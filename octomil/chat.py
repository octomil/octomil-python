"""Interactive chat REPL for octomil.

Provides ``run_chat_repl()`` which drives a terminal conversation against
a local ``octomil serve`` instance via the OpenAI-compatible
``/v1/chat/completions`` endpoint with SSE streaming.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator, Optional

import click
import httpx


def stream_chat(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Iterator[dict[str, Any]]:
    """Stream chat completions from local octomil serve.

    Yields parsed SSE ``data:`` frames from ``/v1/chat/completions``.
    Adapted from ``octomil/demos/code_assistant.py``.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json=payload,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Server returned {response.status_code}: "
                    f"{response.read().decode(errors='replace')}"
                )
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


def _read_input() -> str | None:
    """Read a line of user input. Returns ``None`` on EOF/interrupt."""
    try:
        return input(">>> ")
    except (EOFError, KeyboardInterrupt):
        return None


def run_chat_repl(
    url: str,
    model: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    _input_fn: Any = None,
) -> None:
    """Main interactive REPL loop.

    Parameters
    ----------
    url:
        Base URL of the octomil serve instance (e.g. ``http://localhost:8080``).
    model:
        Model name to pass in the completions request.
    system_prompt:
        Optional system message prepended to the conversation.
    temperature:
        Sampling temperature.
    max_tokens:
        Max tokens per assistant turn.
    _input_fn:
        Override for ``_read_input`` (used in tests).
    """
    read_input = _input_fn or _read_input

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    click.echo(f"Chatting with {model}. Type /exit to quit, /clear to reset.\n")

    while True:
        user_input = read_input()
        if user_input is None:
            # EOF or Ctrl-C
            break

        stripped = user_input.strip()

        if stripped == "/exit":
            break
        if stripped == "/clear":
            messages = [m for m in messages if m["role"] == "system"]
            click.echo("Conversation cleared.")
            continue
        if not stripped:
            continue

        messages.append({"role": "user", "content": stripped})

        # Stream response
        start = time.perf_counter()
        full_response = ""
        token_count = 0

        try:
            for chunk in stream_chat(
                url,
                model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    click.echo(content, nl=False)
                    full_response += content
                    token_count += 1
        except (httpx.ConnectError, httpx.RemoteProtocolError, RuntimeError) as exc:
            click.secho(f"\nError: {exc}", fg="red", err=True)
            # Remove the unanswered user message so conversation stays clean
            messages.pop()
            continue

        elapsed = time.perf_counter() - start
        click.echo()  # newline after streaming
        tok_per_sec = token_count / max(elapsed, 0.01)
        click.echo(
            click.style(
                f"  [{token_count} tokens, {elapsed:.1f}s, {tok_per_sec:.1f} tok/s]",
                dim=True,
            )
        )
        click.echo()

        messages.append({"role": "assistant", "content": full_response})
