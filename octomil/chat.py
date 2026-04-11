"""Interactive chat REPL for octomil.

Provides ``run_chat_repl()`` which drives a terminal conversation against
a local ``octomil serve`` instance.  Inference is delegated to an
``OctomilResponses`` instance via ``stream_chat_via_responses()``.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Iterator

import click

if TYPE_CHECKING:
    from .execution.kernel import ExecutionKernel
    from .responses.responses import OctomilResponses


# ---------------------------------------------------------------------------
# Responses-backed chat streaming
# ---------------------------------------------------------------------------


def _messages_to_input(messages: list[dict[str, str]]) -> list[Any]:
    """Convert OpenAI-style chat messages to Response API InputItems."""
    from .responses.types import SystemInput, TextContent, UserInput

    items: list[Any] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            items.append(SystemInput(content=content))
        elif role == "assistant":
            from .responses.types import AssistantInput

            items.append(AssistantInput(content=[TextContent(text=content)]))
        else:
            items.append(UserInput(content=[TextContent(text=content)]))
    return items


def _response_to_chat_chunk(text_delta: str) -> dict[str, Any]:
    """Build an OpenAI-compatible SSE chunk dict from a text delta."""
    return {
        "choices": [
            {
                "delta": {"content": text_delta},
                "index": 0,
            }
        ]
    }


def stream_chat_via_responses(
    responses: OctomilResponses,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Iterator[dict[str, Any]]:
    """Stream chat completions via OctomilResponses.

    Converts chat messages to a ResponseRequest, streams via
    ``responses.stream()``, and yields OpenAI-compatible SSE chunk dicts.
    """
    from .responses.types import ResponseRequest, TextDeltaEvent

    input_items = _messages_to_input(messages)
    request = ResponseRequest(
        model=model,
        input=input_items,
        stream=True,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    async def _collect() -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        async for event in responses.stream(request):
            if isinstance(event, TextDeltaEvent):
                chunks.append(_response_to_chat_chunk(event.delta))
        return chunks

    loop = asyncio.new_event_loop()
    try:
        chunks = loop.run_until_complete(_collect())
    finally:
        loop.close()

    yield from chunks


def stream_chat_via_kernel(
    kernel: ExecutionKernel,
    model: str,
    messages: list[dict[str, str]],
    *,
    policy: str | None = None,
    app: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Iterator[dict[str, Any]]:
    """Stream chat completions via the shared execution kernel."""

    async def _collect() -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        async for event in kernel.stream_chat_messages(
            messages,
            model=model,
            policy=policy,
            app=app,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ):
            if event.delta:
                chunks.append(_response_to_chat_chunk(event.delta))
        return chunks

    loop = asyncio.new_event_loop()
    try:
        chunks = loop.run_until_complete(_collect())
    finally:
        loop.close()

    yield from chunks


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def _read_input() -> str | None:
    """Read a line of user input. Returns ``None`` on EOF/interrupt."""
    try:
        return input(">>> ")
    except (EOFError, KeyboardInterrupt):
        return None


def run_chat_repl(
    model: str,
    responses: OctomilResponses | ExecutionKernel,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    policy: str | None = None,
    app: str | None = None,
    stream_fn: Any = None,
    _input_fn: Any = None,
) -> None:
    """Main interactive REPL loop.

    Parameters
    ----------
    model:
        Model name to pass in the completions request.
    responses:
        ``OctomilResponses`` instance used for inference.
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
    stream: Any = stream_fn or stream_chat_via_responses

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
            stream_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if stream is stream_chat_via_kernel:
                stream_kwargs["policy"] = policy
                stream_kwargs["app"] = app

            chunk_iter = stream(
                responses,
                model,
                messages,
                **stream_kwargs,
            )
            for chunk in chunk_iter:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    click.echo(content, nl=False)
                    full_response += content
                    token_count += 1
        except RuntimeError as exc:
            click.secho(f"\nError: {exc}", fg="red", err=True)
            # Remove the unanswered user message so conversation stays clean
            messages.pop()
            continue
        except Exception as exc:
            click.secho(f"\nError: {_format_chat_error(exc)}", fg="red", err=True)
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


def _format_chat_error(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    first_line = text.splitlines()[0].strip()
    return first_line or exc.__class__.__name__
