"""Shared helpers for CLI commands.

Extracted from cli.py to avoid circular imports and allow command modules
to share common utilities (auth, client creation, model completion, etc.).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Optional

import click
import httpx

_logger = logging.getLogger(__name__)

# Status codes safe to retry (transient server errors + rate limiting).
_RETRYABLE_STATUS_CODES = {502, 503, 504, 429}


def http_request(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    timeout: float = 15.0,
    headers: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> httpx.Response:
    """HTTP request with automatic retry on transient failures.

    Retries on connection errors, timeouts, 502/503/504/429, and generic
    404s (FastAPI cold-start artifact).  Use this instead of raw
    ``httpx.get()`` / ``httpx.post()`` in CLI commands.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.request(method, url, headers=headers, **kwargs)

            if resp.status_code < 400:
                return resp

            # Generic 404 from cold start — retry.
            if (
                resp.status_code == 404
                and attempt < max_retries - 1
                and resp.text.strip() in ('{"detail":"Not Found"}', "Not Found")
            ):
                wait = backoff_base * (2 ** attempt)
                _logger.debug("Generic 404 on %s %s, retrying in %.1fs", method, url, wait)
                time.sleep(wait)
                continue

            # Retryable server error — backoff and retry.
            if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt)
                _logger.debug("HTTP %d on %s %s, retrying in %.1fs", resp.status_code, method, url, wait)
                time.sleep(wait)
                continue

            return resp  # Non-retryable — let caller handle

        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt)
                click.echo(
                    f"  Connection failed, retrying ({attempt + 1}/{max_retries})...",
                    err=True,
                )
                time.sleep(wait)
            else:
                click.echo(
                    f"Error: failed to connect to Octomil after {max_retries} attempts: {exc}",
                    err=True,
                )
                sys.exit(1)

    # Should not reach here.
    click.echo(f"Error: request failed after {max_retries} attempts", err=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Auth / credentials helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Read API key from env or ~/.octomil/credentials (JSON or legacy format)."""
    import json

    key = os.environ.get("OCTOMIL_API_KEY", "")
    if not key:
        config_path = os.path.expanduser("~/.octomil/credentials")
        if os.path.exists(config_path):
            with open(config_path) as f:
                raw = f.read().strip()
            if not raw:
                return ""
            try:
                data = json.loads(raw)
                key = data.get("api_key", "")
            except (json.JSONDecodeError, ValueError):
                for line in raw.splitlines():
                    if line.startswith("api_key="):
                        key = line.split("=", 1)[1].strip()
                        break
    return key


def _require_api_key() -> str:
    key = _get_api_key()
    if not key:
        click.echo("No API key found. Run `octomil login` first.", err=True)
        sys.exit(1)
    return key


def _get_org_id() -> str | None:
    """Read org_id from env or ~/.octomil/credentials."""
    import json

    oid = os.environ.get("OCTOMIL_ORG_ID", "")
    if not oid:
        config_path = os.path.expanduser("~/.octomil/credentials")
        if os.path.exists(config_path):
            with open(config_path) as f:
                raw = f.read().strip()
            try:
                data = json.loads(raw)
                oid = data.get("org_id", "")
            except (json.JSONDecodeError, ValueError):
                pass
    return oid or None


def _get_client():  # type: ignore[no-untyped-def]
    from .client import Client

    return Client(api_key=_require_api_key(), org_id=_get_org_id())


def _get_telemetry_reporter():  # type: ignore[no-untyped-def]
    """Best-effort TelemetryReporter for funnel events. Returns None if no API key."""
    from .telemetry import TelemetryReporter

    api_key = _get_api_key()
    if not api_key:
        return None
    api_base: str = (
        os.environ.get("OCTOMIL_API_URL")
        or os.environ.get("OCTOMIL_API_BASE")
        or "https://api.octomil.com/api/v1"
    )
    return TelemetryReporter(api_key=api_key, api_base=api_base)


def _save_credentials(
    api_key: str,
    org: Optional[str] = None,
    org_id: Optional[str] = None,
) -> None:
    """Save credentials to ~/.octomil/credentials as JSON with restrictive permissions."""
    import json

    config_dir = os.path.expanduser("~/.octomil")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "credentials")
    data: dict[str, str] = {"api_key": api_key}
    if org:
        data["org_name"] = org
    if org_id:
        data["org_id"] = org_id
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.chmod(config_path, 0o600)


# ---------------------------------------------------------------------------
# Enterprise helpers
# ---------------------------------------------------------------------------


def _get_enterprise_client(api_base: Optional[str] = None):  # type: ignore[no-untyped-def]
    """Build an EnterpriseClient with the current API key."""
    from .enterprise import EnterpriseClient

    key = _require_api_key()
    return EnterpriseClient(api_key=key, api_base=api_base)


def _require_org_id() -> str:
    """Read org_id from config/env or exit."""
    from .enterprise import get_org_id

    org_id = get_org_id()
    if not org_id:
        click.echo(
            "No org_id configured. Run `octomil init <name>` first, or "
            "set OCTOMIL_ORG_ID.",
            err=True,
        )
        sys.exit(1)
    return org_id


# ---------------------------------------------------------------------------
# Auto-optimization helpers
# ---------------------------------------------------------------------------

# Quant variants that can appear after ":" in a model tag
_KNOWN_QUANT_SUFFIXES = frozenset(
    {
        "q2_k",
        "q3_k_s",
        "q3_k_m",
        "q4_0",
        "q4_k_s",
        "q4_k_m",
        "q5_0",
        "q5_k_s",
        "q5_k_m",
        "q6_k",
        "q8_0",
        "4bit",
        "8bit",
        "fp16",
        "f16",
        "f32",
    }
)


# ~200 most common model names for offline shell autocomplete.
# Covers 99%+ of what users type. The API fallback below only adds
# user-specific custom model IDs from their registry.
_KNOWN_MODEL_NAMES = frozenset(
    {
        # Llama (Meta)
        "llama2", "llama2-7b", "llama2-13b", "llama2-70b", "llama2-uncensored",
        "llama3", "llama3-8b", "llama3-70b",
        "llama3.1", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b",
        "llama3.2", "llama3.2-1b", "llama3.2-3b", "llama3.2-vision",
        "llama3.3", "llama3.3-70b",
        "llama4", "llama4-scout", "llama4-maverick",
        "llama-guard3",
        # Octomil catalog aliases (dash style)
        "llama-1b", "llama-3b", "llama-8b",
        "llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b",
        # CodeLlama
        "codellama", "codellama-7b", "codellama-13b", "codellama-34b",
        "codellama-python", "codellama-instruct",
        # Gemma (Google)
        "gemma", "gemma-2b", "gemma-7b",
        "gemma2", "gemma2-2b", "gemma2-9b", "gemma2-27b",
        "gemma3", "gemma3-1b", "gemma3-4b", "gemma3-12b", "gemma3-27b",
        "gemma3n", "gemma3n-e2b", "gemma3n-e4b",
        "codegemma",
        # Octomil catalog keys/aliases
        "gemma-1b", "gemma-4b", "gemma-12b", "gemma-27b",
        "gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b", "gemma-3b",
        # Phi (Microsoft)
        "phi", "phi-2", "phi3", "phi3-mini", "phi3.5",
        "phi4", "phi4-mini", "phi4-reasoning", "phi4-mini-reasoning",
        "phi-4", "phi-mini", "phi-4-mini", "phi-3.5-mini", "phi-mini-3.8b",
        # Qwen (Alibaba)
        "qwen", "qwen2", "qwen2.5",
        "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
        "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b", "qwen2.5-coder",
        "qwen3", "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b",
        "qwen3-14b", "qwen3-32b", "qwen3-coder",
        "qwen3.5", "qwq", "codeqwen",
        # Octomil catalog keys/aliases
        "qwen-1.5b", "qwen-3b", "qwen-7b", "qwen-moe-14b",
        "qwen-2.5-1.5b", "qwen-2.5-3b", "qwen-2.5-7b",
        "qwen-1.5-moe", "qwen-moe", "qwen3-1b",
        # Mistral / Mixtral
        "mistral", "mistral-7b", "mistral-nemo", "mistral-small",
        "mistral-small3.1", "mistral-large",
        "mixtral", "mixtral-8x7b", "mixtral-8x22b",
        "mixtral-instruct", "mixtral-8x22b-instruct",
        "codestral", "devstral", "magistral", "mathstral", "ministral-3",
        # DeepSeek
        "deepseek-r1", "deepseek-r1-1.5b", "deepseek-r1-7b", "deepseek-r1-8b",
        "deepseek-r1-14b", "deepseek-r1-32b", "deepseek-r1-70b",
        "deepseek-v2", "deepseek-v2-lite", "deepseek-v2.5",
        "deepseek-v3", "deepseek-v3.1",
        "deepseek-coder", "deepseek-coder-v2",
        "deepseek-v2-lite-chat",
        # Whisper (OpenAI)
        "whisper-tiny", "whisper-base", "whisper-small",
        "whisper-medium", "whisper-large", "whisper-large-v2",
        "whisper-large-v3", "whisper-turbo",
        # Nous / Hermes
        "nous-hermes", "nous-hermes2", "hermes3", "openhermes",
        # Dolphin
        "dolphin3", "dolphin-llama3", "dolphin-mistral", "dolphin-phi",
        # Yi (01.AI)
        "yi", "yi-coder", "yi-34b",
        # Falcon (TII)
        "falcon", "falcon2", "falcon3",
        # Command-R (Cohere)
        "command-r", "command-r-plus", "command-r7b",
        # StarCoder
        "starcoder", "starcoder2", "starcoder2-3b", "starcoder2-15b",
        # SmolLM (HuggingFace)
        "smollm", "smollm2", "smollm-360m", "smollm-1.7b",
        "smollm2-360m", "smollm2-1.7b",
        # TinyLlama
        "tinyllama",
        # Vicuna / WizardLM
        "vicuna", "wizardlm2", "wizardcoder",
        # GLM (Zhipu)
        "glm4", "glm-4.7",
        # Granite (IBM)
        "granite-code", "granite3.2", "granite4",
        # Aya (Cohere)
        "aya-expanse",
        # OLMo (AI2)
        "olmo2",
        # Multimodal / Vision
        "llava", "llava-llama3", "moondream", "minicpm-v",
        # Embedding
        "nomic-embed-text", "mxbai-embed-large", "bge-m3", "all-minilm",
        "snowflake-arctic-embed",
        # Stable / Flux
        "stable-code", "stablelm2",
        # Other notable
        "dbrx", "dbrx-instruct",
        "nemotron", "nemotron-mini",
        "neural-chat", "openchat", "solar", "solar-pro",
        "sqlcoder", "zephyr", "orca-mini", "orca2",
        "tinydolphin", "internlm2",
    }
)


def _complete_model_name(ctx, param, incomplete):
    """Shell completion callback for model name arguments/options."""
    from click.shell_completion import CompletionItem

    names = set(_KNOWN_MODEL_NAMES)

    # Best-effort: include user's custom models from their registry.
    try:
        import httpx

        api_key = _get_api_key()
        api_base = (
            os.environ.get("OCTOMIL_API_URL")
            or os.environ.get("OCTOMIL_API_BASE")
            or "https://api.octomil.com/api/v1"
        )
        if api_key:
            resp = httpx.get(
                f"{api_base}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=2.0,
            )
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    mid = m.get("model_id") or m.get("id", "")
                    if mid:
                        names.add(mid)
    except Exception:
        pass

    return [CompletionItem(n) for n in sorted(names) if n.startswith(incomplete)]


def _has_explicit_quant(model_tag: str) -> bool:
    """Check if the model tag already specifies a quantization variant."""
    if ":" not in model_tag:
        return False
    variant = model_tag.rsplit(":", 1)[1].lower().replace("-", "_")
    return variant in _KNOWN_QUANT_SUFFIXES


def _auto_optimize(model_tag: str, context_length: int = 4096) -> str | None:
    """Run hardware-aware optimization and print results.

    Returns the recommended quantization string (e.g. "Q6_K"), or ``None``
    if the model size cannot be determined.  Does NOT modify *model_tag*.
    """
    from .hardware import detect_hardware
    from .model_optimizer import ModelOptimizer, _resolve_model_size

    model_size_b = _resolve_model_size(model_tag)
    if model_size_b is None:
        return None

    hw = detect_hardware()
    opt = ModelOptimizer(hw)
    config = opt.pick_quant_and_offload(model_size_b, context_length)
    speed = opt.predict_speed(model_size_b, config)

    click.echo()
    click.secho("  Hardware optimization", bold=True)
    click.echo(f"    Quantization: {config.quantization}")
    click.echo(
        f"    Strategy: {config.strategy.value} ({config.gpu_layers} GPU layers)"
    )
    click.echo(f"    VRAM: {config.vram_gb:.1f} GB  RAM: {config.ram_gb:.1f} GB")
    click.echo(
        f"    Est. speed: {speed.tokens_per_second:.1f} tok/s"
        f" ({speed.confidence} confidence)"
    )
    if config.warning:
        click.secho(f"    Warning: {config.warning}", fg="yellow")

    return config.quantization


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------

WELCOME_MESSAGE = """\
Octomil — on-device AI for consumer apps

  Get started:
    1. octomil login                         authenticate
    2. octomil serve phi-4-mini              local inference server
    3. octomil benchmark phi-4-mini          measure device performance
    4. octomil deploy phi-4-mini --phone     deploy to device fleet

  Useful commands:
    octomil models                           list available models
    octomil push <file> --version 1.0.0      upload a model
    octomil dashboard                        open web dashboard
    octomil launch <agent>                   launch coding agent

  Run octomil <command> --help for details.
  Docs: https://docs.octomil.com
"""
