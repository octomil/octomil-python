"""Shared helpers for CLI commands.

Extracted from cli.py to avoid circular imports and allow command modules
to share common utilities (auth, client creation, model completion, etc.).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import click


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


def _complete_model_name(ctx, param, incomplete):
    """Shell completion callback for model name arguments/options."""
    from click.shell_completion import CompletionItem
    from .models.catalog import CATALOG, MODEL_ALIASES
    from .sources.resolver import _MODEL_ALIASES as _RESOLVER_ALIASES

    names = set(CATALOG) | set(MODEL_ALIASES) | set(_RESOLVER_ALIASES)

    # Best-effort: include server-side model IDs
    try:
        import httpx

        api_key = os.environ.get("OCTOMIL_API_KEY", "")
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
Octomil — run ML on any device

  Get started:
    1. octomil login                         authenticate
    2. octomil serve phi-4-mini              local inference server
    3. octomil push phi-4-mini --version 1.0.0
                                             download, convert, push
    4. octomil deploy phi-4-mini --phone     send to device

  Useful commands:
    octomil benchmark <model>                measure tokens/s
    octomil models                           list available models
    octomil dashboard                        open web dashboard
    octomil launch <agent>                   launch coding agent

  Run octomil <command> --help for details.
  Docs: https://docs.octomil.com
"""
