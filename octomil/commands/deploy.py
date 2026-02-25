"""Deploy, rollback, pair, status, and dashboard commands."""

from __future__ import annotations

import os
import sys
import webbrowser
from typing import Optional

import click

from octomil.cli_helpers import (
    _complete_model_name,
    _get_api_key,
    _get_client,
    _get_org_id,
    _get_telemetry_reporter,
    _require_api_key,
)


def register(cli: click.Group) -> None:
    cli.add_command(deploy)
    cli.add_command(rollback)
    cli.add_command(pair)
    cli.add_command(status)
    cli.add_command(dashboard)


# ---------------------------------------------------------------------------
# octomil deploy
# ---------------------------------------------------------------------------


@click.command()
@click.argument("name", shell_complete=_complete_model_name)
@click.option(
    "--version", "-v", default=None, help="Version to deploy. Defaults to latest."
)
@click.option("--phone", is_flag=True, help="Deploy to your connected phone.")
@click.option("--rollout", "-r", default=100, help="Rollout percentage (1-100).")
@click.option(
    "--strategy",
    "-s",
    default="canary",
    type=click.Choice(["canary", "immediate", "blue_green"]),
    help="Rollout strategy.",
)
@click.option(
    "--target",
    "-t",
    default=None,
    help="Comma-separated target formats: ios, android.",
)
@click.option(
    "--devices",
    default=None,
    help="Comma-separated device IDs to deploy to.",
)
@click.option(
    "--group",
    "-g",
    default=None,
    help="Device group name to deploy to.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview what would happen without deploying.",
)
def deploy(
    name: str,
    version: Optional[str],
    phone: bool,
    rollout: int,
    strategy: str,
    target: Optional[str],
    devices: Optional[str],
    group: Optional[str],
    dry_run: bool,
) -> None:
    """Deploy a model to edge devices.

    Deploys NAME at VERSION to devices. Use --phone for quick
    phone deployment, --devices/--group for targeted deployment,
    or --rollout for fleet percentage rollouts.

    Use the ollama:// URI scheme to deploy directly from your
    local Ollama cache:

    \b
    Examples:

        octomil deploy gemma-1b --phone
        octomil deploy ollama://llama3.2 --phone
        octomil deploy ollama://gemma:2b --phone
        octomil deploy sentiment-v1 --rollout 10 --strategy canary
        octomil deploy gemma-1b --devices device_1,device_2
        octomil deploy gemma-1b --group production
        octomil deploy gemma-1b --group production --dry-run
    """
    # Handle ollama:// URI scheme
    ollama_source_result = None
    if name.startswith("ollama://"):
        ollama_ref = name[len("ollama://") :]
        if not ollama_ref:
            click.echo(
                "Error: ollama:// URI requires a model name, e.g. ollama://llama3.2",
                err=True,
            )
            sys.exit(1)
        click.echo(f"Resolving ollama model: {ollama_ref}")

        from octomil.sources.ollama import OllamaSource

        source = OllamaSource()
        try:
            ollama_source_result = source.resolve(ollama_ref)
        except RuntimeError as exc:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
            sys.exit(1)
        click.echo(click.style(f"  Found: {ollama_source_result.path}", fg="green"))
        # Use the base model name for display / API calls
        name = ollama_ref.split(":")[0]

    if phone:
        import httpx

        from octomil.qr import build_deep_link, render_qr_terminal

        # Detect ollama models before deploying
        from octomil.ollama import get_ollama_model

        ollama_model = get_ollama_model(name)
        if ollama_model:
            click.echo(
                f"Detected ollama model: {ollama_model.name} "
                f"({ollama_model.size_display}, {ollama_model.quantization})"
            )

        # Try mDNS network scan before falling back to QR code
        from octomil.discovery import scan_for_devices

        click.echo("Scanning for Octomil devices on local network...")
        discovered = scan_for_devices(timeout=5.0)

        if discovered:
            if len(discovered) == 1:
                dev = discovered[0]
                click.echo(
                    click.style(
                        f"  \u2713 Found: {dev.name} ({dev.platform}, {dev.ip})",
                        fg="green",
                    )
                )
                if click.confirm(f"\nDeploy {name} to this device?", default=True):
                    # Direct deployment — create pairing session targeting this device
                    pass
            else:
                click.echo(f"  Found {len(discovered)} devices:")
                for i, dev in enumerate(discovered, 1):
                    click.echo(f"    {i}. {dev.name} ({dev.platform}, {dev.ip})")
                choice = click.prompt("Select device", type=int, default=1)
                dev = discovered[choice - 1]
                # Deploy to selected device (pairing session will target it)
        else:
            click.echo("  No devices found. Falling back to QR code pairing.\n")

        api_key = _require_api_key()
        api_base: str = (
            os.environ.get("OCTOMIL_API_URL")
            or os.environ.get("OCTOMIL_API_BASE")
            or "https://api.octomil.com/api/v1"
        )
        dashboard_url = os.environ.get(
            "OCTOMIL_DASHBOARD_URL", "https://app.octomil.com"
        )
        headers = {"Authorization": f"Bearer {api_key}"}

        # Ensure model exists in registry — auto-download, convert, push if needed
        check_resp = httpx.get(
            f"{api_base}/models/{name}",
            headers=headers,
            timeout=10.0,
        )
        if check_resp.status_code == 404:
            click.echo(f"Model '{name}' not in registry — auto-importing...")
            from octomil.sources.resolver import resolve_and_download

            try:
                resolved_path = resolve_and_download(name)
            except Exception as exc:
                click.echo(
                    click.style(f"  Error resolving model: {exc}", fg="red"),
                    err=True,
                )
                sys.exit(1)
            click.echo(click.style(f"  Downloaded to {resolved_path}", fg="green"))

            client = _get_client()
            effective_version = version or "1.0.0"
            resolved_name = name.split("/")[-1].split(":")[-1]
            try:
                client.push(
                    resolved_path,
                    name=resolved_name,
                    version=effective_version,
                )
                click.echo(
                    click.style(
                        f"  Pushed {resolved_name} v{effective_version}", fg="green"
                    )
                )
                name = resolved_name
            except Exception as exc:
                # 402 = plan limit
                if "402" in str(exc) or "limit" in str(exc).lower():
                    click.echo(
                        click.style(
                            "  Plan limit reached — upgrade at "
                            "https://app.octomil.com/settings/billing",
                            fg="red",
                        ),
                        err=True,
                    )
                else:
                    click.echo(
                        click.style(f"  Push failed: {exc}", fg="red"), err=True
                    )
                sys.exit(1)

        click.echo(f"Creating pairing session for {name}...")
        resp = httpx.post(
            f"{api_base}/deploy/pair",
            json={"model_name": name, "model_version": version},
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code >= 400:
            click.echo(
                f"Failed to create pairing session: {resp.status_code}", err=True
            )
            click.echo(resp.text, err=True)
            sys.exit(1)

        session = resp.json()
        code = session["code"]

        # Build the deep link URL for the Octomil mobile app.
        # The octomil:// scheme is handled by DeepLinkHandler in the
        # iOS and Android SDKs, opening the app directly to pairing.
        pair_url = build_deep_link(token=code, host=api_base)

        # Render QR code in a styled box
        qr_art = render_qr_terminal(pair_url)
        qr_lines = qr_art.split("\n")
        # Determine box width: widest QR line or minimum for text
        max_qr_width = max((len(line) for line in qr_lines), default=0)
        box_inner = max(max_qr_width + 4, 45)

        click.echo()
        click.echo("\u256d" + "\u2500" * box_inner + "\u256e")
        click.echo(
            "\u2502"
            + "  Scan this QR code with your phone camera:".ljust(box_inner)
            + "\u2502"
        )
        click.echo("\u2502" + " " * box_inner + "\u2502")
        for line in qr_lines:
            padded = ("  " + line).ljust(box_inner)
            click.echo("\u2502" + padded + "\u2502")
        click.echo("\u2502" + " " * box_inner + "\u2502")
        click.echo(
            "\u2502" + f"  Or open manually: {pair_url}".ljust(box_inner) + "\u2502"
        )
        click.echo("\u2502" + "  Expires in 5 minutes".ljust(box_inner) + "\u2502")
        click.echo("\u2570" + "\u2500" * box_inner + "\u256f")
        click.echo()

        webbrowser.open(pair_url)

        click.echo("Waiting for device to connect (Ctrl+C to cancel)...")
        last_status = ""
        try:
            while True:
                import time

                time.sleep(2)
                poll = httpx.get(f"{api_base}/deploy/pair/{code}", timeout=5.0)
                if poll.status_code != 200:
                    continue
                data = poll.json()
                status_val = data.get("status", "pending")
                if status_val == last_status:
                    continue
                last_status = status_val
                if status_val == "connected":
                    device = data.get("device_name") or data.get("device_id", "unknown")
                    platform = data.get("device_platform", "unknown")
                    click.echo(
                        click.style(
                            f"  \u2713 Device connected: {device} ({platform})",
                            fg="green",
                        )
                    )
                elif status_val == "converting":
                    click.echo(
                        click.style(
                            "  \u2713 Converting model for device...", fg="yellow"
                        )
                    )
                elif status_val == "deploying":
                    click.echo(
                        click.style("  \u2713 Deploying to device...", fg="yellow")
                    )
                elif status_val == "done":
                    device = data.get("device_name") or data.get("device_id", "device")
                    click.echo(
                        click.style(
                            f"  \u2713 Deployment complete! Model running on {device}",
                            fg="green",
                            bold=True,
                        )
                    )
                    click.echo(f"  Open dashboard: {dashboard_url}")
                    try:
                        reporter = _get_telemetry_reporter()
                        if reporter:
                            reporter.report_funnel_event(
                                "first_deploy",
                                success=True,
                                model_id=name,
                                device_id=data.get("device_id"),
                                platform=data.get("device_platform"),
                            )
                    except Exception:
                        pass  # Never break CLI
                    break
                elif status_val in ("expired", "cancelled"):
                    click.echo(f"Session {status_val}.", err=True)
                    sys.exit(1)
        except KeyboardInterrupt:
            click.echo("\nCancelled.")
            httpx.post(
                f"{api_base}/deploy/pair/{code}/cancel",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5.0,
            )
        return

    client = _get_client()
    device_list = [d.strip() for d in devices.split(",")] if devices else None

    # Dry-run: preview deployment plan
    if dry_run:
        click.echo(f"Preparing deployment plan for {name}...")
        plan = client.deploy_prepare(
            name, version=version, devices=device_list, group=group
        )
        click.echo(f"Model: {plan.model_name} v{plan.model_version}")
        click.echo(f"Devices: {len(plan.deployments)}")
        for d in plan.deployments:
            conv = " (conversion needed)" if d.conversion_needed else ""
            click.echo(
                f"  {d.device_id}: {d.format} via {d.executor} [{d.quantization}]{conv}"
            )
        return

    # Targeted deployment
    if device_list or group:
        target_desc = f"devices={devices}" if devices else f"group={group}"
        click.echo(f"Deploying {name} to {target_desc} ({strategy})...")
        result = client.deploy(
            name,
            version=version,
            rollout=rollout,
            strategy=strategy,
            devices=device_list,
            group=group,
        )
        # result is a DeploymentResult
        from octomil.models import DeploymentResult

        if isinstance(result, DeploymentResult):
            click.echo(f"Deployment: {result.deployment_id}")
            click.echo(f"Status: {result.status}")
            for ds in result.device_statuses:
                err = f" — {ds.error}" if ds.error else ""
                click.echo(f"  {ds.device_id}: {ds.status}{err}")
        return

    # Default: rollout-based deploy
    click.echo(f"Deploying {name} at {rollout}% rollout ({strategy})...")
    result = client.deploy(
        name,
        version=version,
        rollout=rollout,
        strategy=strategy,
    )
    click.echo(f"Rollout created: {result.get('id', 'ok')}")
    click.echo(f"Status: {result.get('status', 'started')}")


# ---------------------------------------------------------------------------
# octomil rollback
# ---------------------------------------------------------------------------


@click.command()
@click.argument("name")
@click.option(
    "--to-version",
    default=None,
    help="Version to rollback to. Defaults to previous version.",
)
def rollback(name: str, to_version: Optional[str]) -> None:
    """Rollback a model to a previous version.

    Reverts NAME to the specified version, or the previous version
    if --to-version is not provided.

    Examples:

        octomil rollback gemma-1b
        octomil rollback gemma-1b --to-version 1.0.0
    """
    client = _get_client()
    target = to_version or "previous"
    click.echo(f"Rolling back {name} to {target}...")

    try:
        result = client.rollback(name, to_version=to_version)
    except Exception as exc:
        click.echo(f"Rollback failed: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Rolled back: {result.from_version} -> {result.to_version}")
    click.echo(f"Rollout ID: {result.rollout_id}")
    click.echo(f"Status: {result.status}")


# ---------------------------------------------------------------------------
# octomil pair (device-side: connect to a pairing session)
# ---------------------------------------------------------------------------


@click.command()
@click.argument("code")
@click.option(
    "--device-id", default=None, help="Device identifier. Auto-generated if omitted."
)
@click.option(
    "--platform",
    "-p",
    default=None,
    help="Device platform (ios, android, python). Auto-detected if omitted.",
)
@click.option("--device-name", default=None, help="Friendly device name.")
def pair(
    code: str,
    device_id: Optional[str],
    platform: Optional[str],
    device_name: Optional[str],
) -> None:
    """Connect to a pairing session as a device.

    Enter the CODE displayed by `octomil deploy --phone` to pair
    this device and receive the model deployment.

    Example:

        octomil pair ABC123
    """
    import platform as _platform
    import uuid

    import httpx

    api_base: str = (
        os.environ.get("OCTOMIL_API_URL")
        or os.environ.get("OCTOMIL_API_BASE")
        or "https://api.octomil.com/api/v1"
    )
    device_id = device_id or f"device-{uuid.uuid4().hex[:8]}"
    platform = platform or f"python-{_platform.system().lower()}"
    device_name = device_name or f"{_platform.node()}"

    click.echo(f"Connecting to pairing session {code.upper()}...")
    click.echo(f"Device: {device_name} ({platform})")

    resp = httpx.post(
        f"{api_base}/deploy/pair/{code}/connect",
        json={
            "device_id": device_id,
            "platform": platform,
            "device_name": device_name,
        },
        timeout=10.0,
    )

    if resp.status_code == 404:
        click.echo("Pairing session not found. Check the code and try again.", err=True)
        sys.exit(1)
    elif resp.status_code == 410:
        click.echo("Pairing session has expired.", err=True)
        sys.exit(1)
    elif resp.status_code == 409:
        click.echo(
            f"Session conflict: {resp.json().get('detail', 'already connected')}",
            err=True,
        )
        sys.exit(1)
    elif resp.status_code >= 400:
        click.echo(f"Failed to connect: {resp.status_code}", err=True)
        sys.exit(1)

    session = resp.json()
    click.echo(f"Connected to session for model: {session['model_name']}")
    click.echo(f"Status: {session['status']}")

    try:
        reporter = _get_telemetry_reporter()
        if reporter:
            reporter.report_funnel_event(
                "app_pair",
                success=True,
                device_id=device_id,
                platform=platform,
            )
    except Exception:
        pass  # Never break CLI

    click.echo("Waiting for deployment...")

    import time

    while True:
        time.sleep(2)
        poll = httpx.get(f"{api_base}/deploy/pair/{code}", timeout=5.0)
        if poll.status_code != 200:
            continue
        data = poll.json()
        st = data.get("status", "connected")
        if st == "deploying":
            click.echo("Deployment in progress...")
        elif st == "done":
            click.echo("Deployment complete. Model received.")
            break
        elif st in ("expired", "cancelled"):
            click.echo(f"Session {st}.", err=True)
            sys.exit(1)


# ---------------------------------------------------------------------------
# octomil status
# ---------------------------------------------------------------------------


@click.command()
@click.argument("name", shell_complete=_complete_model_name)
def status(name: str) -> None:
    """Show model status, active rollouts, and inference metrics.

    Example:

        octomil status sentiment-v1
    """
    client = _get_client()
    info = client.status(name)

    model = info.get("model", {})
    click.echo(f"Model: {model.get('name', name)}")
    click.echo(f"ID: {model.get('id', 'unknown')}")
    click.echo(f"Framework: {model.get('framework', 'unknown')}")

    rollouts = info.get("active_rollouts", [])
    if rollouts:
        click.echo(f"\nActive rollouts: {len(rollouts)}")
        for r in rollouts:
            click.echo(
                f"  v{r.get('version', '?')} — "
                f"{r.get('rollout_percentage', 0)}% — "
                f"{r.get('status', 'unknown')}"
            )
    else:
        click.echo("\nNo active rollouts.")


# ---------------------------------------------------------------------------
# octomil dashboard
# ---------------------------------------------------------------------------


@click.command()
def dashboard() -> None:
    """Open the Octomil dashboard in your browser.

    Shows inference metrics across all devices — latency,
    throughput, errors, model versions side-by-side.
    """
    dashboard_url = os.environ.get("OCTOMIL_DASHBOARD_URL", "https://app.octomil.com")
    click.echo(f"Opening dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)
