"""Deploy, rollback, pair, status, and dashboard commands."""

from __future__ import annotations

import os
import sys
import webbrowser
from typing import Optional

import click

from octomil.cli_helpers import (
    _complete_model_name,
    _get_client,
    _get_telemetry_reporter,
    _require_api_key,
    cli_header,
    cli_kv,
    cli_section,
    cli_success,
    cli_warn,
    http_request,
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
@click.option("--version", "-v", default=None, help="Version to deploy. Defaults to latest.")
@click.option("--phone", is_flag=True, help="Deploy to your phone via QR code pairing.")
@click.option("--fleet", is_flag=True, help="Deploy to your full device fleet.")
@click.option("--rollout", "-r", default=None, type=int, help="Rollout percentage (1-100).")
@click.option(
    "--strategy",
    "-s",
    default=None,
    type=click.Choice(["progressive-15m", "progressive-1h", "progressive-1m", "instant"]),
    help="Rollout strategy.",
)
@click.option(
    "--env",
    "-e",
    default=None,
    help="Target environment (e.g. production, staging, development).",
)
@click.option(
    "--sub-env",
    default=None,
    help="Target sub-environment (e.g. us-west, eu-central).",
)
@click.option(
    "--schedule",
    default=None,
    help="Scheduled start (ISO 8601, e.g. 2026-03-15T09:00:00Z).",
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
    fleet: bool,
    rollout: Optional[int],
    strategy: Optional[str],
    env: Optional[str],
    sub_env: Optional[str],
    schedule: Optional[str],
    target: Optional[str],
    devices: Optional[str],
    group: Optional[str],
    dry_run: bool,
) -> None:
    """Deploy a model to edge devices.

    \b
    Deploy Modes
    ─────────────────────────────────────────────
    Phone (default)   Scan QR code, model lands on phone
    Fleet (--fleet)   Rollout to your full device fleet
    Targeted          Deploy to specific --devices or --group

    \b
    Fleet Options
    ─────────────────────────────────────────────
    --rollout          Percentage of devices (default: 10%)
    --strategy         progressive-15m, progressive-1h,
                       progressive-1m, instant
    --env              Target environment (production, staging)
    --sub-env          Sub-environment (us-west, eu-central)
    --schedule         Scheduled start (ISO 8601 datetime)

    \b
    Examples
    ─────────────────────────────────────────────
    octomil deploy gemma3-1b
    octomil deploy ollama://llama3.2
    octomil deploy gemma3-1b --fleet --rollout 10
    octomil deploy gemma3-1b --fleet --strategy progressive-1h
    octomil deploy gemma3-1b --fleet --env staging
    octomil deploy gemma3-1b --devices d1,d2
    octomil deploy gemma3-1b --group production --dry-run
    """
    cli_header(f"Deploy — {name}")

    # Handle ollama:// URI scheme
    ollama_source_result = None
    if name.startswith("ollama://"):
        ollama_ref = name[len("ollama://") :]
        if not ollama_ref:
            click.echo(
                click.style("  ollama:// URI requires a model name, e.g. ollama://llama3.2", fg="red"),
                err=True,
            )
            sys.exit(1)
        click.echo(click.style(f"  Resolving ollama model: {ollama_ref}", dim=True))

        from octomil.sources.ollama import OllamaSource

        source = OllamaSource()
        try:
            ollama_source_result = source.resolve(ollama_ref)
        except RuntimeError as exc:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
            sys.exit(1)
        cli_success(f"Found: {ollama_source_result.path}")
        # Use the base model name for display / API calls
        name = ollama_ref.split(":")[0]

    # Determine deploy mode:
    # - --devices/--group → targeted deploy (no flag needed)
    # - --fleet → fleet-wide rollout
    # - default (no flags, or --phone) → phone deploy via QR
    is_targeted = bool(devices or group)
    is_phone = not fleet and not is_targeted

    if is_phone:
        # Detect ollama models before deploying
        from octomil.ollama import get_ollama_model
        from octomil.qr import build_deep_link, render_qr_terminal, save_qr_svg

        ollama_model = get_ollama_model(name)
        if ollama_model:
            click.echo(
                f"Detected ollama model: {ollama_model.name} ({ollama_model.size_display}, {ollama_model.quantization})"
            )

        # Try mDNS network scan before falling back to QR code
        from octomil.discovery import scan_for_devices

        click.echo(click.style("  Scanning for devices on local network...", dim=True))
        discovered = scan_for_devices(timeout=5.0)

        if discovered:
            if len(discovered) == 1:
                dev = discovered[0]
                cli_success(f"Found: {dev.name} ({dev.platform}, {dev.ip})")
                if click.confirm(f"\n  Deploy {name} to this device?", default=True):
                    # Direct deployment — create pairing session targeting this device
                    pass
            else:
                cli_section(f"Found {len(discovered)} devices")
                for i, dev in enumerate(discovered, 1):
                    click.echo(f"    {click.style(str(i), fg='cyan')}. {dev.name} ({dev.platform}, {dev.ip})")
                choice = click.prompt("  Select device", type=int, default=1)
                dev = discovered[choice - 1]
                # Deploy to selected device (pairing session will target it)
        else:
            cli_warn("No devices found. Falling back to QR code pairing.")

        api_key = _require_api_key()
        api_base: str = (
            os.environ.get("OCTOMIL_API_URL") or os.environ.get("OCTOMIL_API_BASE") or "https://api.octomil.com/api/v1"
        )
        dashboard_url = os.environ.get("OCTOMIL_DASHBOARD_URL", "https://app.octomil.com")
        headers = {"Authorization": f"Bearer {api_key}"}

        # Ensure model exists in v2 catalog — auto-download, convert, push if needed.
        from octomil.models.catalog import CATALOG, _resolve_alias

        canonical = _resolve_alias(name)
        model_found = name in CATALOG or canonical in CATALOG
        if model_found:
            name = canonical

        if not model_found:
            click.echo(click.style(f"  Model '{name}' not in registry — importing...", dim=True))
            client = _get_client()
            effective_version = version or "1.0.0"
            resolved_name = name.split("/")[-1].split(":")[-1]

            # Try server-side HF import first (no local download needed)
            from octomil.sources.resolver import resolve_hf_repo

            hf_repo = resolve_hf_repo(name)
            if hf_repo:
                try:
                    client.import_from_hf(
                        hf_repo,
                        name=resolved_name,
                        version=effective_version,
                    )
                    cli_success(f"Imported {resolved_name} from HuggingFace")
                    name = resolved_name
                except Exception as exc:
                    if "402" in str(exc) or "limit" in str(exc).lower():
                        click.echo(
                            click.style(
                                "  Plan limit reached — upgrade at https://app.octomil.com/settings/billing",
                                fg="red",
                            ),
                            err=True,
                        )
                        sys.exit(1)
                    # HF import failed — fall through to local download
                    cli_warn(f"Server-side import failed ({exc}), trying local download...")
                    hf_repo = None

            # Fallback: download locally and upload
            if not hf_repo:
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

                try:
                    # Push model weights
                    file_size_mb = os.path.getsize(resolved_path) / (1024 * 1024)
                    click.echo(click.style(f"  Uploading to registry ({file_size_mb:.0f} MB)...", dim=True))
                    client.push(
                        resolved_path,
                        name=resolved_name,
                        version=effective_version,
                    )
                    # Push companion files (mmproj, etc.) from same directory
                    model_dir = os.path.dirname(resolved_path)
                    for companion in os.listdir(model_dir):
                        if companion.startswith("mmproj") and companion.endswith(".gguf"):
                            comp_path = os.path.join(model_dir, companion)
                            comp_mb = os.path.getsize(comp_path) / (1024 * 1024)
                            click.echo(click.style(f"  Uploading projector ({comp_mb:.0f} MB)...", dim=True))
                            client.push(
                                comp_path,
                                name=f"{resolved_name}-mmproj",
                                version=effective_version,
                            )
                    click.echo(click.style(f"  Pushed {resolved_name} v{effective_version}", fg="green"))
                    name = resolved_name
                except Exception as exc:
                    if "402" in str(exc) or "limit" in str(exc).lower():
                        click.echo(
                            click.style(
                                "  Plan limit reached — upgrade at https://app.octomil.com/settings/billing",
                                fg="red",
                            ),
                            err=True,
                        )
                    else:
                        click.echo(click.style(f"  Push failed: {exc}", fg="red"), err=True)
                    sys.exit(1)

        click.echo(f"Creating pairing session for {name}...")
        resp = http_request(
            "POST",
            f"{api_base}/deploy/pair",
            json={"model_name": name, "model_version": version},
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code >= 400:
            click.echo(f"Failed to create pairing session: {resp.status_code}", err=True)
            click.echo(resp.text, err=True)
            sys.exit(1)

        session = resp.json()
        code = session["code"]

        # Build short Universal Link URL for QR code.
        pair_url = build_deep_link(token=code, host=api_base)

        # Render QR code in terminal
        qr_art = render_qr_terminal(pair_url)

        # Save SVG fallback for reliable scanning
        import tempfile

        svg_path: str | None = None
        try:
            _svg = os.path.join(tempfile.gettempdir(), f"octomil-pair-{code}.svg")
            save_qr_svg(pair_url, _svg)
            svg_path = _svg
        except Exception:
            pass

        click.echo()
        click.echo(qr_art)
        click.echo()
        click.echo(click.style("  Scan with your phone camera", bold=True))
        click.echo(click.style("  Or enter code manually: ", dim=True) + click.style(code, fg="cyan", bold=True))
        if svg_path:
            click.echo(click.style("  QR image saved: ", dim=True) + click.style(svg_path, fg="cyan"))
        click.echo(click.style("  Expires in 5 minutes", dim=True))
        click.echo()

        click.echo("Waiting for device to connect (Ctrl+C to cancel)...")
        last_status = ""
        try:
            while True:
                import time

                time.sleep(2)
                try:
                    poll = http_request("GET", f"{api_base}/deploy/pair/{code}", timeout=5.0)
                except SystemExit:
                    continue  # retry silently during polling
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
                    # Trigger deployment — server resolves model version from catalog
                    try:
                        deploy_resp = http_request(
                            "POST",
                            f"{api_base}/deploy/pair/{code}/deploy",
                            headers=headers,
                            timeout=10.0,
                        )
                        if deploy_resp.status_code >= 400:
                            click.echo(
                                click.style(
                                    f"  Deploy trigger failed: {deploy_resp.status_code} — {deploy_resp.text}",
                                    fg="red",
                                ),
                                err=True,
                            )
                            sys.exit(1)
                        click.echo(click.style("  \u2713 Deploying to device...", fg="yellow"))
                    except SystemExit:
                        raise
                    except Exception as exc:
                        click.echo(
                            click.style(f"  Deploy trigger error: {exc}", fg="red"),
                            err=True,
                        )
                        sys.exit(1)
                elif status_val == "converting":
                    click.echo(click.style("  \u2713 Converting model for device...", fg="yellow"))
                elif status_val == "deploying":
                    click.echo(click.style("  \u2713 Deploying to device...", fg="yellow"))
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
            try:
                http_request(
                    "POST",
                    f"{api_base}/deploy/pair/{code}/cancel",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=5.0,
                )
            except SystemExit:
                pass  # Best-effort cancel
        return

    # Fleet and targeted deployments share common setup.
    effective_rollout = rollout if rollout is not None else 10
    effective_strategy = strategy or "progressive-15m"

    client = _get_client()
    device_list = [d.strip() for d in devices.split(",")] if devices else None

    # Show deployment summary before proceeding.
    cli_section("Deployment Config")
    cli_kv("Model", f"{name}" + (f" v{version}" if version else ""))
    cli_kv("Strategy", effective_strategy)
    cli_kv("Rollout", f"{effective_rollout}%")
    if env:
        cli_kv("Environment", env)
    if sub_env:
        cli_kv("Sub-environment", sub_env)
    if schedule:
        cli_kv("Scheduled", schedule)
    click.echo()

    # Dry-run: preview deployment plan
    if dry_run:
        click.echo(click.style("  Preparing deployment plan...", dim=True))
        plan = client.deploy_prepare(name, version=version, devices=device_list, group=group)
        cli_section("Deployment Plan")
        cli_kv("Model", f"{plan.model_name} v{plan.model_version}")
        cli_kv("Devices", str(len(plan.deployments)))
        for d in plan.deployments:
            conv = click.style(" (conversion needed)", fg="yellow") if d.conversion_needed else ""
            click.echo(
                f"    {click.style(d.device_id, fg='white')}  {d.format} via {d.executor} [{d.quantization}]{conv}"
            )
        return

    # Targeted deployment (--devices or --group)
    if device_list or group:
        target_desc = f"devices={devices}" if devices else f"group={group}"
        click.echo(click.style(f"  Deploying {name} to {target_desc} ({effective_strategy})...", dim=True))
        result = client.deploy(
            name,
            version=version,
            rollout=effective_rollout,
            strategy=effective_strategy,
            devices=device_list,
            group=group,
            env=env,
            sub_env=sub_env,
            schedule=schedule,
        )
        from octomil.models import DeploymentResult

        if isinstance(result, DeploymentResult):
            cli_kv("Deployment", result.deployment_id)
            cli_kv("Status", result.status)
            for ds in result.device_statuses:
                icon = click.style("\u2713", fg="green") if ds.status == "done" else click.style("\u2022", dim=True)
                err = click.style(f" — {ds.error}", fg="red") if ds.error else ""
                click.echo(f"    {icon} {ds.device_id}: {ds.status}{err}")
        return

    # Fleet-wide rollout (--fleet)
    click.echo(click.style(f"  Deploying {name} at {effective_rollout}% ({effective_strategy})...", dim=True))
    result = client.deploy(
        name,
        version=version,
        rollout=effective_rollout,
        strategy=effective_strategy,
        env=env,
        sub_env=sub_env,
        schedule=schedule,
    )
    cli_success(f"Rollout created: {result.get('id', 'ok')}")
    cli_kv("Status", result.get("status", "started"))


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

        octomil rollback gemma3-1b
        octomil rollback gemma3-1b --to-version 1.0.0
    """
    client = _get_client()
    target = to_version or "previous"
    cli_header(f"Rollback — {name}")
    click.echo(click.style(f"  Rolling back to {target}...", dim=True))

    try:
        result = client.rollback(name, to_version=to_version)
    except Exception as exc:
        click.echo(click.style(f"  Rollback failed: {exc}", fg="red"), err=True)
        sys.exit(1)

    cli_success(f"{result.from_version} \u2192 {result.to_version}")
    cli_kv("Rollout ID", result.rollout_id)
    cli_kv("Status", result.status)


# ---------------------------------------------------------------------------
# octomil pair (device-side: connect to a pairing session)
# ---------------------------------------------------------------------------


@click.command()
@click.argument("code")
@click.option("--device-id", default=None, help="Device identifier. Auto-generated if omitted.")
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

    cli_header("Pair")

    api_base: str = (
        os.environ.get("OCTOMIL_API_URL") or os.environ.get("OCTOMIL_API_BASE") or "https://api.octomil.com/api/v1"
    )
    device_id = device_id or f"device-{uuid.uuid4().hex[:8]}"
    platform = platform or f"python-{_platform.system().lower()}"
    device_name = device_name or f"{_platform.node()}"

    click.echo(f"Connecting to pairing session {code.upper()}...")
    click.echo(f"Device: {device_name} ({platform})")

    resp = http_request(
        "POST",
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
        try:
            poll = http_request("GET", f"{api_base}/deploy/pair/{code}", timeout=5.0)
        except SystemExit:
            continue  # retry silently during polling
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
    cli_header(f"Status — {model.get('name', name)}")
    cli_kv("ID", model.get("id", "unknown"))
    cli_kv("Framework", model.get("framework", "unknown"))

    rollouts = info.get("active_rollouts", [])
    if rollouts:
        click.echo()
        cli_section(f"Active Rollouts ({len(rollouts)})")
        for r in rollouts:
            pct = r.get("rollout_percentage", 0)
            st = r.get("status", "unknown")
            ver = r.get("version", "?")
            ver_styled = click.style("v" + ver, fg="white", bold=True)
            click.echo("    " + ver_styled + "  " + str(pct) + "%  " + click.style(st, dim=True))
    else:
        click.echo()
        click.echo(click.style("  No active rollouts.", dim=True))


# ---------------------------------------------------------------------------
# octomil dashboard
# ---------------------------------------------------------------------------


@click.command()
def dashboard() -> None:
    """Open the Octomil dashboard in your browser.

    Shows inference metrics across all devices — latency,
    throughput, errors, model versions side-by-side.
    """
    cli_header("Dashboard")
    dashboard_url = os.environ.get("OCTOMIL_DASHBOARD_URL", "https://app.octomil.com")
    cli_success(f"Opening: {dashboard_url}")
    webbrowser.open(dashboard_url)
