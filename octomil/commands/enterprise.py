"""Enterprise CLI commands: login, init, org, team, keys, train.

Extracted from cli.py for modularity.  All helpers are imported from
``octomil.cli_helpers`` so they can be shared with other command modules.
"""

from __future__ import annotations

import os
import sys
import webbrowser
from typing import Any, Optional

import click

from octomil.cli_helpers import (
    _get_api_key,
    _get_client,
    _get_enterprise_client,
    _require_org_id,
    _save_credentials,
)


# ---------------------------------------------------------------------------
# Browser-based login helper (local to this module)
# ---------------------------------------------------------------------------


def _browser_login() -> None:
    """Run the browser-based OAuth-style login flow.

    1. Spins up a temporary HTTP server on localhost.
    2. Opens the Octomil dashboard auth page in the browser.
    3. Waits for the dashboard to redirect back with an API key.
    4. Falls back to manual paste on timeout.
    """
    import http.server
    import secrets
    import socket
    import threading
    import urllib.parse

    state = secrets.token_urlsafe(32)

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    dashboard_url = os.environ.get("OCTOMIL_DASHBOARD_URL", "https://app.octomil.com")
    callback_url = f"http://127.0.0.1:{port}"
    auth_url = (
        f"{dashboard_url}/cli/auth"
        f"?callback={urllib.parse.quote(callback_url, safe='')}"
        f"&state={state}"
    )

    received_key: str | None = None
    received_org: str | None = None
    received_org_id: str | None = None
    got_callback = threading.Event()

    class _CallbackHandler(http.server.BaseHTTPRequestHandler):
        """Handle the single GET callback from the dashboard."""

        def do_GET(self) -> None:  # noqa: N802
            nonlocal received_key, received_org, received_org_id
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

            cb_state = params.get("state", [None])[0]
            if cb_state != state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid state parameter.")
                return

            received_key = params.get("key", [None])[0]
            received_org = params.get("org", [None])[0]
            received_org_id = params.get("org_id", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            _CMD = "octomil push phi-4-mini"
            _COPY_ICON = (
                '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/>'
                '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'
            )
            _CHECK_ICON = '<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>'
            _page = (
                '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
                "<title>Octomil CLI</title><style>"
                "*{margin:0;padding:0;box-sizing:border-box}"
                "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;"
                "display:flex;justify-content:center;align-items:center;min-height:100vh;"
                "background:#070d12;color:#e2e8f0}"
                ".card{text-align:center;padding:3rem 2.5rem;max-width:420px}"
                ".ok{width:56px;height:56px;border-radius:50%;background:rgba(34,197,94,.12);"
                "border:1.5px solid rgba(34,197,94,.3);display:flex;align-items:center;"
                "justify-content:center;margin:0 auto 1.5rem}"
                ".ok svg{width:28px;height:28px;stroke:#22c55e;fill:none;stroke-width:2.5;"
                "stroke-linecap:round;stroke-linejoin:round}"
                "h2{font-size:1.25rem;font-weight:600;margin-bottom:.5rem;letter-spacing:-.01em}"
                "p{color:#64748b;font-size:.9rem;line-height:1.5}"
                ".hint{margin-top:2rem;padding:.875rem 1rem;background:#0f1822;"
                "border:1px solid rgba(255,255,255,.06);border-radius:8px;"
                "font-family:'SF Mono','Fira Code',monospace;font-size:.8rem;color:#7dd3fc;"
                "display:flex;align-items:center;justify-content:space-between;gap:.75rem}"
                ".hint code{flex:1;text-align:left}"
                ".cb{background:none;border:1px solid rgba(255,255,255,.1);border-radius:5px;"
                "padding:4px 6px;cursor:pointer;color:#64748b;display:flex;align-items:center;"
                "transition:all .15s ease;flex-shrink:0}"
                ".cb:hover{border-color:rgba(255,255,255,.2);color:#7dd3fc}"
                ".cb svg{width:16px;height:16px;stroke:currentColor;fill:none;"
                "stroke-width:2;stroke-linecap:round;stroke-linejoin:round}"
                "</style></head><body><div class=card>"
                '<div class=ok><svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg></div>'
                "<h2>CLI Authenticated</h2>"
                "<p>You can close this tab and return to your terminal.</p>"
                f'<div class=hint><code>$ {_CMD}</code>'
                f'<button class=cb id=cpb>{_COPY_ICON}</button></div>'
                "<script>"
                f"document.getElementById('cpb').onclick=function(){{navigator.clipboard.writeText('{_CMD}');"
                f"this.innerHTML='{_CHECK_ICON}';var b=this;"
                f"setTimeout(function(){{b.innerHTML='{_COPY_ICON}'}},1500)}}"
                "</script>"
                "</div></body></html>"
            )
            self.wfile.write(_page.encode())
            got_callback.set()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # Suppress server log noise

    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    webbrowser.open(auth_url)
    click.echo("Opening browser for authentication...")
    click.echo("Waiting for authorization (Ctrl+C to cancel)...")

    got_callback.wait(timeout=300)
    server.shutdown()

    if received_key:
        _save_credentials(received_key, org=received_org, org_id=received_org_id)
        org_display = received_org or ""
        if org_display:
            click.echo(
                click.style(f"  Authenticated ({org_display})", fg="green")
            )
        else:
            click.echo(click.style("  Authenticated", fg="green"))
        click.echo("  Credentials saved to ~/.octomil/credentials")
        try:
            from .completions import _install_completions

            _install_completions()
        except Exception:
            click.echo(
                "\n  Tip: Run `octomil completions --install` for tab-completion setup."
            )
        if not received_org_id:
            click.echo(
                click.style(
                    "\n  No organization linked. Run `octomil init <org>` to create one.",
                    fg="yellow",
                )
            )
    else:
        click.echo("Timed out. You can paste your API key manually:")
        manual_key: str = click.prompt("API key", hide_input=True)
        _save_credentials(manual_key)
        click.echo("API key saved to ~/.octomil/credentials")


# ---------------------------------------------------------------------------
# octomil login
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--api-key",
    default=None,
    hide_input=True,
    help="Paste API key directly (skip browser flow).",
)
def login(api_key: Optional[str]) -> None:
    """Authenticate with Octomil Cloud.

    Opens your browser to authenticate, then saves the API key locally.
    Use --api-key to paste a key directly (for CI/headless environments).
    """
    if api_key:
        _save_credentials(api_key)
        click.echo("API key saved to ~/.octomil/credentials")
        return

    _browser_login()


# ---------------------------------------------------------------------------
# octomil init
# ---------------------------------------------------------------------------


@click.command()
@click.argument("org_name")
@click.option(
    "--compliance",
    type=click.Choice(["hipaa", "gdpr", "pci", "soc2"]),
    default=None,
    help="Enable a compliance preset (hipaa, gdpr, pci, soc2).",
)
@click.option("--region", default="us", help="Data region (us, eu, ap).")
@click.option("--api-base", default=None, help="Override API base URL.")
def init(
    org_name: str,
    compliance: Optional[str],
    region: str,
    api_base: Optional[str],
) -> None:
    """Initialize an Octomil organization for enterprise use.

    Creates a new organization, optionally applies a compliance preset,
    and saves the org_id to ~/.octomil/config.json for subsequent commands.

    Examples:

        octomil init "Acme Corp" --compliance hipaa --region us
        octomil init "Startup Inc" --region eu
    """
    from octomil.enterprise import EnterpriseClient, save_config, load_config

    key = _get_api_key()
    if not key:
        click.echo("No API key found. Run `octomil login` first.", err=True)
        sys.exit(1)

    client = EnterpriseClient(api_key=key, api_base=api_base)

    # 1. Create org
    click.echo(f"Creating organization: {org_name} (region={region})")
    try:
        result = client.create_org(org_name, region=region, workspace_type="enterprise")
    except Exception as exc:
        click.echo(f"Failed to create organization: {exc}", err=True)
        sys.exit(1)

    org_id = result.get("org_id", "")
    click.echo(click.style(f"  Organization created: {org_id}", fg="green"))

    # 2. Apply compliance preset if specified
    if compliance:
        click.echo(f"Applying {compliance.upper()} compliance preset...")
        try:
            client.set_compliance(org_id, compliance)
            click.echo(
                click.style(f"  {compliance.upper()} compliance applied", fg="green")
            )
        except Exception as exc:
            click.echo(f"  Warning: compliance preset failed: {exc}", err=True)

    # 3. Save org_id to config
    config = load_config()
    config["org_id"] = org_id
    config["org_name"] = org_name
    config["region"] = region
    if compliance:
        config["compliance"] = compliance
    save_config(config)
    click.echo("  Config saved to ~/.octomil/config.json")

    # 4. Print next steps
    click.echo("")
    click.echo("Next steps:")
    click.echo(
        f"  1. Invite team members:  octomil team add alice@{org_name.lower().replace(' ', '')}.com --role admin"
    )
    click.echo("  2. Create an API key:    octomil keys create deploy-key")
    click.echo("  3. Set security policy:  octomil team set-policy --require-mfa")
    click.echo(
        "  4. Push a model:         octomil push my-model --version 1.0.0"
    )


# ---------------------------------------------------------------------------
# octomil org
# ---------------------------------------------------------------------------


@click.command("org")
def org_info() -> None:
    """Show current organization info and settings."""
    from octomil.enterprise import get_org_id, load_config

    org_id = get_org_id()
    if not org_id:
        click.echo("No organization configured. Run `octomil init <name>` first.")
        return

    config = load_config()
    click.echo(f"Organization: {config.get('org_name', org_id)}")
    click.echo(f"Org ID: {org_id}")
    click.echo(f"Region: {config.get('region', 'unknown')}")
    if config.get("compliance"):
        click.echo(f"Compliance: {config['compliance'].upper()}")

    # Try to fetch live settings
    key = _get_api_key()
    if key:
        try:
            client = _get_enterprise_client()
            settings = client.get_settings(org_id)
            click.echo("")
            click.echo("Settings:")
            click.echo(
                f"  Audit retention:     {settings.get('audit_retention_days', '?')} days"
            )
            click.echo(
                f"  Require MFA:         {settings.get('require_mfa_for_admin', '?')}"
            )
            click.echo(
                f"  Admin approval:      {settings.get('require_admin_approval', '?')}"
            )
            click.echo(
                f"  Model approval:      {settings.get('require_model_approval', '?')}"
            )
            click.echo(
                f"  Auto rollback:       {settings.get('auto_rollback_enabled', '?')}"
            )
            click.echo(
                f"  Session duration:    {settings.get('session_duration_hours', '?')}h"
            )
            click.echo(
                f"  Reauth interval:     {settings.get('reauth_interval_minutes', '?')}min"
            )
        except Exception:
            click.echo("\n  (unable to fetch live settings)")


# ---------------------------------------------------------------------------
# octomil team
# ---------------------------------------------------------------------------


@click.group()
def team() -> None:
    """Manage organization team members."""


@team.command("add")
@click.argument("email")
@click.option(
    "--role",
    type=click.Choice(["member", "admin", "owner"]),
    default="member",
    help="Role to assign.",
)
@click.option("--name", default=None, help="Display name for the member.")
def team_add(email: str, role: str, name: Optional[str]) -> None:
    """Invite a team member to the organization.

    Example:

        octomil team add alice@acme.com --role admin
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    click.echo(f"Inviting {email} as {role}...")
    try:
        result = client.invite_member(org_id, email, role=role, name=name)
        click.echo(
            click.style(
                f"  Invited: {result.get('email', email)} ({result.get('role', role)})",
                fg="green",
            )
        )
    except Exception as exc:
        click.echo(f"Failed to invite member: {exc}", err=True)
        sys.exit(1)


@team.command("list")
def team_list() -> None:
    """List team members.

    Example:

        octomil team list
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    try:
        members = client.list_members(org_id)
    except Exception as exc:
        click.echo(f"Failed to list members: {exc}", err=True)
        sys.exit(1)

    if not members:
        click.echo("No team members found.")
        return

    # Table header
    click.echo(f"{'Email':<35s} {'Role':<10s} {'Name':<20s}")
    click.echo("-" * 65)
    for m in members:
        email = m.get("email", "?")
        role = m.get("role", "?")
        name = m.get("name", "") or ""
        click.echo(f"{email:<35s} {role:<10s} {name:<20s}")
    click.echo(f"\nTotal: {len(members)} member(s)")


@team.command("set-policy")
@click.option(
    "--min-privacy-budget", type=float, default=None, help="Minimum DP epsilon budget."
)
@click.option(
    "--require-mfa", is_flag=True, default=None, help="Require MFA for admins."
)
@click.option(
    "--no-require-mfa",
    is_flag=True,
    default=None,
    help="Disable MFA requirement for admins.",
)
@click.option(
    "--auto-rollback/--no-auto-rollback",
    default=None,
    help="Auto-rollback on model drift.",
)
@click.option(
    "--session-hours", type=int, default=None, help="Session duration in hours."
)
@click.option(
    "--reauth-minutes",
    type=int,
    default=None,
    help="Re-authentication interval in minutes.",
)
@click.option(
    "--audit-retention-days",
    type=int,
    default=None,
    help="Audit log retention in days.",
)
@click.option(
    "--require-model-approval",
    is_flag=True,
    default=None,
    help="Require approval for model deployments.",
)
@click.option(
    "--no-require-model-approval",
    is_flag=True,
    default=None,
    help="Disable model deployment approval.",
)
def team_set_policy(
    min_privacy_budget: Optional[float],
    require_mfa: Optional[bool],
    no_require_mfa: Optional[bool],
    auto_rollback: Optional[bool],
    session_hours: Optional[int],
    reauth_minutes: Optional[int],
    audit_retention_days: Optional[int],
    require_model_approval: Optional[bool],
    no_require_model_approval: Optional[bool],
) -> None:
    """Set organization security policies.

    Examples:

        octomil team set-policy --require-mfa --session-hours 8
        octomil team set-policy --auto-rollback --audit-retention-days 365
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    updates: dict[str, Any] = {}

    # Resolve MFA flag (--require-mfa / --no-require-mfa)
    if require_mfa:
        updates["require_mfa_for_admin"] = True
    elif no_require_mfa:
        updates["require_mfa_for_admin"] = False

    if auto_rollback is not None:
        updates["auto_rollback_enabled"] = auto_rollback

    if session_hours is not None:
        updates["session_duration_hours"] = session_hours

    if reauth_minutes is not None:
        updates["reauth_interval_minutes"] = reauth_minutes

    if audit_retention_days is not None:
        updates["audit_retention_days"] = audit_retention_days

    # Resolve model approval flag
    if require_model_approval:
        updates["require_model_approval"] = True
    elif no_require_model_approval:
        updates["require_model_approval"] = False

    if not updates:
        click.echo("No policy changes specified. Use --help to see options.")
        return

    click.echo("Updating security policies...")
    try:
        client.update_settings(org_id, **updates)
        for key, value in updates.items():
            label = key.replace("_", " ").title()
            click.echo(click.style(f"  {label}: {value}", fg="green"))
    except Exception as exc:
        click.echo(f"Failed to update policies: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# octomil keys
# ---------------------------------------------------------------------------


@click.group()
def keys() -> None:
    """Manage API keys."""


@keys.command("create")
@click.argument("name")
@click.option(
    "--scope",
    multiple=True,
    help="Key scopes (e.g., devices:write, models:read). Repeat for multiple.",
)
def keys_create(name: str, scope: tuple[str, ...]) -> None:
    """Create a new API key.

    Examples:

        octomil keys create deploy-key --scope devices:write --scope models:read
        octomil keys create admin-key
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    # Convert scope tuple to dict format expected by the API
    scopes: dict[str, Any] | None = None
    if scope:
        scopes = {}
        for s in scope:
            if ":" in s:
                resource, permission = s.split(":", 1)
                scopes[resource] = permission
            else:
                scopes[s] = "read"

    click.echo(f"Creating API key: {name}")
    try:
        result = client.create_api_key(org_id, name, scopes=scopes)
        raw_key = result.get("api_key", "")
        prefix = result.get("prefix", "")
        click.echo(click.style(f"  Key created: {prefix}...", fg="green"))
        click.echo("")
        click.echo(click.style(f"  API Key: {raw_key}", fg="yellow", bold=True))
        click.echo("")
        click.echo("  Save this key securely — it will not be shown again.")
        click.echo(f"  Set it as: export OCTOMIL_API_KEY={raw_key}")
    except Exception as exc:
        click.echo(f"Failed to create API key: {exc}", err=True)
        sys.exit(1)


@keys.command("list")
def keys_list() -> None:
    """List API keys.

    Example:

        octomil keys list
    """
    org_id = _require_org_id()
    client = _get_enterprise_client()

    try:
        api_keys = client.list_api_keys(org_id)
    except Exception as exc:
        click.echo(f"Failed to list API keys: {exc}", err=True)
        sys.exit(1)

    if not api_keys:
        click.echo("No API keys found.")
        return

    click.echo(f"{'Name':<25s} {'Prefix':<15s} {'Created':<20s} {'Status':<10s}")
    click.echo("-" * 70)
    for k in api_keys:
        name = k.get("name", "?")
        prefix = k.get("prefix", "?")
        created = k.get("created_at", "?")[:10]
        revoked = k.get("revoked_at")
        status_str = (
            click.style("revoked", fg="red")
            if revoked
            else click.style("active", fg="green")
        )
        click.echo(f"{name:<25s} {prefix:<15s} {created:<20s} {status_str}")
    click.echo(f"\nTotal: {len(api_keys)} key(s)")


@keys.command("revoke")
@click.argument("key_id")
@click.confirmation_option(prompt="Are you sure you want to revoke this API key?")
def keys_revoke(key_id: str) -> None:
    """Revoke an API key.

    Example:

        octomil keys revoke abc-123-def
    """
    client = _get_enterprise_client()

    click.echo(f"Revoking API key: {key_id}")
    try:
        result = client.revoke_api_key(key_id)
        click.echo(
            click.style(
                f"  Revoked: {result.get('name', key_id)} (prefix: {result.get('prefix', '?')})",
                fg="yellow",
            )
        )
    except Exception as exc:
        click.echo(f"Failed to revoke API key: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# octomil train
# ---------------------------------------------------------------------------


@click.group()
def train() -> None:
    """Federated training across deployed devices."""


@train.command("start")
@click.argument("name")
@click.option(
    "--strategy",
    "-s",
    default="fedavg",
    type=click.Choice(
        [
            "fedavg",
            "fedprox",
            "fedopt",
            "fedadam",
            "krum",
            "scaffold",
            "ditto",
            "fedmedian",
            "fedtrimmedavg",
        ]
    ),
    help="Aggregation strategy.",
)
@click.option("--rounds", "-r", default=10, help="Number of training rounds.")
@click.option("--group", "-g", default=None, help="Device group to train on.")
@click.option(
    "--privacy",
    default=None,
    type=click.Choice(["dp-sgd", "none"]),
    help="Privacy mechanism.",
)
@click.option(
    "--epsilon", default=None, type=float, help="Privacy budget (lower = more private)."
)
@click.option("--min-devices", default=2, help="Minimum devices required per round.")
def train_start(
    name: str,
    strategy: str,
    rounds: int,
    group: Optional[str],
    privacy: Optional[str],
    epsilon: Optional[float],
    min_devices: int,
) -> None:
    """Start federated training.

    Example:

        octomil train start sentiment-v1 --strategy fedavg --rounds 50
    """
    client = _get_client()
    click.echo(f"Starting federated training for {name}")
    click.echo(f"Strategy: {strategy} | Rounds: {rounds} | Min devices: {min_devices}")
    if privacy:
        click.echo(f"Privacy: {privacy} (e={epsilon})")

    result = client.train(
        name,
        strategy=strategy,
        rounds=rounds,
        group=group,
        privacy=privacy,
        epsilon=epsilon,
        min_devices=min_devices,
    )
    click.echo(f"Training started: {result.session_id}")
    click.echo(f"Status: {result.status}")
    click.echo(f"Monitor: octomil train status {name}")


@train.command("status")
@click.argument("name")
def train_status_cmd(name: str) -> None:
    """Show training progress.

    Example:

        octomil train status sentiment-v1
    """
    client = _get_client()
    info = client.train_status(name)

    current = info.get("current_round", 0)
    total = info.get("total_rounds", 0)
    devices = info.get("active_devices", 0)
    status_val = info.get("status", "unknown")
    loss = info.get("loss")
    accuracy = info.get("accuracy")

    click.echo(f"Model: {name}")
    click.echo(f"Status: {status_val}")
    click.echo(f"Round: {current}/{total}")
    click.echo(f"Active devices: {devices}")
    if loss is not None:
        click.echo(f"Loss: {loss:.4f}")
    if accuracy is not None:
        click.echo(f"Accuracy: {accuracy:.1%}")


@train.command("stop")
@click.argument("name")
def train_stop_cmd(name: str) -> None:
    """Stop active training.

    Example:

        octomil train stop sentiment-v1
    """
    client = _get_client()
    click.echo(f"Stopping training for {name}...")
    result = client.train_stop(name)
    click.echo(f"Training stopped. Last round: {result.get('last_round', '?')}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    """Register all enterprise commands with the top-level CLI group."""
    cli.add_command(login)
    cli.add_command(init)
    cli.add_command(org_info, "org")
    cli.add_command(team)
    cli.add_command(keys)
    cli.add_command(train)
