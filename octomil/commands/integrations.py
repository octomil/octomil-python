"""CLI commands for managing observability integrations.

    octomil integrations list
    octomil integrations create --type otlp --endpoint http://collector:4318
    octomil integrations delete <id>
    octomil integrations test <id>
    octomil integrations connect-otlp --endpoint http://collector:4318
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import click

from octomil.cli_helpers import _require_api_key, _get_org_id


def _get_integrations_api():
    """Build IntegrationsAPI with current credentials."""
    import os
    from octomil.integrations import IntegrationsAPI

    api_key = _require_api_key()
    org_id = _get_org_id() or "default"
    api_base = (
        os.environ.get("OCTOMIL_API_URL")
        or os.environ.get("OCTOMIL_API_BASE")
        or "https://api.octomil.com/api/v1"
    )
    return IntegrationsAPI(api_key=api_key, org_id=org_id, api_base=api_base)


@click.group()
def integrations() -> None:
    """Manage observability export integrations (metrics + logs)."""


@integrations.command("list")
@click.option("--type", "kind", type=click.Choice(["metrics", "logs", "all"]), default="all",
              help="Filter by integration kind.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def list_integrations(kind: str, as_json: bool) -> None:
    """List configured export integrations.

    Examples:

        octomil integrations list
        octomil integrations list --type metrics
        octomil integrations list --json
    """
    api = _get_integrations_api()
    rows = []

    if kind in ("metrics", "all"):
        try:
            for m in api.list_metrics_integrations():
                rows.append({
                    "id": m.id,
                    "kind": "metrics",
                    "name": m.name,
                    "type": m.integration_type,
                    "enabled": m.enabled,
                    "detail": m.config.get("endpoint", ""),
                })
        except Exception as exc:
            click.echo(f"Failed to fetch metrics integrations: {exc}", err=True)

    if kind in ("logs", "all"):
        try:
            for lg in api.list_log_integrations():
                rows.append({
                    "id": lg.id,
                    "kind": "logs",
                    "name": lg.name,
                    "type": lg.integration_type,
                    "enabled": lg.enabled,
                    "detail": lg.endpoint_url,
                })
        except Exception as exc:
            click.echo(f"Failed to fetch log integrations: {exc}", err=True)

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return

    if not rows:
        click.echo("No integrations configured.")
        click.echo("Run `octomil integrations connect-otlp --endpoint <url>` to get started.")
        return

    click.echo(f"{'Kind':<10s} {'Type':<16s} {'Name':<30s} {'Enabled':<8s} {'Detail'}")
    click.echo("-" * 90)
    for r in rows:
        enabled_str = click.style("yes", fg="green") if r["enabled"] else click.style("no", fg="red")
        click.echo(f"{r['kind']:<10s} {r['type']:<16s} {r['name']:<30s} {enabled_str:<17s} {r['detail']}")
    click.echo(f"\nTotal: {len(rows)} integration(s)")


@integrations.command("create")
@click.option("--kind", type=click.Choice(["metrics", "logs"]), required=True,
              help="Integration kind.")
@click.option("--type", "integration_type", required=True,
              help="Type (prometheus, opentelemetry, datadog, statsd, webhook, splunk, elasticsearch, otlp, cloudwatch).")
@click.option("--name", required=True, help="Integration name.")
@click.option("--endpoint", default=None, help="Endpoint URL.")
@click.option("--config-json", default=None, help="Config as JSON string (for metrics).")
@click.option("--format", "fmt", default="json", help="Log format (json, syslog, hec, otlp).")
def create_integration(
    kind: str,
    integration_type: str,
    name: str,
    endpoint: Optional[str],
    config_json: Optional[str],
    fmt: str,
) -> None:
    """Create an export integration.

    Examples:

        octomil integrations create --kind metrics --type prometheus --name prod-prom \\
            --config-json '{"prefix": "octomil", "scrape_interval": 30}'

        octomil integrations create --kind logs --type otlp --name prod-otlp \\
            --endpoint http://collector:4318/v1/logs --format otlp
    """
    api = _get_integrations_api()

    if kind == "metrics":
        config = json.loads(config_json) if config_json else {}
        if endpoint and "endpoint" not in config:
            config["endpoint"] = endpoint
        try:
            result = api.create_metrics_integration(name, integration_type, config)
            click.echo(click.style(f"Created metrics integration: {result.id}", fg="green"))
        except Exception as exc:
            click.echo(f"Failed: {exc}", err=True)
            sys.exit(1)
    else:
        if not endpoint:
            click.echo("--endpoint is required for log integrations.", err=True)
            sys.exit(1)
        try:
            result = api.create_log_integration(name, integration_type, endpoint, format=fmt)
            click.echo(click.style(f"Created log integration: {result.id}", fg="green"))
        except Exception as exc:
            click.echo(f"Failed: {exc}", err=True)
            sys.exit(1)


@integrations.command("delete")
@click.argument("integration_id")
@click.option("--kind", type=click.Choice(["metrics", "logs"]), required=True,
              help="Integration kind.")
@click.confirmation_option(prompt="Are you sure you want to delete this integration?")
def delete_integration(integration_id: str, kind: str) -> None:
    """Delete an integration.

    Example:

        octomil integrations delete abc-123 --kind metrics
    """
    api = _get_integrations_api()
    try:
        if kind == "metrics":
            api.delete_metrics_integration(integration_id)
        else:
            api.delete_log_integration(integration_id)
        click.echo(click.style(f"Deleted {kind} integration: {integration_id}", fg="yellow"))
    except Exception as exc:
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(1)


@integrations.command("test")
@click.argument("integration_id")
@click.option("--kind", type=click.Choice(["metrics", "logs"]), required=True,
              help="Integration kind.")
def test_integration(integration_id: str, kind: str) -> None:
    """Test an integration.

    Example:

        octomil integrations test abc-123 --kind metrics
    """
    api = _get_integrations_api()
    try:
        if kind == "metrics":
            result = api.test_metrics_integration(integration_id)
        else:
            result = api.test_log_integration(integration_id)
        click.echo(click.style("Test passed!", fg="green"))
        if isinstance(result, dict) and result.get("message"):
            click.echo(f"  {result['message']}")
    except Exception as exc:
        click.echo(click.style(f"Test failed: {exc}", fg="red"), err=True)
        sys.exit(1)


@integrations.command("connect-otlp")
@click.option("--name", default="OTLP Collector", help="Display name.")
@click.option("--endpoint", required=True, help="OTLP collector base URL (e.g. http://collector:4318).")
@click.option("--headers-json", default=None, help='Auth headers as JSON: \'{"Authorization": "Bearer token"}\'')
def connect_otlp(name: str, endpoint: str, headers_json: Optional[str]) -> None:
    """Connect an OTLP collector for both metrics and logs.

    Single command to configure your OpenTelemetry collector. Creates both
    metrics and log integrations pointing at the same endpoint.

    Examples:

        octomil integrations connect-otlp --endpoint http://collector:4318
        octomil integrations connect-otlp --endpoint https://otlp.grafana.net \\
            --headers-json '{"Authorization": "Basic abc123"}'
    """
    api = _get_integrations_api()

    headers = json.loads(headers_json) if headers_json else None

    click.echo(f"Connecting OTLP collector: {endpoint}")
    try:
        result = api.connect_otlp_collector(name, endpoint, headers=headers)
        click.echo(click.style("  Metrics integration created", fg="green"))
        click.echo(click.style("  Log integration created", fg="green"))
        click.echo(f"\n  Metrics → {endpoint}/v1/metrics")
        click.echo(f"  Logs    → {endpoint.rstrip('/')}/v1/logs")
        click.echo("\nVerify with: octomil integrations list")
    except Exception as exc:
        click.echo(f"Failed: {exc}", err=True)
        sys.exit(1)


def register(cli: click.Group) -> None:
    """Register integrations commands with the top-level CLI group."""
    cli.add_command(integrations)
