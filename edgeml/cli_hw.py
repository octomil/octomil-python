from __future__ import annotations

import json as json_mod
import logging

import click

logger = logging.getLogger(__name__)


@click.group()
def hw() -> None:
    """Hardware detection and model recommendations."""


@hw.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option(
    "--priority",
    "-p",
    type=click.Choice(["speed", "quality", "balanced"]),
    default="balanced",
    help="Recommendation priority (default: balanced).",
)
def detect(as_json: bool, priority: str) -> None:
    """Detect hardware and show recommended models.

    Prints GPU, CPU, and RAM info, then recommends which models
    your hardware can run — sorted by the chosen priority.
    """
    from .hardware import detect_hardware
    from .model_optimizer import ModelOptimizer

    profile = detect_hardware(force=True)
    optimizer = ModelOptimizer(profile)
    recs = optimizer.recommend(priority=priority)
    env = optimizer.env_vars()

    if as_json:
        import dataclasses

        def _serialize(obj: object) -> object:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return dataclasses.asdict(obj)
            if hasattr(obj, "value"):
                return obj.value
            return str(obj)

        click.echo(
            json_mod.dumps(
                {
                    "hardware": dataclasses.asdict(profile),
                    "priority": priority,
                    "recommendations": [dataclasses.asdict(r) for r in recs],
                    "env_vars": env,
                },
                indent=2,
                default=_serialize,
            )
        )
        return

    # Pretty print hardware
    click.secho("\n  Hardware Profile\n", bold=True)

    # GPU
    if profile.gpu and profile.gpu.gpus:
        click.secho("  GPU", bold=True)
        for gpu in profile.gpu.gpus:
            click.echo(f"    [{gpu.index}] {gpu.name}")
            click.echo(
                f"        VRAM: {gpu.memory.total_gb:.1f} GB "
                f"(free: {gpu.memory.free_gb:.1f} GB)"
            )
            if gpu.compute_capability:
                click.echo(f"        Compute: {gpu.compute_capability}")
            if gpu.architecture:
                click.echo(f"        Arch: {gpu.architecture}")
        click.echo(f"    Total VRAM: {profile.gpu.total_vram_gb:.1f} GB")
        click.echo(f"    Backend: {profile.gpu.backend}")
        if profile.gpu.driver_version:
            click.echo(f"    Driver: {profile.gpu.driver_version}")
        if profile.gpu.cuda_version:
            click.echo(f"    CUDA: {profile.gpu.cuda_version}")
        if profile.gpu.rocm_version:
            click.echo(f"    ROCm: {profile.gpu.rocm_version}")
        if profile.gpu.detection_method:
            click.echo(f"    Detection: {profile.gpu.detection_method}")
    else:
        click.secho("  GPU: None detected", fg="yellow")

    click.echo()

    # CPU
    click.secho("  CPU", bold=True)
    click.echo(f"    {profile.cpu.brand}")
    click.echo(f"    Cores: {profile.cpu.cores} ({profile.cpu.threads} threads)")
    click.echo(f"    Speed: {profile.cpu.base_speed_ghz:.2f} GHz")
    click.echo(f"    Arch: {profile.cpu.architecture}")
    features = []
    if profile.cpu.has_avx512:
        features.append("AVX-512")
    if profile.cpu.has_avx2:
        features.append("AVX2")
    if profile.cpu.has_neon:
        features.append("NEON")
    if features:
        click.echo(f"    Features: {', '.join(features)}")
    click.echo(f"    Est. GFLOPS: {profile.cpu.estimated_gflops:.1f}")

    click.echo()

    # RAM
    click.secho("  Memory", bold=True)
    click.echo(f"    Total: {profile.total_ram_gb:.1f} GB")
    click.echo(f"    Available: {profile.available_ram_gb:.1f} GB")

    click.echo()
    click.echo(f"  Best backend: {profile.best_backend}")

    # Diagnostics
    if profile.diagnostics:
        click.echo()
        click.secho("  Diagnostics", bold=True, fg="yellow")
        for d in profile.diagnostics:
            click.echo(f"    \u2022 {d}")

    # Recommendations
    click.echo()
    click.secho(f"  Recommended models ({priority})\n", bold=True)
    for rec in recs:
        if rec.speed.tokens_per_second >= 15:
            speed_color = "green"
        elif rec.speed.tokens_per_second >= 5:
            speed_color = "yellow"
        else:
            speed_color = "red"
        click.echo(f"    {rec.model_size} @ {rec.quantization}")
        click.echo(f"      {rec.reason}")
        click.secho(
            f"      Speed: {rec.speed.tokens_per_second:.1f} tok/s "
            f"({rec.speed.confidence})",
            fg=speed_color,
        )
        click.echo(f"      $ {rec.serve_command}")
        click.echo()

    if env:
        click.secho("  Environment variables:", bold=True)
        for k, v in env.items():
            click.echo(f"    export {k}={v}")
    click.echo()


# Command to register in main CLI
def interactive_cmd_factory(cli_group: click.Group) -> click.Command:
    """Create the interactive command that needs a reference to the CLI group."""

    @click.command("interactive")
    def interactive() -> None:
        """Open interactive command panel."""
        from .interactive import launch_interactive

        launch_interactive(cli_group)

    return interactive
