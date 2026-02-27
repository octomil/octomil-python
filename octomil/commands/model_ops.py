"""Model operations — push, pull, check, convert, list, models, scan."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import click

from octomil.cli_helpers import (
    _auto_optimize,
    _complete_model_name,
    _get_api_key,
    _get_client,
    _get_org_id,
    _has_explicit_quant,
)


def register(cli: click.Group) -> None:
    cli.add_command(push)
    cli.add_command(pull)
    cli.add_command(check)
    cli.add_command(scan)
    cli.add_command(convert)
    cli.add_command(list_models_cmd)
    cli.add_command(models)


@click.command()
@click.argument(
    "path", required=False, default=None, shell_complete=_complete_model_name
)
@click.option(
    "--model-id",
    "-m",
    default=None,
    help="Model ID in the registry. Inferred from path or model name if omitted.",
    shell_complete=_complete_model_name,
)
@click.option(
    "--version", "-v", default="1.0.0", help="Semantic version (default: 1.0.0)."
)
@click.option("--description", "-d", default=None, help="Version description.")
@click.option(
    "--formats",
    "-f",
    default=None,
    help="Comma-separated target formats for server-side conversion.",
)
@click.option(
    "--use-case",
    default=None,
    help="Model use case (e.g. text_generation, image_classification, nlp). Defaults to auto-detect.",
)
def push(
    path: Optional[str],
    model_id: Optional[str],
    version: str,
    description: Optional[str],
    formats: Optional[str],
    use_case: Optional[str],
) -> None:
    """Push a model to Octomil.

    PATH can be a local file, or a model name for server-side import:

    \b
        octomil push phi-4-mini
        octomil push deepseek-r1-7b --version 2.0.0
        octomil push hf:microsoft/Phi-4-mini-instruct
        octomil push ./model.safetensors --model-id my-model
    """
    if not path and not model_id:
        click.echo(
            "Error: provide a path or model name.\n\n"
            "  octomil push phi-4-mini\n"
            "  octomil push ./model.safetensors --model-id my-model",
            err=True,
        )
        sys.exit(1)

    # ── Local file or directory upload ────────────────────────────────
    if path and (os.path.isfile(path) or os.path.isdir(path)):
        if os.path.isdir(path):
            from octomil.client import _find_model_file

            model_file = _find_model_file(path)
            if not model_file:
                click.echo(
                    f"Error: no model file found in {path}\n"
                    "  Expected: .safetensors, .gguf, .pt, .pth, .bin, .onnx",
                    err=True,
                )
                sys.exit(1)
            upload_path = model_file
        else:
            upload_path = path
        resolved_name = model_id or os.path.splitext(os.path.basename(upload_path))[0]
        client = _get_client()
        click.echo(f"  Uploading {resolved_name} v{version}...")
        push_kwargs: dict[str, Any] = {
            "name": resolved_name,
            "version": version,
            "description": description,
            "formats": formats,
        }
        if use_case:
            push_kwargs["use_case"] = use_case
        result = client.push(upload_path, **push_kwargs)
        click.echo(click.style(f"\n  Done — {resolved_name} v{version}", fg="green"))
        for fmt, info in result.get("formats", {}).items():
            click.echo(f"    {fmt}: {info}")
        _print_sdk_snippet(resolved_name, version)
        return

    # ── Model name → server-side HuggingFace import ───────────────────
    model_name = path or model_id
    assert model_name is not None
    resolved_name = model_id or model_name.split("/")[-1].split(":")[-1]

    from octomil.sources.resolver import resolve_hf_repo

    hf_repo = resolve_hf_repo(model_name)
    if not hf_repo:
        click.echo(f"Error: unknown model '{model_name}'", err=True)
        click.echo(
            "  Use a known model name (phi-4-mini, gemma-4b, llama-8b, ...)\n"
            "  or a HuggingFace repo: hf:org/model",
            err=True,
        )
        sys.exit(1)

    click.echo(f"  Importing {resolved_name} from {hf_repo}...")

    client = _get_client()
    try:
        result = client.import_from_hf(
            repo_id=hf_repo,
            name=resolved_name,
            version=version,
            description=description,
            use_case=use_case,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(click.style(f"\n  Done — {resolved_name} v{version}", fg="green"))
    if result.get("files"):
        click.echo(f"    Files: {', '.join(result['files'])}")
    if result.get("storage_type"):
        click.echo(f"    Storage: {result['storage_type']}")
    _print_sdk_snippet(resolved_name, version)


def _print_sdk_snippet(model_name: str, version: str) -> None:
    """Print ready-to-paste SDK snippets with real credentials."""

    api_key = _get_api_key()
    org_id = _get_org_id() or "<your-org-id>"

    click.echo("\n  Add to your app:\n")
    click.secho("  Swift (iOS)", bold=True)
    click.echo(
        f"    // Swift Package Manager: https://github.com/octomil/octomil-ios\n"
        f"    import OctomilSDK\n"
        f"\n"
        f'    let client = OctomilClient(apiKey: "{api_key}", orgId: "{org_id}")\n'
        f'    let model = try await client.pull("{model_name}", version: "{version}")\n'
    )
    click.secho("  Kotlin (Android)", bold=True)
    click.echo(
        f'    // implementation("com.octomil:octomil-android:1.0.0")\n'
        f"    import com.octomil.sdk.OctomilClient\n"
        f"\n"
        f'    val client = OctomilClient(apiKey = "{api_key}", orgId = "{org_id}", context = this)\n'
        f'    val model = client.pull("{model_name}", version = "{version}")\n'
    )
    click.secho("  Python", bold=True)
    click.echo(
        f"    # pip install octomil-sdk\n"
        f"    import octomil\n"
        f"\n"
        f'    client = octomil.Client(api_key="{api_key}", org_id="{org_id}")\n'
        f'    text = client.predict("{model_name}", [{{"role": "user", "content": "Hello"}}])\n'
    )
    click.secho("  Node.js", bold=True)
    click.echo(
        f"    // npm install @octomil/sdk\n"
        f'    import {{ OctomilClient }} from "@octomil/sdk";\n'
        f"\n"
        f'    const client = new OctomilClient({{ apiKey: "{api_key}", orgId: "{org_id}" }});\n'
        f'    const result = await client.predict("{model_name}", {{ text: "Hello" }});\n'
    )


@click.command()
@click.argument("name", shell_complete=_complete_model_name)
@click.option(
    "--version", "-v", default=None, help="Version to download. Defaults to latest."
)
@click.option(
    "--format",
    "-f",
    "fmt",
    default=None,
    help="Model format (onnx, coreml, tflite).",
)
@click.option("--output", "-o", default=".", help="Output directory.")
def pull(name: str, version: Optional[str], fmt: Optional[str], output: str) -> None:
    """Download a model from the registry.

    NAME accepts Ollama-style model:variant syntax:

    \b
        octomil pull gemma-1b                  # auto-picks best quant for your hw
        octomil pull gemma-1b:8bit             # 8-bit quantization (explicit)
        octomil pull sentiment-v1 --version 1.0.0 --format coreml
    """
    if not _has_explicit_quant(name):
        best_quant = _auto_optimize(name)
        if best_quant:
            name = f"{name}:{best_quant.lower()}"
            click.echo(f"    Pulling as: {name}")

    client = _get_client()
    ver_str = version or "latest"
    click.echo(f"Pulling {name} v{ver_str}...")

    result = client.pull(name, version=version, format=fmt, destination=output)
    click.echo(f"Downloaded: {result['model_path']}")


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--devices",
    "-d",
    default=None,
    help="Comma-separated device profiles (e.g. iphone_15_pro,pixel_8).",
)
def check(model_path: str, devices: Optional[str]) -> None:
    """Check device compatibility for a local model file.

    Analyzes the model and reports file size, format, and basic
    compatibility with edge devices.
    """
    file_size = os.path.getsize(model_path)
    ext = os.path.splitext(model_path)[1].lower()

    click.echo(f"Checking: {model_path}")
    click.echo(f"Size: {file_size / 1024 / 1024:.1f} MB")

    format_map = {
        ".pt": "PyTorch",
        ".pth": "PyTorch",
        ".onnx": "ONNX",
        ".mlmodel": "CoreML",
        ".mlpackage": "CoreML",
        ".tflite": "TFLite",
        ".gguf": "GGUF",
    }
    fmt = format_map.get(ext, f"Unknown ({ext})")
    click.echo(f"Format: {fmt}")

    size_mb = file_size / 1024 / 1024
    click.echo("\nDevice compatibility:")
    if size_mb < 50:
        click.echo("  iPhone 15 Pro:  compatible (NPU)")
        click.echo("  Pixel 8:        compatible (NNAPI)")
        click.echo("  Raspberry Pi 4: compatible (CPU)")
    elif size_mb < 500:
        click.echo("  iPhone 15 Pro:  compatible (NPU)")
        click.echo("  Pixel 8:        compatible (NNAPI)")
        click.echo("  Raspberry Pi 4: may require quantization")
    else:
        click.echo("  iPhone 15 Pro:  may require quantization")
        click.echo("  Pixel 8:        may require quantization")
        click.echo("  Raspberry Pi 4: too large — quantize or prune")

    click.echo("\nRecommendations:")
    if ext in (".pt", ".pth"):
        click.echo("  - Convert to ONNX: octomil convert model.pt --target onnx")
        click.echo(
            "  - Convert to CoreML (iOS): octomil convert model.pt --target coreml"
        )
        click.echo(
            "  - Convert to TFLite (Android): octomil convert model.pt --target tflite"
        )
    elif ext == ".onnx":
        click.echo("  - ONNX is cross-platform — ready for deployment")
        click.echo(
            "  - Convert to CoreML (iOS): octomil convert model.onnx --target coreml"
        )
    elif ext == ".gguf":
        click.echo("  - GGUF models work with llama.cpp backend")
        click.echo("  - Serve locally: octomil serve model.gguf")
    else:
        click.echo("  - No specific recommendations for this format")


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format (text or json).",
)
@click.option(
    "--platform",
    "-p",
    default=None,
    type=click.Choice(["ios", "android", "python"]),
    help="Filter by platform. Default: scan all platforms.",
)
def scan(path: str, output_format: str, platform: str | None) -> None:
    """Scan a codebase for ML inference points.

    Walks PATH looking for model loading, inference calls, and model files
    across iOS (CoreML), Android (TFLite), and Python (PyTorch, ONNX,
    OpenAI, HuggingFace, MLX).

    Reports where you can add Octomil.wrap() for telemetry.

    \b
    Examples:
        octomil scan ./MyApp
        octomil scan ./MyApp --format json
        octomil scan ./MyApp --platform ios
        octomil scan ./MyApp --platform android
    """
    from octomil.scanner import format_json, format_text, scan_directory

    click.echo("Scanning for inference points...\n")

    try:
        points = scan_directory(path, platform=platform)
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(format_json(points))
    else:
        click.echo(format_text(points))


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--target",
    "-t",
    default="onnx",
    help="Comma-separated target formats: onnx, coreml, tflite.",
)
@click.option("--output", "-o", default="./converted", help="Output directory.")
@click.option(
    "--input-shape",
    default="1,3,224,224",
    help="Comma-separated input tensor shape (e.g. 1,3,224,224).",
)
def convert(model_path: str, target: str, output: str, input_shape: str) -> None:
    """Convert a model to edge formats locally.

    Converts MODEL_PATH (PyTorch .pt/.pth) to target formats. Runs entirely
    on your machine — no account needed.

    Examples:

        octomil convert model.pt --target onnx
        octomil convert model.pt --target onnx,coreml --input-shape 1,3,224,224
    """
    formats = [f.strip().lower() for f in target.split(",")]
    shape = [int(d) for d in input_shape.split(",")]
    os.makedirs(output, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    click.echo(f"Converting {model_path} → {', '.join(formats)}")
    click.echo(f"Input shape: {shape}")
    click.echo(f"Output: {output}")

    results: dict[str, str] = {}

    if "onnx" in formats:
        try:
            import torch

            model = torch.jit.load(model_path)
            model.eval()
            sample = torch.randn(*shape)
            onnx_path = os.path.join(output, f"{model_name}.onnx")
            torch.onnx.export(model, sample, onnx_path, opset_version=13)
            results["onnx"] = onnx_path
        except ImportError:
            click.echo("  onnx: requires torch — pip install torch", err=True)
        except Exception as exc:
            click.echo(f"  onnx: failed — {exc}", err=True)

    if "coreml" in formats:
        try:
            import coremltools as ct  # type: ignore[import-untyped]

            onnx_src = results.get("onnx")
            if not onnx_src:
                click.echo(
                    "  coreml: requires onnx conversion first — "
                    "include onnx in --target",
                    err=True,
                )
            else:
                ml_model = ct.converters.onnx.convert(model=onnx_src)
                coreml_path = os.path.join(output, f"{model_name}.mlmodel")
                ml_model.save(coreml_path)
                results["coreml"] = coreml_path
        except ImportError:
            click.echo(
                "  coreml: requires coremltools — pip install coremltools", err=True
            )
        except Exception as exc:
            click.echo(f"  coreml: failed — {exc}", err=True)

    if "tflite" in formats:
        try:
            import onnx  # type: ignore[import-untyped]
            from onnx_tf.backend import prepare  # type: ignore[import-untyped]

            onnx_src = results.get("onnx")
            if not onnx_src:
                click.echo(
                    "  tflite: requires onnx conversion first — "
                    "include onnx in --target",
                    err=True,
                )
            else:
                import tensorflow as tf  # type: ignore[import-untyped]

                onnx_model = onnx.load(onnx_src)
                tf_rep = prepare(onnx_model)
                tf_dir = os.path.join(output, f"{model_name}_tf")
                tf_rep.export_graph(tf_dir)
                converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
                tflite_model = converter.convert()
                tflite_path = os.path.join(output, f"{model_name}.tflite")
                with open(tflite_path, "wb") as f:
                    f.write(tflite_model)
                results["tflite"] = tflite_path
        except ImportError:
            click.echo(
                "  tflite: requires onnx, onnx-tf, tensorflow — "
                "pip install onnx onnx-tf tensorflow",
                err=True,
            )
        except Exception as exc:
            click.echo(f"  tflite: failed — {exc}", err=True)

    if results:
        click.echo("\nConverted:")
        for fmt, path in results.items():
            size_mb = os.path.getsize(path) / 1024 / 1024
            click.echo(f"  {fmt}: {path} ({size_mb:.1f} MB)")
    else:
        click.echo("\nNo conversions succeeded.", err=True)
        sys.exit(1)


@click.command("list")
@click.argument("model_family", required=False, default=None)
def list_models_cmd(model_family: Optional[str]) -> None:
    """List available model families and their variants.

    Without arguments, shows all available model families with
    default variant, parameter count, and supported engines.

    With a model family name, shows all variants for that family
    with engine-specific artifacts (MLX repos, GGUF files).

    \b
    Examples:
        octomil list
        octomil list gemma-4b
        octomil list llama-8b

    \b
    Use model:variant syntax with serve/benchmark/pull:
        octomil serve gemma-4b:8bit
        octomil benchmark llama-8b:fp16
    """
    from octomil.models.catalog import CATALOG, get_model

    if model_family is None:
        click.echo(
            f"{'Model':<18s} {'Publisher':<14s} {'Params':<8s} "
            f"{'Default':<10s} {'Variants'}"
        )
        click.echo("-" * 76)
        for name, entry in sorted(CATALOG.items()):
            variant_tags = ", ".join(sorted(entry.variants.keys()))
            click.echo(
                f"  {name:<16s} {entry.publisher:<14s} {entry.params:<8s} "
                f"{entry.default_quant:<10s} {variant_tags}"
            )
        click.echo(f"\n{len(CATALOG)} model families available.")
        click.echo("Use `octomil list <model>` to see variants and engine artifacts.")
        click.echo("Use `octomil serve <model>:<variant>` to serve a specific variant.")
    else:
        entry = get_model(model_family)
        if entry is None:
            import difflib

            suggestions = difflib.get_close_matches(
                model_family, CATALOG.keys(), n=3, cutoff=0.4
            )
            click.echo(f"Unknown model family: {model_family}", err=True)
            if suggestions:
                click.echo(f"Did you mean: {', '.join(suggestions)}?", err=True)
            click.echo(
                f"Available: {', '.join(sorted(CATALOG))}",
                err=True,
            )
            sys.exit(1)

        click.echo(f"{model_family} ({entry.publisher}, {entry.params})")
        click.echo(f"Default variant: {entry.default_quant}")
        engines_str = ", ".join(sorted(entry.engines))
        click.echo(f"Engines: {engines_str}")
        click.echo("")

        for quant, variant in sorted(entry.variants.items()):
            default_marker = " (default)" if quant == entry.default_quant else ""
            click.echo(f"  {model_family}:{quant}{default_marker}")
            if variant.mlx:
                click.echo(f"    mlx-lm:    {variant.mlx}")
            if variant.gguf:
                click.echo(
                    f"    llama.cpp: {variant.gguf.repo} ({variant.gguf.filename})"
                )
            if variant.source_repo:
                click.echo(f"    source:    {variant.source_repo}")
            click.echo("")


@click.command()
@click.option(
    "--source",
    type=click.Choice(["all", "ollama", "registry"]),
    default="all",
    help="Filter model source.",
)
def models(source: str) -> None:
    """List available models from ollama and the Octomil registry.

    \b
    Deploy ollama models directly:
        octomil deploy ollama://llama3.2 --phone
        octomil deploy ollama://gemma:2b --phone
    """
    if source in ("all", "ollama"):
        from octomil.ollama import is_ollama_running, list_ollama_models

        if is_ollama_running():
            ollama_models = list_ollama_models()
            if ollama_models:
                click.echo("Local (ollama):")
                click.echo(
                    f"  {'NAME':<20s}{'SIZE':>8s}   "
                    f"{'QUANT':<9s}{'FAMILY':<12s}DEPLOY URI"
                )
                for m in ollama_models:
                    deploy_uri = f"ollama://{m.name}"
                    click.echo(
                        f"  {m.name:<20s}{m.size_display:>8s}   "
                        f"{m.quantization:<9s}{m.family:<12s}{deploy_uri}"
                    )
            else:
                click.echo("Local (ollama): no models found")
        else:
            click.echo("Local (ollama): not running")

    if source in ("all", "registry"):
        api_key = _get_api_key()
        if api_key:
            try:
                client = _get_client()
                registry_data = client.list_models()
                registry_models: list[dict[str, Any]] = registry_data.get("models", [])
                if registry_models:
                    click.echo("Registry (octomil):")
                    for rm in registry_models:
                        name_val = rm.get("name", "unknown")
                        size = rm.get("size", 0)
                        fmt = rm.get("format", "unknown")
                        framework = rm.get("framework", "unknown")
                        size_mb = size / (1024 * 1024) if size else 0
                        click.echo(
                            f"  {name_val:<20s}{size_mb:>5.0f} MB   {fmt:<9s}{framework}"
                        )
                else:
                    click.echo("Registry (octomil): no models found")
            except Exception:
                click.echo("Registry (octomil): unable to fetch", err=True)
        elif source == "registry":
            click.echo("Registry (octomil): no API key — run `octomil login` first")
