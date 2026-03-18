#!/usr/bin/env python3
"""Generate octomil/models/_embedded_catalog.py from the server catalog seed.

Reads the server's catalog_seed.py (or a running server endpoint) and
transforms the seed data into the embedded catalog format used by the
Python SDK for offline bootstrap.

Only **blessed** families are included by default.

Usage:
    # From the server seed file (default path assumes sibling repos)
    python scripts/sync_embedded_catalog.py --from-seed ../octomil-server/server/app/services/catalog_seed.py

    # From a running server
    python scripts/sync_embedded_catalog.py --from-url https://api.octomil.com

    # Filter to specific families
    python scripts/sync_embedded_catalog.py --from-seed ... --families gemma-3 whisper
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo layout: octomil-python/scripts/sync_embedded_catalog.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = _REPO_ROOT / "octomil" / "models" / "_embedded_catalog.py"

_HEADER = textwrap.dedent('''\
    """Embedded minimal catalog — offline fallback when server and disk cache are unavailable.

    Contains a small set of blessed models so the SDK can bootstrap without
    network access. This data is intentionally minimal: just enough to resolve
    the most common model names to downloadable artifacts.

    Format matches the server's ``GET /api/v2/catalog/manifest`` response:
    nested ``{family_name: {variants: {variant_name: {versions: ...}}}}``.

    This file is auto-generated from the server manifest. Do not edit by hand.
    """

    from __future__ import annotations

''')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_id(name: str) -> str:
    """Turn a variant/family name into a slug suitable for IDs.

    gemma3-1b -> gemma3_1b
    qwen2.5-3b -> qwen25_3b
    whisper-large-v3 -> whisper_large_v3
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _make_version_id(variant_name: str) -> str:
    """embedded-<sanitized>-v1"""
    return f"embedded-{_sanitize_id(variant_name)}-v1"


def _make_package_id(variant_name: str, fmt: str, quant: str) -> str:
    """pkg_<sanitized>_<format>_<quant>"""
    return f"pkg_{_sanitize_id(variant_name)}_{_sanitize_id(fmt)}_{_sanitize_id(quant)}"


def _infer_resource_path(uri: str, fmt: str) -> str:
    """Extract the filename from a hf:// URI, or '.' for directory URIs."""
    if fmt == "mlx":
        return "."
    # hf://org/repo/file.gguf -> file.gguf
    parts = uri.removeprefix("hf://").split("/")
    if len(parts) >= 3:
        return "/".join(parts[2:])
    return "."


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")


# ---------------------------------------------------------------------------
# Seed data extraction
# ---------------------------------------------------------------------------


def _extract_seed_families_from_file(seed_path: Path) -> list[dict[str, Any]]:
    """Parse _SEED_FAMILIES from catalog_seed.py using AST.

    We parse the Python file's AST, find the assignment to _SEED_FAMILIES,
    then evaluate it as a literal (safe — no arbitrary code execution).
    """
    source = seed_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        # Handle both `_SEED_FAMILIES = [...]` and `_SEED_FAMILIES: list[...] = [...]`
        target_name: str | None = None
        value_node: ast.expr | None = None

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_SEED_FAMILIES":
                    target_name = target.id
                    value_node = node.value
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "_SEED_FAMILIES":
                target_name = node.target.id
                value_node = node.value

        if target_name and value_node is not None:
            value_source = ast.get_source_segment(source, value_node)
            if value_source is None:
                raise ValueError("Could not extract _SEED_FAMILIES source segment")
            return ast.literal_eval(value_source)  # type: ignore[return-value]

    raise ValueError(f"_SEED_FAMILIES not found in {seed_path}")


def _fetch_seed_families_from_url(base_url: str) -> list[dict[str, Any]]:
    """Fetch the catalog manifest from a running server and convert to seed format."""
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx is required for --from-url mode. pip install httpx", file=sys.stderr)
        sys.exit(1)

    url = f"{base_url.rstrip('/')}/api/v2/catalog/manifest"
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    manifest = resp.json()

    # The server manifest is already in the nested format.
    # Convert it back to seed-like format for uniform processing.
    families: list[dict[str, Any]] = []
    for family_name, family_data in manifest.items():
        fd: dict[str, Any] = {
            "name": family_name,
            "vendor": family_data.get("vendor", ""),
            "description": family_data.get("description", ""),
            "modalities": family_data.get("modalities", ["text"]),
            "license": family_data.get("license", ""),
            "homepage_url": family_data.get("homepage_url", ""),
            "support_tier": "blessed",  # server only returns blessed by default
            "variants": [],
        }
        for variant_name, variant_data in family_data.get("variants", {}).items():
            vd: dict[str, Any] = {
                "name": variant_name,
                "parameter_count": variant_data.get("parameter_count", ""),
                "context_length": variant_data.get("context_length"),
                "quantizations": variant_data.get("quantizations", []),
                "packages": [],
            }
            for _ver_key, ver_data in variant_data.get("versions", {}).items():
                for pkg in ver_data.get("packages", []):
                    pd: dict[str, Any] = {
                        "platform": pkg.get("platform", "macos"),
                        "format": pkg.get("artifact_format", "gguf"),
                        "executor": pkg.get("runtime_executor", "llamacpp"),
                        "quant": pkg.get("quantization", "Q4_K_M"),
                        "default": pkg.get("is_default", False),
                    }
                    resources = pkg.get("resources", [])
                    if resources:
                        pd["uri"] = resources[0].get("uri", "")
                    vd["packages"].append(pd)
            fd["variants"].append(vd)
        families.append(fd)

    return families


# ---------------------------------------------------------------------------
# Transform seed -> embedded catalog
# ---------------------------------------------------------------------------


def _transform_package(
    variant_name: str,
    pkg: dict[str, Any],
    family_tier: str,
) -> dict[str, Any]:
    """Transform a seed package dict into embedded catalog format."""
    fmt = pkg["format"]
    quant = pkg["quant"]

    # Multi-resource packages (vision projectors, sherpa encoder/decoder/joiner)
    if "resources" in pkg:
        resources = [
            {
                "kind": r["kind"],
                "uri": r["uri"],
                "path": _infer_resource_path(r["uri"], fmt),
                "required": True,
                "load_order": r.get("load_order", 0),
            }
            for r in pkg["resources"]
        ]
    else:
        uri = pkg["uri"]
        resources = [
            {
                "kind": "weights",
                "uri": uri,
                "path": _infer_resource_path(uri, fmt),
                "required": True,
            }
        ]

    return {
        "id": _make_package_id(variant_name, fmt, quant),
        "platform": pkg["platform"],
        "artifact_format": fmt,
        "runtime_executor": pkg["executor"],
        "quantization": quant,
        "support_tier": pkg.get("support_tier", family_tier),
        "is_default": pkg.get("default", False),
        "resources": resources,
    }


def _transform_variant(
    variant: dict[str, Any],
    family_modalities: list[str],
    family_tier: str,
    released_at: str,
) -> dict[str, Any]:
    """Transform a seed variant dict into embedded catalog format."""
    name = variant["name"]
    packages = [_transform_package(name, pkg, family_tier) for pkg in variant["packages"]]

    return {
        "id": name,
        "parameter_count": variant["parameter_count"],
        "context_length": variant.get("context_length"),
        "modalities": family_modalities,
        "quantizations": variant.get("quantizations", []),
        "versions": {
            "1.0.0": {
                "id": _make_version_id(name),
                "version": "1.0.0",
                "lifecycle": "active",
                "released_at": released_at,
                "min_sdk_version": None,
                "packages": packages,
            }
        },
    }


def _transform_family(family: dict[str, Any], released_at: str) -> dict[str, Any]:
    """Transform a seed family dict into embedded catalog format."""
    modalities = family.get("modalities", ["text"])
    tier = family.get("support_tier", "blessed")

    variants = {}
    for vd in family["variants"]:
        variants[vd["name"]] = _transform_variant(vd, modalities, tier, released_at)

    return {
        "id": family["name"],
        "vendor": family["vendor"],
        "description": family.get("description", ""),
        "modalities": modalities,
        "license": family.get("license", ""),
        "homepage_url": family.get("homepage_url", ""),
        "variants": variants,
    }


def build_manifest(
    seed_families: list[dict[str, Any]],
    *,
    families_filter: list[str] | None = None,
    blessed_only: bool = True,
    released_at: str | None = None,
) -> dict[str, Any]:
    """Build the EMBEDDED_MANIFEST dict from seed data.

    Args:
        seed_families: List of seed family dicts (from _SEED_FAMILIES).
        families_filter: If set, only include these family names.
        blessed_only: If True (default), only include blessed families.
        released_at: ISO timestamp for version release date. Defaults to today.

    Returns:
        Nested dict matching the embedded catalog format.
    """
    if released_at is None:
        released_at = _today_iso()

    manifest: dict[str, Any] = {}
    for fd in seed_families:
        name = fd["name"]
        tier = fd.get("support_tier", "experimental")

        if blessed_only and tier != "blessed":
            continue
        if families_filter and name not in families_filter:
            continue

        manifest[name] = _transform_family(fd, released_at)

    return manifest


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _quote(s: str) -> str:
    """Double-quote a string, escaping internal double quotes."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_value(value: Any, indent: int = 0) -> str:
    """Format a Python value as source code with proper indentation."""
    prefix = "    " * indent

    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(v, str) for v in value):
            items = ", ".join(_quote(v) for v in value)
            line = f"[{items}]"
            if len(line) <= 80:
                return line
        items_str = ",\n".join(f"{prefix}    {_format_value(v, indent + 1)}" for v in value)
        return f"[\n{items_str},\n{prefix}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        entries: list[str] = []
        for k, v in value.items():
            formatted_v = _format_value(v, indent + 1)
            entries.append(f"{prefix}    {_quote(k)}: {formatted_v}")
        items_str = ",\n".join(entries)
        return f"{{\n{items_str},\n{prefix}}}"

    return repr(value)


def generate_source(manifest: dict[str, Any]) -> str:
    """Generate the full _embedded_catalog.py source string."""
    body = _format_value(manifest, 0)
    return f"{_HEADER}EMBEDDED_MANIFEST: dict = {body}\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate _embedded_catalog.py from server catalog seed.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--from-seed",
        type=Path,
        metavar="PATH",
        help="Path to catalog_seed.py",
    )
    source_group.add_argument(
        "--from-url",
        type=str,
        metavar="URL",
        help="Base URL of running Octomil server",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        metavar="NAME",
        help="Only include these families (default: all blessed)",
    )
    parser.add_argument(
        "--all-tiers",
        action="store_true",
        help="Include all support tiers, not just blessed",
    )
    parser.add_argument(
        "--released-at",
        type=str,
        default=None,
        help="ISO timestamp for version release date (default: today)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_PATH,
        help=f"Output file (default: {_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print to stdout instead of writing to file",
    )

    args = parser.parse_args()

    # Load seed data
    if args.from_seed:
        seed_path = Path(args.from_seed).resolve()
        if not seed_path.exists():
            print(f"ERROR: Seed file not found: {seed_path}", file=sys.stderr)
            sys.exit(1)
        seed_families = _extract_seed_families_from_file(seed_path)
        print(f"Loaded {len(seed_families)} families from {seed_path}")
    else:
        seed_families = _fetch_seed_families_from_url(args.from_url)
        print(f"Loaded {len(seed_families)} families from {args.from_url}")

    # Build manifest
    manifest = build_manifest(
        seed_families,
        families_filter=args.families,
        blessed_only=not args.all_tiers,
        released_at=args.released_at,
    )

    if not manifest:
        print("WARNING: No families matched the filter criteria.", file=sys.stderr)
        sys.exit(1)

    # Sort families alphabetically for stable output
    manifest = dict(sorted(manifest.items()))

    family_names = list(manifest.keys())
    variant_count = sum(len(f["variants"]) for f in manifest.values())
    print(f"Generating embedded catalog: {len(manifest)} families, {variant_count} variants")
    print(f"  Families: {', '.join(family_names)}")

    # Generate source
    source = generate_source(manifest)

    if args.dry_run:
        print("\n--- Generated source ---")
        print(source)
        return

    # Write
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source)
    print(f"Wrote {output_path} ({len(source)} bytes)")


if __name__ == "__main__":
    main()
