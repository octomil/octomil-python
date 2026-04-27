# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for octomil standalone binary.

Build with::

    pyinstaller octomil.spec

The resulting binary lands in ``dist/octomil``.
"""


block_cipher = None

# ---------------------------------------------------------------------------
# Hidden imports — engines are lazily imported, PyInstaller can't detect them.
# ---------------------------------------------------------------------------
hidden_imports = [
    # Top-level octomil modules that shadow inner octomil.python.octomil.*
    # Must be listed first so PyInstaller resolves them correctly.
    "octomil.auth",
    # Generated contract types (some are symlinks PyInstaller can't follow)
    "octomil._generated",
    "octomil._generated.auth_type",
    "octomil._generated.compatibility_level",
    "octomil._generated.device_class",
    "octomil._generated.error_code",
    "octomil._generated.finish_reason",
    "octomil._generated.model_status",
    "octomil._generated.otlp_resource_attributes",
    "octomil._generated.principal_type",
    "octomil._generated.scope",
    "octomil._generated.telemetry_events",
    # SDK modules imported by __init__.py
    "octomil.model",
    "octomil.client",
    "octomil.telemetry",
    "octomil.enterprise",
    "octomil.decomposer",
    "octomil.routing",
    # Runtime — lazily imported, PyInstaller can't detect them.
    "octomil.runtime.core.base",
    "octomil.runtime.core.model_runtime",
    "octomil.runtime.core.registry",
    "octomil.runtime.core.adapter",
    "octomil.runtime.core.engine_bridge",
    "octomil.runtime.core.types",
    "octomil.runtime.core.policy",
    "octomil.runtime.core.router",
    "octomil.runtime.core.cloud_runtime",
    "octomil.runtime.engines.registry",
    "octomil.runtime.engines.mlx.engine",
    "octomil.runtime.engines.llamacpp.engine",
    "octomil.runtime.engines.ort.engine",
    "octomil.runtime.engines.whisper.engine",
    "octomil.runtime.engines.sherpa.engine",
    "octomil.runtime.engines.echo.engine",
    "octomil.runtime.engines.experimental.mlc.engine",
    "octomil.runtime.engines.experimental.mnn.engine",
    "octomil.runtime.engines.experimental.executorch.engine",
    "octomil.runtime.engines.experimental.cactus.engine",
    "octomil.runtime.engines.experimental.samsung_one.engine",
    "octomil.models.catalog",
    "octomil.models.parser",
    "octomil.models.resolver",
    "octomil.models._types",
    "octomil.cli",
    "octomil.cli_helpers",
    "octomil.commands",
    "octomil.commands.serve",
    "octomil.commands.model_ops",
    "octomil.commands.deploy",
    "octomil.commands.benchmark",
    "octomil.commands.enterprise",
    "octomil.commands.federation",
    "octomil.commands.interactive",
    "octomil.commands.completions",
    "octomil.commands.setup",
    "octomil.setup",
    "octomil.demos",
    "octomil.cli_hw",
    "octomil.sources",
    "octomil.sources.resolver",
    "huggingface_hub",
    # click needs these for shell completion
    "click.shell_completion",
    # httpx transport backends
    "httpcore",
    "httpcore._async",
    "httpcore._sync",
    "h11",
    # Other runtime deps
    "psutil",
    "segno",
    # Serve deps (fastapi + uvicorn + pydantic)
    "fastapi",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "fastapi.responses",
    "starlette",
    "starlette.routing",
    "starlette.responses",
    "starlette.middleware",
    "uvicorn",
    "uvicorn.config",
    "uvicorn.main",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "pydantic",
    "pydantic._internal",
    "anyio",
    "anyio._backends",
    "anyio._backends._asyncio",
]

# ---------------------------------------------------------------------------
# Data files — model catalog is pure Python so no extra data needed.
# Include the package metadata for importlib.metadata / pkg_resources.
# ---------------------------------------------------------------------------
datas = []

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
import os
REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(REPO_ROOT, "octomil", "__main__.py")],
    pathex=[REPO_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy optional deps that users install separately.
        # The binary is for the CLI (serve, deploy, scan, etc.) — not for
        # the training/FL SDK which requires torch anyway.
        "torch",
        "tensorflow",
        "onnxruntime",
        "mlx",
        "mlx_lm",
        "mlc_llm",
        "llama_cpp",
        "pywhispercpp",
        "executorch",
        "MNN",
        "numpy",
        "scipy",
        "matplotlib",
        "tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ---------------------------------------------------------------------------
# Bundle as directory (--onedir) — instant startup, no temp extraction.
# The install script archives the whole directory for distribution.
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="octomil",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="octomil",
)
