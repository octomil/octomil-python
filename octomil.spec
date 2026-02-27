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
    # SDK modules imported by __init__.py
    "octomil.model",
    "octomil.client",
    "octomil.telemetry",
    "octomil.enterprise",
    "octomil.decomposer",
    "octomil.routing",
    # Engines — lazily imported, PyInstaller can't detect them.
    "octomil.engines.mlx_engine",
    "octomil.engines.mlc_engine",
    "octomil.engines.llamacpp_engine",
    "octomil.engines.mnn_engine",
    "octomil.engines.executorch_engine",
    "octomil.engines.ort_engine",
    "octomil.engines.whisper_engine",
    "octomil.engines.echo_engine",
    "octomil.engines.cactus_engine",
    "octomil.engines.samsung_one_engine",
    "octomil.engines.base",
    "octomil.engines.registry",
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
    "qrcode",
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
a = Analysis(
    ["octomil/__main__.py"],
    pathex=[],
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
