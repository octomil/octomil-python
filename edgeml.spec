# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for edgeml standalone binary.

Build with::

    pyinstaller edgeml.spec

The resulting binary lands in ``dist/edgeml``.
"""


block_cipher = None

# ---------------------------------------------------------------------------
# Hidden imports — engines are lazily imported, PyInstaller can't detect them.
# ---------------------------------------------------------------------------
hidden_imports = [
    "edgeml.engines.mlx_engine",
    "edgeml.engines.mlc_engine",
    "edgeml.engines.llamacpp_engine",
    "edgeml.engines.mnn_engine",
    "edgeml.engines.executorch_engine",
    "edgeml.engines.ort_engine",
    "edgeml.engines.whisper_engine",
    "edgeml.engines.echo_engine",
    "edgeml.engines.base",
    "edgeml.engines.registry",
    "edgeml.models.catalog",
    "edgeml.models.parser",
    "edgeml.models.resolver",
    "edgeml.models._types",
    "edgeml.demos",
    "edgeml.sources",
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
    ["edgeml/cli.py"],
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
# Bundle into a single file
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="edgeml",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
