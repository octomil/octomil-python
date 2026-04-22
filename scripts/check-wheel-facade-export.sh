#!/usr/bin/env bash
#
# Build the local source distribution as a wheel, install it into a clean
# virtualenv, and verify the public Octomil facade export matches the SDK
# quickstart surface.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
TMPDIR_PATH="$(mktemp -d)"

cleanup() {
    rm -rf "$TMPDIR_PATH"
}
trap cleanup EXIT

cd "$ROOT_DIR"

"$PYTHON_BIN" -m pip wheel --no-deps --wheel-dir "$TMPDIR_PATH/dist" .
"$PYTHON_BIN" -m venv "$TMPDIR_PATH/venv"
"$TMPDIR_PATH/venv/bin/python" -m pip install --upgrade pip >/dev/null
"$TMPDIR_PATH/venv/bin/python" -m pip install "$TMPDIR_PATH"/dist/octomil-*.whl >/dev/null

cd "$TMPDIR_PATH"
"$TMPDIR_PATH/venv/bin/python" - <<'PY'
import inspect
import importlib.metadata as metadata

import octomil
from octomil import Octomil
from octomil.edge import Octomil as LegacyEdgeOctomil
from octomil.facade import Octomil as FacadeOctomil

assert metadata.version("octomil") == octomil.__version__
assert Octomil is FacadeOctomil
assert hasattr(Octomil, "from_env")
assert inspect.getsourcefile(Octomil) == inspect.getsourcefile(FacadeOctomil)
assert hasattr(LegacyEdgeOctomil, "from_env")

print("octomil_file=", octomil.__file__)
print("Octomil_file=", inspect.getsourcefile(Octomil))
print("from_env=", hasattr(Octomil, "from_env"))
print("version=", metadata.version("octomil"))
PY
