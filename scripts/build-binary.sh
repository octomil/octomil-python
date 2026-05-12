#!/usr/bin/env bash
#
# Build octomil as a standalone binary using PyInstaller.
#
# Usage:
#   ./scripts/build-binary.sh
#
# Requirements:
#   python -m pip install -e ".[serve,binary]"
#
# Output:
#   dist/octomil-v<version>-<os>-<arch>.tar.gz
#   dist/octomil-v<version>-<os>-<arch>.tar.gz.sha256

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------
detect_platform() {
    local os arch

    case "$(uname -s)" in
        Darwin)  os="darwin" ;;
        Linux)   os="linux" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *)
            echo "ERROR: Unsupported OS: $(uname -s)" >&2
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        arm64|aarch64) arch="arm64" ;;
        x86_64|amd64)  arch="amd64" ;;
        *)
            echo "ERROR: Unsupported architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac

    echo "${os}-${arch}"
}

PYTHON="${PYTHON:-python3}"
PLATFORM="$(detect_platform)"
echo "==> Platform: ${PLATFORM}"

detect_version() {
    local version

    if [ -n "${OCTOMIL_VERSION:-}" ]; then
        version="${OCTOMIL_VERSION}"
    else
        version="$(sed -nE 's/^version = "([^"]+)"/\1/p' pyproject.toml | head -n1)"
    fi

    case "$version" in
        v*) printf '%s\n' "$version" ;;
        *)  printf 'v%s\n' "$version" ;;
    esac
}

VERSION="$(detect_version)"
echo "==> Version: ${VERSION}"

# ---------------------------------------------------------------------------
# Check PyInstaller is installed
# ---------------------------------------------------------------------------
if ! "$PYTHON" -m PyInstaller --version >/dev/null 2>&1; then
    echo "ERROR: PyInstaller not found. Install build dependencies with:" >&2
    echo "  ${PYTHON} -m pip install -e \".[serve,binary]\"" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Clean previous build artifacts
# ---------------------------------------------------------------------------
echo "==> Cleaning previous build..."
rm -rf build/ dist/octomil dist/octomil.pkg dist/octomil-*.tar.gz dist/octomil-*.tar.gz.sha256

# ---------------------------------------------------------------------------
# Run PyInstaller
# ---------------------------------------------------------------------------
echo "==> Building binary with PyInstaller..."
"$PYTHON" -m PyInstaller packaging/octomil.spec --noconfirm

# Verify the binary was created
BUNDLE_DIR="dist/octomil"
BINARY="${BUNDLE_DIR}/octomil"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at ${BINARY}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify the binary runs
# ---------------------------------------------------------------------------
echo "==> Verifying binary..."
if ! "$BINARY" --version; then
    echo "ERROR: Binary failed to run --version" >&2
    exit 1
fi
if ! "$BINARY" --help >/dev/null; then
    echo "ERROR: Binary failed to run --help" >&2
    exit 1
fi
if ! "$BINARY" serve --help >/dev/null; then
    echo "ERROR: Binary failed to run serve --help" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Report bundle size
# ---------------------------------------------------------------------------
BUNDLE_SIZE=$(du -sh "$BUNDLE_DIR" | cut -f1)
echo "==> Bundle size: ${BUNDLE_SIZE}"

# ---------------------------------------------------------------------------
# Create archive
# ---------------------------------------------------------------------------
ARCHIVE="octomil-${VERSION}-${PLATFORM}.tar.gz"

echo "==> Creating archive..."
tar -czf "dist/${ARCHIVE}" -C "$BUNDLE_DIR" .

ARCHIVE_SIZE=$(du -sh "dist/${ARCHIVE}" | cut -f1)
echo "==> Archive: dist/${ARCHIVE} (${ARCHIVE_SIZE})"

# ---------------------------------------------------------------------------
# Generate SHA256 checksum
# ---------------------------------------------------------------------------
if command -v sha256sum &>/dev/null; then
    SHA256=$(sha256sum "dist/${ARCHIVE}" | awk '{print $1}')
elif command -v shasum &>/dev/null; then
    SHA256=$(shasum -a 256 "dist/${ARCHIVE}" | awk '{print $1}')
else
    echo "ERROR: sha256sum or shasum is required to checksum release artifacts" >&2
    exit 1
fi
echo "==> SHA256: ${SHA256}"
printf '%s  %s\n' "$SHA256" "$ARCHIVE" > "dist/${ARCHIVE}.sha256"

echo ""
echo "Build complete."
echo "  Bundle:  dist/octomil"
echo "  Archive: dist/${ARCHIVE}"
echo "  Checksum: dist/${ARCHIVE}.sha256"
echo "  SHA256:  ${SHA256}"
