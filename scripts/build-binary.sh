#!/usr/bin/env bash
#
# Build edgeml as a standalone binary using PyInstaller.
#
# Usage:
#   ./scripts/build-binary.sh
#
# Requirements:
#   pip install pyinstaller
#
# Output:
#   dist/edgeml-<os>-<arch>.tar.gz   (macOS / Linux)
#   dist/edgeml-<os>-<arch>.zip      (Windows)

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

PLATFORM="$(detect_platform)"
echo "==> Platform: ${PLATFORM}"

# ---------------------------------------------------------------------------
# Check PyInstaller is installed
# ---------------------------------------------------------------------------
if ! command -v pyinstaller &>/dev/null; then
    echo "ERROR: pyinstaller not found. Install it with:" >&2
    echo "  pip install pyinstaller" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Clean previous build artifacts
# ---------------------------------------------------------------------------
echo "==> Cleaning previous build..."
rm -rf build/ dist/edgeml dist/edgeml.pkg

# ---------------------------------------------------------------------------
# Run PyInstaller
# ---------------------------------------------------------------------------
echo "==> Building binary with PyInstaller..."
pyinstaller edgeml.spec --noconfirm

# Verify the binary was created
BINARY="dist/edgeml"
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

# ---------------------------------------------------------------------------
# Report binary size
# ---------------------------------------------------------------------------
BINARY_SIZE=$(du -sh "$BINARY" | cut -f1)
echo "==> Binary size: ${BINARY_SIZE}"

# ---------------------------------------------------------------------------
# Create archive
# ---------------------------------------------------------------------------
ARCHIVE_NAME="edgeml-${PLATFORM}"

echo "==> Creating archive..."
case "$PLATFORM" in
    windows-*)
        # zip for Windows
        ARCHIVE="${ARCHIVE_NAME}.zip"
        (cd dist && zip -9 "$ARCHIVE" edgeml.exe 2>/dev/null || zip -9 "$ARCHIVE" edgeml)
        ;;
    *)
        # tar.gz for macOS and Linux
        ARCHIVE="${ARCHIVE_NAME}.tar.gz"
        tar -czf "dist/${ARCHIVE}" -C dist edgeml
        ;;
esac

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
    SHA256="(sha256sum not available)"
fi
echo "==> SHA256: ${SHA256}"

echo ""
echo "Build complete."
echo "  Binary:  dist/edgeml"
echo "  Archive: dist/${ARCHIVE}"
echo "  SHA256:  ${SHA256}"
