#!/usr/bin/env sh
#
# EdgeML installer — download the correct binary for this platform.
#
# Usage:
#   curl -fsSL https://get.edgeml.io | sh
#
# Environment variables:
#   EDGEML_VERSION   — specific version to install (default: latest)
#   EDGEML_INSTALL   — installation directory (default: /usr/local/bin or ~/.local/bin)

set -eu

REPO="edgeml-ai/edgeml-python"
BINARY_NAME="edgeml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info() {
    printf '\033[1;34m==>\033[0m %s\n' "$1"
}

error() {
    printf '\033[1;31merror:\033[0m %s\n' "$1" >&2
    exit 1
}

warn() {
    printf '\033[1;33mwarning:\033[0m %s\n' "$1" >&2
}

# ---------------------------------------------------------------------------
# Detect OS and architecture
# ---------------------------------------------------------------------------

detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Darwin) OS="darwin" ;;
        Linux)  OS="linux" ;;
        *)      error "Unsupported operating system: $OS. EdgeML supports macOS and Linux." ;;
    esac

    case "$ARCH" in
        arm64|aarch64)  ARCH="arm64" ;;
        x86_64|amd64)   ARCH="amd64" ;;
        *)              error "Unsupported architecture: $ARCH. EdgeML supports arm64 and x86_64." ;;
    esac

    PLATFORM="${OS}-${ARCH}"
}

# ---------------------------------------------------------------------------
# Determine version
# ---------------------------------------------------------------------------

get_version() {
    if [ -n "${EDGEML_VERSION:-}" ]; then
        VERSION="$EDGEML_VERSION"
        # Prepend v if missing
        case "$VERSION" in
            v*) ;;
            *)  VERSION="v${VERSION}" ;;
        esac
        return
    fi

    info "Fetching latest release..."

    if command -v curl >/dev/null 2>&1; then
        VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | \
            grep '"tag_name"' | head -1 | sed -E 's/.*"tag_name":\s*"([^"]+)".*/\1/')
    elif command -v wget >/dev/null 2>&1; then
        VERSION=$(wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" | \
            grep '"tag_name"' | head -1 | sed -E 's/.*"tag_name":\s*"([^"]+)".*/\1/')
    else
        error "curl or wget is required to download edgeml."
    fi

    if [ -z "$VERSION" ]; then
        error "Could not determine latest version. Set EDGEML_VERSION manually."
    fi
}

# ---------------------------------------------------------------------------
# Determine install directory
# ---------------------------------------------------------------------------

get_install_dir() {
    if [ -n "${EDGEML_INSTALL:-}" ]; then
        INSTALL_DIR="$EDGEML_INSTALL"
        return
    fi

    # Prefer /usr/local/bin if writable, otherwise ~/.local/bin
    if [ -w "/usr/local/bin" ]; then
        INSTALL_DIR="/usr/local/bin"
    elif [ "$(id -u)" = "0" ]; then
        INSTALL_DIR="/usr/local/bin"
    else
        INSTALL_DIR="${HOME}/.local/bin"
    fi
}

# ---------------------------------------------------------------------------
# Download and install
# ---------------------------------------------------------------------------

download_and_install() {
    ARCHIVE_NAME="${BINARY_NAME}-${PLATFORM}.tar.gz"
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${ARCHIVE_NAME}"

    TMPDIR_PATH="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR_PATH"' EXIT

    info "Downloading ${BINARY_NAME} ${VERSION} for ${PLATFORM}..."

    if command -v curl >/dev/null 2>&1; then
        HTTP_CODE=$(curl -fSL --write-out '%{http_code}' -o "${TMPDIR_PATH}/${ARCHIVE_NAME}" "$DOWNLOAD_URL" 2>/dev/null) || true
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${TMPDIR_PATH}/${ARCHIVE_NAME}" "$DOWNLOAD_URL" 2>/dev/null && HTTP_CODE="200" || HTTP_CODE="404"
    fi

    if [ "${HTTP_CODE:-}" != "200" ] || [ ! -f "${TMPDIR_PATH}/${ARCHIVE_NAME}" ]; then
        error "Download failed (HTTP ${HTTP_CODE:-unknown}).
  URL: ${DOWNLOAD_URL}

  Possible causes:
    - Version ${VERSION} does not have a binary for ${PLATFORM}
    - The release has not been published yet

  Available platforms: darwin-arm64, darwin-amd64, linux-amd64"
    fi

    info "Extracting..."
    tar -xzf "${TMPDIR_PATH}/${ARCHIVE_NAME}" -C "$TMPDIR_PATH"

    if [ ! -f "${TMPDIR_PATH}/${BINARY_NAME}" ]; then
        error "Archive did not contain expected binary '${BINARY_NAME}'."
    fi

    # Create install directory if needed
    if [ ! -d "$INSTALL_DIR" ]; then
        info "Creating ${INSTALL_DIR}..."
        mkdir -p "$INSTALL_DIR"
    fi

    info "Installing to ${INSTALL_DIR}/${BINARY_NAME}..."
    mv "${TMPDIR_PATH}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
    chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
}

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------

verify() {
    if ! "${INSTALL_DIR}/${BINARY_NAME}" --version >/dev/null 2>&1; then
        warn "Binary installed but --version check failed."
        warn "This may happen if required shared libraries are missing."
    else
        INSTALLED_VERSION=$("${INSTALL_DIR}/${BINARY_NAME}" --version 2>&1 || true)
        info "Installed: ${INSTALLED_VERSION}"
    fi

    # Check if install dir is in PATH
    case ":${PATH}:" in
        *":${INSTALL_DIR}:"*) ;;
        *)
            warn "${INSTALL_DIR} is not in your PATH."
            echo ""
            echo "  Add it to your shell profile:"
            echo ""
            echo "    export PATH=\"${INSTALL_DIR}:\$PATH\""
            echo ""
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    detect_platform
    get_version
    get_install_dir
    download_and_install
    verify

    echo ""
    info "EdgeML ${VERSION} installed successfully."
    echo ""
    echo "  Get started:"
    echo "    edgeml serve gemma-1b"
    echo "    edgeml --help"
    echo ""
}

main
