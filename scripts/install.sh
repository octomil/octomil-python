#!/usr/bin/env sh
#
# Octomil installer — download the correct binary for this platform.
#
# Usage:
#   curl -fsSL https://get.octomil.com | sh
#
# Environment variables:
#   OCTOMIL_VERSION   — specific version to install (default: latest)
#   OCTOMIL_INSTALL   — installation directory (default: /usr/local/bin or ~/.local/bin)

set -eu

REPO="octomil/octomil-python"
BINARY_NAME="octomil"

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
        *)      error "Unsupported operating system: $OS. Octomil supports macOS and Linux." ;;
    esac

    case "$ARCH" in
        arm64|aarch64)  ARCH="arm64" ;;
        x86_64|amd64)   ARCH="amd64" ;;
        *)              error "Unsupported architecture: $ARCH. Octomil supports arm64 and x86_64." ;;
    esac

    PLATFORM="${OS}-${ARCH}"

    # Intel macOS: no pre-built binary — fall back to pip
    if [ "$PLATFORM" = "darwin-amd64" ]; then
        warn "Pre-built binaries are only available for Apple Silicon (arm64)."
        echo ""
        echo "  Install via pip instead:"
        echo ""
        echo "    pip install octomil[serve]"
        echo ""
        exit 0
    fi
}

# ---------------------------------------------------------------------------
# Determine version
# ---------------------------------------------------------------------------

get_version() {
    if [ -n "${OCTOMIL_VERSION:-}" ]; then
        VERSION="$OCTOMIL_VERSION"
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
            grep '"tag_name"' | head -1 | sed -E 's/.*"tag_name":[[:space:]]*"([^"]+)".*/\1/')
    elif command -v wget >/dev/null 2>&1; then
        VERSION=$(wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" | \
            grep '"tag_name"' | head -1 | sed -E 's/.*"tag_name":[[:space:]]*"([^"]+)".*/\1/')
    else
        error "curl or wget is required to download octomil."
    fi

    if [ -z "$VERSION" ]; then
        error "Could not determine latest version. Set OCTOMIL_VERSION manually."
    fi
}

# ---------------------------------------------------------------------------
# Determine install directory
# ---------------------------------------------------------------------------

get_install_dir() {
    if [ -n "${OCTOMIL_INSTALL:-}" ]; then
        INSTALL_DIR="$OCTOMIL_INSTALL"
        return
    fi

    # bin dir: symlink target (must be on PATH)
    if [ -w "/usr/local/bin" ]; then
        INSTALL_DIR="/usr/local/bin"
    elif [ "$(id -u)" = "0" ]; then
        INSTALL_DIR="/usr/local/bin"
    else
        INSTALL_DIR="${HOME}/.local/bin"
    fi

    # lib dir: where the full binary bundle lives
    LIB_DIR="${HOME}/.local/lib/octomil"
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

  Available platforms: darwin-arm64, linux-amd64"
    fi

    info "Extracting..."
    tar -xzf "${TMPDIR_PATH}/${ARCHIVE_NAME}" -C "$TMPDIR_PATH"

    if [ ! -f "${TMPDIR_PATH}/${BINARY_NAME}/${BINARY_NAME}" ]; then
        error "Archive did not contain expected binary '${BINARY_NAME}/${BINARY_NAME}'."
    fi

    # Remove previous installation
    if [ -d "$LIB_DIR" ]; then
        rm -rf "$LIB_DIR"
    fi

    # Install bundle to lib dir
    mkdir -p "$(dirname "$LIB_DIR")"
    mv "${TMPDIR_PATH}/${BINARY_NAME}" "$LIB_DIR"
    chmod +x "${LIB_DIR}/${BINARY_NAME}"

    # Create bin directory and symlink
    if [ ! -d "$INSTALL_DIR" ]; then
        info "Creating ${INSTALL_DIR}..."
        mkdir -p "$INSTALL_DIR"
    fi

    ln -sf "${LIB_DIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
    info "Installed to ${INSTALL_DIR}/${BINARY_NAME}"
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
# Shell completions
# ---------------------------------------------------------------------------

COMPLETIONS_MARKER="# octomil shell completions"

setup_completions() {
    CURRENT_SHELL="$(basename "${SHELL:-/bin/sh}")"

    case "$CURRENT_SHELL" in
        zsh)
            COMP_LINE='eval "$(_OCTOMIL_COMPLETE=zsh_source octomil)"'
            RC_FILE="${HOME}/.zshrc"
            ;;
        fish)
            COMP_LINE='_OCTOMIL_COMPLETE=fish_source octomil | source'
            RC_FILE="${HOME}/.config/fish/conf.d/octomil.fish"
            ;;
        bash|*)
            COMP_LINE='eval "$(_OCTOMIL_COMPLETE=bash_source octomil)"'
            RC_FILE="${HOME}/.bashrc"
            ;;
    esac

    # Skip if already installed
    if [ -f "$RC_FILE" ] && grep -qF "$COMPLETIONS_MARKER" "$RC_FILE" 2>/dev/null; then
        return
    fi

    # For fish, ensure conf.d directory exists
    if [ "$CURRENT_SHELL" = "fish" ]; then
        mkdir -p "$(dirname "$RC_FILE")"
    fi

    printf '\n%s\n%s\n' "$COMPLETIONS_MARKER" "$COMP_LINE" >> "$RC_FILE"
    info "Shell completions added to ${RC_FILE}"
}

# ---------------------------------------------------------------------------
# Install Python SDK
# ---------------------------------------------------------------------------

install_python_sdk() {
    # Find a working pip/pip3
    PIP_CMD=""
    if command -v pip3 >/dev/null 2>&1; then
        PIP_CMD="pip3"
    elif command -v pip >/dev/null 2>&1; then
        PIP_CMD="pip"
    fi

    if [ -z "$PIP_CMD" ]; then
        return
    fi

    info "Installing Python SDK..."
    if $PIP_CMD install --quiet --upgrade octomil-sdk 2>/dev/null; then
        info "Python SDK installed (import octomil)"
    else
        warn "Could not install Python SDK. Install manually: pip install octomil-sdk"
    fi
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
    setup_completions
    install_python_sdk

    echo ""
    info "Octomil ${VERSION} installed successfully."
    echo ""
    echo "  Get started:"
    echo "    octomil serve gemma-1b"
    echo "    octomil --help"
    echo ""
}

main
