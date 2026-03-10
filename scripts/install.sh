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

    # Fetch recent releases and pick the first one that has a binary for
    # this platform.  This handles the race where the latest tag exists
    # but the binary hasn't finished building yet.
    ASSET_NAME="${BINARY_NAME}-${PLATFORM}.tar.gz"
    RELEASES_JSON=""
    if command -v curl >/dev/null 2>&1; then
        RELEASES_JSON=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases?per_page=5" 2>/dev/null) || true
    elif command -v wget >/dev/null 2>&1; then
        RELEASES_JSON=$(wget -qO- "https://api.github.com/repos/${REPO}/releases?per_page=5" 2>/dev/null) || true
    else
        error "curl or wget is required to download octomil."
    fi

    if [ -z "$RELEASES_JSON" ]; then
        error "Could not fetch releases. Set OCTOMIL_VERSION manually."
    fi

    # Extract tag names from the JSON (lightweight grep, no jq needed)
    TAGS=$(printf '%s' "$RELEASES_JSON" | grep '"tag_name"' | sed -E 's/.*"tag_name":[[:space:]]*"([^"]+)".*/\1/')

    VERSION=""
    for tag in $TAGS; do
        CHECK_URL="https://github.com/${REPO}/releases/download/${tag}/${ASSET_NAME}"
        if command -v curl >/dev/null 2>&1; then
            HTTP_STATUS=$(curl -sI -o /dev/null -w '%{http_code}' -L "$CHECK_URL" 2>/dev/null) || true
        else
            HTTP_STATUS=$(wget --spider -S "$CHECK_URL" 2>&1 | grep 'HTTP/' | tail -1 | awk '{print $2}') || true
        fi
        if [ "${HTTP_STATUS:-}" = "200" ]; then
            VERSION="$tag"
            break
        fi
    done

    if [ -z "$VERSION" ]; then
        error "No release found with a binary for ${PLATFORM}. Set OCTOMIL_VERSION manually."
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
# Engine setup (background)
# ---------------------------------------------------------------------------

setup_engine_background() {
    OCTOMIL_BIN="${INSTALL_DIR}/${BINARY_NAME}"
    STATE_FILE="${HOME}/.octomil/setup_state.json"

    # Skip if setup already completed
    if [ -f "$STATE_FILE" ]; then
        PHASE=$(grep '"phase"' "$STATE_FILE" 2>/dev/null | sed -E 's/.*"phase":[[:space:]]*"([^"]+)".*/\1/' || true)
        if [ "$PHASE" = "complete" ]; then
            info "Engine already set up."
            return
        fi
    fi

    # Verify the binary can run setup
    if ! "$OCTOMIL_BIN" setup --status >/dev/null 2>&1; then
        warn "Could not run 'octomil setup'. Skipping engine auto-setup."
        echo ""
        echo "  You can set up manually later:"
        echo "    octomil setup"
        echo ""
        return
    fi

    info "Setting up inference engine in background..."
    echo "  Run 'octomil setup --status' to check progress."

    mkdir -p "${HOME}/.octomil"
    nohup "$OCTOMIL_BIN" setup --foreground > "${HOME}/.octomil/setup.log" 2>&1 &
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
    setup_engine_background

    # --- Post-install banner ---
    CYAN='\033[1;36m'
    DIM='\033[2m'
    WHITE='\033[1;37m'
    GREEN='\033[1;32m'
    YELLOW='\033[1;33m'
    RESET='\033[0m'

    echo ""
    printf "  ${CYAN}🐙 Octomil ${VERSION}${RESET}\n"
    printf "  ${DIM}on-device AI inference${RESET}\n"
    echo ""
    printf "  ${GREEN}✓${RESET} Installed successfully\n"
    echo ""
    printf "  ${YELLOW}Get started${RESET}\n"
    printf "    ${WHITE}octomil serve qwen-7b${RESET}        ${DIM}Start a local model server${RESET}\n"
    printf "    ${WHITE}octomil launch${RESET}               ${DIM}Launch a coding agent (Claude, Codex, …)${RESET}\n"
    printf "    ${WHITE}octomil deploy phi-4-mini${RESET}    ${DIM}Deploy a model to devices${RESET}\n"
    printf "    ${WHITE}octomil benchmark gemma-1b${RESET}   ${DIM}Benchmark inference performance${RESET}\n"
    echo ""
    printf "  ${YELLOW}More${RESET}\n"
    printf "    ${WHITE}octomil --help${RESET}               ${DIM}All commands${RESET}\n"
    printf "    ${DIM}https://docs.octomil.com${RESET}       ${DIM}Documentation${RESET}\n"
    echo ""
}

main
