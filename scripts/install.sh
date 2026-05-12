#!/usr/bin/env sh
#
# Octomil installer — download the self-contained binary bundle for this platform.
#
# Usage:
#   curl -fsSL https://get.octomil.com | sh
#
# Environment variables:
#   OCTOMIL_VERSION       specific version to install (default: latest)
#   OCTOMIL_INSTALL_DIR   bin directory for the octomil symlink (default: /usr/local/bin or ~/.local/bin)
#   OCTOMIL_INSTALL       legacy alias for OCTOMIL_INSTALL_DIR
#   OCTOMIL_LIB_DIR       bundle directory (default: ~/.local/lib/octomil)

set -eu

REPO="octomil/octomil-python"
BINARY_NAME="octomil"

info() {
    printf '\033[1;34m==>\033[0m %s\n' "$1"
}

warn() {
    printf '\033[1;33mwarning:\033[0m %s\n' "$1" >&2
}

error() {
    printf '\033[1;31merror:\033[0m %s\n' "$1" >&2
    exit 1
}

download() {
    url="$1"
    dest="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$dest"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$dest" "$url"
    else
        error "curl or wget is required to download Octomil."
    fi
}

asset_exists() {
    url="$1"

    if command -v curl >/dev/null 2>&1; then
        curl -fsIL "$url" >/dev/null 2>&1
    elif command -v wget >/dev/null 2>&1; then
        wget --spider -q "$url" >/dev/null 2>&1
    else
        error "curl or wget is required to download Octomil."
    fi
}

detect_platform() {
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Darwin) os="darwin" ;;
        Linux) os="linux" ;;
        *) error "Unsupported operating system: $os. Octomil supports macOS and Linux." ;;
    esac

    case "$arch" in
        arm64|aarch64) arch="arm64" ;;
        x86_64|amd64) arch="amd64" ;;
        *) error "Unsupported architecture: $arch. Octomil supports arm64 and x86_64." ;;
    esac

    PLATFORM="${os}-${arch}"
}

artifact_name() {
    printf 'octomil-%s-%s.tar.gz\n' "$1" "$PLATFORM"
}

release_base_url() {
    printf 'https://github.com/%s/releases/download/%s\n' "$REPO" "$1"
}

get_version() {
    if [ -n "${OCTOMIL_VERSION:-}" ]; then
        VERSION="$OCTOMIL_VERSION"
        case "$VERSION" in
            v*) ;;
            *) VERSION="v${VERSION}" ;;
        esac
        ARTIFACT="$(artifact_name "$VERSION")"
        return
    fi

    info "Fetching latest release..."
    tmp_releases="${TMPDIR_PATH}/releases.json"
    download "https://api.github.com/repos/${REPO}/releases?per_page=10" "$tmp_releases"

    tags="$(grep '"tag_name"' "$tmp_releases" | sed -E 's/.*"tag_name":[[:space:]]*"([^"]+)".*/\1/' || true)"
    VERSION=""
    ARTIFACT=""

    for tag in $tags; do
        candidate="$(artifact_name "$tag")"
        if asset_exists "$(release_base_url "$tag")/${candidate}"; then
            VERSION="$tag"
            ARTIFACT="$candidate"
            break
        fi
    done

    if [ -z "$VERSION" ]; then
        error "No release found with a binary for ${PLATFORM}. Set OCTOMIL_VERSION manually."
    fi
}

get_install_dirs() {
    if [ -n "${OCTOMIL_INSTALL_DIR:-}" ]; then
        INSTALL_DIR="$OCTOMIL_INSTALL_DIR"
    elif [ -n "${OCTOMIL_INSTALL:-}" ]; then
        INSTALL_DIR="$OCTOMIL_INSTALL"
    elif [ -w "/usr/local/bin" ] || [ "$(id -u)" = "0" ]; then
        INSTALL_DIR="/usr/local/bin"
    else
        INSTALL_DIR="${HOME}/.local/bin"
    fi

    LIB_DIR="${OCTOMIL_LIB_DIR:-${HOME}/.local/lib/octomil}"
}

download_checksums() {
    base="$1"
    dest="$2"

    if download "${base}/SHA256SUMS" "$dest" 2>/dev/null; then
        return
    fi

    if download "${base}/checksums.txt" "$dest" 2>/dev/null; then
        return
    fi

    error "Could not download SHA256SUMS for ${VERSION}."
}

verify_checksum() {
    checksum_file="$1"
    archive_path="$2"
    archive_name="$(basename "$archive_path")"
    line="$(grep " ${archive_name}\$" "$checksum_file" || true)"

    if [ -z "$line" ]; then
        error "${archive_name} is not listed in SHA256SUMS."
    fi

    info "Verifying checksum..."
    (
        cd "$(dirname "$archive_path")"
        if command -v sha256sum >/dev/null 2>&1; then
            printf '%s\n' "$line" | sha256sum -c -
        elif command -v shasum >/dev/null 2>&1; then
            printf '%s\n' "$line" | shasum -a 256 -c -
        else
            error "sha256sum or shasum is required to verify Octomil."
        fi
    ) >/dev/null
}

download_and_install() {
    base="$(release_base_url "$VERSION")"
    archive_path="${TMPDIR_PATH}/${ARTIFACT}"
    checksum_path="${TMPDIR_PATH}/SHA256SUMS"
    bundle_path="${TMPDIR_PATH}/bundle"

    info "Downloading Octomil ${VERSION} for ${PLATFORM}..."
    download "${base}/${ARTIFACT}" "$archive_path"
    download_checksums "$base" "$checksum_path"
    verify_checksum "$checksum_path" "$archive_path"

    info "Installing bundle..."
    mkdir -p "$bundle_path"
    tar -xzf "$archive_path" -C "$bundle_path"

    if [ ! -f "${bundle_path}/${BINARY_NAME}" ]; then
        error "Archive did not contain expected binary '${BINARY_NAME}'."
    fi

    rm -rf "$LIB_DIR"
    mkdir -p "$LIB_DIR"
    tar -xzf "$archive_path" -C "$LIB_DIR"
    chmod +x "${LIB_DIR}/${BINARY_NAME}"

    mkdir -p "$INSTALL_DIR"
    ln -sf "${LIB_DIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"

    info "Installed to ${INSTALL_DIR}/${BINARY_NAME}"
}

verify_installation() {
    installed="${INSTALL_DIR}/${BINARY_NAME}"

    if ! "$installed" --version >/dev/null 2>&1; then
        warn "Binary installed but --version check failed."
        warn "This may happen if required shared libraries are missing."
    else
        installed_version="$("$installed" --version 2>&1 || true)"
        info "Installed: ${installed_version}"
    fi

    resolved="$(command -v "$BINARY_NAME" || true)"
    if [ "$resolved" != "$installed" ]; then
        echo ""
        warn "Octomil installed successfully, but your shell resolves '${BINARY_NAME}' to:"
        echo "    ${resolved:-not found}"
        echo ""
        echo "Add this to your shell config:"
        echo "    export PATH=\"${INSTALL_DIR}:\$PATH\""
        echo ""
        echo "Or run directly:"
        echo "    ${installed} --help"
    fi
}

main() {
    TMPDIR_PATH="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR_PATH"' EXIT

    detect_platform
    get_version
    get_install_dirs
    download_and_install
    verify_installation

    CYAN='\033[1;36m'
    DIM='\033[2m'
    WHITE='\033[1;37m'
    GREEN='\033[1;32m'
    YELLOW='\033[1;33m'
    RESET='\033[0m'

    echo ""
    printf "  ${CYAN}Octomil ${VERSION}${RESET}\n"
    printf "  ${DIM}on-device AI inference${RESET}\n"
    echo ""
    printf "  ${GREEN}Installed successfully${RESET}\n"
    echo ""
    printf "  ${YELLOW}Get started${RESET}\n"
    printf "    ${WHITE}%s serve qwen-7b${RESET}        ${DIM}Start a local model server${RESET}\n" "${INSTALL_DIR}/${BINARY_NAME}"
    printf "    ${WHITE}%s launch${RESET}               ${DIM}Launch a coding agent${RESET}\n" "${INSTALL_DIR}/${BINARY_NAME}"
    printf "    ${WHITE}%s benchmark gemma-1b${RESET}   ${DIM}Benchmark inference performance${RESET}\n" "${INSTALL_DIR}/${BINARY_NAME}"
    echo ""
    printf "  ${DIM}The installer did not use Python, pip, or virtualenvs.${RESET}\n"
    echo ""
}

main
