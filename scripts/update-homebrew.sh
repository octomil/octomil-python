#!/usr/bin/env bash
# Update Homebrew formula SHA256 hashes after a release.
# Usage: ./scripts/update-homebrew.sh v0.1.0

set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"
TAG="${VERSION#v}"
FORMULA="homebrew-tap/Formula/octomil.rb"
BASE_URL="https://github.com/octomil/octomil-python/releases/download/${VERSION}"

echo "Updating formula for ${VERSION}..."

# Fetch SHA256 hashes for each platform binary
echo "  Fetching SHA256 for darwin-arm64..."
SHA_ARM64=$(curl -fsSL "${BASE_URL}/octomil-darwin-arm64.tar.gz" | shasum -a 256 | cut -d' ' -f1)

echo "  Fetching SHA256 for darwin-amd64..."
SHA_AMD64=$(curl -fsSL "${BASE_URL}/octomil-darwin-amd64.tar.gz" | shasum -a 256 | cut -d' ' -f1)

echo "  Fetching SHA256 for linux-amd64..."
SHA_LINUX=$(curl -fsSL "${BASE_URL}/octomil-linux-amd64.tar.gz" | shasum -a 256 | cut -d' ' -f1)

# Replace sha256 values positionally (1st = arm64, 2nd = amd64, 3rd = linux).
# Matches any existing sha256 "..." value including placeholders, making the
# script idempotent across releases.
awk -v arm64="${SHA_ARM64}" -v amd64="${SHA_AMD64}" -v linux="${SHA_LINUX}" '
  /sha256 "/ {
    n++
    if (n == 1) sub(/sha256 ".*"/, "sha256 \"" arm64 "\"")
    else if (n == 2) sub(/sha256 ".*"/, "sha256 \"" amd64 "\"")
    else if (n == 3) sub(/sha256 ".*"/, "sha256 \"" linux "\"")
  }
  { print }
' "${FORMULA}" > "${FORMULA}.tmp" && mv "${FORMULA}.tmp" "${FORMULA}"

sed -i.bak "s/version \".*\"/version \"${TAG}\"/" "${FORMULA}"
rm -f "${FORMULA}.bak"

echo "Done. Review ${FORMULA} and push to homebrew-tap repo."
