#!/usr/bin/env bash
# Update Homebrew formula SHA256 hashes after a release.
# Usage: ./scripts/update-homebrew.sh v0.1.0

set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"
TAG="${VERSION#v}"
FORMULA="homebrew-tap/Formula/octomil.rb"
BASE_URL="https://github.com/octomil/octomil-python/releases/download/${VERSION}"

echo "Updating formula for ${VERSION}..."

# Fetch SHA256 hashes for each platform binary from the release manifest.
CHECKSUMS="$(mktemp)"
trap 'rm -f "$CHECKSUMS"' EXIT
curl -fsSL "${BASE_URL}/SHA256SUMS" -o "$CHECKSUMS"

sha_for() {
  awk -v name="$1" '$2 == name { print $1 }' "$CHECKSUMS"
}

SHA_DARWIN_ARM64="$(sha_for "octomil-${VERSION}-darwin-arm64.tar.gz")"
SHA_DARWIN_AMD64="$(sha_for "octomil-${VERSION}-darwin-amd64.tar.gz")"
SHA_LINUX_ARM64="$(sha_for "octomil-${VERSION}-linux-arm64.tar.gz")"
SHA_LINUX_AMD64="$(sha_for "octomil-${VERSION}-linux-amd64.tar.gz")"

# Replace sha256 values positionally.
# Matches any existing sha256 "..." value including placeholders, making the
# script idempotent across releases.
awk \
  -v darwin_arm64="${SHA_DARWIN_ARM64}" \
  -v darwin_amd64="${SHA_DARWIN_AMD64}" \
  -v linux_arm64="${SHA_LINUX_ARM64}" \
  -v linux_amd64="${SHA_LINUX_AMD64}" '
  /sha256 "/ {
    n++
    if (n == 1) sub(/sha256 ".*"/, "sha256 \"" darwin_arm64 "\"")
    else if (n == 2) sub(/sha256 ".*"/, "sha256 \"" darwin_amd64 "\"")
    else if (n == 3) sub(/sha256 ".*"/, "sha256 \"" linux_arm64 "\"")
    else if (n == 4) sub(/sha256 ".*"/, "sha256 \"" linux_amd64 "\"")
  }
  { print }
' "${FORMULA}" > "${FORMULA}.tmp" && mv "${FORMULA}.tmp" "${FORMULA}"

sed -i.bak "s/version \".*\"/version \"${TAG}\"/" "${FORMULA}"
rm -f "${FORMULA}.bak"

echo "Done. Review ${FORMULA} and push to homebrew-tap repo."
