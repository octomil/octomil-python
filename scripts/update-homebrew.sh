#!/usr/bin/env bash
# Update Homebrew formula SHA256 hashes after a release.
# Usage: ./scripts/update-homebrew.sh v0.1.0

set -eu

VERSION="${1:?Usage: $0 <version>}"
TAG="${VERSION#v}"
FORMULA="homebrew-tap/Formula/edgeml.rb"
BASE_URL="https://github.com/edgeml-ai/edgeml-python/releases/download/${VERSION}"

echo "Updating formula for ${VERSION}..."

for platform in darwin-arm64 darwin-amd64 linux-amd64; do
    URL="${BASE_URL}/edgeml-${platform}.tar.gz"
    echo "  Fetching SHA256 for ${platform}..."
    SHA=$(curl -fsSL "${URL}" | shasum -a 256 | cut -d' ' -f1)

    case "${platform}" in
        darwin-arm64) PLACEHOLDER="PLACEHOLDER_SHA256_ARM64" ;;
        darwin-amd64) PLACEHOLDER="PLACEHOLDER_SHA256_AMD64" ;;
        linux-amd64)  PLACEHOLDER="PLACEHOLDER_SHA256_LINUX" ;;
    esac

    sed -i.bak "s/${PLACEHOLDER}/${SHA}/" "${FORMULA}"
done

sed -i.bak "s/version \".*\"/version \"${TAG}\"/" "${FORMULA}"
rm -f "${FORMULA}.bak"

echo "Done. Review ${FORMULA} and push to homebrew-tap repo."
