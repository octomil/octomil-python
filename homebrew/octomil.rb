# typed: false
# frozen_string_literal: true

# Homebrew formula for octomil CLI.
#
# This is a template — update the version, URLs, and SHA256 hashes before
# publishing to a Homebrew tap.
#
# Usage:
#   brew install octomil/tap/octomil
#
# To update after a new release:
#   1. Replace VERSION, url, and sha256 values
#   2. Commit to the homebrew-tap repository

class Octomil < Formula
  desc "Serve, deploy, and observe ML models on edge devices"
  homepage "https://octomil.com"
  version "2.1.7"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/octomil/octomil-python/releases/download/v#{version}/octomil-darwin-arm64.tar.gz"
      sha256 "PLACEHOLDER_DARWIN_ARM64_SHA256"
    else
      url "https://github.com/octomil/octomil-python/releases/download/v#{version}/octomil-darwin-amd64.tar.gz"
      sha256 "PLACEHOLDER_DARWIN_AMD64_SHA256"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/octomil/octomil-python/releases/download/v#{version}/octomil-linux-arm64.tar.gz"
      sha256 "PLACEHOLDER_LINUX_ARM64_SHA256"
    else
      url "https://github.com/octomil/octomil-python/releases/download/v#{version}/octomil-linux-amd64.tar.gz"
      sha256 "PLACEHOLDER_LINUX_AMD64_SHA256"
    end
  end

  def install
    bin.install "octomil"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/octomil --version")
  end
end
