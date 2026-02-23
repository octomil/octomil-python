# typed: false
# frozen_string_literal: true

class Octomil < Formula
  desc "On-device ML inference — serve models locally with one command"
  homepage "https://octomil.com"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/octomil/octomil-python/releases/download/v#{version}/octomil-darwin-arm64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_ARM64"
    else
      url "https://github.com/octomil/octomil-python/releases/download/v#{version}/octomil-darwin-amd64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_AMD64"
    end
  end

  on_linux do
    url "https://github.com/octomil/octomil-python/releases/download/v#{version}/octomil-linux-amd64.tar.gz"
    sha256 "PLACEHOLDER_SHA256_LINUX"
  end

  def install
    bin.install "octomil"
  end

  test do
    assert_match "octomil", shell_output("#{bin}/octomil --version")
  end
end
