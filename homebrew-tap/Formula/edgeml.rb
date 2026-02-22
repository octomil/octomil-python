# typed: false
# frozen_string_literal: true

class Edgeml < Formula
  desc "On-device ML inference — serve models locally with one command"
  homepage "https://edgeml.io"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/edgeml-ai/edgeml-python/releases/download/v#{version}/edgeml-darwin-arm64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_ARM64"
    else
      url "https://github.com/edgeml-ai/edgeml-python/releases/download/v#{version}/edgeml-darwin-amd64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_AMD64"
    end
  end

  on_linux do
    url "https://github.com/edgeml-ai/edgeml-python/releases/download/v#{version}/edgeml-linux-amd64.tar.gz"
    sha256 "PLACEHOLDER_SHA256_LINUX"
  end

  def install
    bin.install "edgeml"
  end

  test do
    assert_match "edgeml", shell_output("#{bin}/edgeml --version")
  end
end
