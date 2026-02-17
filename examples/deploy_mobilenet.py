#!/usr/bin/env python3
"""Deploy MobileNet to iOS and Android from a single script.

Demonstrates the full engine router workflow:
1. Connect to EdgeML platform
2. Check device compatibility
3. Optimize model (pruning, quantization, format conversion)
4. Roll out to iOS (CoreML) and Android (TFLite) devices
5. Monitor and advance rollout

Usage:
    export EDGEML_API_KEY="edgeml_..."
    python deploy_mobilenet.py
"""

import os
import sys

import edgeml


def main():
    api_key = os.environ.get("EDGEML_API_KEY")
    if not api_key:
        print("Set EDGEML_API_KEY environment variable first.")
        sys.exit(1)

    # --- Step 1: Connect to EdgeML platform ---
    edgeml.connect(api_key=api_key)
    print("Connected to EdgeML platform.\n")

    # --- Step 2: Deploy to both platforms ---
    # This single call:
    #   - Checks model compatibility with target devices
    #   - Runs optimization (pruning → quantization → format conversion)
    #   - Creates a gradual rollout starting at 10%
    print("Deploying mobilenet-v3 to iOS and Android...")
    deployment = edgeml.deploy_remote(
        model="mobilenet-v3",
        version="1.0.0",
        targets=["iphone_15_pro", "pixel_8"],
        optimize=True,
        accuracy_threshold=0.95,  # keep >=95% of original accuracy
        rollout=10,               # start at 10%
        target_rollout=100,       # ramp to 100%
        increment_step=10,        # 10% increments
    )

    print(f"\n{deployment}")
    print("\nPer-device status:")
    for device, info in deployment.status.items():
        fmt = info["format"]
        pct = info["rollout"]
        size = info.get("size_mb", "?")
        ratio = info.get("compression_ratio", "?")
        print(f"  {device}: {fmt} | {size}MB | {ratio}x compressed | {pct}% rolled out")

    # --- Step 3: Advance rollout ---
    print("\nAdvancing rollout to 50%...")
    deployment.advance(50)
    for device, info in deployment.status.items():
        print(f"  {device}: {info['rollout']}%")

    print("\nDone. Monitor rollout progress in the EdgeML dashboard.")


if __name__ == "__main__":
    main()
