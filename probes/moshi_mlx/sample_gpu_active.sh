#!/usr/bin/env bash
# Sample GPU HW active residency on Apple Silicon while the slice-2B
# Moshi/MLX probe streams. Prints the top 5 active-residency percents
# observed during the sampling window. Use the peak as
# OCTOMIL_PROBE_GPU_PCT for the gating probe run.
#
# Usage:
#   ./probes/moshi_mlx/sample_gpu_active.sh
#
# Requires sudo (powermetrics needs root). Will prompt for password.

set -euo pipefail

cd "$(dirname "$0")/../.."

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/octomil-artifacts/moshi-v0.2}"
PROBE_VENV="probes/moshi_mlx/.venv-probe/bin/python"
PROBE="probes/moshi_mlx/probe.py"

if [[ ! -x "$PROBE_VENV" ]]; then
  echo "error: probe venv not found at $PROBE_VENV" >&2
  exit 1
fi
if [[ ! -d "$ARTIFACT_ROOT" ]]; then
  echo "error: artifact root not found at $ARTIFACT_ROOT" >&2
  exit 1
fi

echo "Caching sudo credentials..."
sudo -v

echo "Spinning probe loop in background (10 iterations, ~25s wall time)..."
(
  for i in 1 2 3 4 5 6 7 8 9 10; do
    "$PROBE_VENV" "$PROBE" \
      --artifact-root "$ARTIFACT_ROOT" \
      --output /tmp/probe-during-sample.json \
      > /dev/null 2>&1 || true
  done
) &
LOOP_PID=$!

# Give the first probe time to load + start streaming.
sleep 3

echo "Sampling GPU HW active residency for ~15s..."
SAMPLES=$(sudo powermetrics --samplers gpu_power -i 1000 -n 15 2>/dev/null \
  | grep -E "GPU HW active residency" \
  | sed -E 's/.*residency:[[:space:]]+([0-9.]+)%.*/\1/' \
  || true)

# Best effort: stop the probe loop if still running.
kill "$LOOP_PID" 2>/dev/null || true
wait "$LOOP_PID" 2>/dev/null || true

if [[ -z "$SAMPLES" ]]; then
  echo "error: no GPU HW active residency samples captured" >&2
  exit 2
fi

echo ""
echo "All samples (% active):"
echo "$SAMPLES"

PEAK=$(echo "$SAMPLES" | sort -g | tail -1)
AVG=$(echo "$SAMPLES" | awk '{s+=$1; n++} END{printf "%.2f", s/n}')

echo ""
echo "PEAK_GPU_ACTIVE_PCT=$PEAK"
echo "AVG_GPU_ACTIVE_PCT=$AVG"
echo ""
echo "Re-run the probe with the peak value:"
echo "  OCTOMIL_PROBE_GPU_PCT=$PEAK probes/moshi_mlx/.venv-probe/bin/python probes/moshi_mlx/probe.py \\"
echo "    --artifact-root $ARTIFACT_ROOT \\"
echo "    --output probes/moshi_mlx/probe-results.json"
