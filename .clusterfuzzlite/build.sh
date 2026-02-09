#!/bin/bash -eu

cd /src/edgeml-python

# Build fuzz targets
for fuzzer in .clusterfuzzlite/fuzz_*.py; do
  target_name=$(basename "$fuzzer" .py)
  compile_python_fuzzer "$fuzzer" --add-binary "$(which python3):."
done
