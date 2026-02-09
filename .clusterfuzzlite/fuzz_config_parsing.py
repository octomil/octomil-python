#!/usr/bin/env python3
import atheris
import sys

with atheris.instrument_imports():
    import json

def TestOneInput(data):
    """Fuzz test for configuration and data parsing."""
    fdp = atheris.FuzzedDataProvider(data)

    # Fuzz JSON config parsing
    try:
        raw = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 4096))
        config = json.loads(raw)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        pass

    # Fuzz model metadata parsing
    try:
        metadata_str = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 2048))
        parts = metadata_str.split(":")
        if len(parts) >= 2:
            model_id = parts[0]
            version = parts[1]
    except (ValueError, IndexError):
        pass

def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()

if __name__ == "__main__":
    main()
