# Schema reference

The JSON schema for parity artifacts is canonical in
`octomil-runtime/release/parity/_schema.json`. The SDK does not
duplicate the schema — instead, `scripts/release_run_parity.py`
invokes the runtime-side scripts which write schema-conformant JSON
directly into this directory.

To validate locally:

```bash
python3 -c "
import json, jsonschema
schema = json.load(open('../octomil-runtime/release/parity/_schema.json'))
data   = json.load(open('release/parity/audio.transcription/audio.transcription.parity.json'))
jsonschema.validate(data, schema)
print('OK')
"
```
