# Python SDK Bloat Reduction Track

Reviewer: @tai

## Goal

Reduce Python SDK drift by making generated contracts authoritative and retiring legacy duplicate client surfaces.

## Findings

- Hand-maintained runtime planner schemas duplicate generated runtime planner contract types and drop fields such as route correlation and gate metadata.
- SDK parity fixtures are stale relative to the current contract.
- A legacy inner Python SDK still carries duplicate clients, package metadata, and model registry surfaces.
- `requirements.txt` reintroduces heavy deps that `pyproject.toml` moved out of core.
- The Makefile contains a stale `app.main:app` dev target.
- Local `.venv`, coverage, mypy, pytest, and pycache artifacts slow scans.

## Proposed Cleanup

- Parse runtime planner responses through generated contract types or keep a single adapter layer that preserves all fields.
- Refresh SDK parity fixtures from `octomil-contracts` and extend conformance coverage for runtime plan response schemas.
- Deprecate or remove the legacy inner SDK package surface after import compatibility tests pass.
- Align `requirements.txt` with pyproject extras and remove accidental heavy core deps.
- Fix or remove stale Makefile targets.
- Teach contract audit tooling to skip ignored cache directories.

## Validation

```bash
python3 -m pytest tests/test_runtime_planner.py tests/test_output_quality_gates.py tests/test_contract_conformance.py tests/test_doctor_and_lazy_imports.py
python3 -m pytest tests/test_doctor_and_lazy_imports.py::test_pandas_pyarrow_moved_off_core_dependencies tests/test_doctor_and_lazy_imports.py::test_import_octomil_does_not_load_pandas_pyarrow_or_torch
make -n dev test install
```
