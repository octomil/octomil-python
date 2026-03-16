.PHONY: dev install lint test sync-catalog

dev:
	doppler run -- uvicorn app.main:app --reload --port 8000

install:
	uv venv && uv pip install -e ".[dev]"

lint:
	ruff check . && mypy .

test:
	doppler run -- pytest

sync-catalog:
	python3 scripts/sync_embedded_catalog.py --from-seed ../octomil-server/server/app/services/catalog_seed.py
