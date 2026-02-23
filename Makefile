.PHONY: dev install lint test

dev:
	doppler run -- uvicorn app.main:app --reload --port 8000

install:
	uv venv && uv pip install -e ".[dev]"

lint:
	ruff check . && mypy .

test:
	doppler run -- pytest
