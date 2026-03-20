.PHONY: dev install lint test bench bench-update

dev:
	doppler run -- uvicorn app.main:app --reload --port 8000

install:
	uv venv && uv pip install -e ".[dev]"

lint:
	ruff check . && mypy .

test:
	doppler run -- pytest

bench:
	python3 scripts/run_benchmark.py --iterations 30 --warmup 3

bench-update:
	python3 scripts/run_benchmark.py --iterations 30 --warmup 3 --update-baseline
