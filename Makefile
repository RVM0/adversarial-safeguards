# Convenience targets. Run `make help` for a list.

.PHONY: help setup install lint format test test-unit test-integration smoke pilot sweep clean docker

help:
	@echo "Available targets:"
	@echo "  setup           Bootstrap the .venv and install deps (calls scripts/setup_env.sh)"
	@echo "  install         Re-install advsafe in dev mode"
	@echo "  lint            Run ruff + mypy"
	@echo "  format          Run ruff format"
	@echo "  test            Run unit tests (fast)"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests (downloads models)"
	@echo "  smoke           Run advsafe-smoke on Llama 3.1 8B"
	@echo "  pilot           Run the Week 2 pilot"
	@echo "  sweep           Run the full Week 3 sweep (warns; cloud cost)"
	@echo "  docker          Build the Docker image"
	@echo "  clean           Remove build artifacts (NOT results or data)"

setup:
	bash scripts/setup_env.sh

# Backend for smoke/pilot/sweep. Default mlx (local Apple Silicon); override with BACKEND=hf.
BACKEND ?= mlx

install:
	pip install -e ".[dev]"

install-mlx:
	pip install -e ".[mlx,dev]"   # Apple Silicon, torch-free (installs on Python 3.14)

install-hf:
	pip install -e ".[hf,dev]"    # torch/transformers stack (CUDA/Linux reproduction)

lint:
	ruff check src tests
	mypy src --ignore-missing-imports || true

format:
	ruff format src tests
	ruff check --fix src tests

test test-unit:
	pytest tests/unit -v -m "not slow"

test-integration:
	pytest tests/integration -v

smoke:
	advsafe-smoke --model llama-3.1-8b --prompt "Hello, are you online?" --backend $(BACKEND)

pilot:
	advsafe-pilot --config configs/experiments/pilot.yaml --output results/pilot --backend $(BACKEND)

sweep:
	@echo "Launching the full sweep on the '$(BACKEND)' backend."
	@echo "  BACKEND=mlx → local on this Mac, \$$0.  BACKEND=hf → cloud A100s, ~\$$77."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ]
	advsafe-sweep --config configs/experiments/sweep.yaml --output results/sweep --backend $(BACKEND)

docker:
	docker build -t advsafe:latest .

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	@echo "Note: data/, results/, checkpoints/ are intentionally preserved."
