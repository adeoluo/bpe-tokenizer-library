.PHONY: help setup lint format type test coverage evaluation run diagram docker-build docker-up clean all \ changelog cut-release
	rust-check rust-develop parity bench-compare

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and pre-commit hooks
	uv sync --all-extras --dev
	uv run pre-commit install

format: ## Format automatically
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (no fixes)
	uv run ruff format --check .
	uv run ruff check .

type: ## Type-check
	uv run mypy src

test: ## Run tests
	uv run pytest

coverage: ## Tests with coverage gate
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

evaluation: ## Reproduce the results table
	uv run python -m bpe_tokenizer_library train

run: ## Run the HTTP service
	uv run python -m bpe_tokenizer_library serve

rust-check: ## Rust core: fmt, clippy, unit tests
	cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test

rust-develop: ## Rebuild the Rust core into .venv after editing rust/ (uv sync also builds it)
	uv sync --group rust
	uv run --no-sync maturin develop --uv --release

parity: ## Rust vs Python parity tests
	BPE_REQUIRE_RUST=1 uv run --no-sync pytest tests/test_rust_parity.py tests/test_backend.py

bench-compare: ## Encode throughput: Python vs Rust vs tiktoken
	uv run --no-sync python benchmarks/compare.py

diagram: ## Render every diagram in docs/architecture to SVG (house theme, white background)
	@for f in docs/architecture/*.mmd; do \
		npx -y @mermaid-js/mermaid-cli -q -i "$$f" -o "$${f%.mmd}.svg" -c mermaid.config.json -b white || exit 1; \
	done

docker-build: ## Build the image
	docker build -t bpe-tokenizer-library .

docker-up: ## Run via docker compose
	docker compose up --build

clean: ## Remove caches and artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint type coverage ## Full local gate (mirror of continuous integration)

changelog: ## Regenerate CHANGELOG.md from Conventional Commits (git-cliff)
	git cliff --output CHANGELOG.md && perl -0pi -e 's/\n+\z/\n/' CHANGELOG.md

cut-release: ## Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional)
	scripts/release.sh $(RELEASE_VERSION)
