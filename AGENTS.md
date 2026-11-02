# AGENTS.md

Working notes for coding agents and contributors in **BPE Tokenizer Library**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

A byte-level byte-pair-encoding tokenizer implementing the GPT-2 algorithm, with a reference Python implementation and an optional Rust core that produces identical token ids.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `rust/`: Rust core compiled into a Python extension with PyO3
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `rust/src/bytelevel.rs`: GPT-2's reversible map between the 256 byte values and printable glyphs.
- `rust/src/lib.rs`: Rust core of `bpe-tokenizer-library`.
- `rust/src/pretokenize.rs`: Pre-tokenization that reproduces the Python pattern exactly.
- `rust/src/python.rs`: PyO3 bindings, compiled only with the `python` feature (enabled by maturin).
- `rust/src/tokenizer.rs`: Encoding and decoding with an ordered merge list.
- `rust/src/trainer.rs`: Training merges, identical to the Python `train_bpe`.
- `rust/src/unicode_tables/mod.rs`: Character-class tables, one per Unicode version of the Python `re` module.
- `rust/src/unicode_tables/u14_0_0.rs`
- `rust/src/unicode_tables/u15_0_0.rs`
- `rust/src/unicode_tables/u15_1_0.rs`
- `rust/src/unicode_tables/u16_0_0.rs`
- `src/bpe_tokenizer_library/__init__.py`: BPE Tokenizer Library: a byte-level byte-pair-encoding tokenizer with an optional Rust core.
- `src/bpe_tokenizer_library/backend.py`: Optional Rust backend, loaded only when the ``bpe_tokenizer_rs`` extension is built.
- `src/bpe_tokenizer_library/bytelevel.py`: The reversible byte-to-unicode mapping used by GPT-2 style byte-level BPE.
- `src/bpe_tokenizer_library/cli.py`: Command line interface for the byte-level BPE tokenizer.
- `src/bpe_tokenizer_library/data.py`: A small built-in corpus for training and demonstrations.
- `src/bpe_tokenizer_library/evaluation.py`: Evaluation: compression, lossless round-trip, and encode throughput.
- `src/bpe_tokenizer_library/observability.py`: In-process request metrics for the HTTP service (Prometheus text exposition).
- `src/bpe_tokenizer_library/pretokenize.py`: Pre-tokenization: split text into word-like chunks before byte-level BPE.
- `src/bpe_tokenizer_library/service.py`: A standard-library HTTP service exposing the trained tokenizer.
- `src/bpe_tokenizer_library/tokenizer.py`: The byte-level BPE tokenizer: encode, decode, save, and load.
- `src/bpe_tokenizer_library/trainer.py`: Training byte-pair-encoding merges from a corpus.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the HTTP service |
| `make rust-check` | Rust core: fmt, clippy, unit tests |
| `make rust-develop` | Build the Rust extension into .venv (opt-in rust group) |
| `make parity` | Rust vs Python parity tests (requires rust-develop) |
| `make bench-compare` | Encode throughput: Python vs Rust vs tiktoken (requires rust-develop) |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut by the maintainer with `make cut-release`, never from CI.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

## Rust notes

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` must pass.
- No `unwrap()` in library code paths that can fail at runtime; return a typed error.
- Unsafe code is limited to the places that need a syscall and carries a `SAFETY:` comment.
