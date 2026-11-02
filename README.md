<h1 align="center">BPE Tokenizer Library</h1>

<p align="center">
  <em>A byte-level byte-pair-encoding tokenizer implementing the GPT-2 algorithm, with a reference Python implementation and an optional Rust core that produces identical token ids.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/bpe-tokenizer-library/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/bpe-tokenizer-library/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/bpe-tokenizer-library/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/bpe-tokenizer-library"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <img alt="Rust" src="https://img.shields.io/badge/rust-stable-orange">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Motivation

Every language model starts with a tokenizer, and the one used by GPT-2 and its successors is
**byte-level byte-pair encoding**. It has three parts worth understanding: a reversible map from
bytes to printable glyphs that makes the tokenizer lossless for any input, a regular expression
that splits text into word-like chunks, and a learned list of merges applied by rank. This
library implements all three, matching the GPT-2 algorithm. The Python implementation is the
readable reference and needs nothing beyond the standard library.

Because it is byte-level, it round-trips **any** string exactly, including emoji, accented text,
and non-Latin scripts, which tokenizers that split on characters cannot guarantee.

The Python encoder runs at roughly one megabyte per second or less. An optional Rust core, exposed
through PyO3, encodes the same text 13 to 19 times faster and is tested to produce byte-for-byte identical
output, so it can be switched on without changing any result.

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="860">
</p>

<sub>Legend: amber = entry points and external systems, indigo = core logic, green = data and state, blue = validation and control, grey = observability and outputs.</sub>

Training counts adjacent symbol pairs across a pre-tokenized, byte-level corpus and repeatedly
merges the most frequent pair, producing an ordered merge list that is the model. Encoding applies
those merges by rank within each chunk. `BPETokenizer` is the single Python API: with
`backend="rust"` it delegates to the `bpe_tokenizer_rs` extension (Rust core via PyO3), and when
the extension is not installed it falls back to the Python implementation. Parity tests hold the
two paths to identical output. The HTTP service validates text length and id ranges before calling
the same API and exposes Prometheus metrics, and the evaluation reports compression, round-trip
losslessness, and throughput. See the [ADRs](docs/adr/) for the key decisions.

### What it implements

- **Reversible byte-level mapping**: GPT-2's fixed map from the 256 byte values to printable code
  points, so any UTF-8 text becomes carry-able glyphs and back with no loss.
- **GPT-2 pre-tokenization**: the standard regular expression that keeps leading spaces attached,
  separates letters, digits, and punctuation (underscores included), and never merges across
  chunk boundaries.
- **BPE training**: iterative most-frequent-pair merging with a deterministic tie-break.
- **Rank-based encoding**: the same lowest-rank-first merge application GPT-2 uses, plus JSON
  save and load.
- **Rust core** (`rust/`): the same pre-tokenizer, byte map, merge loop, and trainer in Rust. The
  pre-tokenizer uses character-class tables generated from Python's own `re` module for each
  supported Unicode version, and the merge loop switches to a heap for long chunks.

## Results

> Reproduce with `python -m bpe_tokenizer_library train`. Trained on the built-in corpus; metrics
> are on a disjoint held-out passage.

| Vocabulary size | Compression (bytes/token) | Lossless |
|---|---:|:--:|
| 256 (no merges) | 1.00 | yes |
| 400 | 2.42 | yes |
| 600 (requested) | 3.74 | yes |

The small bundled corpus saturates at 237 merges (vocabulary 493), so requesting a larger
vocabulary does not add tokens; a larger corpus continues to improve compression. Compression
improves from one token per byte to **3.74 bytes per token** as merges are learned, and every
configuration round-trips the full battery of adversarial strings (emoji, Japanese, accents,
control whitespace, empty) **exactly**.

### Encode throughput: Python, Rust, tiktoken

> Reproduce with `uv sync --group rust`, `uv run maturin develop --uv --release -m rust/Cargo.toml`,
> then `uv run python benchmarks/compare.py`. Apple M1, macOS, Python 3.13, single thread, best of
> five runs on about 1 MB of text. The machine was shared and under load during the run, so the
> absolute figures are conservative; the ratios are the stable part.

| Workload | Python backend | Rust backend | tiktoken (gpt2) | Rust / Python |
|---|---:|---:|---:|---:|
| English text, 1 MB | 0.47 MB/s | 8.85 MB/s | 8.88 MB/s | 19x |
| Random Unicode, 1 MB | 1.22 MB/s | 15.45 MB/s | 5.14 MB/s | 13x |

The Python and Rust columns use the same merges (vocabulary 493) and return identical token ids;
the benchmark asserts this before timing. tiktoken runs its own GPT-2 vocabulary of 50,257 tokens,
so its column is a throughput reference only: the vocabularies, and therefore the token ids and
the amount of merge work per byte, differ, and the figures do not show that one tokenizer is
faster than the other on equal terms.

### Parity guarantee

The Rust backend is required to match the Python implementation byte for byte: the same
pre-tokenization chunks, the same token ids, the same decoded text (including how invalid UTF-8
is replaced), the same errors for unknown ids, and the same trained merges. The parity suite
(`tests/test_rust_parity.py`) checks this on:

| Check | Inputs | Mismatches |
|---|---:|---:|
| Pre-tokenization | 12,223 strings, plus every Unicode scalar value in four arrangements | 0 |
| Encode ids and round-trip decode | 12,223 strings x 4 merge sets (0 to 944 merges) | 0 |
| Decode of random id sequences | 3,000 sequences, mostly invalid UTF-8 | 0 |
| Training | 3 corpora x 3 vocabulary sizes | 0 |

The strings are the bundled corpus, the round-trip battery, 8,000 seeded random Unicode strings
(half spanning the full code point range), 4,000 seeded strings built from whitespace,
contraction, underscore, digit, emoji, and multi-script fragments, and 200 long chunks that
exercise the heap merge path. Python's `re` classifies characters with the Unicode database of
the running interpreter, so the extension ships one generated table per version (Unicode 14.0,
15.0, 15.1, 16.0; Python 3.11 through 3.14) and the suite passed under each. On any other
interpreter the tokenizer stays on the Python backend rather than risk a difference.

### Lossless on inputs that break naive tokenizers

| Input | Round-trips exactly |
|---|:--:|
| `café naïve résumé` | yes |
| `emoji: 😀🚀🧠 end` | yes |
| `日本語のテキスト` | yes |
| `tabs\tand\nnewlines` | yes |
| `snake_case __dunder__` | yes |
| `a_b _ _x_ trailing_` | yes |

Losslessness is enforced by a property test that pre-tokenizes and round-trips **3,000 random
Unicode strings** (seeded, spanning Latin, punctuation, accented, and CJK ranges) and asserts the
concatenated chunks rebuild each input exactly. An earlier version of the pre-tokenizer silently
dropped underscores (and any other word-class character that is neither a letter nor a digit),
which broke the round trip; the regular expression and the property test were added together so
the class of defect cannot recur. See [the pre-tokenizer notes](src/bpe_tokenizer_library/pretokenize.py).

### Scale (Python backend)

> Reproduce with `uv run python benchmarks/scale.py`. Trains once, then encodes held-out passages
> of increasing size; peak memory from `tracemalloc`.

| Input size | Encode time | Throughput | Tokens | Peak memory | Lossless |
|---|---:|---:|---:|---:|:--:|
| 23,200 bytes | 0.08 s | ~300,000 bytes/s | 6,200 | 0.3 MiB | yes |
| 116,000 bytes | 0.39 s | ~301,000 bytes/s | 31,000 | 1.5 MiB | yes |
| 464,000 bytes | 1.68 s | ~277,000 bytes/s | 124,000 | 5.8 MiB | yes |
| 1,160,000 bytes | 4.44 s | ~261,000 bytes/s | 310,000 | 14.5 MiB | yes |

Encode throughput stays roughly constant per byte and memory grows linearly with input size, and
the round trip stays lossless at over a megabyte of input. This table measures how the Python
implementation scales; for wall-clock speed see the throughput comparison above.

### Evaluation

The corpus is bundled and training is deterministic, so every number reproduces offline. Full
method and tables: [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md).

### Comparison scope

Compression is measured on a small bundled corpus against a raw-byte baseline, not against an
established tokenizer's vocabulary. Throughput is compared with
[tiktoken](https://github.com/openai/tiktoken) as a reference point only, since the vocabularies
differ; [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) has not been
measured. What the results establish is a faithful, lossless implementation of the GPT-2
byte-level BPE algorithm, and a Rust core that reproduces it exactly at roughly the speed class of
a compiled tokenizer.

## Quickstart

```bash
git clone https://github.com/adeoluo/bpe-tokenizer-library.git && cd bpe-tokenizer-library
make setup                      # install dev tooling + pre-commit hooks
make all                        # lint + type-check + test

python -m bpe_tokenizer_library train              # train and report metrics
python -m bpe_tokenizer_library encode "the dog"    # encode a string and show the round-trip
```

```python
from bpe_tokenizer_library import BPETokenizer, sample_corpus

tok = BPETokenizer.train(sample_corpus(), vocab_size=600)
ids = tok.encode("café 😀")
assert tok.decode(ids) == "café 😀"
```

## Usage

### Command line

```bash
python -m bpe_tokenizer_library train              # train and report metrics
python -m bpe_tokenizer_library encode "the dog"    # encode a string and show the round-trip
python -m bpe_tokenizer_library serve               # start the HTTP service on :8000
```

### Library API

`BPETokenizer.train(corpus, vocab_size)` learns merges; `encode`, `decode`, `encode_batch`, and
`tokens` apply them; `to_json` and `BPETokenizer.from_json` save and load the merge list.

### Rust backend

The Rust core is an optional extension module, built from `rust/` with maturin. The package
installs and works without it.

```bash
uv sync --group rust                                     # maturin and tiktoken (opt-in)
uv run maturin develop --uv --release -m rust/Cargo.toml # build bpe_tokenizer_rs into .venv
```

```python
from bpe_tokenizer_library import BPETokenizer, rust_available, sample_corpus

tok = BPETokenizer.train(sample_corpus(), vocab_size=600, backend="rust")
print(rust_available(), tok.backend)  # True rust (or False python without the extension)
ids = tok.encode("café 😀")  # identical ids to the Python backend
fast = BPETokenizer.from_json(tok.to_json(), backend="rust")  # or tok.with_backend("rust")
```

`backend="rust"` is a request, not a requirement: when the extension is missing, or has no
character table for the interpreter's Unicode version, the tokenizer uses the Python backend and
`tok.backend` reports `"python"`. `encode_batch` on the Rust backend releases the GIL while
encoding.

### HTTP service

A standard-library HTTP service trains once at startup, then encodes and decodes text. The routing
logic is a pure function, tested without a socket. Under a concurrent stress benchmark
(`uv run python benchmarks/stress.py`, 10000 requests, 8 workers) the `/encode` handler sustains
roughly **6,200 requests/second** at **p50 0.11 ms, p95 0.23 ms**.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness and vocabulary size |
| GET | `/ready` | readiness once the tokenizer is trained |
| GET | `/metrics` | per-route request, error, and latency metrics (Prometheus text) |
| POST | `/encode` | body `{"text": str}`; returns token ids and glyph tokens |
| POST | `/decode` | body `{"ids": [int]}`; returns the decoded text |
| POST | `/stats` | body `{"text": str}`; returns byte length, token count, bytes per token |

### Demonstration

```text
$ python -m bpe_tokenizer_library train
Trained a byte-level BPE tokenizer: vocab 493, 237 merges.
  held-out compression: 3.74 bytes/token (31 tokens)
  lossless on the round-trip battery: True
  encode throughput: 564,232 bytes/sec
```

### Demos

Recorded from real runs with [vhs](https://github.com/charmbracelet/vhs); each tape is in `docs/demo/`.

Training the byte-level BPE, then encoding text with it.

<p align="center">
  <img src="docs/demo/train-encode.gif" alt="Training the byte-level BPE, then encoding text with it" width="860">
</p>

## Design decisions

- **Byte-level, not character-level**: operating on bytes through a reversible mapping makes the
  tokenizer lossless for arbitrary input. See [ADR 0002](docs/adr/0002-byte-level.md).
- **Merge ranks for encoding**: applying merges lowest-rank-first reproduces GPT-2's segmentation
  exactly and keeps encoding deterministic. See [ADR 0003](docs/adr/0003-merge-ranks.md).
- **Optional Rust core with a Python fallback**: the Python implementation stays the reference
  and the default; the Rust extension is opt-in and must match it exactly. See
  [ADR 0004](docs/adr/0004-rust-core.md).

## Testing

```bash
make all                                   # ruff, mypy --strict, pytest with an 80% coverage gate
cd rust && cargo fmt --check && cargo clippy -- -D warnings && cargo test
BPE_REQUIRE_RUST=1 uv run pytest tests/test_rust_parity.py   # after maturin develop
```

- **Python tests** cover the byte map, pre-tokenizer, trainer, tokenizer, evaluation helpers, CLI,
  HTTP service, and the backend fallback; they pass without the extension, and the parity module
  skips itself when the extension is not built.
- **Property tests** round-trip thousands of seeded random Unicode strings.
- **Rust unit tests** (`cargo test`) cover the byte map, pre-tokenizer branches, per-version
  Unicode tables, merge ranks, decode errors, the trainer, and agreement between the scan and heap
  merge loops.
- **Parity tests** compare the two backends as described under Results. CI runs them in a
  dedicated `rust` job (fmt, clippy, cargo test, maturin develop, parity) with
  `BPE_REQUIRE_RUST=1`, so a missing extension fails instead of skipping.

## Responsible AI

- **Model card:** [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- **Data card:** [`docs/DATA_CARD.md`](docs/DATA_CARD.md)
- **Threat model:** [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)

## Limitations

- The bundled corpus is small, so learned merges saturate at a few hundred; compression plateaus
  accordingly. Training on a larger corpus continues to improve it.
- The Python backend runs at roughly one megabyte per second or less; use the Rust backend where
  throughput matters.
- The Rust extension is built from source with maturin; prebuilt wheels are attached to releases
  for macOS arm64 only.
- Rust parity depends on a generated character table per Unicode version. A new Python release
  with a newer Unicode database uses the Python backend until a table is generated for it
  (`scripts/gen_unicode_tables.py`).
- The implementation follows the GPT-2 algorithm; it does not ship GPT-2's published vocabulary
  file, so it reproduces the mechanism rather than that specific vocabulary.

## Roadmap

- [x] Rust core with PyO3 bindings and byte-for-byte parity tests.
- [x] Encode throughput compared with tiktoken.
- [ ] Load GPT-2's published merges to compare token ids directly with tiktoken.
- [ ] Benchmark against Hugging Face Tokenizers.
- [ ] Publish wheels for Linux and Windows.
- [ ] Add special-token handling and an added-tokens table.

## License

Released under the [MIT License](LICENSE).
