# Evaluation Report - BPE Tokenizer Library

> Reproduce with `python -m bpe_tokenizer_library train` and `benchmarks/benchmark.py`. Training
> is deterministic and offline.

## Setup

- **Corpus:** a small bundled set of English sentences with repetition (no downloads).
- **Held-out text:** a short passage not identical to any training sentence.
- **Metrics:** compression (bytes per token), lossless round-trip over an adversarial battery, and
  encode throughput.
- **Hardware / cost:** CPU only; training is tens of milliseconds. No accelerators, no external
  services.

## Correctness: lossless round-trip

Every string in the battery decodes back exactly, including emoji, accented text, Japanese,
control whitespace, and the empty string. This holds at every vocabulary size because the
byte-level mapping is reversible and merges never cross chunk boundaries. It is enforced as a test.

| Input | Round-trips exactly |
|---|:--:|
| `the quick brown fox` | yes |
| `Hello, World! 123` | yes |
| `café naïve résumé` | yes |
| `emoji: 😀🚀🧠 end` | yes |
| `日本語のテキスト` | yes |
| `tabs\tand\nnewlines` | yes |
| `   leading and trailing   ` | yes |
| `` (empty) | yes |

## Compression by vocabulary size

| Vocabulary | Compression (bytes/token) | Tokens on held-out text |
|---|---:|---:|
| 256 (no merges) | 1.00 | 116 |
| 400 | 2.42 | 48 |
| 493 (corpus saturates) | 3.74 | 31 |

Compression improves as merges are learned. The bundled corpus saturates at 237 merges
(vocabulary 493); a larger corpus would continue to improve compression.

## Throughput

Encoding the held-out passage repeatedly (`benchmarks/benchmark.py`, which reports each backend by
name) yields about 2.0 MB and 540,000 tokens per second with the Python backend and about 32 MB and
8.6 million tokens per second with the Rust backend. Under a concurrent service stress benchmark the
`/encode` handler, on the default Rust backend, sustains about 69,000 requests per second with p50
near 0.006 ms and p95 near 0.45 ms.

### Rust backend and tiktoken

`benchmarks/compare.py` encodes about 1 MB of text single-threaded (best of five runs, Apple M1,
Python 3.12; absolute figures move with machine load, the ratios are the stable part). The Python and Rust backends share the same merges and return identical ids;
tiktoken uses its own GPT-2 vocabulary, so its column is a throughput reference only.

| Workload | Python backend | Rust backend | tiktoken (gpt2) |
|---|---:|---:|---:|
| English text, 1 MB | 2.03 MB/s | 32.96 MB/s | 31.03 MB/s |
| Random Unicode, 1 MB | 4.60 MB/s | 53.27 MB/s | 22.35 MB/s |

## Parity of the Rust backend

`tests/test_rust_parity.py` compares the backends on 12,223 strings (bundled corpus, round-trip
battery, seeded random Unicode, adversarial fragments, long chunks) against four merge sets, every
Unicode scalar value through the pre-tokenizer, 3,000 random id sequences through the decoder,
and nine training runs. All outputs were identical under Python 3.12, 3.13, 3.14 and 3.15.

## Reproducibility

- Training is deterministic: the same corpus and vocabulary size produce identical merges.
- No network access, no external data.
- Regenerate every number with the CLI and `benchmarks/benchmark.py`.
