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

Encoding the held-out passage repeatedly yields roughly 560,000 to 680,000 bytes per second and
over 180,000 tokens per second in pure Python. Under a concurrent service stress benchmark the
`/encode` handler sustains about 6,200 requests per second with p50 near 0.11 ms.

## Reproducibility

- Training is deterministic: the same corpus and vocabulary size produce identical merges.
- No network access, no external data.
- Regenerate every number with the CLI and `benchmarks/benchmark.py`.
