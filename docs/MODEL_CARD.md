# Model / System Card - BPE Tokenizer Library

## Overview
- **What it is:** a byte-level byte-pair-encoding tokenizer implementing the GPT-2 algorithm
  (reversible byte mapping, regex pre-tokenization, rank-based merges), with a reference Python
  implementation and an optional Rust core (PyO3 extension) that produces identical token ids.
- **Base model:** none. Merges are trained from a corpus.
- **Version and date:** 0.2.0, 2026.

## Intended use
- **Primary use case:** tokenizing text into subword units for a language model, and a readable
  reference implementation of byte-level BPE.
- **Primary users:** engineers building or studying tokenization pipelines.

## Out-of-scope / prohibited use
- Not a drop-in for GPT-2's exact vocabulary unless its published merges are loaded; by default it
  reproduces the algorithm, not that specific table.
- The Python backend encodes at roughly one megabyte per second; the optional Rust backend is 13 to
  19 times faster on the bundled benchmark but is built from source or from a release wheel.

## Performance
- Lossless round-trip on an adversarial battery at every vocabulary size.
- Compression up to 3.74 bytes per token on the bundled corpus; ~600,000 bytes/sec encode with the Python
  backend on a short passage; 32 to 51 MB/s with the Rust backend on 1 MB inputs.
- Full method and tables: [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Limitations and failure modes
- The bundled corpus is small, so merges saturate; a larger corpus improves compression.
- No special-token handling in the default configuration.

## Ethical considerations
- No personal data, no network access. The corpus is a short set of public-domain style sentences.
