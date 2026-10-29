# Model / System Card - BPE Tokenizer Library

## Overview
- **What it is:** a from-scratch byte-level byte-pair-encoding tokenizer implementing the GPT-2
  algorithm (reversible byte mapping, regex pre-tokenization, rank-based merges). No third-party
  dependencies.
- **Base model:** none. Merges are trained from a corpus.
- **Version and date:** 0.1.0, 2026.

## Intended use
- **Primary use case:** tokenizing text into subword units for a language model, and a readable
  reference implementation of byte-level BPE.
- **Primary users:** engineers building or studying tokenization pipelines.

## Out-of-scope / prohibited use
- Not a drop-in for GPT-2's exact vocabulary unless its published merges are loaded; by default it
  reproduces the algorithm, not that specific table.
- Not an optimized production kernel; pure-Python encoding is fast but not C-speed.

## Performance
- Lossless round-trip on an adversarial battery at every vocabulary size.
- Compression up to 3.74 bytes per token on the bundled corpus; ~600,000 bytes/sec encode.
- Full method and tables: [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Limitations and failure modes
- The bundled corpus is small, so merges saturate; a larger corpus improves compression.
- No special-token handling in the default configuration.

## Ethical considerations
- No personal data, no network access. The corpus is a short set of public-domain style sentences.
