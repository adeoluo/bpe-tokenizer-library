# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.3.0] 2026-11-02

### Performance

- Heap-based trainer, parallel encode_batch (#3)

### Documentation

- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Remove authorship statements from docs and scripts
- Recorded demos of more commands, with their vhs tapes (#1)
- Recorded demos of the HTTP service and more commands (#2)

### Maintenance

- Add release automation, AGENTS.md, llms.txt and contributor docs
- Classify squash merges in the changelog by the commits they contain

## [0.2.0] 2026-11-02

### Features

- Implement byte-level BPE tokenizer with GPT-2 algorithm (#1)
- Add optional Rust core with PyO3 bindings and parity tests

### Bug fixes

- Stop pre-tokenizer dropping underscores (data-loss defect) (#3)

### Documentation

- Add explicit comparison scope and honest baseline framing (#2)

### Maintenance

- Initialize repository with license and overview
- Remove Dependabot configuration
