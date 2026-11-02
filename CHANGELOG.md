# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## Unreleased

### Documentation

- Horizontal layered architecture diagram
- Compact horizontal architecture diagram

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

### Bench

- Add large-workload scale benchmark (#4)
