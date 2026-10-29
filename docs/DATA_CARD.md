# Data Card - BPE Tokenizer Library

## Source
- **Bundled, in-repository.** The training corpus is a small set of English sentences defined in
  code. There is no external dataset, download, or network access.

## Composition
- Ten short public-domain style sentences, each repeated to give merge statistics signal.
- A separate held-out passage, not identical to any training sentence, is used for evaluation.

## Labels and supervision
- Tokenization is unsupervised; training only counts symbol-pair frequencies. There are no labels.

## Provenance and licensing
- Authored for this repository under the MIT license. No third-party text is included.

## Filtering and contamination
- The held-out passage is distinct from the training sentences, so compression is measured on text
  the merges were not fit to verbatim.

## Known limitations
- The corpus is intentionally tiny so training is instant and deterministic. It is not
  representative of natural language at scale, and merges saturate quickly.
