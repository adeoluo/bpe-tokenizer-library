# ADR-0003: Encoding by merge rank

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
Once merges are learned, a chunk of glyphs must be segmented into tokens. Applying merges in the
wrong order would produce a different, inconsistent segmentation.

## Decision
Assign each merge a rank equal to its position in the training order (earlier merges are more
frequent and rank lower). To encode a chunk, repeatedly apply the available merge with the lowest
rank until none remain. This is the same procedure GPT-2 uses.

## Consequences
- Encoding is deterministic and reproduces the segmentation implied by the training order.
- The encoder is a simple loop over the current symbols; it is easy to read and to test.
- Saving the ordered merge list is enough to reconstruct the tokenizer; the vocabulary is derived
  from it on load.
