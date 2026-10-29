# ADR-0002: Byte-level encoding for lossless tokenization

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
A tokenizer must handle any input a user can type: emoji, accented characters, non-Latin scripts,
and control whitespace. A character-level tokenizer has to decide what to do with characters it
never saw in training, and a naive byte split produces unprintable symbols that are awkward to
store and inspect.

## Decision
Operate at the byte level through GPT-2's reversible map from the 256 byte values to printable
Unicode code points. Text is encoded as UTF-8 bytes, mapped to glyphs, tokenized, and mapped back
on decode. Merges are learned and applied over these glyphs.

## Consequences
- The tokenizer is lossless for arbitrary input, because every byte has a reversible glyph and
  merges never cross chunk boundaries. This is verified by a round-trip battery in the tests.
- There are no out-of-vocabulary characters: any byte sequence is representable.
- Tokens are byte fragments, not whole words, which is the expected behavior for this family of
  tokenizers.
