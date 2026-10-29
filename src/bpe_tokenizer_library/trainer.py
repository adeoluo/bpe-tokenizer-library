"""Training byte-pair-encoding merges from a corpus.

Starting from individual byte-level glyphs, the trainer repeatedly finds the most
frequent adjacent pair across the corpus and records a merge, until the vocabulary
reaches the requested size. The ordered list of merges is the model: applying them
in order (lowest rank first) reproduces the learned segmentation. This is the same
training procedure as GPT-2's tokenizer, implemented from scratch.
"""

from __future__ import annotations

import itertools
from collections import Counter

from bpe_tokenizer_library.bytelevel import encode_bytes
from bpe_tokenizer_library.pretokenize import pretokenize

Pair = tuple[str, str]


def _word_frequencies(corpus: list[str]) -> dict[tuple[str, ...], int]:
    counts: Counter[tuple[str, ...]] = Counter()
    for document in corpus:
        for chunk in pretokenize(document):
            glyphs = tuple(encode_bytes(chunk))
            if glyphs:
                counts[glyphs] += 1
    return dict(counts)


def _pair_counts(words: dict[tuple[str, ...], int]) -> Counter[Pair]:
    pairs: Counter[Pair] = Counter()
    for symbols, freq in words.items():
        for a, b in itertools.pairwise(symbols):
            pairs[(a, b)] += freq
    return pairs


def _merge_word(symbols: tuple[str, ...], pair: Pair, merged: str) -> tuple[str, ...]:
    out: list[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(corpus: list[str], vocab_size: int) -> list[Pair]:
    """Return the ordered list of merges learned from ``corpus``.

    ``vocab_size`` is the target total vocabulary: 256 base byte glyphs plus the
    learned merges. A deterministic tie-break (lexicographic on the pair) keeps
    training reproducible.
    """

    target_merges = max(0, vocab_size - 256)
    words = _word_frequencies(corpus)
    merges: list[Pair] = []
    for _ in range(target_merges):
        pairs = _pair_counts(words)
        if not pairs:
            break
        best_count = max(pairs.values())
        best = min(pair for pair, count in pairs.items() if count == best_count)
        merged = best[0] + best[1]
        words = {_merge_word(symbols, best, merged): freq for symbols, freq in words.items()}
        merges.append(best)
    return merges
