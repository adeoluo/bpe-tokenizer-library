"""The byte-level BPE tokenizer: encode, decode, save, and load.

Encoding pre-tokenizes the text, maps each chunk to byte-level glyphs, and applies
the learned merges within each chunk using merge ranks, exactly as GPT-2 does. The
result is lossless for arbitrary input because the byte-level mapping is reversible
and merges never cross chunk boundaries. Decoding maps token ids back to glyphs and
then back to the original bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from bpe_tokenizer_library.bytelevel import decode_bytes, encode_bytes
from bpe_tokenizer_library.pretokenize import pretokenize
from bpe_tokenizer_library.trainer import Pair, train_bpe


@dataclass
class BPETokenizer:
    """A trained byte-level byte-pair-encoding tokenizer."""

    merges: list[Pair]
    vocab: dict[str, int]
    _ranks: dict[Pair, int]
    _id_to_token: dict[int, str]

    @classmethod
    def from_merges(cls, merges: list[Pair]) -> BPETokenizer:
        """Build a tokenizer from an ordered merge list, deriving the vocabulary."""

        from bpe_tokenizer_library.bytelevel import bytes_to_unicode

        vocab: dict[str, int] = {}
        for glyph in bytes_to_unicode().values():
            vocab.setdefault(glyph, len(vocab))
        for a, b in merges:
            token = a + b
            if token not in vocab:
                vocab[token] = len(vocab)
        ranks = {pair: index for index, pair in enumerate(merges)}
        id_to_token = {index: token for token, index in vocab.items()}
        return cls(merges=merges, vocab=vocab, _ranks=ranks, _id_to_token=id_to_token)

    @classmethod
    def train(cls, corpus: list[str], vocab_size: int) -> BPETokenizer:
        """Train merges on ``corpus`` and return a ready tokenizer."""

        return cls.from_merges(train_bpe(corpus, vocab_size))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _bpe(self, glyphs: str) -> list[str]:
        symbols = list(glyphs)
        if len(symbols) < 2:
            return symbols
        while True:
            best_rank = None
            best_index = -1
            for i in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i
            if best_rank is None:
                break
            merged = symbols[best_index] + symbols[best_index + 1]
            symbols[best_index : best_index + 2] = [merged]
        return symbols

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token ids."""

        ids: list[int] = []
        for chunk in pretokenize(text):
            for token in self._bpe(encode_bytes(chunk)):
                ids.append(self.vocab[token])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back into the original string."""

        glyphs = "".join(self._id_to_token[i] for i in ids)
        return decode_bytes(glyphs)

    def tokens(self, text: str) -> list[str]:
        """Return the glyph tokens (not ids) for inspection."""

        return [self._id_to_token[i] for i in self.encode(text)]

    def to_json(self) -> str:
        """Serialize the merges to JSON (the vocabulary is derived on load)."""

        return json.dumps({"merges": [list(p) for p in self.merges]})

    @classmethod
    def from_json(cls, blob: str) -> BPETokenizer:
        """Load a tokenizer from :meth:`to_json` output."""

        data = json.loads(blob)
        merges = [(a, b) for a, b in data["merges"]]
        return cls.from_merges(merges)
