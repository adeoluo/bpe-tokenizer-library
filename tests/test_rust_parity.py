"""Parity between the Rust core and the Python reference implementation.

Every check compares the two backends on the same input and requires identical
output: the same pre-tokenization chunks, the same token ids, the same decoded
text (including the replacement of invalid UTF-8), the same errors, and the same
trained merges. The module is skipped when the extension is not built, unless
``BPE_REQUIRE_RUST=1`` is set (as in CI), in which case a missing extension fails.
"""

from __future__ import annotations

import os
import unicodedata
from functools import cache
from random import Random

import pytest

from bpe_tokenizer_library import backend
from bpe_tokenizer_library.data import held_out_text, sample_corpus
from bpe_tokenizer_library.evaluation import ROUND_TRIP_CASES, random_unicode_cases
from bpe_tokenizer_library.pretokenize import pretokenize
from bpe_tokenizer_library.tokenizer import BPETokenizer
from bpe_tokenizer_library.trainer import train_bpe

if os.environ.get("BPE_REQUIRE_RUST") == "1":
    assert backend.rust_available(), "BPE_REQUIRE_RUST=1 but the Rust extension is unusable"

pytestmark = pytest.mark.skipif(
    not backend.rust_available(), reason="Rust extension bpe_tokenizer_rs is not built"
)

# Characters that stress every branch of the pre-tokenizer: ASCII and Unicode
# whitespace, contractions, underscores, digits from several scripts, numerics
# that Python counts as letters, combining marks, emoji, and CJK.
_PIECES = (
    " ", "  ", "\t", "\n", "\r\n", "\x0b", "\x1c", "\x85", "\xa0", "\u2003", "\u3000",
    "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'", "''", "'S",
    "_", "__", "a_b", "x", "the", "Quick", "ﬁ", "é", "e\u0301", "ß", "İ",
    "0", "42", "\u0663\u0664", "\u0967", "²", "½", "Ⅻ",
    "!", "?!", "...", "#", "$", "/", "\\", "😀", "🚀🧠", "👩‍💻", "\U0001f1fa\U0001f1f8",
    "日本語", "한국어", "Привет", "مرحبا", "\u200b", "\ufeff", "\x00", "\x7f",
)  # fmt: skip


def _piece_cases(count: int, seed: int) -> list[str]:
    rng = Random(seed)
    return ["".join(rng.choice(_PIECES) for _ in range(rng.randint(0, 24))) for _ in range(count)]


def _long_cases(count: int, seed: int) -> list[str]:
    # Long single chunks exercise the Rust heap-based merge path (64+ bytes).
    rng = Random(seed)
    alphabet = "etaoinshrdlu"
    return [
        " " + "".join(rng.choice(alphabet) for _ in range(rng.randint(64, 400)))
        for _ in range(count)
    ]


@cache
def _all_cases() -> tuple[str, ...]:
    cases: list[str] = list(ROUND_TRIP_CASES)
    cases += sample_corpus(repeat=1)
    cases.append(held_out_text())
    cases.append(" ".join(sample_corpus()))
    cases += random_unicode_cases(count=4000, seed=1)
    cases += random_unicode_cases(count=4000, seed=2, max_length=64, max_codepoint=0x10FFFF)
    cases += _piece_cases(4000, seed=3)
    cases += _long_cases(200, seed=4)
    return tuple(cases)


@cache
def _mixed_merges() -> list[tuple[str, str]]:
    # Merges learned from non-ASCII text, so multi-byte glyph merges are covered.
    corpus = sample_corpus(repeat=2) + _piece_cases(400, seed=5) + list(_long_cases(20, seed=6))
    return train_bpe(corpus, 1200)


@cache
def _merge_sets() -> tuple[tuple[str, list[tuple[str, str]]], ...]:
    corpus = sample_corpus()
    return (
        ("base", []),
        ("corpus-400", train_bpe(corpus, 400)),
        ("corpus-1000", train_bpe(corpus, 1000)),
        ("mixed-1200", _mixed_merges()),
    )


def _pair(merges: list[tuple[str, str]]) -> tuple[BPETokenizer, BPETokenizer]:
    python = BPETokenizer.from_merges(merges)
    rust = BPETokenizer.from_merges(merges, backend="rust")
    assert python.backend == "python"
    assert rust.backend == "rust"
    return python, rust


def test_case_count_is_in_the_thousands() -> None:
    assert len(_all_cases()) > 12_000


def test_pretokenize_matches_python() -> None:
    ext = backend._EXTENSION
    version = unicodedata.unidata_version
    for case in _all_cases():
        assert ext.pretokenize(case, version) == pretokenize(case), repr(case)


def test_pretokenize_matches_python_on_every_code_point() -> None:
    ext = backend._EXTENSION
    version = unicodedata.unidata_version
    scalars = [chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    for text in ("".join(scalars), " ".join(scalars), "a".join(scalars), "1 ".join(scalars)):
        assert ext.pretokenize(text, version) == pretokenize(text)


@pytest.mark.parametrize("index", range(4), ids=lambda i: f"merges{i}")
def test_encode_ids_and_round_trip_match_python(index: int) -> None:
    _, merges = _merge_sets()[index]
    python, rust = _pair(merges)
    mismatches = []
    for case in _all_cases():
        ids = rust.encode(case)
        if ids != python.encode(case) or rust.decode(ids) != case:
            mismatches.append(case)
    assert mismatches == []


def test_encode_batch_matches_encode() -> None:
    python, rust = _pair(_mixed_merges())
    cases = list(_all_cases()[:2000])
    assert rust.encode_batch(cases) == python.encode_batch(cases)
    assert rust.encode_batch(cases) == [python.encode(c) for c in cases]


def test_decode_matches_python_on_arbitrary_ids() -> None:
    # Random id sequences produce invalid UTF-8; both sides must replace it the same way.
    python, rust = _pair(_mixed_merges())
    rng = Random(8)
    for _ in range(3000):
        ids = [rng.randrange(python.vocab_size) for _ in range(rng.randint(0, 12))]
        assert rust.decode(ids) == python.decode(ids), ids


@pytest.mark.parametrize("bad", [-1, 10**6, 2**70])
def test_unknown_ids_raise_key_error_on_both(bad: int) -> None:
    python, rust = _pair(_mixed_merges())
    for tok in (python, rust):
        with pytest.raises(KeyError):
            tok.decode([1, bad])


def test_training_matches_python() -> None:
    corpora = (
        sample_corpus(),
        sample_corpus(repeat=1) + _piece_cases(300, seed=9),
        list(random_unicode_cases(count=300, seed=10)),
    )
    for corpus in corpora:
        for vocab_size in (256, 300, 700):
            assert backend.rust_train(corpus, vocab_size) == train_bpe(corpus, vocab_size)


def test_trained_rust_tokenizer_equals_python() -> None:
    python = BPETokenizer.train(sample_corpus(), 600)
    rust = BPETokenizer.train(sample_corpus(), 600, backend="rust")
    assert rust.backend == "rust"
    assert rust == python
    text = held_out_text()
    assert rust.encode(text) == python.encode(text)
    assert rust.tokens(text) == python.tokens(text)


def test_json_round_trip_keeps_backend() -> None:
    tok = BPETokenizer.train(sample_corpus(repeat=5), 400)
    restored = BPETokenizer.from_json(tok.to_json(), backend="rust")
    assert restored.backend == "rust"
    assert restored.encode("the quick dog") == tok.encode("the quick dog")
