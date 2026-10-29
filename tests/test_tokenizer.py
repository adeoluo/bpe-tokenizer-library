"""Tests for byte-level mapping, pretokenization, training, and the tokenizer."""

from __future__ import annotations

from bpe_tokenizer_library.bytelevel import bytes_to_unicode, decode_bytes, encode_bytes
from bpe_tokenizer_library.data import held_out_text, sample_corpus
from bpe_tokenizer_library.evaluation import (
    ROUND_TRIP_CASES,
    compression,
    encode_throughput,
    fuzz_round_trip_rate,
    is_lossless,
    pretokenize_is_lossless,
    random_unicode_cases,
    round_trip_rate,
)
from bpe_tokenizer_library.pretokenize import pretokenize
from bpe_tokenizer_library.tokenizer import BPETokenizer
from bpe_tokenizer_library.trainer import train_bpe


def test_byte_mapping_is_reversible_over_all_bytes() -> None:
    table = bytes_to_unicode()
    assert len(table) == 256
    assert len(set(table.values())) == 256
    for raw in (b"hello", bytes(range(256))):
        text = raw.decode("latin-1")
        assert encode_bytes(text) is not None
    # round-trip arbitrary UTF-8
    for s in ("abc", "café", "日本語", "😀"):
        assert decode_bytes(encode_bytes(s)) == s


def test_pretokenize_keeps_leading_spaces() -> None:
    chunks = pretokenize("the quick dog")
    assert chunks == ["the", " quick", " dog"]


def test_pretokenize_separates_digits_and_punctuation() -> None:
    assert pretokenize("abc123!!!") == ["abc", "123", "!!!"]


def test_pretokenize_preserves_underscores() -> None:
    # Regression: underscores were previously dropped because they are word
    # characters excluded from both the letter class and the naive "other" class.
    assert pretokenize("a_b") == ["a", "_", "b"]
    assert pretokenize("__init__") == ["__", "init", "__"]
    assert pretokenize("_") == ["_"]
    for case in ("a_b", "__init__", "_", "x_y 1_2", "snake_case"):
        assert "".join(pretokenize(case)) == case


def test_pretokenize_is_lossless_over_random_unicode() -> None:
    # Property test: the pre-tokenizer must regroup, never drop or add characters.
    cases = random_unicode_cases(count=3000, seed=0)
    assert pretokenize_is_lossless(cases)


def test_tokenizer_round_trips_random_unicode() -> None:
    tok = BPETokenizer.train(sample_corpus(), vocab_size=600)
    cases = random_unicode_cases(count=1000, seed=7)
    assert fuzz_round_trip_rate(tok, cases) == 1.0


def test_training_is_deterministic() -> None:
    corpus = sample_corpus(repeat=5)
    assert train_bpe(corpus, 400) == train_bpe(corpus, 400)


def test_merges_count_matches_vocab_target() -> None:
    tok = BPETokenizer.train(sample_corpus(repeat=5), vocab_size=320)
    assert tok.vocab_size >= 256
    assert len(tok.merges) <= 64


def test_encode_decode_is_lossless_on_battery() -> None:
    tok = BPETokenizer.train(sample_corpus(), vocab_size=600)
    for case in ROUND_TRIP_CASES:
        assert tok.decode(tok.encode(case)) == case
    assert is_lossless(tok)
    assert round_trip_rate(tok) == 1.0


def test_compression_improves_with_vocab_size() -> None:
    corpus = sample_corpus()
    text = held_out_text()
    small = compression(BPETokenizer.train(corpus, 256), text).bytes_per_token
    large = compression(BPETokenizer.train(corpus, 600), text).bytes_per_token
    assert small == 1.0  # a 256-vocab tokenizer has no merges, one token per byte
    assert large > small


def test_merges_lower_token_count_than_raw_bytes() -> None:
    tok = BPETokenizer.train(sample_corpus(), vocab_size=600)
    text = held_out_text()
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_json_round_trip_preserves_encoding() -> None:
    tok = BPETokenizer.train(sample_corpus(repeat=5), vocab_size=400)
    restored = BPETokenizer.from_json(tok.to_json())
    assert restored.encode("the quick dog") == tok.encode("the quick dog")
    assert restored.merges == tok.merges


def test_tokens_helper_matches_ids() -> None:
    tok = BPETokenizer.train(sample_corpus(repeat=5), vocab_size=400)
    text = "the lazy dog"
    ids = tok.encode(text)
    assert tok.tokens(text) == [tok._id_to_token[i] for i in ids]


def test_single_character_and_empty_inputs() -> None:
    tok = BPETokenizer.train(sample_corpus(repeat=5), vocab_size=400)
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.decode(tok.encode("a")) == "a"


def test_throughput_report_positive() -> None:
    tok = BPETokenizer.train(sample_corpus(repeat=5), vocab_size=400)
    report = encode_throughput(tok, held_out_text(), repeats=10)
    assert report.bytes_per_second > 0
    assert report.tokens_per_second > 0
