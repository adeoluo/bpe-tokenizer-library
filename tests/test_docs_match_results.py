"""The compression table in the README is what training on the bundled corpus gives today."""

from __future__ import annotations

from pathlib import Path

import pytest

from bpe_tokenizer_library.data import held_out_text, sample_corpus
from bpe_tokenizer_library.evaluation import compression, is_lossless
from bpe_tokenizer_library.tokenizer import BPETokenizer

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()


@pytest.mark.parametrize(
    ("vocab", "label"), [(256, "256 (no merges)"), (400, "400"), (600, "600 (requested)")]
)
def test_readme_compression_row_matches_a_fresh_run(vocab: int, label: str) -> None:
    tokenizer = BPETokenizer.train(sample_corpus(), vocab_size=vocab)
    report = compression(tokenizer, held_out_text())
    assert is_lossless(tokenizer)
    assert f"| {label} | {report.bytes_per_token:.2f} | yes |" in README
