"""Backend selection and the Python fallback, independent of the Rust extension."""

from __future__ import annotations

import pytest

from bpe_tokenizer_library import backend
from bpe_tokenizer_library.data import held_out_text, sample_corpus
from bpe_tokenizer_library.tokenizer import BPETokenizer


@pytest.fixture
def no_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "_EXTENSION", None)


def test_default_backend_is_the_fastest_available() -> None:
    expected = "rust" if backend.rust_available() else "python"
    assert BPETokenizer.train(sample_corpus(repeat=2), 300).backend == expected
    assert BPETokenizer.train(sample_corpus(repeat=2), 300, backend="python").backend == "python"


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        BPETokenizer.from_merges([], backend="cuda")  # type: ignore[arg-type]


@pytest.mark.usefixtures("no_extension")
def test_rust_backend_falls_back_to_python_without_extension() -> None:
    assert not backend.rust_available()
    assert backend.rust_core([]) is None
    assert backend.rust_train(["abc"], 300) is None
    reference = BPETokenizer.train(sample_corpus(), 500, backend="python")
    tok = BPETokenizer.train(sample_corpus(), 500, backend="rust")
    assert tok.backend == "python"
    assert tok == reference
    text = held_out_text()
    assert tok.encode(text) == reference.encode(text)
    assert tok.decode(tok.encode(text)) == text
    assert tok.encode_batch([text, ""]) == [reference.encode(text), []]


def test_rust_backend_requires_matching_unicode_version(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeExtension:
        SUPPORTED_UNICODE_VERSIONS = ("0.0.0",)

    monkeypatch.setattr(backend, "_EXTENSION", FakeExtension())
    assert not backend.rust_available()
    assert BPETokenizer.from_merges([], backend="rust").backend == "python"


def test_with_backend_preserves_merges() -> None:
    tok = BPETokenizer.train(sample_corpus(repeat=2), 320)
    other = tok.with_backend("rust")
    assert other == tok
    assert other.backend == ("rust" if backend.rust_available() else "python")
    assert other.with_backend("python").backend == "python"
