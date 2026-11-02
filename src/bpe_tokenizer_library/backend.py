"""The Rust backend, the default whenever the ``bpe_tokenizer_rs`` extension is present.

The Rust core lives in ``rust/`` and is compiled by maturin into the package as
``bpe_tokenizer_library.bpe_tokenizer_rs`` on every install. It reproduces the
Python encoder and decoder exactly. The pre-tokenizer's character classes come from the Unicode
database of the running interpreter, which changes between Python releases, so
the extension ships one table per Unicode version and is only used when it has
the table for this interpreter. In every other case the library falls back to
the Python implementation, which is always available.
"""

from __future__ import annotations

import importlib
import unicodedata
from typing import Any, Literal, Protocol

Backend = Literal["python", "rust"]
BACKENDS: tuple[Backend, ...] = ("python", "rust")


class RustCore(Protocol):
    """The interface of ``bpe_tokenizer_rs.Tokenizer``."""

    def encode(self, text: str) -> list[int]: ...

    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...

    def decode(self, ids: list[int]) -> str: ...


def _load_extension() -> Any:
    # Inside the package for normal installs; top level for an older `maturin develop`.
    for name in ("bpe_tokenizer_library.bpe_tokenizer_rs", "bpe_tokenizer_rs"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


_EXTENSION: Any = _load_extension()


def rust_available() -> bool:
    """Return whether the Rust backend can be used under this interpreter."""

    return (
        _EXTENSION is not None
        and unicodedata.unidata_version in _EXTENSION.SUPPORTED_UNICODE_VERSIONS
    )


def check_backend(backend: str) -> Backend:
    """Validate a backend name."""

    if backend == "python":
        return "python"
    if backend == "rust":
        return "rust"
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def rust_core(merges: list[tuple[str, str]]) -> RustCore | None:
    """Build a Rust tokenizer for ``merges``, or return ``None`` if unavailable."""

    if not rust_available():
        return None
    core: RustCore = _EXTENSION.Tokenizer(merges, unicodedata.unidata_version)
    return core


def rust_train(corpus: list[str], vocab_size: int) -> list[tuple[str, str]] | None:
    """Train merges with the Rust core, or return ``None`` if unavailable."""

    if not rust_available():
        return None
    merges: list[tuple[str, str]] = _EXTENSION.train_bpe(
        corpus, vocab_size, unicodedata.unidata_version
    )
    return merges
