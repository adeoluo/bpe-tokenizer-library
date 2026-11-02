"""Encode throughput: Python backend vs Rust backend vs tiktoken.

Compares single-threaded encode throughput in bytes per second on the same
input. The Python and Rust backends use the same merges and produce identical
token ids. tiktoken runs its own GPT-2 vocabulary (``encode_ordinary``), so it is
included as a throughput reference only: its token ids and vocabulary differ.

Requires the ``rust`` dependency group (for tiktoken); installing builds the extension:

    uv sync --group rust
    uv run python benchmarks/compare.py

Each figure is the best of several timed runs after a warm-up run.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from random import Random

from bpe_tokenizer_library.backend import rust_available
from bpe_tokenizer_library.data import sample_corpus
from bpe_tokenizer_library.evaluation import random_unicode_cases
from bpe_tokenizer_library.tokenizer import BPETokenizer

TARGET_BYTES = 1_000_000
RUNS = 5


def _english() -> str:
    rng = Random(0)
    sentences = sample_corpus(repeat=1)
    parts: list[str] = []
    size = 0
    while size < TARGET_BYTES:
        sentence = rng.choice(sentences).capitalize() + rng.choice((". ", "! ", "? ", ".\n"))
        parts.append(sentence)
        size += len(sentence.encode("utf-8"))
    return "".join(parts)


def _unicode() -> str:
    parts: list[str] = []
    size = 0
    seed = 0
    while size < TARGET_BYTES:
        chunk = " ".join(random_unicode_cases(count=1000, seed=seed, max_codepoint=0x10FFFF))
        parts.append(chunk)
        size += len(chunk.encode("utf-8"))
        seed += 1
    return " ".join(parts)


def _best_rate(encode: Callable[[str], object], text: str) -> float:
    encode(text)
    size = len(text.encode("utf-8"))
    best = float("inf")
    for _ in range(RUNS):
        start = time.perf_counter()
        encode(text)
        best = min(best, time.perf_counter() - start)
    return size / best


def main() -> None:
    if not rust_available():
        raise SystemExit("build the extension first: uv run maturin develop --uv --release")
    import tiktoken

    merges = BPETokenizer.train(sample_corpus(), 1000, backend="rust").merges
    python = BPETokenizer.from_merges(merges)
    rust = BPETokenizer.from_merges(merges, backend="rust")
    gpt2 = tiktoken.get_encoding("gpt2")

    print(f"{platform.machine()} {platform.system()}, Python {platform.python_version()}")
    print(f"vocab: ours {python.vocab_size} (trained on the bundled corpus), tiktoken gpt2")
    print("| Workload | Python (MB/s) | Rust (MB/s) | tiktoken gpt2 (MB/s) | Rust / Python |")
    print("|---|---:|---:|---:|---:|")
    for name, text in (("English text, 1 MB", _english()), ("Random Unicode, 1 MB", _unicode())):
        assert rust.encode(text) == python.encode(text)
        py_rate = _best_rate(python.encode, text)
        rs_rate = _best_rate(rust.encode, text)
        tk_rate = _best_rate(gpt2.encode_ordinary, text)
        print(
            f"| {name} | {py_rate / 1e6:.2f} | {rs_rate / 1e6:.2f} | {tk_rate / 1e6:.2f} "
            f"| {rs_rate / py_rate:.0f}x |"
        )


if __name__ == "__main__":
    main()
