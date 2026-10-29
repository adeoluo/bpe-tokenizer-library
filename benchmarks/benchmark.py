"""A deterministic micro-benchmark for the BPE tokenizer.

Trains tokenizers at several vocabulary sizes and reports compression on a
held-out passage, lossless round-trip, and encode throughput. Deterministic and
offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from bpe_tokenizer_library.data import held_out_text, sample_corpus
from bpe_tokenizer_library.evaluation import compression, encode_throughput, is_lossless
from bpe_tokenizer_library.tokenizer import BPETokenizer


def main() -> None:
    corpus = sample_corpus()
    text = held_out_text()
    print("Compression on a held-out passage by vocabulary size:")
    for vocab in (256, 400, 600, 1000):
        start = time.perf_counter()
        tok = BPETokenizer.train(corpus, vocab_size=vocab)
        train_ms = (time.perf_counter() - start) * 1000
        report = compression(tok, text)
        print(
            f"  vocab {vocab:5d}: {report.bytes_per_token:.2f} bytes/token "
            f"({report.token_count} tokens), lossless={is_lossless(tok)}, "
            f"trained in {train_ms:.0f} ms"
        )
    tok = BPETokenizer.train(corpus, vocab_size=600)
    tp = encode_throughput(tok, text, repeats=200)
    print(
        f"Encode throughput (vocab 600): {tp.bytes_per_second:,.0f} bytes/sec, "
        f"{tp.tokens_per_second:,.0f} tokens/sec"
    )


if __name__ == "__main__":
    main()
