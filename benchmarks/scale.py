"""A large-workload scaling benchmark for the byte-level BPE tokenizer.

Trains once, then encodes held-out passages of increasing size and reports
wall-clock time, encode throughput, peak memory, and the lossless round-trip
check. This is a pure-Python encoder, so the throughput measures how the
implementation scales, not competitive wall-clock speed against a compiled
tokenizer such as tiktoken. Deterministic and offline; run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import time
import tracemalloc

from bpe_tokenizer_library.data import held_out_text, sample_corpus
from bpe_tokenizer_library.tokenizer import BPETokenizer


def main() -> None:
    tokenizer = BPETokenizer.train(sample_corpus(), vocab_size=600)
    passage = held_out_text()
    print(
        "BPE encode at scale (pure-Python; throughput measures scaling, "
        "not competitive wall-clock speed against a compiled tokenizer):"
    )
    for multiplier in (200, 1000, 4000, 10000):
        text = passage * multiplier
        byte_length = len(text.encode("utf-8"))
        tracemalloc.start()
        start = time.perf_counter()
        ids = tokenizer.encode(text)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        lossless = tokenizer.decode(ids) == text
        throughput = byte_length / elapsed if elapsed else 0.0
        print(
            f"  {byte_length:>9,} bytes: {elapsed:6.2f}s "
            f"({throughput:11,.0f} bytes/sec), {len(ids):>8,} tokens, "
            f"peak {peak / (1024 * 1024):6.1f} MiB, lossless={lossless}"
        )


if __name__ == "__main__":
    main()
