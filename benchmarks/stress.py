"""A production-style stress benchmark for the tokenizer service.

Trains once, then drives the /encode handler under sustained, concurrent load and
reports throughput with p50, p95, and p99 latency. Deterministic and offline; run
with:

    uv run python benchmarks/stress.py
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

from bpe_tokenizer_library.data import sample_corpus
from bpe_tokenizer_library.service import TokenizerService, handle


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    service = TokenizerService.trained()
    rng = random.Random(0)
    corpus = sample_corpus(repeat=1)
    requests = [{"text": rng.choice(corpus)} for _ in range(10000)]

    def one(body: dict[str, object]) -> float:
        start = time.perf_counter()
        handle("POST", "/encode", body, service)
        return (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        latencies = list(pool.map(one, requests))
    total_s = time.perf_counter() - start

    print(f"Requests: {len(requests)} | Workers: 8")
    print(f"Throughput: {len(requests) / total_s:.0f} req/sec")
    print(
        "Latency ms  "
        f"p50 {_percentile(latencies, 0.50):.3f}  "
        f"p95 {_percentile(latencies, 0.95):.3f}  "
        f"p99 {_percentile(latencies, 0.99):.3f}"
    )


if __name__ == "__main__":
    main()
