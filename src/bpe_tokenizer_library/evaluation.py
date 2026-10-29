"""Evaluation: compression, lossless round-trip, and encode throughput."""

from __future__ import annotations

import time
from dataclasses import dataclass

from bpe_tokenizer_library.tokenizer import BPETokenizer

# A battery of strings a lossless tokenizer must round-trip exactly, including
# emoji, accented text, non-Latin scripts, control whitespace, and raw digits.
ROUND_TRIP_CASES = (
    "the quick brown fox",
    "Hello, World! 123",
    "café naïve résumé",
    "emoji: 😀🚀🧠 end",
    "日本語のテキスト",
    "tabs\tand\nnewlines",
    "   leading and trailing   ",
    "",
)


@dataclass(frozen=True)
class CompressionReport:
    """Compression statistics for a tokenizer on a text."""

    byte_length: int
    token_count: int

    @property
    def bytes_per_token(self) -> float:
        return self.byte_length / self.token_count if self.token_count else 0.0


def compression(tokenizer: BPETokenizer, text: str) -> CompressionReport:
    """Return the byte length and token count of ``text`` under ``tokenizer``."""

    return CompressionReport(
        byte_length=len(text.encode("utf-8")), token_count=len(tokenizer.encode(text))
    )


def round_trip_rate(tokenizer: BPETokenizer, cases: tuple[str, ...] = ROUND_TRIP_CASES) -> float:
    """Return the fraction of ``cases`` the tokenizer decodes back exactly."""

    exact = sum(1 for case in cases if tokenizer.decode(tokenizer.encode(case)) == case)
    return exact / len(cases)


def is_lossless(tokenizer: BPETokenizer, cases: tuple[str, ...] = ROUND_TRIP_CASES) -> bool:
    """Return whether every case round-trips exactly."""

    return round_trip_rate(tokenizer, cases) == 1.0


@dataclass(frozen=True)
class ThroughputReport:
    """Encode throughput over a corpus."""

    bytes_per_second: float
    tokens_per_second: float


def encode_throughput(tokenizer: BPETokenizer, text: str, *, repeats: int = 50) -> ThroughputReport:
    """Measure encode throughput by encoding ``text`` ``repeats`` times."""

    total_bytes = len(text.encode("utf-8")) * repeats
    start = time.perf_counter()
    total_tokens = 0
    for _ in range(repeats):
        total_tokens += len(tokenizer.encode(text))
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        elapsed = 1e-9
    return ThroughputReport(
        bytes_per_second=total_bytes / elapsed,
        tokens_per_second=total_tokens / elapsed,
    )
