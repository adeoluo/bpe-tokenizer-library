"""Evaluation: compression, lossless round-trip, and encode throughput."""

from __future__ import annotations

import time
from dataclasses import dataclass
from random import Random

from bpe_tokenizer_library.pretokenize import pretokenize
from bpe_tokenizer_library.tokenizer import BPETokenizer

# A battery of strings a lossless tokenizer must round-trip exactly, including
# emoji, accented text, non-Latin scripts, control whitespace, raw digits, and
# underscores (a regression guard for the pre-tokenizer dropping word punctuation).
ROUND_TRIP_CASES = (
    "the quick brown fox",
    "Hello, World! 123",
    "café naïve résumé",
    "emoji: 😀🚀🧠 end",
    "日本語のテキスト",
    "tabs\tand\nnewlines",
    "   leading and trailing   ",
    "snake_case_name and __dunder__",
    "a_b _ _x_ trailing_",
    "path/to_file.ext?q=1&y=2",
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


def random_unicode_cases(
    count: int = 500, *, seed: int = 0, max_length: int = 16, max_codepoint: int = 0x2FFF
) -> tuple[str, ...]:
    """Generate deterministic random Unicode strings for adversarial testing.

    Codepoints span Latin, punctuation, accented, and CJK ranges. Surrogate
    halves are filtered because they are not valid scalar values. The same
    ``seed`` always yields the same corpus, so failures are reproducible.
    """

    rng = Random(seed)
    cases: list[str] = []
    while len(cases) < count:
        length = rng.randint(0, max_length)
        raw = "".join(chr(rng.randint(0, max_codepoint)) for _ in range(length))
        scalar = raw.encode("utf-8", "ignore").decode("utf-8")
        cases.append(scalar)
    return tuple(cases)


def pretokenize_is_lossless(cases: tuple[str, ...]) -> bool:
    """Return whether concatenating the pre-tokenizer's chunks rebuilds each input.

    This is the property the pre-tokenizer must satisfy for the tokenizer to be
    lossless: the chunker may regroup characters but must never drop or add any.
    """

    return all("".join(pretokenize(case)) == case for case in cases)


def fuzz_round_trip_rate(tokenizer: BPETokenizer, cases: tuple[str, ...]) -> float:
    """Return the fraction of adversarial ``cases`` the tokenizer decodes exactly."""

    if not cases:
        return 1.0
    exact = sum(1 for case in cases if tokenizer.decode(tokenizer.encode(case)) == case)
    return exact / len(cases)


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
