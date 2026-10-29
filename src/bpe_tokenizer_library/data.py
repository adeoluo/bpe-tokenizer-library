"""A small built-in corpus for training and demonstrations.

The corpus is a handful of public-domain style English sentences with repetition,
enough to learn meaningful merges deterministically and offline. It is not meant
to be representative of natural language at scale; it exists so the library can be
trained and evaluated with no downloads.
"""

from __future__ import annotations

_SENTENCES = (
    "the quick brown fox jumps over the lazy dog",
    "the early bird catches the worm but the second mouse gets the cheese",
    "a journey of a thousand miles begins with a single step",
    "to be or not to be that is the question",
    "all that glitters is not gold and not all who wander are lost",
    "the only thing we have to fear is fear itself",
    "tokenization turns raw text into a sequence of tokens for a model",
    "byte pair encoding merges frequent pairs of symbols into new tokens",
    "the cat sat on the mat while the dog ran across the yard",
    "practice makes perfect so keep practicing every single day",
)


def sample_corpus(repeat: int = 20) -> list[str]:
    """Return the built-in corpus, each sentence repeated to give merges signal."""

    return [sentence for sentence in _SENTENCES for _ in range(repeat)]


def held_out_text() -> str:
    """Return a short held-out passage not identical to any training sentence."""

    return (
        "the quick dog and the lazy fox wander across the yard every day, "
        "turning raw text into tokens for the model to read."
    )
