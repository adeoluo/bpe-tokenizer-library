"""Pre-tokenization: split text into word-like chunks before byte-level BPE.

Byte-pair encoding is applied within chunks, never across them, so the chunker
decides where merges can happen. This uses the GPT-2 regular expression, which
keeps leading spaces attached to the following word, groups letters and digits
separately, and isolates punctuation. Only the standard-library ``re`` module is
used.
"""

from __future__ import annotations

import re

# The GPT-2 pre-tokenization pattern: contractions, letters, digits, other
# non-space runs, and trailing whitespace, each with an optional leading space.
#
# The "other" class must cover every non-whitespace character that is neither a
# letter nor a digit. GPT-2 expresses this with the Unicode property class
# ``[^\s\p{L}\p{N}]``. The standard-library ``re`` module has no ``\p{...}``
# support, so the class is approximated. The subtlety is the underscore: Python
# counts ``_`` as a word character (``\w``), so it is excluded from the letter
# class ``[^\W\d_]`` and, unless handled explicitly, also from the naive "other"
# class ``[^\s\w]``. That gap silently dropped underscores (and any other ``\w``
# character that is neither a letter nor a digit) during pre-tokenization, which
# broke lossless round-tripping. Adding ``_`` to the "other" alternation closes
# the gap so that every character falls into exactly one branch.
_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into GPT-2 style chunks."""

    return _PATTERN.findall(text)
