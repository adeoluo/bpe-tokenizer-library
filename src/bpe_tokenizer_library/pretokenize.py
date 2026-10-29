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
_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into GPT-2 style chunks."""

    return _PATTERN.findall(text)
