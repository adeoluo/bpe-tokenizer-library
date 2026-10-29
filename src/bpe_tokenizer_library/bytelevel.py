"""The reversible byte-to-unicode mapping used by GPT-2 style byte-level BPE.

Byte-level BPE operates on all 256 byte values, but several of them are control
or whitespace characters that are awkward to carry around as text. GPT-2 solves
this with a fixed, reversible mapping from bytes to a set of printable Unicode
code points. Encoding any text as UTF-8 bytes and then through this mapping makes
the tokenizer lossless for arbitrary input, including emoji and non-English text,
which tokenizers that split on characters cannot guarantee.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Return GPT-2's reversible map from the 256 byte values to printable glyphs."""

    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapping = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapping.append(256 + extra)
            extra += 1
    return {byte: chr(code) for byte, code in zip(printable, mapping, strict=True)}


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    """Return the inverse of :func:`bytes_to_unicode`."""

    return {glyph: byte for byte, glyph in bytes_to_unicode().items()}


def encode_bytes(text: str) -> str:
    """Map a string to its byte-level glyph representation."""

    table = bytes_to_unicode()
    return "".join(table[byte] for byte in text.encode("utf-8"))


def decode_bytes(glyphs: str) -> str:
    """Invert :func:`encode_bytes`, recovering the original string."""

    table = unicode_to_bytes()
    raw = bytes(table[glyph] for glyph in glyphs)
    return raw.decode("utf-8", errors="replace")
