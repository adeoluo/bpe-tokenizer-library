"""BPE Tokenizer Library - a byte-level byte-pair-encoding tokenizer from scratch."""

from __future__ import annotations

from bpe_tokenizer_library.bytelevel import bytes_to_unicode, decode_bytes, encode_bytes
from bpe_tokenizer_library.data import held_out_text, sample_corpus
from bpe_tokenizer_library.evaluation import (
    CompressionReport,
    ThroughputReport,
    compression,
    encode_throughput,
    is_lossless,
    round_trip_rate,
)
from bpe_tokenizer_library.pretokenize import pretokenize
from bpe_tokenizer_library.tokenizer import BPETokenizer
from bpe_tokenizer_library.trainer import train_bpe

__version__ = "0.1.0"

__all__ = [
    "BPETokenizer",
    "CompressionReport",
    "ThroughputReport",
    "__version__",
    "bytes_to_unicode",
    "compression",
    "decode_bytes",
    "encode_bytes",
    "encode_throughput",
    "held_out_text",
    "is_lossless",
    "pretokenize",
    "round_trip_rate",
    "sample_corpus",
    "train_bpe",
]
